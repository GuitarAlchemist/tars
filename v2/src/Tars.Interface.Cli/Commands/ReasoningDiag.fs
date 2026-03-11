module Tars.Interface.Cli.Commands.ReasoningDiag

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Configuration
open Serilog
open Tars.Core
open Tars.Cortex.Patterns
open Tars.Knowledge
open Tars.Interface.Cli.Commands.AgentHelpers

/// Workflow/Tree/Graph of Thoughts diagnostics with trace + ledger persistence.
[<RequireQualifiedAccess>]
type ReasoningMode =
    | WorkflowOfThoughts
    | TreeOfThoughts
    | GraphOfThoughts

let private modeLabel =
    function
    | ReasoningMode.WorkflowOfThoughts -> "wot"
    | ReasoningMode.TreeOfThoughts -> "tot"
    | ReasoningMode.GraphOfThoughts -> "got"

let private tryParseReasoningMode (value: string) =
    match value.Trim().ToLowerInvariant() with
    | "wot"
    | "workflow"
    | "workflow-of-thoughts"
    | "workflowofthoughts" -> Some ReasoningMode.WorkflowOfThoughts
    | "tot"
    | "tree"
    | "tree-of-thoughts"
    | "treeofthoughts" -> Some ReasoningMode.TreeOfThoughts
    | "got"
    | "graph"
    | "graph-of-thoughts"
    | "graphofthoughts" -> Some ReasoningMode.GraphOfThoughts
    | _ -> None

type ReasoningArgs =
    { Mode: ReasoningMode
      Goal: string
      EvidencePath: string
      UseLedger: bool
      AutoRetry: bool }

let private defaultEvidenceDir =
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "diagnostics", "reasoning")

let private defaultEvidencePath (mode: ReasoningMode) =
    Directory.CreateDirectory(defaultEvidenceDir) |> ignore

    let fileName =
        $"tars-reasoning-{modeLabel mode}-{DateTime.UtcNow:yyyyMMddHHmmss}.json"

    Path.Combine(defaultEvidenceDir, fileName)

let private truncate (value: string) (max: int) =
    if String.IsNullOrWhiteSpace(value) then ""
    elif value.Length <= max then value
    else value.Substring(0, max) + "..."

let private splitGoalAndOptions (tokens: string list) =
    let rec loop (acc: string list) (remaining: string list) =
        match remaining with
        | [] -> List.rev acc, []
        | head :: _ when head.StartsWith("--") -> List.rev acc, remaining
        | head :: tail -> loop (head :: acc) tail

    loop [] tokens

let private parseOptions (tokens: string list) : Result<bool * string option * bool, string> =
    let rec loop remaining (useLedger, evidencePath, autoRetry) =
        match remaining with
        | [] -> Result.Ok(useLedger, evidencePath, autoRetry)
        | "--ledger" :: rest -> loop rest (true, evidencePath, autoRetry)
        | "--no-ledger" :: rest -> loop rest (false, evidencePath, autoRetry)
        | "--evidence" :: path :: rest -> loop rest (useLedger, Some path, autoRetry)
        | "--evidence" :: [] -> Result.Error "Missing path after --evidence"
        | "--auto-retry" :: rest -> loop rest (useLedger, evidencePath, true)
        | unknown :: _ -> Result.Error $"Unknown option '{unknown}'"

    loop tokens (true, None, false)

let parseArgs (args: string list) : Result<ReasoningArgs, string> =
    match args with
    | modeToken :: rest ->
        match tryParseReasoningMode modeToken with
        | None -> Result.Error $"Unknown reasoning mode '{modeToken}'. Use wot/tot/got."
        | Some mode ->
            let goalTokens, optionTokens = splitGoalAndOptions rest

            if List.isEmpty goalTokens then
                Result.Error "Please provide a diagnostic goal after the reasoning mode."
            else
                match parseOptions optionTokens with
                | Result.Error err -> Result.Error err
                | Result.Ok(useLedger, evidencePathOpt, autoRetry) ->
                    let evidencePath =
                        evidencePathOpt |> Option.defaultWith (fun () -> defaultEvidencePath mode)

                    Result.Ok
                        { Mode = mode
                          Goal = (goalTokens |> String.concat " ").Trim()
                          EvidencePath = evidencePath
                          UseLedger = useLedger
                          AutoRetry = autoRetry }
    | [] -> Result.Error "Missing reasoning mode. Please use 'wot', 'tot', or 'got'."

let private ledgerFromConfig (logger: ILogger) (config: TarsConfig) =
    let tryPostgres =
        config.Memory.PostgresConnectionString
        |> Option.orElse (Environment.GetEnvironmentVariable("TARS_POSTGRES_CONNECTION") |> Option.ofObj)

    match tryPostgres with
    | Some connStr ->
        try
            KnowledgeLedger(PostgresLedgerStorage.createWithConnectionString connStr)
        with ex ->
            logger.Warning("Postgres ledger unavailable ({Message}); falling back to in-memory.", [| box ex.Message |])
            KnowledgeLedger.createInMemory ()
    | None -> KnowledgeLedger.createInMemory ()

let private statusTagFor outcome =
    match outcome with
    | Success _ -> "success"
    | PartialSuccess _ -> "partial"
    | Failure _ -> "failure"

let recordTraceToLedger
    (ledger: KnowledgeLedger)
    (trace: Trace)
    (mode: ReasoningMode)
    (goal: string)
    (tracePath: string option)
    (outcome: ExecutionOutcome<string>)
    (auditSummary: string option)
    : Task<Result<BeliefId, string>> =
    async {
        let runId = RunId.New()

        let provenance =
            { Provenance.FromRun(runId, AgentId.System) with
                Confidence =
                    match outcome with
                    | Success _ -> 0.95
                    | PartialSuccess _ -> 0.7
                    | Failure _ -> 0.25 }

        let outcomeSummary =
            match outcome with
            | Success value -> $"success: {truncate value 120}"
            | PartialSuccess(value, warnings) ->
                let warningsTxt =
                    warnings |> List.map (fun w -> $"%A{w}") |> String.concat "; "

                $"partial: {truncate value 80} | warnings: {warningsTxt}"
            | Failure errors ->
                let errorTxt = errors |> List.map (fun e -> $"%A{e}") |> String.concat "; "

                $"failure: {truncate errorTxt 120}"

        let pathTag =
            match tracePath with
            | Some path -> [ $"tracePath:{path}" ]
            | None -> []

        let baseTags =
            [ $"mode:{modeLabel mode}"
              $"status:{statusTagFor outcome}"
              $"summary:{truncate outcomeSummary 120}"
              $"goal:{truncate goal 80}"
              $"events:{trace.Events.Length}"
              $"traceId:{trace.Id}" ]
            @ pathTag

        let tags =
            match auditSummary with
            | Some summary -> baseTags @ [ $"audit:{summary}" ]
            | None -> baseTags

        let subject = $"diag:{modeLabel mode}:{trace.Id}"
        let predicate = RelationType.Mentions
        let obj = truncate goal 160

        let belief =
            { Belief.create subject predicate obj provenance with
                Tags = tags }

        let! result = ledger.Assert(belief, AgentId.System, runId) |> Async.AwaitTask
        return result
    }
    |> Async.StartAsTask

let private defaultGoTConfigFor mode =
    match mode with
    | ReasoningMode.WorkflowOfThoughts -> defaultWoTConfig.BaseConfig
    | _ -> defaultGoTConfig

let private relaxGoTConfig (config: GoTConfig) =
    { config with
        ScoreThreshold = min config.ScoreThreshold 0.2
        MinConfidence = min config.MinConfidence 0.25
        TopK = max config.TopK 3
        BranchingFactor = max config.BranchingFactor 4
        MaxDepth = max config.MaxDepth 4 }

let private buildWorkflow mode llm goal config =
    match mode with
    | ReasoningMode.WorkflowOfThoughts ->
        workflowOfThought
            llm
            { defaultWoTConfig with
                BaseConfig = config }
            goal
    | ReasoningMode.TreeOfThoughts -> treeOfThoughts llm config goal
    | ReasoningMode.GraphOfThoughts -> graphOfThoughts llm config goal

let private printUsage () =
    printfn "Usage: tars diag reasoning <wot|tot|got> <goal> [--ledger|--no-ledger] [--evidence <path>] [--auto-retry]"
    printfn "  --ledger     (default) persist traces to the knowledge ledger"
    printfn "  --no-ledger  skip ledger writes (still emits evidence)"
    printfn "  --evidence <path>  override the trace output file"
    printfn "  --auto-retry rerun with relaxed thresholds when heuristic fallback is used"

let private retryEvidencePath (path: string) =
    if Path.HasExtension(path) then
        let directory = Path.GetDirectoryName(path)
        let fileName = Path.GetFileNameWithoutExtension(path)
        let extension = Path.GetExtension(path)
        let retryName = fileName + "-retry" + extension

        if String.IsNullOrWhiteSpace directory then
            retryName
        else
            Path.Combine(directory, retryName)
    else
        path

let private normalizeOutcome (log: string -> unit) (outcome: ExecutionOutcome<string>) =
    let isNoSolution (value: string) =
        let trimmed = value.Trim()

        String.IsNullOrWhiteSpace trimmed
        || trimmed.Equals("No solution found.", StringComparison.OrdinalIgnoreCase)
        || trimmed.Equals("No solution found", StringComparison.OrdinalIgnoreCase)

    match outcome with
    | Success answer when isNoSolution answer ->
        log "Treating empty/no-solution answer as failure."
        Failure [ PartialFailure.Error "No solution found." ]
    | PartialSuccess(answer, warnings) when isNoSolution answer ->
        log "Treating empty/no-solution answer as failure."
        Failure(warnings @ [ PartialFailure.Error "No solution found." ])
    | _ -> outcome

let run (logger: ILogger) (config: IConfiguration) (tarsConfig: TarsConfig) (args: string list) =
    async {
        match parseArgs args with
        | Result.Error err ->
            printfn $"Error: %s{err}"
            printUsage ()
            return 1
        | Result.Ok opts ->
            let! ledgerOpt =
                if opts.UseLedger then
                    async {
                        let ledger = ledgerFromConfig logger tarsConfig
                        do! ledger.Initialize() |> Async.AwaitTask
                        return Some ledger
                    }
                else
                    async { return None }

            match createLlmService config with
            | Result.Error msg ->
                logger.Error("LLM configuration error: {Message}", [| box msg |])
                return 1
            | Result.Ok(llm, _) ->
                let log msg =
                    logger.Information("[ReasoningDiag] {Message}", [| box msg |])

                let baseConfig = defaultGoTConfigFor opts.Mode

                let runDiagnostic label evidencePath config =
                    async {
                        let options =
                            { defaultOptions with
                                EvidencePath = Some evidencePath }

                        let llmWithEvidence, traceHandle = attachEvidence label llm options

                        let! traceEvidence =
                            match traceHandle with
                            | Some handle ->
                                async {
                                    let! value = handle |> Async.AwaitTask
                                    return Some value
                                }
                            | None -> async { return None }

                        let reasoningAudit = ReasoningAudit.create ()
                        let ctx = createAgentContext log llmWithEvidence (Some reasoningAudit)
                        let workflow = buildWorkflow opts.Mode llmWithEvidence opts.Goal config
                        let! outcome = workflow ctx
                        let outcome = normalizeOutcome log outcome

                        let auditRecords = ReasoningAudit.snapshot reasoningAudit

                        let auditSummary =
                            if auditRecords.IsEmpty then
                                None
                            else
                                Some(ReasoningAudit.summary reasoningAudit)

                        match auditSummary with
                        | Some summary -> log $"Decision audit: {summary}"
                        | None -> ()

                        let auditStats = ReasoningAudit.stats reasoningAudit

                        let traceInfo =
                            match traceEvidence with
                            | Some(recorder, path) -> Some(recorder, path)
                            | None -> None

                        match traceInfo with
                        | Some(recorder, path) ->
                            do! (recorder :> ITraceRecorder).EndTraceAsync()
                            do! recorder.SaveToFileAsync(path)
                            log $"Evidence saved to {path}"
                        | None -> ()

                        let! traceOpt =
                            match traceInfo with
                            | Some(recorder, _) ->
                                async {
                                    let! trace = (recorder :> ITraceRecorder).GetTraceAsync()
                                    return trace
                                }
                            | None -> async { return None }

                        let tracePathOpt = traceInfo |> Option.map snd
                        return outcome, auditSummary, auditStats, traceOpt, tracePathOpt
                    }

                let persistLedger outcome auditSummary traceOpt tracePathOpt =
                    async {
                        match ledgerOpt, traceOpt with
                        | Some ledger, Some trace ->
                            let! result =
                                recordTraceToLedger ledger trace opts.Mode opts.Goal tracePathOpt outcome auditSummary
                                |> Async.AwaitTask

                            match result with
                            | Result.Ok beliefId -> log $"Ledger belief recorded: {beliefId}"
                            | Result.Error err ->
                                // Detect PostgreSQL schema mismatch (error code 42703 = undefined column)
                                if err.Contains("42703") && err.Contains("created_by") then
                                    logger.Warning(
                                        "Belief persistence skipped: PostgreSQL schema mismatch detected. "
                                        + "The 'beliefs' table is missing the 'created_by' column. "
                                        + "Run 'dotnet run --project src/Tars.Interface.Cli -- init-db' to update schema, "
                                        + "or execute: ALTER TABLE beliefs ADD COLUMN created_by TEXT NOT NULL DEFAULT 'system';"
                                    )
                                elif err.Contains("42703") then
                                    logger.Warning(
                                        "Belief persistence skipped: PostgreSQL column not found. "
                                        + "Schema may be outdated. Error: {Message}. "
                                        + "Consider running schema migrations.",
                                        [| box err |]
                                    )
                                else
                                    logger.Warning("Failed to persist reasoning diag belief: {Message}", [| box err |])
                        | _ -> ()
                    }

                let reportOutcome outcome =
                    match outcome with
                    | Success answer ->
                        log $"Reasoning succeeded. Answer: {answer}"
                        0
                    | PartialSuccess(answer, warnings) ->
                        log $"Partial success. Answer: {answer}"

                        if not (List.isEmpty warnings) then
                            for warn in warnings do
                                printfn $"[ReasoningDiag Warning] %A{warn}"

                        0
                    | Failure errors ->
                        log "Reasoning failed."

                        for err in errors do
                            printfn $"[ReasoningDiag Error] %A{err}"

                        1

                let! outcome, auditSummary, auditStats, traceOpt, tracePathOpt =
                    runDiagnostic $"diag-{modeLabel opts.Mode}" opts.EvidencePath baseConfig

                do! persistLedger outcome auditSummary traceOpt tracePathOpt

                let shouldRetry =
                    opts.AutoRetry
                    && (match outcome with
                        | Failure _ -> true
                        | _ -> false)
                    && (auditStats.HeuristicFallbacks > 0 || auditStats.ParseFailures > 0)

                if shouldRetry then
                    log "Auto-retry enabled: retrying with relaxed thresholds after heuristic fallback."
                    let retryPath = retryEvidencePath opts.EvidencePath
                    let relaxedConfig = relaxGoTConfig baseConfig

                    let! retryOutcome, retrySummary, _, retryTraceOpt, retryTracePathOpt =
                        runDiagnostic $"diag-{modeLabel opts.Mode}-retry" retryPath relaxedConfig

                    do! persistLedger retryOutcome retrySummary retryTraceOpt retryTracePathOpt

                    return reportOutcome retryOutcome
                else
                    return reportOutcome outcome
    }
    |> Async.StartAsTask
