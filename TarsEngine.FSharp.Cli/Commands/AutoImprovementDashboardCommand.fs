namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open System.Globalization
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Commands.CommandResult

[<CLIMutable>]
type ConsensusAgentSnapshot =
    { agentId: string
      role: string
      outcome: string
      confidence: float option
      notes: string option
      producedAt: DateTime }

[<CLIMutable>]
type ConsensusSnapshot =
    { status: string
      message: string option
      agents: ConsensusAgentSnapshot list }

[<CLIMutable>]
type CriticSnapshot =
    { status: string
      message: string option }

[<CLIMutable>]
type ConsensusRuleSnapshot =
    { minimumPassCount: int
      requiredRoles: string list
      allowNeedsReview: bool
      minimumConfidence: float option
      maxFailureCount: int option }

[<CLIMutable>]
type PolicySnapshot =
    { requireConsensus: bool
      requireCritic: bool
      stopOnFailure: bool
      captureLogs: bool
      patchCommands: int
      validationCommands: int
      benchmarkCommands: int
      hasAgentProvider: bool
      hasTraceProvider: bool
      hasFeedbackSink: bool
      hasReasoningCritic: bool
      consensusRule: ConsensusRuleSnapshot option }

[<CLIMutable>]
type PolicyDeltaSnapshot =
    { field: string
      previousValue: string
      currentValue: string
      rationale: string option }

[<CLIMutable>]
type AgentFeedbackSnapshot =
    { agentId: string
      role: string
      verdict: string
      detail: string option
      confidence: float option
      notes: string option
      suggestedActions: string list
      recordedAt: DateTime }

[<CLIMutable>]
type CommandSnapshot =
    { Name: string
      Stage: string
      ExitCode: int
      DurationSeconds: float
      StartedAt: DateTime
      CompletedAt: DateTime }

[<CLIMutable>]
type HarnessSnapshot =
    { status: string
      failureReason: string option
      failedCommandCount: int option }

[<CLIMutable>]
type GovernanceLedgerEntry =
    { runId: Guid
      timestamp: DateTime
      specId: string
      specPath: string
      description: string option
      status: string
      consensus: ConsensusSnapshot option
      critic: CriticSnapshot option
      policyBefore: PolicySnapshot
      policyAfter: PolicySnapshot
      policyChanges: PolicyDeltaSnapshot list
      agentFeedback: AgentFeedbackSnapshot list
      harness: HarnessSnapshot option
      artifacts: string list
      commands: CommandSnapshot list
      nextSteps: string list
      metrics: Dictionary<string, JsonElement> option
      inferenceTelemetry: Dictionary<string, JsonElement> option
      appendPath: string }

type LedgerRecord =
    { FilePath: string
      FileName: string
      Entry: GovernanceLedgerEntry }

type CommandMode =
    | Overview of refreshSeconds: int option
    | RunDetail of runId: string option

type AutoImprovementDashboardCommand(logger: ILogger<AutoImprovementDashboardCommand>) =

    let serializerOptions =
        lazy (
            let options = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
            options
        )

    let escape (text: string) = Markup.Escape(text)

    let tryClearConsole () =
        try
            if not Console.IsOutputRedirected then
                AnsiConsole.Clear()
        with
        | :? IOException
        | :? InvalidOperationException -> ()

    let parseMode (options: CommandOptions) =
        match options.Arguments with
        | [] -> Overview None
        | first :: rest ->
            let key = first.Trim().ToLowerInvariant()

            if key = "watch" then
                match rest with
                | seconds :: _ ->
                    match Int32.TryParse seconds with
                    | true, value when value > 0 -> Overview (Some value)
                    | _ -> Overview (Some 5)
                | [] -> Overview (Some 5)
            elif key = "run" then
                match rest with
                | runId :: _ when not (String.IsNullOrWhiteSpace(runId)) -> RunDetail (Some runId)
                | _ -> RunDetail None
            else
                Overview None

    let formatStatus (status: string) =
        if String.IsNullOrWhiteSpace(status) then
            "[grey]unknown[/]"
        else
            let normalized = status.Trim().ToLowerInvariant()
            let escaped = escape status
            if normalized.StartsWith("pass") || normalized.StartsWith("success") then
                $"[green]{escaped}[/]"
            elif normalized.StartsWith("fail") || normalized.StartsWith("error") then
                $"[red]{escaped}[/]"
            elif normalized.Contains("pending") || normalized.Contains("running") then
                $"[yellow]{escaped}[/]"
            else
                $"[blue]{escaped}[/]"

    let formatConfidence (value: float option) =
        match value with
        | Some v when v >= 0.0 && v <= 1.5 ->
            // Use percentage when value looks normalized.
            let percent = Math.Round(v * 100.0, 1)
            $"{percent:F1}%%"
        | Some v ->
            v.ToString("F2")
        | None -> "—"

    let safeList (items: 'a list) =
        if obj.ReferenceEquals(items, null) then [] else items

    let tryMetric (entry: GovernanceLedgerEntry) (key: string) =
        entry.metrics
        |> Option.bind (fun metrics ->
            match metrics.TryGetValue(key) with
            | true, element -> Some element
            | _ -> None)

    let tryMetricDouble (entry: GovernanceLedgerEntry) (key: string) =
        tryMetric entry key
        |> Option.bind (fun element ->
            match element.ValueKind with
            | JsonValueKind.Number ->
                let mutable parsed = 0.0
                if element.TryGetDouble(&parsed) then Some parsed else None
            | JsonValueKind.String ->
                let text = element.GetString()
                match Double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture) with
                | true, value -> Some value
                | _ -> None
            | _ -> None)

    let formatPercent (value: float) =
        let percent = Math.Round(value * 100.0, 1)
        $"{percent:F1}%%"

    let tryTelemetry (entry: GovernanceLedgerEntry) (key: string) =
        entry.inferenceTelemetry
        |> Option.bind (fun telemetry ->
            match telemetry.TryGetValue(key) with
            | true, element -> Some element
            | _ -> None)

    let tryTelemetryDouble (entry: GovernanceLedgerEntry) (key: string) =
        tryTelemetry entry key
        |> Option.bind (fun element ->
            match element.ValueKind with
            | JsonValueKind.Number ->
                let mutable parsed = 0.0
                if element.TryGetDouble(&parsed) then Some parsed else None
            | JsonValueKind.String ->
                let text = element.GetString()
                match Double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture) with
                | true, value -> Some value
                | _ -> None
            | _ -> None)

    let tryTelemetryString (entry: GovernanceLedgerEntry) (key: string) =
        tryTelemetry entry key
        |> Option.bind (fun element ->
            let value =
                match element.ValueKind with
                | JsonValueKind.String -> element.GetString()
                | JsonValueKind.Number -> element.ToString()
                | JsonValueKind.True -> "true"
                | JsonValueKind.False -> "false"
                | JsonValueKind.Null -> null
                | _ -> element.GetRawText()

            if String.IsNullOrWhiteSpace(value) then None else Some value)

    let tryTelemetryBool (entry: GovernanceLedgerEntry) (key: string) =
        tryTelemetry entry key
        |> Option.bind (fun element ->
            match element.ValueKind with
            | JsonValueKind.True -> Some true
            | JsonValueKind.False -> Some false
            | JsonValueKind.String ->
                match element.GetString() with
                | null -> None
                | text ->
                    match text.Trim().ToLowerInvariant() with
                    | "true"
                    | "yes"
                    | "1"
                    | "gpu" -> Some true
                    | "false"
                    | "no"
                    | "0"
                    | "cpu" -> Some false
                    | _ -> None
            | JsonValueKind.Number ->
                let mutable parsed = 0.0
                if element.TryGetDouble(&parsed) then Some (Math.Abs(parsed) > Double.Epsilon) else None
            | _ -> None)

    let formatLatency (value: float) =
        if value >= 1000.0 then
            let seconds = value / 1000.0
            $"{seconds:F2}s"
        else
            let milliseconds = Math.Round(value) |> int64
            $"{milliseconds}ms"

    let formatInteger (value: float) =
        Math.Round(value)
        |> int64
        |> fun v -> v.ToString("N0", CultureInfo.InvariantCulture)

    let averageMetric (entries: GovernanceLedgerEntry list) (key: string) (count: int) =
        entries
        |> List.truncate count
        |> List.choose (fun entry -> tryMetricDouble entry key)
        |> fun values ->
            if List.isEmpty values then None else Some (values |> List.average)

    let criticRejectRate (entries: GovernanceLedgerEntry list) (count: int) =
        let statuses =
            entries
            |> List.truncate count
            |> List.choose (fun entry ->
                entry.metrics
                |> Option.bind (fun metrics ->
                    match metrics.TryGetValue("safety.critic_status") with
                    | true, element when element.ValueKind = JsonValueKind.String -> Some(element.GetString())
                    | _ -> None))
        match statuses with
        | [] -> None
        | _ ->
            let rejects =
                statuses
                |> List.filter (fun status -> status.Equals("reject", StringComparison.OrdinalIgnoreCase))
                |> List.length
            Some (float rejects / float statuses.Length)

    let formatMetricValue (element: JsonElement) =
        match element.ValueKind with
        | JsonValueKind.String -> element.GetString() |> escape
        | JsonValueKind.Number -> element.ToString() |> escape
        | JsonValueKind.True -> "true"
        | JsonValueKind.False -> "false"
        | JsonValueKind.Null -> "null"
        | _ -> element.GetRawText() |> escape

    let consensusSummary (entry: GovernanceLedgerEntry) =
        entry.consensus
        |> Option.map (fun consensus ->
            let approved =
                consensus.agents
                |> List.filter (fun agent -> agent.outcome.Equals("pass", StringComparison.OrdinalIgnoreCase))
                |> List.length

            let total = List.length consensus.agents
            let status = formatStatus consensus.status
            $"{status} [grey]({approved}/{total} approvals)[/]"
        )
        |> Option.defaultValue "[grey]n/a[/]"

    let harnessSummary (entry: GovernanceLedgerEntry) =
        entry.harness
        |> Option.map (fun harness ->
            let status = formatStatus harness.status
            let failureDetail =
                match harness.failureReason with
                | Some reason when not (String.IsNullOrWhiteSpace(reason)) -> $" [grey]({escape reason})[/]"
                | _ -> ""
            $"{status}{failureDetail}"
        )
        |> Option.defaultValue "[grey]n/a[/]"

    let loadLedgerRecords () =
        let ledgerDirectory = Path.Combine(Environment.CurrentDirectory, ".specify", "ledger", "iterations")
        if not (Directory.Exists(ledgerDirectory)) then
            []
        else
            Directory.GetFiles(ledgerDirectory, "*.json", SearchOption.TopDirectoryOnly)
            |> Array.filter (fun path -> not (path.EndsWith("latest.json", StringComparison.OrdinalIgnoreCase)))
            |> Array.choose (fun path ->
                try
                    let json = File.ReadAllText(path)
                    let entry = JsonSerializer.Deserialize<GovernanceLedgerEntry>(json, serializerOptions.Value)
                    if obj.ReferenceEquals(entry, null) then
                        None
                    else
                        Some
                            { FilePath = path
                              FileName = Path.GetFileName(path)
                              Entry = entry }
                with ex ->
                    logger.LogWarning(ex, "Failed to load governance ledger entry from {Path}", path)
                    None)
            |> Array.sortBy (fun record -> record.Entry.timestamp)
            |> Array.rev
            |> Array.toList

    let loadLedgerEntries () =
        loadLedgerRecords ()
        |> List.map (fun record -> record.Entry)

    let renderSummary (entries: GovernanceLedgerEntry list) =
        let summaryTable = Table()
        summaryTable.Title <- TableTitle("[bold]Auto-Improvement Overview[/]")
        summaryTable.AddColumn(TableColumn("[grey]Metric[/]")) |> ignore
        summaryTable.AddColumn(TableColumn("[grey]Value[/]")) |> ignore

        let totalIterations = entries.Length
        summaryTable.AddRow("Total iterations", totalIterations.ToString()) |> ignore

        let statusBreakdown =
            entries
            |> List.countBy (fun entry -> entry.status.Trim().ToLowerInvariant())
            |> List.sortByDescending snd
            |> List.map (fun (status, count) -> $"{formatStatus status} [grey]({count})[/]")
            |> String.concat "  "

        if not (String.IsNullOrWhiteSpace(statusBreakdown)) then
            summaryTable.AddRow("Status breakdown", statusBreakdown) |> ignore

        match entries with
        | latest :: _ ->
            let timestamp = latest.timestamp.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss")
            summaryTable.AddRow("Latest run", $"{escape latest.specId} @ {escape timestamp}") |> ignore
            summaryTable.AddRow("Latest status", formatStatus latest.status) |> ignore
            summaryTable.AddRow("Latest consensus", consensusSummary latest) |> ignore
            summaryTable.AddRow("Latest harness", harnessSummary latest) |> ignore
            let latestCommands = safeList latest.commands
            summaryTable.AddRow("Latest commands", latestCommands.Length.ToString()) |> ignore

            tryMetricDouble latest "capability.pass_ratio"
            |> Option.iter (fun value ->
                summaryTable.AddRow("Harness pass ratio", formatPercent value) |> ignore)

            tryMetricDouble latest "safety.consensus_avg_confidence"
            |> Option.iter (fun value ->
                summaryTable.AddRow("Consensus avg confidence", formatPercent value) |> ignore)

            averageMetric entries "capability.pass_ratio" 10
            |> Option.iter (fun value ->
                summaryTable.AddRow("Pass ratio trend (10)", formatPercent value) |> ignore)

            averageMetric entries "safety.consensus_avg_confidence" 10
            |> Option.iter (fun value ->
                summaryTable.AddRow("Consensus trend (10)", formatPercent value) |> ignore)

            criticRejectRate entries 10
            |> Option.iter (fun value ->
                summaryTable.AddRow("Critic reject trend (10)", formatPercent value) |> ignore)

            tryTelemetryDouble latest "inference.metrics.analysis_elapsed_ms"
            |> Option.iter (fun value ->
                summaryTable.AddRow("Inference latency", escape (formatLatency value)) |> ignore)

            tryTelemetryDouble latest "inference.metrics.token_count"
            |> Option.iter (fun value ->
                summaryTable.AddRow("Prompt tokens", escape (formatInteger value)) |> ignore)

            tryTelemetryDouble latest "inference.metrics.context_length"
            |> Option.iter (fun value ->
                summaryTable.AddRow("Context length", escape (formatInteger value)) |> ignore)

            tryTelemetryString latest "inference.metrics.top_terms"
            |> Option.iter (fun value ->
                let formatted = value.Replace(",", ", ")
                summaryTable.AddRow("Top terms", escape formatted) |> ignore)

            tryTelemetryBool latest "inference.metrics.used_cuda"
            |> Option.iter (fun usedCuda ->
                let deviceLabel = if usedCuda then "[green]GPU[/]" else "[yellow]CPU[/]"
                summaryTable.AddRow("Inference device", deviceLabel) |> ignore)

            averageMetric entries "inference.metrics.analysis_elapsed_ms" 10
            |> Option.iter (fun value ->
                summaryTable.AddRow("Latency trend (10)", escape (formatLatency value)) |> ignore)

            averageMetric entries "inference.metrics.token_count" 10
            |> Option.iter (fun value ->
                summaryTable.AddRow("Token trend (10)", escape (formatInteger value)) |> ignore)
        | [] -> ()

        AnsiConsole.Write(summaryTable)

    let renderRecentHistory (entries: GovernanceLedgerEntry list) =
        let historyTable = Table()
        historyTable.Title <- TableTitle("[bold]Recent Iterations[/]")
        historyTable.AddColumn(TableColumn("[grey]Run[/]")) |> ignore
        historyTable.AddColumn(TableColumn("[grey]Spec[/]")) |> ignore
        historyTable.AddColumn(TableColumn("[grey]Status[/]")) |> ignore
        historyTable.AddColumn(TableColumn("[grey]Consensus[/]")) |> ignore
        historyTable.AddColumn(TableColumn("[grey]Harness[/]")) |> ignore
        historyTable.AddColumn(TableColumn("[grey]Latency[/]")) |> ignore
        historyTable.AddColumn(TableColumn("[grey]Tokens[/]")) |> ignore

        entries
        |> List.truncate 10
        |> List.iter (fun entry ->
            let runLabel =
                entry.timestamp.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss")
                |> escape
            let latencyText =
                tryTelemetryDouble entry "inference.metrics.analysis_elapsed_ms"
                |> Option.map formatLatency
                |> Option.defaultValue "—"
            let tokenText =
                tryTelemetryDouble entry "inference.metrics.token_count"
                |> Option.map formatInteger
                |> Option.defaultValue "—"

            historyTable.AddRow(
                runLabel,
                escape entry.specId,
                formatStatus entry.status,
                consensusSummary entry,
                harnessSummary entry,
                escape latencyText,
                escape tokenText) |> ignore)

        AnsiConsole.Write(historyTable)

    let renderMetrics (latest: GovernanceLedgerEntry) =
        let metrics =
            latest.metrics
            |> Option.defaultValue (Dictionary<string, JsonElement>())

        let metricsTable = Table()
        metricsTable.Title <- TableTitle("[bold]Evolution Metrics[/]")
        metricsTable.AddColumn(TableColumn("[grey]Key[/]")) |> ignore
        metricsTable.AddColumn(TableColumn("[grey]Value[/]")) |> ignore

        if metrics.Count = 0 then
            metricsTable.AddRow("[grey]No metrics recorded[/]", "") |> ignore
        else
            metrics
            |> Seq.sortBy (fun kv -> kv.Key)
            |> Seq.iter (fun kv ->
                metricsTable.AddRow(escape kv.Key, formatMetricValue kv.Value) |> ignore)

        AnsiConsole.Write(metricsTable)

    let renderAgentFeedback (latest: GovernanceLedgerEntry) =
        let feedback = latest.agentFeedback |> safeList |> List.truncate 8

        let feedbackTable = Table()
        feedbackTable.Title <- TableTitle("[bold]Agent Feedback[/]")
        feedbackTable.AddColumn(TableColumn("[grey]Agent[/]")) |> ignore
        feedbackTable.AddColumn(TableColumn("[grey]Role[/]")) |> ignore
        feedbackTable.AddColumn(TableColumn("[grey]Verdict[/]")) |> ignore
        feedbackTable.AddColumn(TableColumn("[grey]Confidence[/]")) |> ignore
        feedbackTable.AddColumn(TableColumn("[grey]Notes[/]")) |> ignore

        if List.isEmpty feedback then
            feedbackTable.AddRow("[grey]No agent feedback[/]", "", "", "", "") |> ignore
        else
            feedback
            |> List.iter (fun item ->
                let verdict = formatStatus item.verdict
                let notes =
                    match item.notes with
                    | Some value when not (String.IsNullOrWhiteSpace(value)) -> escape value
                    | _ -> "—"

                feedbackTable.AddRow(
                    escape item.agentId,
                    escape item.role,
                    verdict,
                    formatConfidence item.confidence,
                    notes) |> ignore)

        AnsiConsole.Write(feedbackTable)

    let renderCommandTimeline (entry: GovernanceLedgerEntry) =
        let commands = safeList entry.commands

        if not (List.isEmpty commands) then
            let timeline = Table()
            timeline.Title <- TableTitle("[bold]Command Timeline[/]")
            timeline.AddColumn(TableColumn("[grey]Stage[/]")) |> ignore
            timeline.AddColumn(TableColumn("[grey]Command[/]")) |> ignore
            timeline.AddColumn(TableColumn("[grey]Exit[/]")) |> ignore
            timeline.AddColumn(TableColumn("[grey]Duration[/]")) |> ignore
            timeline.AddColumn(TableColumn("[grey]Started[/]")) |> ignore

            commands
            |> List.sortBy (fun command -> command.StartedAt)
            |> List.iter (fun command ->
                let exitCode =
                    if command.ExitCode = 0 then "[green]0[/]" else $"[red]{command.ExitCode}[/]"
                let durationText = $"{command.DurationSeconds:F2}s"
                let startedAt = command.StartedAt.ToLocalTime().ToString("HH:mm:ss")

                timeline.AddRow(
                    escape command.Stage,
                    escape command.Name,
                    exitCode,
                    durationText,
                    escape startedAt) |> ignore)

            AnsiConsole.Write(timeline)

    let renderArtifacts (entry: GovernanceLedgerEntry) =
        let artifacts = safeList entry.artifacts

        if not (List.isEmpty artifacts) then
            let artifactTable = Table()
            artifactTable.Title <- TableTitle("[bold]Artifacts[/]")
            artifactTable.AddColumn(TableColumn("[grey]Path[/]")) |> ignore

            artifacts
            |> List.iter (fun path -> artifactTable.AddRow(escape path) |> ignore)

            AnsiConsole.Write(artifactTable)

    let renderNextSteps (entry: GovernanceLedgerEntry) =
        let steps = safeList entry.nextSteps

        if not (List.isEmpty steps) then
            let stepTable = Table()
            stepTable.Title <- TableTitle("[bold]Next Steps[/]")
            stepTable.AddColumn(TableColumn("[grey]Action[/]")) |> ignore

            steps
            |> List.iter (fun step -> stepTable.AddRow(escape step) |> ignore)

            AnsiConsole.Write(stepTable)

    let renderPolicies (latest: GovernanceLedgerEntry) =
        let beforeSnapshot = latest.policyBefore
        let afterSnapshot = latest.policyAfter

        let policyTable = Table()
        policyTable.Title <- TableTitle("[bold]Policy Guardrails[/]")
        policyTable.AddColumn(TableColumn("[grey]Policy[/]")) |> ignore
        policyTable.AddColumn(TableColumn("[grey]Before[/]")) |> ignore
        policyTable.AddColumn(TableColumn("[grey]After[/]")) |> ignore

        let boolText value = if value then "[green]yes[/]" else "[red]no[/]"

        policyTable.AddRow("Require consensus", boolText beforeSnapshot.requireConsensus, boolText afterSnapshot.requireConsensus) |> ignore
        policyTable.AddRow("Require critic", boolText beforeSnapshot.requireCritic, boolText afterSnapshot.requireCritic) |> ignore
        policyTable.AddRow("Stop on failure", boolText beforeSnapshot.stopOnFailure, boolText afterSnapshot.stopOnFailure) |> ignore
        policyTable.AddRow("Capture logs", boolText beforeSnapshot.captureLogs, boolText afterSnapshot.captureLogs) |> ignore
        policyTable.AddRow("Validation commands", beforeSnapshot.validationCommands.ToString(), afterSnapshot.validationCommands.ToString()) |> ignore
        policyTable.AddRow("Benchmark commands", beforeSnapshot.benchmarkCommands.ToString(), afterSnapshot.benchmarkCommands.ToString()) |> ignore

        let ruleSummary snapshot =
            snapshot.consensusRule
            |> Option.map (fun rule ->
                let roles = rule.requiredRoles |> String.concat ", "
                $"min-pass: {rule.minimumPassCount}, roles: {escape roles}")
            |> Option.defaultValue "disabled"

        policyTable.AddRow("Consensus rule", ruleSummary beforeSnapshot, ruleSummary afterSnapshot) |> ignore

        if not (List.isEmpty latest.policyChanges) then
            let changeTable = Table()
            changeTable.Title <- TableTitle("[bold]Policy Changes[/]")
            changeTable.AddColumn(TableColumn("[grey]Field[/]")) |> ignore
            changeTable.AddColumn(TableColumn("[grey]Previous[/]")) |> ignore
            changeTable.AddColumn(TableColumn("[grey]Current[/]")) |> ignore
            changeTable.AddColumn(TableColumn("[grey]Rationale[/]")) |> ignore

            latest.policyChanges
            |> List.iter (fun change ->
                let rationale =
                    change.rationale
                    |> Option.filter (fun r -> not (String.IsNullOrWhiteSpace(r)))
                    |> Option.map escape
                    |> Option.defaultValue "—"
                changeTable.AddRow(
                    escape change.field,
                    escape change.previousValue,
                    escape change.currentValue,
                    rationale) |> ignore)

            let panel = Panel(changeTable)
            panel.Border <- BoxBorder.Rounded
            panel.BorderStyle <- Style(Color.Yellow)
            AnsiConsole.Write(panel)

        AnsiConsole.Write(policyTable)

    let renderRunDetail (record: LedgerRecord) =
        tryClearConsole ()

        let entry = record.Entry

        let header =
            Rule($"[bold purple]🧩 TARS Iteration Detail[/]  [grey]{escape entry.specId}[/]")
        header.Justification <- Justify.Center
        AnsiConsole.Write(header)
        AnsiConsole.WriteLine()

        let summaryTable = Table()
        summaryTable.Title <- TableTitle("[bold]Run Summary[/]")
        summaryTable.AddColumn(TableColumn("[grey]Field[/]")) |> ignore
        summaryTable.AddColumn(TableColumn("[grey]Value[/]")) |> ignore

        summaryTable.AddRow("Run id", escape (entry.runId.ToString())) |> ignore
        summaryTable.AddRow("Timestamp", escape (entry.timestamp.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss"))) |> ignore
        summaryTable.AddRow("Status", formatStatus entry.status) |> ignore
        summaryTable.AddRow("Consensus", consensusSummary entry) |> ignore
        summaryTable.AddRow("Harness", harnessSummary entry) |> ignore
        summaryTable.AddRow("Spec path", escape entry.specPath) |> ignore
        summaryTable.AddRow("Ledger file", escape record.FilePath) |> ignore
        summaryTable.AddRow("Adaptive memory path", escape entry.appendPath) |> ignore

        tryTelemetryDouble entry "inference.metrics.analysis_elapsed_ms"
        |> Option.iter (fun value ->
            summaryTable.AddRow("Inference latency", escape (formatLatency value)) |> ignore)

        tryTelemetryDouble entry "inference.metrics.token_count"
        |> Option.iter (fun value ->
            summaryTable.AddRow("Prompt tokens", escape (formatInteger value)) |> ignore)

        tryTelemetryDouble entry "inference.metrics.context_length"
        |> Option.iter (fun value ->
            summaryTable.AddRow("Context length", escape (formatInteger value)) |> ignore)

        tryTelemetryString entry "inference.metrics.top_terms"
        |> Option.iter (fun value ->
            let formatted = value.Replace(",", ", ")
            summaryTable.AddRow("Top terms", escape formatted) |> ignore)

        tryTelemetryBool entry "inference.metrics.used_cuda"
        |> Option.iter (fun usedCuda ->
            let deviceLabel = if usedCuda then "[green]GPU[/]" else "[yellow]CPU[/]"
            summaryTable.AddRow("Inference device", deviceLabel) |> ignore)

        tryTelemetryString entry "inference.metrics.model_name"
        |> Option.iter (fun value ->
            summaryTable.AddRow("Model name", escape value) |> ignore)

        tryTelemetryString entry "inference.metrics.model_checksum"
        |> Option.iter (fun value ->
            summaryTable.AddRow("Model checksum", escape value) |> ignore)

        AnsiConsole.Write(summaryTable)
        AnsiConsole.WriteLine()

        renderCommandTimeline entry
        AnsiConsole.WriteLine()

        renderMetrics entry
        AnsiConsole.WriteLine()

        renderArtifacts entry
        AnsiConsole.WriteLine()

        renderNextSteps entry
        AnsiConsole.WriteLine()

        renderAgentFeedback entry
        AnsiConsole.WriteLine()

        renderPolicies entry

    let renderDashboard entries =
        tryClearConsole ()

        let title = Rule("[bold purple]🧩 TARS Auto-Improvement Dashboard[/]")
        title.Justification <- Justify.Center
        AnsiConsole.Write(title)
        AnsiConsole.WriteLine()

        if List.isEmpty entries then
            AnsiConsole.MarkupLine("[yellow]No governance ledger entries found. Run a Spec Kit iteration to populate the ledger.[/]")
        else
            renderSummary entries
            AnsiConsole.WriteLine()

            renderRecentHistory entries
            AnsiConsole.WriteLine()

            match entries with
            | latest :: _ ->
                renderMetrics latest
                AnsiConsole.WriteLine()

                renderAgentFeedback latest
                AnsiConsole.WriteLine()

                renderPolicies latest
            | [] -> ()

    let rec executeLoop (refreshSeconds: int option) =
        task {
            let entries = loadLedgerEntries ()
            renderDashboard entries

            match refreshSeconds with
            | Some seconds when seconds > 0 ->
                AnsiConsole.MarkupLine("[grey](Press Ctrl+C to exit watch mode)[/]")
                do! Task.Delay(TimeSpan.FromSeconds(float seconds))
                return! executeLoop refreshSeconds
            | _ ->
                return ()
        }

    let tryFindRunRecord (runId: string option) =
        let records = loadLedgerRecords ()

        let matchRecord normalized (record: LedgerRecord) =
            let entry = record.Entry
            let runId = entry.runId
            let candidates =
                [ runId.ToString("D")
                  runId.ToString("N")
                  runId.ToString("B")
                  record.FileName
                  entry.timestamp.ToString("yyyyMMddHHmmss")
                  entry.specId ]

            candidates
            |> List.exists (fun value ->
                let candidate = value.Trim().ToLowerInvariant()
                candidate = normalized || candidate.StartsWith(normalized))

        let selected =
            match runId with
            | None -> records |> List.tryHead
            | Some key ->
                let trimmed = key.Trim()
                if String.IsNullOrEmpty(trimmed) || trimmed.Equals("latest", StringComparison.OrdinalIgnoreCase) then
                    records |> List.tryHead
                else
                    let normalized = trimmed.ToLowerInvariant()
                    records |> List.tryFind (matchRecord normalized)

        records, selected

    interface ICommand with
        member _.Name = "auto-ui"
        member _.Description = "Interactive dashboard for the TARS autonomous improvement ledger."
        member _.Usage = "tars auto-ui [watch [seconds] | run [run-id|latest]]"
        member _.Examples =
            [ "tars auto-ui"
              "tars auto-ui watch"
              "tars auto-ui watch 10"
              "tars auto-ui run"
              "tars auto-ui run 9f798c26"
              "tars auto-ui run latest" ]

        member _.ValidateOptions(_options: CommandOptions) = true

        member this.ExecuteAsync(options: CommandOptions) =
            task {
                try
                    match parseMode options with
                    | Overview refreshSeconds ->
                        do! executeLoop refreshSeconds

                        let message =
                            match refreshSeconds with
                            | Some _ -> "Auto-improvement dashboard streaming."
                            | None -> "Auto-improvement overview rendered."

                        return success message

                    | RunDetail runKey ->
                        let records, selected = tryFindRunRecord runKey

                        match selected with
                        | Some record ->
                            renderRunDetail record
                            return success "Auto-improvement run detail rendered."
                        | None ->
                            if List.isEmpty records then
                                AnsiConsole.MarkupLine("[yellow]No governance ledger entries available yet. Run a Spec Kit iteration first.[/]")
                            else
                                let hint =
                                    records
                                    |> List.truncate 5
                                    |> List.map (fun record ->
                                        record.Entry.runId.ToString("N").Substring(0, 8))
                                    |> String.concat ", "

                                match runKey with
                                | Some key when not (String.IsNullOrWhiteSpace(key)) ->
                                    AnsiConsole.MarkupLine($"[yellow]Run '{escape key}' not found. Try one of: [grey]{escape hint}[/][/]")
                                | _ ->
                                    AnsiConsole.MarkupLine("[yellow]Unable to locate the requested run.[/]")

                            return failure "Requested governance run not located."

                with ex ->
                    logger.LogError(ex, "Failed to render auto-improvement dashboard.")
                    AnsiConsole.MarkupLine($"[red]Dashboard rendering failed: {escape ex.Message}[/]")
                    return failure "Auto-improvement dashboard failed."
            }
