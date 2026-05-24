namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Abstractions
open TarsEngine.FSharp.Core.Specs
open TarsEngine.FSharp.Core.Services.MetascriptClosureIntegrationService
open TarsEngine.FSharp.Core.Closures.UnifiedEvolutionaryClosureFactory
open TarsEngine.FSharp.Core.Services.CrossAgentValidation
open TarsEngine.FSharp.Core.Services.ReasoningTrace
open ExecutionHarness
open TarsEngine.FSharp.SelfImprovement.CrossAgentFeedback
open TarsEngine.FSharp.SelfImprovement.ValidatorCoordination

/// Spec-guided autonomous execution harness that applies patches, validates them, and rolls back on failure.
module AutonomousSpecHarness =

    let private isPatchLike (fragment: string) =
        let trimmed = fragment.Trim().Trim('"').Trim('\'').Trim('`')
        if String.IsNullOrWhiteSpace(trimmed) then
            None
        else
            let extension = Path.GetExtension(trimmed)
            let looksLikePatch =
                extension.Equals(".patch", StringComparison.OrdinalIgnoreCase)
                || extension.Equals(".diff", StringComparison.OrdinalIgnoreCase)

            if not looksLikePatch then
                None
            else
                let candidate =
                    if Path.IsPathRooted(trimmed) then
                        trimmed
                    else
                        Path.Combine(Environment.CurrentDirectory, trimmed)
                        |> Path.GetFullPath

                if File.Exists(candidate) then Some candidate else None

    let private discoverPatchArtifacts (artifacts: string list) =
        artifacts
        |> List.collect (fun artifact ->
            artifact.Split([| ' '; '\t'; '\r'; '\n'; '"'; '\''; '`'; '('; ')'; '['; ']'; '{'; '}' |], StringSplitOptions.RemoveEmptyEntries)
            |> Array.toList
            |> List.choose isPatchLike)
        |> List.distinct

    let private makeApplyCommand (path: string) =
        { Name = $"apply_patch:{Path.GetFileName(path)}"
          Executable = "git"
          Arguments = $"apply --3way --whitespace=fix \"{path}\""
          WorkingDirectory = Some Environment.CurrentDirectory
          Timeout = Some(TimeSpan.FromMinutes 3.0)
          Environment = Map.empty }

    let private makeRollbackCommand (path: string) =
        { Name = $"rollback_patch:{Path.GetFileName(path)}"
          Executable = "git"
          Arguments = $"apply --reverse --whitespace=fix \"{path}\""
          WorkingDirectory = Some Environment.CurrentDirectory
          Timeout = Some(TimeSpan.FromMinutes 3.0)
          Environment = Map.empty }

    let internal discoverPatchArtifactsForTesting artifacts =
        discoverPatchArtifacts artifacts

    type SpecDrivenIterationConfig =
        { SpecPath: string
          Description: string option
          PatchCommands: CommandSpec list
          ValidationCommands: CommandSpec list
          BenchmarkCommands: CommandSpec list
          RollbackCommands: CommandSpec list
          StopOnFailure: bool
          CaptureLogs: bool
          AutoApplyPatchArtifacts: bool
          ConsensusRule: ConsensusRule option
          AgentResultProvider: (MetascriptClosureResult -> AgentValidationResult list) option
          RequireConsensusForExecution: bool
          ReasoningTraceProvider: (MetascriptClosureResult -> ReasoningTrace list) option
          ReasoningCritic: (ReasoningTrace list -> CriticVerdict) option
          RequireCriticApproval: bool
          ReasoningFeedbackSink: (ReasoningTrace list -> CriticVerdict option -> unit) option
          AgentFeedbackProvider: (MetascriptClosureResult -> AgentFeedback list) option
          AdaptiveMemoryPath: string option }

    type SpecDrivenIterationResult =
        { SpecSummary: SpecKitSummary
          ClosureResult: MetascriptClosureResult
          HarnessReport: HarnessReport option
          Consensus: ConsensusOutcome option
          ReasoningTraces: ReasoningTrace list option
          CriticVerdict: CriticVerdict option
          CrossAgentFeedback: AgentFeedback list option
          ValidatorSnapshot: CoordinationSnapshot option }

    type CommandRollbackHandler(logger: ILogger, executor: ICommandExecutor, commands: CommandSpec list) =
        interface IRollbackHandler with
            member _.RollbackAsync() =
                async {
                    for command in commands do
                        try
                            logger.LogInformation("Running rollback command: {Name}", command.Name)
                            let! result = executor.RunCommandAsync(command)
                            if result.ExitCode <> 0 then
                                logger.LogWarning("Rollback command {Name} failed with {ExitCode}", command.Name, result.ExitCode)
                        with ex ->
                            logger.LogError(ex, "Rollback command {Name} threw an exception.", command.Name)
                }

    let private ensureSpecPath (path: string) =
        if String.IsNullOrWhiteSpace(path) then
            invalidArg (nameof path) "Spec path must not be empty."
        if not (File.Exists(path)) then
            invalidArg (nameof path) $"Spec file not found: %s{path}"

    let private buildRollbackHandler
        (loggerFactory: ILoggerFactory)
        (executor: ICommandExecutor)
        (commands: CommandSpec list) =
        if commands.IsEmpty then
            None
        else
            let logger: ILogger = loggerFactory.CreateLogger("AutonomousSpecHarness.Rollback")
            Some(CommandRollbackHandler(logger, executor, commands) :> IRollbackHandler)

    let private runIterationInternal
        (loggerFactory: ILoggerFactory)
        (config: SpecDrivenIterationConfig)
        (commandExecutor: ICommandExecutor option) =
        async {
            ensureSpecPath config.SpecPath

            let factoryLogger = loggerFactory.CreateLogger<UnifiedEvolutionaryClosureFactory>()
            let closureFactory = UnifiedEvolutionaryClosureFactory(factoryLogger)
            let serviceLogger = loggerFactory.CreateLogger<MetascriptClosureIntegrationService>()
            let metascriptService = MetascriptClosureIntegrationService(serviceLogger, closureFactory)

            let specSummary = SpecKitParser.loadFromFile config.SpecPath
            let specId = Path.GetFileNameWithoutExtension(config.SpecPath)

            let closureCommandLine =
                $"CLOSURE_CREATE DYNAMIC_METASCRIPT \"%s{specId}\" spec=\"%s{config.SpecPath}\""

            let closureCommand =
                metascriptService.ParseClosureCommand(closureCommandLine)
                |> Option.defaultWith (fun () -> invalidOp $"Failed to parse metascript command: %s{closureCommandLine}"
                )

            let! closureResult =
                metascriptService.ExecuteClosureCommand(closureCommand)
                |> Async.Catch

            match closureResult with
            | Choice2Of2 ex ->
                return raise (InvalidOperationException("Metascript execution failed.", ex))
            | Choice1Of2 closureResult when not closureResult.Success ->
                return
                    { SpecSummary = specSummary
                      ClosureResult = closureResult
                      HarnessReport = None
                      Consensus = None
                      ReasoningTraces = None
                      CriticVerdict = None
                      CrossAgentFeedback = None
                      ValidatorSnapshot = None }
            | Choice1Of2 closureResult ->
                let patchArtifacts =
                    if config.AutoApplyPatchArtifacts then
                        discoverPatchArtifacts closureResult.Artifacts
                    else
                        []

                let closureResult =
                    if patchArtifacts.IsEmpty then
                        closureResult
                    else
                        let enrichedEvolution =
                            patchArtifacts
                            |> List.mapi (fun idx path -> $"auto.patch[{idx}]", box path)
                            |> List.fold (fun acc (key, value) -> acc |> Map.add key value) closureResult.EvolutionData

                        { closureResult with EvolutionData = enrichedEvolution }

                let consensusOutcome =
                    match config.ConsensusRule, config.AgentResultProvider with
                    | Some rule, Some provider ->
                        let agentResults = provider closureResult
                        Some (evaluate rule agentResults)
                    | _ -> None

                let reasoningTraces =
                    config.ReasoningTraceProvider
                    |> Option.map (fun provider -> provider closureResult)

                let criticVerdict =
                    match config.ReasoningCritic, reasoningTraces with
                    | Some critic, Some traces -> Some (critic traces)
                    | Some critic, None -> Some (critic [])
                    | _ -> None

                let agentFeedback =
                    config.AgentFeedbackProvider
                    |> Option.map (fun provider -> provider closureResult)

                let shouldStopForConsensus =
                    match consensusOutcome with
                    | Some (ConsensusFailed _)
                    | Some (ConsensusNeedsReview _) when config.RequireConsensusForExecution -> true
                    | _ -> false

                let shouldStopForCritic =
                    match criticVerdict with
                    | Some (CriticVerdict.Reject _)
                    | Some (CriticVerdict.NeedsReview _) when config.RequireCriticApproval -> true
                    | _ -> false

                config.ReasoningFeedbackSink
                |> Option.iter (fun sink -> sink (reasoningTraces |> Option.defaultValue []) criticVerdict)

                if shouldStopForConsensus || shouldStopForCritic then
                    return
                        { SpecSummary = specSummary
                          ClosureResult = closureResult
                          HarnessReport = None
                          Consensus = consensusOutcome
                          ReasoningTraces = reasoningTraces
                          CriticVerdict = criticVerdict
                          CrossAgentFeedback = agentFeedback
                          ValidatorSnapshot = None }
                else
                    let harnessLogger = loggerFactory.CreateLogger<AutonomousExecutionHarness>()
                    let executor =
                        match commandExecutor with
                        | Some exec -> exec
                        | None -> new ProcessCommandExecutor(harnessLogger) :> ICommandExecutor

                    let harness = new AutonomousExecutionHarness(harnessLogger, executor)

                    let harnessConfig =
                        let patchCommands =
                            if config.PatchCommands.IsEmpty && not patchArtifacts.IsEmpty then
                                patchArtifacts |> List.map makeApplyCommand
                            else
                                config.PatchCommands

                        let rollbackCommands =
                            if config.RollbackCommands.IsEmpty && not patchArtifacts.IsEmpty then
                                patchArtifacts |> List.map makeRollbackCommand
                            else
                                config.RollbackCommands

                        let rollbackHandler = buildRollbackHandler loggerFactory executor rollbackCommands

                        { Description = config.Description |> Option.defaultValue $"Spec-driven iteration for %s{specSummary.Title}"
                          PreValidation = patchCommands
                          Validation = config.ValidationCommands
                          Benchmarks = config.BenchmarkCommands
                          Rollback =
                            match rollbackHandler with
                            | Some _ -> rollbackHandler
                            | None when not rollbackCommands.IsEmpty ->
                                buildRollbackHandler loggerFactory executor rollbackCommands
                            | None -> None
                          StopOnFailure = config.StopOnFailure
                          CaptureLogs = config.CaptureLogs }

                    let! report = harness.RunAsync(harnessConfig)

                    return { SpecSummary = specSummary
                             ClosureResult = closureResult
                             HarnessReport = Some report
                             Consensus = consensusOutcome
                             ReasoningTraces = reasoningTraces
                             CriticVerdict = criticVerdict
                             CrossAgentFeedback = agentFeedback
                             ValidatorSnapshot = None }
        }


    /// Runs a single spec-driven iteration using default process execution.
    let runIteration loggerFactory config =
        runIterationInternal loggerFactory config None

    /// Runs a spec-driven iteration using the provided command executor (useful for testing).
    let runIterationWithExecutor loggerFactory config executor =
        runIterationInternal loggerFactory config (Some executor)

    /// Convenience helper that uses a null logger factory.
    let runIterationWithNullLogging config =
        runIteration (NullLoggerFactory.Instance :> ILoggerFactory) config


