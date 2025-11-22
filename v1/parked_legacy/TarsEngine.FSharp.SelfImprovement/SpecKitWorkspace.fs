namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Text.RegularExpressions
open TarsEngine.FSharp.Core.Specs
open TarsEngine.FSharp.Core.Services.CrossAgentValidation
open TarsEngine.FSharp.Core.Services.ReasoningTrace
open TarsEngine.FSharp.SelfImprovement.ExecutionHarness
open TarsEngine.FSharp.SelfImprovement.AutonomousSpecHarness

/// Helpers for discovering and leveraging Spec Kit feature directories.
module SpecKitWorkspace =

    type SpecKitTask = {
        LineNumber: int
        Phase: string option
        Status: string
        TaskId: string option
        Priority: string option
        StoryTag: string option
        Description: string
        Raw: string
    }

    type SpecKitFeature = {
        Id: string
        Directory: string
        SpecPath: string
        PlanPath: string option
        TasksPath: string option
        Summary: SpecKitSummary
        Tasks: SpecKitTask list
    }

    type SpecKitSelection = {
        Feature: SpecKitFeature
        Task: SpecKitTask
        PriorityRank: int
    }

    type SpecKitHarnessCommands = {
        Patch: CommandSpec list
        Validation: CommandSpec list
        Benchmarks: CommandSpec list
        Rollback: CommandSpec list
        StopOnFailure: bool
        CaptureLogs: bool
    }

    type SpecKitHarnessOptions = {
        Commands: SpecKitHarnessCommands
        Description: string option
        ConsensusRule: ConsensusRule option
        PersistAdaptiveMemory: bool
        EnableAutoPatch: bool
        RequireCriticApproval: bool
        ReasoningCritic: (ReasoningTrace list -> CriticVerdict) option
    }

    let private baseHarnessCommands =
        { Patch = []
          Validation = []
          Benchmarks = []
          Rollback = []
          StopOnFailure = true
          CaptureLogs = true }

    let private baseHarnessOptions =
        { Commands = baseHarnessCommands
          Description = None
          ConsensusRule = None
          PersistAdaptiveMemory = true
          EnableAutoPatch = true
          RequireCriticApproval = false
          ReasoningCritic = None }

    let private ensureDirectory (path: string) =
        if Directory.Exists(path) then Some path else None

    let private readTasks (path: string) =
        let lines = File.ReadAllLines(path)
        let taskPattern = Regex(@"^-\s*\[(?<status>[ xX])\]\s*(?<rest>.+)$")
        let idPattern = Regex(@"\b(T\d{3,})\b", RegexOptions.IgnoreCase)
        let priorityPattern = Regex(@"\[(P\d)\]")
        let storyPattern = Regex(@"\[(US\d+)\]", RegexOptions.IgnoreCase)

        let mutable currentPhase = None

        lines
        |> Array.mapi (fun idx line ->
            let trimmed = line.Trim()
            if trimmed.StartsWith("## ") then
                currentPhase <- Some(trimmed.TrimStart('#').Trim())
                None
            else
                let m = taskPattern.Match(trimmed)
                if m.Success then
                    let rest = m.Groups.["rest"].Value.Trim()
                    let id =
                        let idMatch = idPattern.Match(rest)
                        if idMatch.Success then Some (idMatch.Value.ToUpperInvariant()) else None
                    let priority =
                        let priorityMatch = priorityPattern.Match(rest)
                        if priorityMatch.Success then Some (priorityMatch.Groups.[1].Value.ToUpperInvariant()) else None
                    let story =
                        let storyMatch = storyPattern.Match(rest)
                        if storyMatch.Success then Some (storyMatch.Groups.[1].Value.ToUpperInvariant()) else None
                    Some {
                        LineNumber = idx + 1
                        Phase = currentPhase
                        Status = if m.Groups.["status"].Value.Trim().ToLowerInvariant() = "x" then "done" else "todo"
                        TaskId = id
                        Priority = priority
                        StoryTag = story
                        Description = rest
                        Raw = line
                    }
                else None)
        |> Array.choose id
        |> Array.toList

    let private parseSpec (specPath: string) =
        let content = File.ReadAllText(specPath)
        let extension = Path.GetExtension(specPath)
        SpecKitParser.parse content extension

    let private discoverFeature (directory: string) =
        let specPath = Path.Combine(directory, "spec.md")
        if not (File.Exists(specPath)) then
            None
        else
            let planPath = Path.Combine(directory, "plan.md")
            let tasksPath = Path.Combine(directory, "tasks.md")

            let tasks =
                if File.Exists(tasksPath) then readTasks tasksPath else []

            let summary = parseSpec specPath

            let id =
                let name = Path.GetFileName(directory)
                if String.IsNullOrWhiteSpace(name) then Guid.NewGuid().ToString("N") else name

            Some {
                Id = id
                Directory = directory
                SpecPath = specPath
                PlanPath = if File.Exists(planPath) then Some planPath else None
                TasksPath = if File.Exists(tasksPath) then Some tasksPath else None
                Summary = summary
                Tasks = tasks
            }

    let discoverFeatures (root: string option) =
        let basePath =
            root
            |> Option.defaultValue (Path.Combine(Environment.CurrentDirectory, ".specify", "specs"))

        ensureDirectory basePath
        |> Option.map (fun rootPath ->
            Directory.GetDirectories(rootPath)
            |> Array.choose discoverFeature
            |> Array.sortBy (fun feature -> feature.Id)
            |> Array.toList)
        |> Option.defaultValue []

    let private isPendingTask (task: SpecKitTask) =
        not (String.Equals(task.Status, "done", StringComparison.OrdinalIgnoreCase))

    let private rankPriority (priority: string option) =
        match priority |> Option.map (fun value -> value.Trim().ToUpperInvariant()) with
        | Some "P0" -> 0
        | Some "P1" -> 1
        | Some "P2" -> 2
        | Some "P3" -> 3
        | Some value when value.StartsWith("P") ->
            match Int32.TryParse(value.Substring(1)) with
            | true, number when number >= 0 -> number
            | _ -> 99
        | _ -> 100

    let private evaluateFeature (feature: SpecKitFeature) =
        feature.Tasks
        |> List.filter isPendingTask
        |> List.sortBy (fun task -> rankPriority task.Priority, task.LineNumber)
        |> List.tryHead
        |> Option.map (fun task ->
            { Feature = feature
              Task = task
              PriorityRank = rankPriority task.Priority })

    let private tryOverrideSelection (features: SpecKitFeature list) =
        match Environment.GetEnvironmentVariable("TARS_SPEC_OVERRIDE") with
        | null
        | "" -> None
        | value ->
            features
            |> List.tryFind (fun feature -> feature.Id.Equals(value, StringComparison.OrdinalIgnoreCase))
            |> Option.bind evaluateFeature

    let selectNextFeature (features: SpecKitFeature list) =
        match tryOverrideSelection features with
        | Some selection -> Some selection
        | None ->
            features
            |> List.choose evaluateFeature
            |> List.sortBy (fun selection -> selection.PriorityRank, selection.Task.LineNumber, selection.Feature.Id)
            |> List.tryHead

    let tryGetFeature (root: string option) (featureId: string) =
        discoverFeatures root
        |> List.tryFind (fun feature -> String.Equals(feature.Id, featureId, StringComparison.OrdinalIgnoreCase))

    let defaultHarnessCommands = baseHarnessCommands
    let defaultHarnessOptions = baseHarnessOptions

    let buildIterationConfig (feature: SpecKitFeature) (options: SpecKitHarnessOptions option) =
        let opts = defaultArg options defaultHarnessOptions

        let enableAutoPatch = opts.EnableAutoPatch

        { SpecPath = feature.SpecPath
          Description = opts.Description |> Option.orElse feature.Summary.FeatureBranch
          PatchCommands = opts.Commands.Patch
          ValidationCommands = opts.Commands.Validation
          BenchmarkCommands = opts.Commands.Benchmarks
          RollbackCommands = opts.Commands.Rollback
          StopOnFailure = opts.Commands.StopOnFailure
          CaptureLogs = opts.Commands.CaptureLogs
          AutoApplyPatchArtifacts = enableAutoPatch
          ConsensusRule = opts.ConsensusRule
          AgentResultProvider = None
          RequireConsensusForExecution = opts.ConsensusRule |> Option.isSome
          ReasoningTraceProvider = None
          ReasoningCritic = opts.ReasoningCritic
          RequireCriticApproval = opts.RequireCriticApproval
          ReasoningFeedbackSink = None
          AgentFeedbackProvider = None
          AdaptiveMemoryPath = None }
