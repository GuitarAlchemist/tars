namespace TarsEngine.FSharp.Cli.Core

open System
open System.IO
open System.Text.Json

/// Represents model identifiers used by the Flux Codex workflow.
type FluxModels = {
    Fast: string
    Reasoning: string
    Review: string
}

/// Represents execution capabilities granted to the sandbox.
type FluxCapabilities = {
    Tools: Map<string, string>
    NetworkAccess: bool
    MaxWriteFiles: int
}

/// Flux configuration section loaded from tars.config.json.
type FluxConfig = {
    Enabled: bool
    BaseBranch: string
    Runner: string
    DockerImage: string option
    AllowedCommands: string list
    Capabilities: FluxCapabilities
    Models: FluxModels
}

/// Classification used to reason about plan steps.
type FluxPlanStepKind =
    | Analysis
    | Branching
    | EnvironmentSetup
    | Build
    | Test
    | Diff
    | Publish
    | Manual
    | Validation

/// Represents an executable command issued to a sandbox runner.
type SandboxCommand = {
    Executable: string
    Arguments: string list
    WorkingDirectory: string option
    Environment: Map<string, string>
}

/// Result of executing a command inside the sandbox.
type SandboxCommandResult = {
    ExitCode: int
    StandardOutput: string
    StandardError: string
    Duration: TimeSpan
    StartedAt: DateTime
    CompletedAt: DateTime
    Command: SandboxCommand
}

/// A single plan step created by the planner.
type FluxPlanStep = {
    Id: string
    Title: string
    Details: string
    Kind: FluxPlanStepKind
    Command: SandboxCommand option
    AllowFailure: bool
}

/// Represents an ordered plan derived from the task description.
type FluxPlan = {
    TaskDescription: string
    BranchName: string
    Steps: FluxPlanStep list
}

/// Outcome of a plan step after execution.
type FluxStepOutcome =
    | Completed of SandboxCommandResult option
    | Skipped of string
    | Failed of string

/// Execution metadata for a single plan step.
type FluxStepExecution = {
    Step: FluxPlanStep
    Outcome: FluxStepOutcome
    StartedAt: DateTime
    CompletedAt: DateTime
}

/// Summary of a complete Flux run.
type FluxRunResult = {
    RunId: string
    WorkspacePath: string option
    ArtifactDirectory: string
    Plan: FluxPlan
    StepResults: FluxStepExecution list
    DraftPrPath: string option
    StartedAt: DateTime
    CompletedAt: DateTime
    TaskDescription: string
    Success: bool
    Messages: string list
}

/// Options passed to a Flux run.
type FluxRunParameters = {
    Task: string
    RepoRoot: string
    BaseBranch: string
    EnablePullRequest: bool
    SkipBuild: bool
    SkipTests: bool
}

/// Context routed to the model invoker when generating a plan.
type FluxPlanningContext = {
    Task: string
    BaseBranch: string
    RunId: string
    Timestamp: DateTime
    RepoRoot: string
    Config: FluxConfig
}

/// Sandbox runner abstraction.
type ISandboxRunner =
    abstract member Name: string
    abstract member CreateWorkspace: runId: string * baseBranch: string -> Result<string, string>
    abstract member Run: workspace: string * command: SandboxCommand -> Result<SandboxCommandResult, string>
    abstract member CleanupWorkspace: workspace: string -> unit

/// Draft pull request publisher abstraction.
type IPrPublisher =
    abstract member PublishDraft:
        repoRoot: string *
        workspace: string *
        plan: FluxPlan *
        stepResults: FluxStepExecution list *
        artifactDirectory: string ->
            Result<string option, string>

/// Planner abstraction that can produce Flux plans.
type IModelInvoker =
    abstract member GeneratePlan: FluxPlanningContext -> FluxPlan

/// Persisted summary written to `run.json`.
type FluxRunSummary = {
    RunId: string
    Task: string
    Success: bool
    BaseBranch: string
    BranchName: string
    DraftPrPath: string option
    StartedAt: DateTime
    CompletedAt: DateTime
    Steps: FluxRunSummaryStep list
}

/// Persisted per-step summary.
and FluxRunSummaryStep = {
    Id: string
    Title: string
    Kind: string
    Outcome: string
    ExitCode: int option
    DurationMs: float
    Notes: string option
}

/// Flux configuration loader with resilient defaults.
module FluxConfigLoader =

    let defaultConfig =
        let capabilities = {
            Tools = Map.ofList [ "git", ">=2.46"; "dotnet", "9.0"; "pnpm", "9" ]
            NetworkAccess = false
            MaxWriteFiles = 200
        }
        {
            Enabled = true
            BaseBranch = "main"
            Runner = "local"
            DockerImage = None
            AllowedCommands = [ "git"; "dotnet"; "pnpm"; "npm"; "yarn" ]
            Capabilities = capabilities
            Models = {
                Fast = "qwen2.5-coder:7b"
                Reasoning = "llama3.1:70b"
                Review = "mixtral:8x22b"
            }
        }

    let private tryGetProperty (element: JsonElement) (propertyName: string) =
        match element.TryGetProperty(propertyName) with
        | true, value -> Some value
        | _ -> None

    let private toStringList (element: JsonElement) =
        element.EnumerateArray()
        |> Seq.choose (fun item ->
            match item.ValueKind with
            | JsonValueKind.String -> Some(item.GetString())
            | _ -> None)
        |> Seq.map (fun value -> value |> Option.ofObj |> Option.defaultValue "")
        |> Seq.filter (fun value -> not (String.IsNullOrWhiteSpace value))
        |> Seq.toList

    let private toToolsMap (element: JsonElement) =
        element.EnumerateObject()
        |> Seq.map (fun prop ->
            let value =
                if prop.Value.ValueKind = JsonValueKind.String then
                    match prop.Value.GetString() with
                    | null | "" -> ""
                    | str -> str
                else
                    prop.Value.ToString()
            prop.Name, value)
        |> Map.ofSeq

    let private readCapabilities (fluxNode: JsonElement) =
        match tryGetProperty fluxNode "capabilities" with
        | None -> defaultConfig.Capabilities
        | Some capabilitiesNode ->
            let tools =
                match tryGetProperty capabilitiesNode "tools" with
                | Some toolsNode when toolsNode.ValueKind = JsonValueKind.Object -> toToolsMap toolsNode
                | _ -> defaultConfig.Capabilities.Tools

            let netAccess =
                match tryGetProperty capabilitiesNode "netAccess" with
                | Some value when value.ValueKind = JsonValueKind.True || value.ValueKind = JsonValueKind.False ->
                    value.GetBoolean()
                | _ -> defaultConfig.Capabilities.NetworkAccess

            let maxWrites =
                match tryGetProperty capabilitiesNode "maxWriteFiles" with
                | Some value when value.ValueKind = JsonValueKind.Number ->
                    match value.TryGetInt32() with
                    | true, number -> number
                    | _ -> defaultConfig.Capabilities.MaxWriteFiles
                | _ -> defaultConfig.Capabilities.MaxWriteFiles

            {
                Tools = tools
                NetworkAccess = netAccess
                MaxWriteFiles = maxWrites
            }

    let private readModels (fluxNode: JsonElement) =
        match tryGetProperty fluxNode "models" with
        | None -> defaultConfig.Models
        | Some modelsNode ->
            let tryString name fallback =
                match tryGetProperty modelsNode name with
                | Some value when value.ValueKind = JsonValueKind.String ->
                    match value.GetString() with
                    | null | "" -> fallback
                    | str -> str
                | _ -> fallback

            {
                Fast = tryString "fast" defaultConfig.Models.Fast
                Reasoning = tryString "reasoning" defaultConfig.Models.Reasoning
                Review = tryString "review" defaultConfig.Models.Review
            }

    /// Loads the flux configuration from the repository root. Falls back to defaults on failure.
    let load (repoRoot: string) =
        let configPath = Path.Combine(repoRoot, "tars.config.json")
        if not (File.Exists(configPath)) then
            defaultConfig
        else
            try
                use document = JsonDocument.Parse(File.ReadAllText(configPath))
                match tryGetProperty document.RootElement "flux" with
                | None ->
                    Console.WriteLine("Warning: Flux config missing in tars.config.json – using defaults.")
                    defaultConfig
                | Some fluxNode ->
                    let enabled =
                        match tryGetProperty fluxNode "enabled" with
                        | Some value when value.ValueKind = JsonValueKind.True || value.ValueKind = JsonValueKind.False -> value.GetBoolean()
                        | _ -> defaultConfig.Enabled

                    let baseBranch =
                        match tryGetProperty fluxNode "baseBranch" with
                        | Some value when value.ValueKind = JsonValueKind.String ->
                            match value.GetString() with
                            | null | "" -> defaultConfig.BaseBranch
                            | str -> str
                        | _ -> defaultConfig.BaseBranch

                    let runner =
                        match tryGetProperty fluxNode "runner" with
                        | Some value when value.ValueKind = JsonValueKind.String ->
                            match value.GetString() with
                            | null | "" -> defaultConfig.Runner
                            | str -> str
                        | _ -> defaultConfig.Runner

                    let dockerImage =
                        match tryGetProperty fluxNode "dockerImage" with
                        | Some value when value.ValueKind = JsonValueKind.String ->
                            match value.GetString() with
                            | null | "" -> None
                            | str -> Some str
                        | _ -> None

                    let allowedCommands =
                        match tryGetProperty fluxNode "allowedCommands" with
                        | Some value when value.ValueKind = JsonValueKind.Array -> toStringList value
                        | _ -> defaultConfig.AllowedCommands

                    {
                        Enabled = enabled
                        BaseBranch = baseBranch
                        Runner = runner
                        DockerImage = dockerImage
                        AllowedCommands = if allowedCommands.IsEmpty then defaultConfig.AllowedCommands else allowedCommands
                        Capabilities = readCapabilities fluxNode
                        Models = readModels fluxNode
                    }
            with
            | ex ->
                Console.WriteLine(sprintf "Warning: Failed to parse flux config (%s). Using defaults." ex.Message)
                defaultConfig
