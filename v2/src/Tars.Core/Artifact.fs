/// Artifact System - Stage output management with blueprint distillation
/// Inspired by DeepCode paper's information-flow optimization
module Tars.Core.Artifact

open System
open System.IO

// ============================================================================
// Artifact Types
// ============================================================================

/// Format of an artifact
type ArtifactFormat =
    | Markdown
    | JSON
    | Code of language: string
    | Binary
    | Directory

/// An artifact produced by a pipeline stage
type StageArtifact =
    { Id: string
      Name: string
      Path: string
      Format: ArtifactFormat
      Blueprint: string option // Compressed summary (DeepCode pattern)
      Size: int64
      CreatedBy: string // Stage name
      CreatedAt: DateTime
      Metadata: Map<string, string> }

/// Collection of artifacts from a stage
type StageArtifacts =
    { Stage: string
      Artifacts: StageArtifact list
      Summary: string option // Overall stage summary
      CompletedAt: DateTime }

// ============================================================================
// Blueprint Distillation
// ============================================================================

/// Blueprint generator for compressing stage output
type BlueprintConfig =
    { MaxLength: int
      IncludeFileList: bool
      IncludeKeyDecisions: bool }

let defaultBlueprintConfig =
    { MaxLength = 500
      IncludeFileList = true
      IncludeKeyDecisions = true }

/// Generate a blueprint from artifacts (simplified version)
/// In production, this would use LLM summarization
let generateBlueprint (config: BlueprintConfig) (artifacts: StageArtifact list) : string =
    let fileList =
        if config.IncludeFileList then
            artifacts
            |> List.map (fun a -> $"- {a.Name} ({a.Format})")
            |> String.concat "\n"
        else
            ""

    let totalSize = artifacts |> List.sumBy (fun a -> a.Size)

    let summary =
        $"Stage produced {artifacts.Length} artifacts ({totalSize} bytes total)"

    let blueprint = $"{summary}\n\n{fileList}"

    if blueprint.Length > config.MaxLength then
        blueprint.Substring(0, config.MaxLength - 3) + "..."
    else
        blueprint

// ============================================================================
// Artifact Manager
// ============================================================================

type ArtifactManager(artifactsRoot: string) =

    let mutable stageArtifacts: Map<string * string, StageArtifacts> = Map.empty

    /// Get artifact directory for a project/stage
    member _.GetArtifactPath(projectId: string, stageName: string, fileName: string) =
        Path.Combine(artifactsRoot, projectId, stageName, fileName)

    /// Ensure directory exists
    member _.EnsureDirectory(path: string) =
        let dir = Path.GetDirectoryName(path)

        if not (String.IsNullOrEmpty dir) && not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore

    /// Record an artifact
    member this.RecordArtifact(projectId: string, stageName: string, artifact: StageArtifact) =
        let key = (projectId, stageName)

        let existing =
            stageArtifacts
            |> Map.tryFind key
            |> Option.defaultValue
                { Stage = stageName
                  Artifacts = []
                  Summary = None
                  CompletedAt = DateTime.UtcNow }

        let updated =
            { existing with
                Artifacts = artifact :: existing.Artifacts }

        stageArtifacts <- stageArtifacts |> Map.add key updated

    /// Get artifacts for a stage
    member _.GetStageArtifacts(projectId: string, stageName: string) : StageArtifact list =
        stageArtifacts
        |> Map.tryFind (projectId, stageName)
        |> Option.map (fun sa -> sa.Artifacts)
        |> Option.defaultValue []

    /// Complete a stage and generate blueprint
    member this.CompleteStage(projectId: string, stageName: string) : StageArtifacts option =
        let key = (projectId, stageName)

        match Map.tryFind key stageArtifacts with
        | None -> None
        | Some existing ->
            let blueprint = generateBlueprint defaultBlueprintConfig existing.Artifacts

            let completed =
                { existing with
                    Summary = Some blueprint
                    CompletedAt = DateTime.UtcNow }

            stageArtifacts <- stageArtifacts |> Map.add key completed
            Some completed

    /// Get handoff context for next stage (blueprint + key artifacts)
    member this.GetHandoffContext(projectId: string, fromStage: string) : string option =
        match this.CompleteStage(projectId, fromStage) with
        | None -> None
        | Some stageOutput ->
            let context =
                stageOutput.Summary
                |> Option.map (fun s -> $"## Previous Stage: {fromStage}\n\n{s}")

            context

    /// Clear all artifacts for a project
    member _.ClearProject(projectId: string) =
        stageArtifacts <- stageArtifacts |> Map.filter (fun (pid, _) _ -> pid <> projectId)

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a simple artifact
let createArtifact id name path format =
    { Id = id
      Name = name
      Path = path
      Format = format
      Blueprint = None
      Size = 0L
      CreatedBy = ""
      CreatedAt = DateTime.UtcNow
      Metadata = Map.empty }

/// Create artifact from file
let createArtifactFromFile id name path format size stageName =
    { Id = id
      Name = name
      Path = path
      Format = format
      Blueprint = None
      Size = size
      CreatedBy = stageName
      CreatedAt = DateTime.UtcNow
      Metadata = Map.empty }
