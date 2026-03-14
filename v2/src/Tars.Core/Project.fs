/// Project Pipeline Management - Core Types
/// Enables TARS to manage multiple projects with persona-based development pipelines
module Tars.Core.Project

open System

// ============================================================================
// Pipeline Stages
// ============================================================================

/// Stages in the development pipeline
type PipelineStage =
    | Vision
    | Specification
    | Development
    | QualityAssurance
    | Demo
    // Agile variants
    | Backlog
    | Sprint
    | Review
    | Deploy
    // Research variants
    | Hypothesis
    | Experiment
    | Analysis
    | Report

/// Get display name for a stage
let stageName =
    function
    | Vision -> "Vision"
    | Specification -> "Specification"
    | Development -> "Development"
    | QualityAssurance -> "QA"
    | Demo -> "Demo"
    | Backlog -> "Backlog"
    | Sprint -> "Sprint"
    | Review -> "Review"
    | Deploy -> "Deploy"
    | Hypothesis -> "Hypothesis"
    | Experiment -> "Experiment"
    | Analysis -> "Analysis"
    | Report -> "Report"

// ============================================================================
// Execution Modes
// ============================================================================

/// How the pipeline should execute
type ExecutionMode =
    /// Pause at each stage for human approval
    | HumanInLoop
    /// Auto-proceed unless errors
    | Continuous
    /// Auto-proceed for some stages, pause at specified stages
    | Hybrid of pauseAt: PipelineStage list

// ============================================================================
// Stage Configuration
// ============================================================================

/// Configuration for a pipeline stage
type StageConfig =
    { Stage: PipelineStage
      Personas: string list // Persona IDs to use at this stage
      RequiredArtifacts: string list // Input artifacts needed
      OutputArtifacts: string list // Artifacts produced
      CompletionCriteria: string option }

/// Status of a stage in execution
type StageStatus =
    | NotStarted
    | InProgress
    | WaitingForApproval
    | Completed of completedAt: DateTime
    | Failed of error: string

/// Runtime state for a stage
type StageState =
    { Config: StageConfig
      Status: StageStatus
      StartedAt: DateTime option
      Artifacts: Map<string, string> } // artifact name -> path

// ============================================================================
// Pipeline Templates
// ============================================================================

/// Predefined pipeline templates
type PipelineTemplate =
    | StandardSDLC
    | AgileSprint
    | Research
    | Custom of string

/// Get stages for a template
let templateStages =
    function
    | StandardSDLC -> [ Vision; Specification; Development; QualityAssurance; Demo ]
    | AgileSprint -> [ Backlog; Sprint; Review; Deploy ]
    | Research -> [ Hypothesis; Experiment; Analysis; Report ]
    | Custom _ -> []

/// Create default stage configs for a template
let createTemplateConfigs template =
    templateStages template
    |> List.map (fun stage ->
        { Stage = stage
          Personas = []
          RequiredArtifacts = []
          OutputArtifacts = []
          CompletionCriteria = None })

// ============================================================================
// Project Definition
// ============================================================================

/// A project managed by TARS
type Project =
    { Id: string
      Name: string
      Description: string option
      Repository: string option // Git URL
      RootPath: string // Local working directory
      Template: PipelineTemplate
      Stages: StageConfig list
      ExecutionMode: ExecutionMode
      GraphitiNamespace: string // Isolated memory namespace
      CreatedAt: DateTime
      UpdatedAt: DateTime }

/// Runtime state of a project
type ProjectState =
    { Project: Project
      CurrentStage: PipelineStage option
      StageStates: Map<PipelineStage, StageState>
      StartedAt: DateTime option
      CompletedAt: DateTime option }

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a new project with template
let createProject id name rootPath template executionMode =
    let namespace' = $"project_{id}"

    { Id = id
      Name = name
      Description = None
      Repository = None
      RootPath = rootPath
      Template = template
      Stages = createTemplateConfigs template
      ExecutionMode = executionMode
      GraphitiNamespace = namespace'
      CreatedAt = DateTime.UtcNow
      UpdatedAt = DateTime.UtcNow }

/// Initialize project state
let initProjectState (project: Project) =
    let stageStates =
        project.Stages
        |> List.map (fun cfg ->
            cfg.Stage,
            { Config = cfg
              Status = NotStarted
              StartedAt = None
              Artifacts = Map.empty })
        |> Map.ofList

    { Project = project
      CurrentStage = None
      StageStates = stageStates
      StartedAt = None
      CompletedAt = None }

/// Check if a stage requires human approval
let requiresApproval (mode: ExecutionMode) (stage: PipelineStage) =
    match mode with
    | HumanInLoop -> true
    | Continuous -> false
    | Hybrid pauseAt -> List.contains stage pauseAt

/// Get next stage in pipeline
let nextStage (stages: PipelineStage list) (current: PipelineStage) =
    let idx = List.tryFindIndex ((=) current) stages

    match idx with
    | Some i when i < stages.Length - 1 -> Some stages.[i + 1]
    | _ -> None
