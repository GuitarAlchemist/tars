namespace Tars.Metascript

open System

module Domain =

    type StepType =
        | Agent
        | Tool
        | Loop
        | Decision
        | MapStep
        | Switch

    type StepContextRef = { StepId: string; OutputName: string }

    type StepDependency =
        { StepId: string
          Condition: string option }

    type WorkflowStep =
        { Id: string
          Type: string // "agent", "tool", "map", "switch", etc.
          DependsOn: StepDependency list option
          Agent: string option
          Tool: string option
          Instruction: string option
          Params: Map<string, string> option
          Context: StepContextRef list option
          Outputs: string list option
          Tools: string list option }

    type WorkflowInput =
        { Name: string
          Type: string
          Description: string }

    type Workflow =
        { Name: string
          Description: string
          Version: string
          Inputs: WorkflowInput list
          Steps: WorkflowStep list }

    type StepExecutionTrace =
        { StepId: string
          StartedAt: DateTime
          Duration: TimeSpan
          Outputs: Map<string, obj>
          Notes: string list }

    /// Runtime state of a workflow execution
    type WorkflowState =
        { Workflow: Workflow
          CurrentStepIndex: int
          Variables: Map<string, obj> // Global variables and inputs
          StepOutputs: Map<string, Map<string, obj>> // StepId -> OutputName -> Value
          ExecutionTrace: StepExecutionTrace list }
