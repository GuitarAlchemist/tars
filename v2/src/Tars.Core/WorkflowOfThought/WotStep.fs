namespace Tars.Core.WorkflowOfThought

// Define StepAction and Step types that mirror what WotCompiler produces, 
// OR define them as shared types in Core for Compiler to use.
// For now, I will define them here to unblock compilation.
// Ideally WotCompiler should depend on Core types.

type StepAction = 
    | Work of WorkOperation
    | Reason of ReasonOperation

type Step = {
    Id: string
    Inputs: string list
    Outputs: string list
    Action: StepAction
}

// Re-exporting for clarity if needed by other modules
// or simply assume these are the 'Step' and 'Action' expected by executor
