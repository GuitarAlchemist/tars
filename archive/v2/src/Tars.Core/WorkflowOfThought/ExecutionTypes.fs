namespace Tars.Core.WorkflowOfThought

// =============================================================================
// GENERIC METADATA
// =============================================================================

/// Generic JSON-like value for extensible metadata
type MetaValue =
    | MStr of string
    | MNum of decimal
    | MBool of bool
    | MArr of MetaValue list
    | MObj of Map<string, MetaValue>

/// Metadata map for workflows, nodes, and edges
type Meta = Map<string, MetaValue>

/// Context for execution
type ExecContext =
    { Inputs: Map<string, string>
      Vars: Map<string, obj> }

/// Platform-agnostic interface for invoking tools
type IToolInvoker =
    abstract Invoke: toolName: string * args: Map<string, string> -> Async<Result<obj, string>>

/// Token usage for reasoning
type TokenUsage = { Prompt: int; Completion: int }

/// Result of reasoning operation
type ReasoningResult =
    { Content: string
      Usage: TokenUsage option }

/// Interface for reasoning capabilities (LLM)
type IReasoner =
    abstract Reason:
        stepId: string * context: ExecContext * goal: string option * instruction: string option * agent: string option ->
            Async<Result<ReasoningResult, string>>

type ReasonStepMode =
    | Stub
    | Llm
    | Replay
