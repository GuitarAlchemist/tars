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

/// The result of invoking a tool through the invoker seam. Unlike a flat
/// Result<obj,string>, this admits the distinct failure modes the invoker knows
/// about, so callers no longer have to string-match to tell them apart.
type ToolOutcome =
    | Succeeded of output: string
    | NotFound
    | CircuitOpen
    | Failed of category: string * message: string

/// Platform-agnostic interface for invoking tools, with resilience (circuit
/// breaker) and recording owned by the implementation.
type IToolInvoker =
    abstract Invoke: toolName: string * args: Map<string, string> -> Async<ToolOutcome>

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
