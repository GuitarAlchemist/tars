namespace Tars.Core.WorkflowOfThought

open System

/// Context for execution
type ExecContext =
  { Inputs : Map<string,string>
    Vars   : Map<string,obj> }

/// Platform-agnostic interface for invoking tools
type IToolInvoker =
  abstract Invoke : toolName:string * args:Map<string,string> -> Async<Result<obj,string>>

/// Interface for reasoning capabilities (LLM)
type IReasoner =
  abstract Reason : stepId:string * context:ExecContext * goal:string option * instruction:string option -> Async<Result<string,string>>

type ReasonStepMode = 
  | Stub 
  | Llm
  | Replay
