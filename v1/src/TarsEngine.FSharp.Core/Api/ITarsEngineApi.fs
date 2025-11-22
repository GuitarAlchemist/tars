namespace TarsEngine.FSharp.Core.Api

open TarsEngine.FSharp.Core.Types

/// TARS Engine API interface
module ITarsEngineApi =
    
    /// Main TARS Engine API
    type ITarsEngineApi =
        abstract member SearchVector: string * int -> Async<{| Id: int; Content: string; Score: float |} list>
        abstract member AskLlm: string * string -> Async<string>
        abstract member SpawnAgent: string * AgentConfig -> string
        abstract member WriteFile: string * string -> bool
        abstract member ExecuteMetascript: string -> Async<ExecutionResult>
