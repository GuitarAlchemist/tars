namespace Tars.Metascript

open System
open Tars.Core
open Tars.Core.HybridBrain
open Tars.Core.WorkflowOfThought

/// <summary>
/// Compiles Hybrid Brain Plans (Typed IR) into executable Workflow of Thought graphs.
/// This is the bridge between neural planning and governed symbolic execution.
/// </summary>
module WotCompiler =

    /// Map a Step ID from Plan IR to a NodeId for WoT
    let private mapId (stepId: int) = NodeId.create()

    /// Compile a Plan Step into a WoT Node
    let compileStep (step: Step) (nodeId: NodeId) : WotNode =
        match step.Action with
        | Action.ExtractFunction(name, s, e) ->
            Work {
                Id = nodeId
                Name = step.Name
                Operation = WorkOperation.ToolCall ("refactor_extract_function", 
                    Map.ofList [("functionName", box name); ("startLine", box s); ("endLine", box e)])
                Input = NodeContent.ofText step.Description
                Output = None
                Budget = { NodeBudget.default' with MaxTokens = 0 } // Tools don't use reasoning tokens
                Policy = PolicyGate.strict
                Evidence = None
                SideEffects = ["filesystem"; "code"]
            }
        
        | Action.RemoveDeadCode(s, e) ->
            Work {
                Id = nodeId
                Name = step.Name
                Operation = WorkOperation.ToolCall ("refactor_remove_dead_code", 
                    Map.ofList [("startLine", box s); ("endLine", box e)])
                Input = NodeContent.ofText step.Description
                Output = None
                Budget = NodeBudget.minimal
                Policy = PolicyGate.strict
                Evidence = None
                SideEffects = ["filesystem"]
            }

        | Action.AddDocumentation(doc, line) ->
             Work {
                Id = nodeId
                Name = step.Name
                Operation = WorkOperation.ToolCall ("refactor_add_documentation", 
                    Map.ofList [("docText", box doc); ("lineNumber", box line)])
                Input = NodeContent.ofText step.Description
                Output = None
                Budget = NodeBudget.minimal
                Policy = PolicyGate.permissive
                Evidence = None
                SideEffects = ["filesystem"]
            }

        | Action.Summarize(content, maxLen) ->
            Reason {
                Id = nodeId
                Name = step.Name
                Operation = ReasonOperation.Synthesize [] // Will need context resolution
                Input = NodeContent.ofText content
                Output = None
                Model = None
                ModelHint = None
                Budget = NodeBudget.default'
                Policy = PolicyGate.permissive
                Evidence = None
            }

        | Action.UseTool tool ->
            let (toolName, toolArgs) = 
                match tool with
                | Tars.Core.HybridBrain.Tool.WebSearch q -> "search_web", Map.ofList [("query", box q)]
                | Tars.Core.HybridBrain.Tool.FileRead p -> "read_file_content", Map.ofList [("path", box p)]
                | Tars.Core.HybridBrain.Tool.FileWrite (p, c) -> "write_file", Map.ofList [("path", box p); ("content", box c)]
                | Tars.Core.HybridBrain.Tool.LlmCall (p, m) -> "llm_complete", Map.ofList [("prompt", box p); ("model", box (m |> Option.toObj))]
                | Tars.Core.HybridBrain.Tool.DatabaseQuery q -> "db_query", Map.ofList [("query", box q)]
                | Tars.Core.HybridBrain.Tool.SandboxExec (c, t) -> "sandbox_exec", Map.ofList [("command", box c); ("timeout", box t.TotalSeconds)]
                | Tars.Core.HybridBrain.Tool.Custom (n, a) -> n, a
                | _ -> "unknown_tool", Map.empty

            Work {
                Id = nodeId
                Name = step.Name
                Operation = WorkOperation.ToolCall (toolName, toolArgs)
                Input = NodeContent.ofText step.Description
                Output = None
                Budget = NodeBudget.default'
                Policy = PolicyGate.permissive
                Evidence = None
                SideEffects = []
            }

        | _ ->
            // Default to a Reason node for unknown actions (assume LLM-based execution)
            Reason {
                Id = nodeId
                Name = step.Name
                Operation = ReasonOperation.Explain (sprintf "Execute action: %A" step.Action)
                Input = NodeContent.ofText step.Description
                Output = None
                Model = None
                ModelHint = None
                Budget = NodeBudget.default'
                Policy = PolicyGate.permissive
                Evidence = None
            }

    /// Compile a full Plan into a WorkflowGraph
    let compile (plan: Plan<'State>) : WorkflowGraph =
        let idMap = plan.Steps |> List.map (fun s -> s.Id, mapId s.Id) |> Map.ofList
        
        let nodes = 
            plan.Steps 
            |> List.map (fun s -> 
                let nid = idMap.[s.Id]
                nid, compileStep s nid)
            |> Map.ofList

        // Simple linear dependency for now based on step order
        let mutable edges = []
        for i in 0 .. plan.Steps.Length - 2 do
            let fromId = idMap.[plan.Steps.[i].Id]
            let toId = idMap.[plan.Steps.[i+1].Id]
            edges <- edges @ [(toId, NodeEdge.DependsOn, fromId)]

        let entryPoint = if plan.Steps.IsEmpty then NodeId.create() else idMap.[plan.Steps.[0].Id]

        {
            Id = plan.Id
            Name = sprintf "WoT: %s" plan.Description
            Description = sprintf "Compiled from Plan %s" (plan.Goal.ToString())
            Nodes = nodes
            Edges = edges
            EntryPoint = entryPoint
            ExitPoints = if plan.Steps.IsEmpty then [] else [idMap.[plan.Steps |> List.last |> fun s -> s.Id]]
            GlobalBudget = { 
                Tars.Core.Budget.default' with 
                    MaxTokens = Some (plan.Budget.MaxTokens * 1<token>)
                    MaxCalls = Some (plan.Budget.MaxApiCalls * 1<requests> )
                    MaxRam = Some (plan.Budget.MaxMemory * 1L<bytes>)
            }
            GlobalPolicy = PolicyGate.permissive
            AuditLog = []
        }
