/// Bridge adapter that wraps TARS Tool instances as MAF AIFunction/AITool objects.
/// This allows all 152 TARS tools to be surfaced in MAF orchestrations without
/// rewriting any tool implementation.
module Tars.Tools.MafToolAdapter

open System
open System.Threading.Tasks
open Microsoft.Extensions.AI
open Tars.Core

/// Wraps a single TARS Tool as a MAF AIFunction.
/// The resulting AIFunction accepts a single "input" string parameter
/// and returns the tool's output string (or error message).
let toAIFunction (tool: Tool) : AIFunction =
    let wrapper =
        Func<string, Task<string>>(fun (input: string) ->
            task {
                let! result = tool.Execute input |> Async.StartAsTask
                return
                    match result with
                    | Result.Ok value -> value
                    | Result.Error msg -> $"Error: {msg}"
            })

    AIFunctionFactory.Create(wrapper, tool.Name, tool.Description)

/// Converts all tools in an IToolRegistry to MAF AIFunction instances.
let toAIFunctions (registry: IToolRegistry) : AIFunction list =
    registry.GetAll() |> List.map toAIFunction

/// Converts all tools in an IToolRegistry to MAF AITool instances.
/// (AIFunction inherits from AITool, so this is a simple upcast.)
let toAITools (registry: IToolRegistry) : AITool list =
    registry.GetAll() |> List.map (fun t -> toAIFunction t :> AITool)
