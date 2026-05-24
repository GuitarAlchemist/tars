namespace Tars.Tools.Standard

open System
open Tars.Tools

module ReasoningTools =

    /// Internal reasoning/planning tool - helps agent think step by step
    [<TarsToolAttribute("think_step_by_step",
                        "Use this tool to reason through a problem step by step before acting. Input: your thoughts/reasoning as a string.")>]
    let thinkStepByStep (args: string) =
        task {
            let thoughts = ToolHelpers.parseStringArg args "thoughts"
            printfn $"💭 THINKING: {thoughts.Substring(0, Math.Min(100, thoughts.Length))}..."
            // Return the thoughts back - this helps the agent structure its reasoning
            return $"Reasoning recorded: {thoughts}\n\nNow proceed with your next action."
        }

    /// Plan decomposition tool - breaks a complex task into steps
    [<TarsToolAttribute("plan_task",
                        "Decomposes a complex task into numbered steps. Input JSON: { \"task\": \"description\", \"context\": \"optional context\" }")>]
    let planTask (args: string) =
        task {
            try
                let taskDesc = ToolHelpers.parseStringArg args "task"
                printfn $"📋 PLANNING: {taskDesc.Substring(0, Math.Min(80, taskDesc.Length))}..."

                // Return a planning template for the agent to fill in
                return
                    $"""Task Analysis for: {taskDesc}

Please break this down into steps:
1. First, understand the requirements
2. Identify the files/code that need to be modified
3. Plan the specific changes
4. Implement each change
5. Verify the changes work

Now use read_code and explore_project to gather information, then execute your plan."""
            with ex ->
                return $"plan_task error: {ex.Message}"
        }

    /// Summary tool - helps agent summarize findings
    [<TarsToolAttribute("summarize", "Summarize your findings or results. Input: text to summarize.")>]
    let summarize (args: string) =
        task {
            let text = ToolHelpers.parseStringArg args "text"
            printfn $"📊 SUMMARY: {text.Substring(0, Math.Min(100, text.Length))}..."
            return $"Summary recorded. Now provide your final answer or take the next action."
        }

module ResearchTools =

    /// Documentation lookup for F# and .NET
    [<TarsToolAttribute("lookup_docs",
                        "Look up F# or .NET documentation. Input: topic to look up (e.g., 'List.map', 'async')")>]
    let lookupDocs (args: string) =
        task {
            let topic = ToolHelpers.parseStringArg args "topic"
            printfn $"📚 DOCS: Looking up '{topic}'"

            // Provide inline documentation for common F# topics
            let docs =
                match topic.ToLowerInvariant() with
                | t when t.Contains("list.map") ->
                    "List.map: ('T -> 'U) -> 'T list -> 'U list\nApplies a function to each element of a list, returning a new list."
                | t when t.Contains("async") ->
                    "F# Async:\n- async { } creates async computations\n- let! binds async results\n- do! runs async for side effects\n- Async.RunSynchronously runs async to completion"
                | t when t.Contains("option") ->
                    "F# Option:\n- Some value | None\n- Option.map, Option.bind, Option.defaultValue\n- Pattern match with | Some x -> ... | None -> ..."
                | t when t.Contains("result") ->
                    "F# Result:\n- Ok value | Error err\n- Result.map, Result.bind, Result.mapError\n- For railway-oriented programming"
                | t when t.Contains("task") ->
                    "F# Task:\n- task { } for .NET Tasks\n- let! for awaiting\n- Similar to async but interops better with C#"
                | t when t.Contains("union") || t.Contains("discriminated") ->
                    "Discriminated Unions (DU):\n- type Shape = | Circle of float | Rect of float * float\n- Powerful for modeling domain data\n- Always exhaustive when pattern matched"
                | t when t.Contains("record") ->
                    "F# Records:\n- type Person = { Name: string; Age: int }\n- Immutable by default\n- Structural equality and 'with' syntax for copy-and-update"
                | t when t.Contains("extension") ->
                    "Type Extensions:\n- type String with member x.IsLong = x.Length > 10\n- Add members to existing types without inheritance"
                | t when t.Contains("computation") || t.Contains("ce") ->
                    "Computation Expressions (CE):\n- Custom { } blocks (like async, task, seq)\n- Define Bind, Return, Yield etc. to customize behavior"
                | t when t.Contains("pattern") || t.Contains("match") ->
                    "F# Pattern Matching:\n- match x with | pattern -> result\n- Patterns: literals, wildcards (_), tuples, records, unions, lists, active patterns"
                | _ ->
                    $"No local quick-reference for '{topic}'.\n\nSearch Microsoft Learn: https://learn.microsoft.com/en-us/dotnet/fsharp/\nOr use search_web tool for real-time web search."

            return docs
        }
