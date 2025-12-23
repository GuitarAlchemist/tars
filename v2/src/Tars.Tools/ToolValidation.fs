namespace Tars.Tools.Standard

open System
open System.Collections.Generic
open Tars.Core
open Tars.Tools

module ToolValidation =

    /// Track tool execution results for validation
    let private toolExecutionLog = ResizeArray<string * bool * string>() // (toolName, success, message)

    /// Record a tool execution result
    let recordExecution (toolName: string) (success: bool) (message: string) =
        toolExecutionLog.Add((toolName, success, message))

        if toolExecutionLog.Count > 100 then
            toolExecutionLog.RemoveAt(0)

    /// Dynamic list of tools (injected at runtime)
    let mutable private dynamicTools: Tool list = []

    /// Initialize the validation module with actual tools
    let setKnownTools (tools: Tool list) = dynamicTools <- tools

    /// Helper to get tool info (case-insensitive)
    let private getToolInfo (name: string) =
        let target = name.Trim().ToLowerInvariant()

        dynamicTools
        |> List.tryFind (fun t -> t.Name.ToLowerInvariant() = target)
        |> Option.map (fun t -> (t.Description, "Checking implementation..."))

    [<TarsToolAttribute("validate_tool",
                        "Validates that a tool exists and shows its signature. Input: tool name to validate")>]
    let validateTool (args: string) =
        task {
            let toolName = ToolHelpers.parseStringArg args "tool_name"
            let name = toolName.Trim()
            printfn $"🔍 VALIDATING TOOL: %s{name}"

            match getToolInfo name with
            | Some(desc, _) ->
                return
                    $"✅ Tool '%s{name}' is valid.\n\n  Description: %s{desc}\n\nUse test_tool to run it with sample input."
            | None ->
                let suggestions =
                    dynamicTools
                    |> Seq.map (fun t -> t.Name)
                    |> Seq.filter (fun k -> k.Contains(name) || name.Contains(k))
                    |> Seq.truncate 3
                    |> String.concat ", "

                if suggestions.Length > 0 then
                    return $"❌ Tool '%s{name}' not found. Did you mean: %s{suggestions}?"
                else
                    return $"❌ Tool '%s{name}' not found. Use list_all_tools to see available tools."
        }

    [<TarsToolAttribute("test_tool",
                        "Tests a tool with sample input (records test request). Input JSON: { \"tool\": \"tool_name\", \"input\": \"test input\" }")>]
    let testTool (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let toolName = root.GetProperty("tool").GetString()
                let input = root.GetProperty("input").GetString()

                printfn $"🧪 TESTING TOOL: %s{toolName}"

                match getToolInfo toolName with
                | Some _ ->
                    recordExecution toolName true $"Test input: %s{input.Substring(0, min 30 input.Length)}"

                    return
                        $"Test recorded for '%s{toolName}'.\n\nTo execute: Call the tool directly with your input.\nExample: %s{toolName} \"%s{input}\""
                | None -> return $"Tool '%s{toolName}' not found. Use list_all_tools to see available tools."
            with ex ->
                return "test_tool error: " + ex.Message
        }

    [<TarsToolAttribute("list_tool_errors",
                        "Lists recent tool execution records. Input: optional filter (tool name or 'failed')")>]
    let listToolErrors (args: string) =
        task {
            let filter = ToolHelpers.parseStringArg args "filter"
            printfn "📋 LISTING TOOL EXECUTION LOG"

            let filtered =
                if String.IsNullOrWhiteSpace(filter) then
                    toolExecutionLog |> Seq.toList
                elif filter.ToLower() = "failed" then
                    toolExecutionLog
                    |> Seq.filter (fun (_, success, _) -> not success)
                    |> Seq.toList
                else
                    toolExecutionLog
                    |> Seq.filter (fun (name, _, _) -> name.Contains(filter))
                    |> Seq.toList

            if filtered.Length = 0 then
                return "No tool execution records yet. Records are created when tools execute."
            else
                let logEntries =
                    filtered
                    |> List.mapi (fun i (name, success, msg) ->
                        let status = if success then "✅" else "❌"
                        $"  %d{i + 1}. %s{status} %s{name}: %s{msg}")
                    |> String.concat "\n"

                return $"Tool execution log (%d{filtered.Length} entries):\n%s{logEntries}"
        }

    [<TarsToolAttribute("introspect_tool", "Shows detailed information about a tool. Input: tool name")>]
    let introspectTool (args: string) =
        task {
            let toolName = ToolHelpers.parseStringArg args "tool_name"
            let name = toolName.Trim()
            printfn $"🔬 INTROSPECTING: %s{name}"

            match getToolInfo name with
            | Some(desc, _) ->
                printfn $"   ✅ FOUND: %s{desc}"
                // Make the output unmistakably clear for the agent
                return $"### TOOL DEFINITION: %s{name} ###\n\n%s{desc}\n\n(This is the full definition)"
            | None ->
                printfn $"   ❌ NOT FOUND. Searching in %d{dynamicTools.Length} tools..."
                return $"Tool '%s{name}' not found. Use list_all_tools to see available tools."
        }

    [<TarsToolAttribute("list_all_tools", "Lists all registered tools with descriptions. No input required.")>]
    let listAllTools (_: string) =
        task {
            printfn $"📋 LISTING ALL %d{dynamicTools.Length} TOOLS"

            let toolList =
                dynamicTools
                |> List.sortBy (fun t -> t.Name)
                |> List.mapi (fun i t -> $"  %2d{i + 1}. %-22s{t.Name} %s{t.Description}")
                |> String.concat "\n"

            return $"Registered Tools (%d{dynamicTools.Length} total):\n%s{toolList}"
        }
