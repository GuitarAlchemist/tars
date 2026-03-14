namespace Tars.Tools.Standard

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

    /// Global registry reference for validation tools
    let mutable private globalRegistry: IToolRegistry option = None

    /// Initialize the validation module with the actual registry
    let setRegistry (registry: IToolRegistry) = globalRegistry <- Some registry

    /// Helper to get tool info (case-insensitive)
    let private getToolInfo (name: string) =
        let target = name.Trim().ToLowerInvariant()
        
        match globalRegistry with
        | Some reg ->
            reg.GetAll()
            |> List.tryFind (fun (t: Tars.Core.Tool) -> t.Name.ToLowerInvariant() = target)
            |> Option.map (fun t -> (t.Description, "Tool implementation verified."))
        | None -> None

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
                    match globalRegistry with
                    | Some reg ->
                        reg.GetAll()
                        |> Seq.map (fun t -> t.Name)
                        |> Seq.filter (fun k -> k.ToLowerInvariant().Contains(name.ToLowerInvariant()))
                        |> Seq.truncate 3
                        |> String.concat ", "
                    | None -> ""

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
                return $"Tool '%s{name}' not found. Use list_all_tools to see available tools."
        }

    [<TarsToolAttribute("list_all_tools", "Lists all registered tools with descriptions. No input required.")>]
    let listAllTools (_: string) =
        task {
            match globalRegistry with
            | Some reg ->
                let tools = reg.GetAll()
                printfn $"📋 LISTING ALL %d{tools.Length} TOOLS"

                let toolList =
                    tools
                    |> List.sortBy (fun t -> t.Name)
                    |> List.mapi (fun i t -> $"  %2d{i + 1}. %-22s{t.Name} %s{t.Description}")
                    |> String.concat "\n"

                return $"Registered Tools (%d{tools.Length} total):\n%s{toolList}"
            | None -> return "Tool registry not initialized in Validation module."
        }
