namespace Tars.Tools.Standard

open System
open System.Collections.Generic
open Tars.Tools

module ToolValidation =

    /// Track tool execution results for validation
    let private toolExecutionLog = ResizeArray<string * bool * string>() // (toolName, success, message)

    /// Record a tool execution result
    let recordExecution (toolName: string) (success: bool) (message: string) =
        toolExecutionLog.Add((toolName, success, message))

        if toolExecutionLog.Count > 100 then
            toolExecutionLog.RemoveAt(0)

    /// Known tools database
    let private knownTools =
        dict
            [ ("explore_project", ("Explore project structure", "path: string"))
              ("read_code", ("Read code file contents", "path: string"))
              ("patch_code", ("Patch/edit code files", "JSON: { file, search, replace }"))
              ("write_code", ("Write new code files", "JSON: { path, content }"))
              ("git_commit", ("Create git commit", "JSON: { message }"))
              ("git_status", ("Check git status", "none"))
              ("git_diff", ("Show git diff", "none"))
              ("run_tests", ("Run project tests", "optional path"))
              ("build_project", ("Build the project", "optional path"))
              ("generate_test", ("Generate test code", "JSON: { file, function }"))
              ("analyze_code", ("Analyze code quality", "path"))
              ("think_step_by_step", ("Chain-of-thought reasoning", "problem description"))
              ("plan_task", ("Create task plan", "task description"))
              ("summarize", ("Summarize content", "content to summarize"))
              ("lookup_docs", ("Search documentation", "topic"))
              ("improve_prompt", ("Enhance prompts", "prompt text"))
              ("reflect_on_task", ("Task reflection", "task description"))
              ("report_progress", ("Progress report", "none"))
              ("run_metascript", ("Execute metascript", "metascript code"))
              ("parse_metascript", ("Parse metascript", "metascript code"))
              ("create_metascript", ("Create metascript", "task description"))
              ("list_files", ("List directory files", "JSON: { path, pattern }"))
              ("search_code", ("Search in code", "JSON: { pattern, path }"))
              ("count_lines", ("Count code lines", "JSON: { path, pattern }"))
              ("find_todos", ("Find TODO comments", "path"))
              ("delegate_task", ("Delegate to agent", "JSON: { agent, task }"))
              ("request_review", ("Request code review", "JSON: { code, focus }"))
              ("query_agent", ("Query agent info", "JSON: { agent, question }"))
              ("list_agents", ("List all agents", "none"))
              ("agent_status", ("Get agent status", "agent name"))
              ("debug_hint", ("Get debug hints", "error message"))
              ("trace_error", ("Trace error context", "JSON: { file, line }"))
              ("explain_error", ("Explain error codes", "error code"))
              ("generate_docs", ("Generate documentation", "JSON: { code, style }"))
              ("update_readme", ("Suggest README updates", "changes description"))
              ("save_note", ("Save session note", "JSON: { key, content }"))
              ("recall_note", ("Recall saved note", "note key"))
              ("list_notes", ("List all notes", "none"))
              ("list_models", ("List LLM models", "none"))
              ("switch_model", ("Switch LLM model", "model name"))
              ("recommend_model", ("Recommend model", "task description"))
              ("pull_model", ("Download model", "model name"))
              ("model_info", ("Get model info", "model name"))
              ("get_active_model", ("Current model", "none"))
              ("validate_tool", ("Validate tool exists", "tool name"))
              ("test_tool", ("Test tool execution", "JSON: { tool, input }"))
              ("list_tool_errors", ("List tool errors", "optional filter"))
              ("introspect_tool", ("Tool introspection", "tool name"))
              ("list_all_tools", ("List all tools", "none")) ]

    [<TarsToolAttribute("validate_tool",
                        "Validates that a tool exists and shows its signature. Input: tool name to validate")>]
    let validateTool (toolName: string) =
        task {
            let name = toolName.Trim()
            printfn "🔍 VALIDATING TOOL: %s" name

            if knownTools.ContainsKey(name) then
                let (desc, input) = knownTools.[name]

                return
                    sprintf
                        "✅ Tool '%s' is valid.\n\n  Description: %s\n  Input: %s\n\nUse test_tool to run it with sample input."
                        name
                        desc
                        input
            else
                let suggestions =
                    knownTools.Keys
                    |> Seq.filter (fun k -> k.Contains(name) || name.Contains(k))
                    |> Seq.truncate 3
                    |> String.concat ", "

                if suggestions.Length > 0 then
                    return sprintf "❌ Tool '%s' not found. Did you mean: %s?" name suggestions
                else
                    return sprintf "❌ Tool '%s' not found. Use list_all_tools to see available tools." name
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

                printfn "🧪 TESTING TOOL: %s" toolName

                if knownTools.ContainsKey(toolName) then
                    recordExecution toolName true (sprintf "Test input: %s" (input.Substring(0, min 30 input.Length)))

                    return
                        sprintf
                            "Test recorded for '%s'.\n\nTo execute: Call the tool directly with your input.\nExample: %s \"%s\""
                            toolName
                            toolName
                            input
                else
                    return sprintf "Tool '%s' not found. Use list_all_tools to see available tools." toolName
            with ex ->
                return "test_tool error: " + ex.Message
        }

    [<TarsToolAttribute("list_tool_errors",
                        "Lists recent tool execution records. Input: optional filter (tool name or 'failed')")>]
    let listToolErrors (filter: string) =
        task {
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
                        sprintf "  %d. %s %s: %s" (i + 1) status name msg)
                    |> String.concat "\n"

                return sprintf "Tool execution log (%d entries):\n%s" filtered.Length logEntries
        }

    [<TarsToolAttribute("introspect_tool", "Shows detailed information about a tool. Input: tool name")>]
    let introspectTool (toolName: string) =
        task {
            let name = toolName.Trim()
            printfn "🔬 INTROSPECTING: %s" name

            if knownTools.ContainsKey(name) then
                let (desc, input) = knownTools.[name]

                return
                    sprintf
                        "Tool: %s\n\nDescription: %s\nExpected Input: %s\n\nUse validate_tool to confirm availability."
                        name
                        desc
                        input
            else
                return sprintf "Tool '%s' not found. Use list_all_tools to see available tools." name
        }

    [<TarsToolAttribute("list_all_tools", "Lists all registered tools with descriptions. No input required.")>]
    let listAllTools (_: string) =
        task {
            printfn "📋 LISTING ALL %d TOOLS" knownTools.Count

            let toolList =
                knownTools
                |> Seq.sortBy (fun kvp -> kvp.Key)
                |> Seq.mapi (fun i kvp ->
                    let (desc, _) = kvp.Value
                    sprintf "  %2d. %-22s %s" (i + 1) kvp.Key desc)
                |> String.concat "\n"

            return
                sprintf "Registered Tools (%d total):\n%s\n\nUse introspect_tool for details." knownTools.Count toolList
        }
