namespace Tars.Tools.Standard

open System
open System.IO
open System.Collections.Generic
open Tars.Tools

module DebugTools =

    [<TarsToolAttribute("debug_hint",
                        "Provides debugging hints for common F# errors. Input: error message or description")>]
    let debugHint (args: string) =
        task {
            let errorMessage = ToolHelpers.parseStringArg args "error"
            printfn $"🔍 DEBUG HINT for: %s{errorMessage.Substring(0, min 50 errorMessage.Length)}"

            let hints =
                let msg = errorMessage.ToLower()

                if msg.Contains("not defined") || msg.Contains("undefined") then
                    "Possible causes:\n"
                    + "  1. Missing 'open' statement for the module\n"
                    + "  2. Typo in function/variable name\n"
                    + "  3. File order in .fsproj - F# requires dependencies first\n"
                    + "  4. Missing project reference\n\n"
                    + "Try: Check for typos, verify imports, check file order"

                elif msg.Contains("type mismatch") || msg.Contains("expected") then
                    "Type mismatch hints:\n"
                    + "  1. Check return type annotations\n"
                    + "  2. Verify function argument types\n"
                    + "  3. Look for implicit conversions needed\n"
                    + "  4. Check Option/Result unwrapping\n\n"
                    + "Try: Add explicit type annotations to narrow down the issue"

                elif msg.Contains("null") then
                    "Null reference hints:\n"
                    + "  1. Use Option type instead of null\n"
                    + "  2. Add null checks at boundaries\n"
                    + "  3. Use 'isNull' for interop code\n"
                    + "  4. Consider using 'Option.ofObj'\n\n"
                    + "Try: Wrap nullable values in Option types"

                elif msg.Contains("async") || msg.Contains("task") then
                    "Async/Task hints:\n"
                    + "  1. Use 'let!' to await async operations\n"
                    + "  2. Use 'do!' for side effects\n"
                    + "  3. 'Async.RunSynchronously' blocks - use sparingly\n"
                    + "  4. Check Task vs Async mixing\n\n"
                    + "Try: Ensure async operations are properly awaited"

                elif msg.Contains("pattern") || msg.Contains("match") then
                    "Pattern matching hints:\n"
                    + "  1. Ensure all cases are covered\n"
                    + "  2. Use wildcard '_' for catch-all\n"
                    + "  3. Check discriminated union completeness\n"
                    + "  4. Verify pattern syntax\n\n"
                    + "Try: Add missing cases or wildcard pattern"

                else
                    "General debugging tips:\n"
                    + "  1. Read the full error message carefully\n"
                    + "  2. Check the line number and context\n"
                    + "  3. Simplify the code to isolate the issue\n"
                    + "  4. Add type annotations to clarify intent\n"
                    + "  5. Use 'printfn' for quick debugging output"

            return hints
        }

    [<TarsToolAttribute("trace_error",
                        "Traces through code to find error sources. Input JSON: { \"file\": \"path.fs\", \"line\": 42, \"error\": \"description\" }")>]
    let traceError (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let file = root.GetProperty("file").GetString()
                let line = root.GetProperty("line").GetInt32()

                let mutable errorProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let error =
                    if root.TryGetProperty("error", &errorProp) then
                        errorProp.GetString()
                    else
                        "unknown error"

                printfn $"🔎 TRACING ERROR at %s{file}:%d{line}"

                let fullPath = Path.GetFullPath(file)

                if not (File.Exists fullPath) then
                    return $"File not found: %s{fullPath}"
                else
                    let lines = File.ReadAllLines(fullPath)

                    if line < 1 || line > lines.Length then
                        return $"Line %d{line} out of range (file has %d{lines.Length} lines)"
                    else
                        let startLine = max 0 (line - 5)
                        let endLine = min (lines.Length - 1) (line + 5)

                        let context =
                            [| for i = startLine to endLine do
                                   let marker = if i = line - 1 then ">>> " else "    "
                                   yield $"%s{marker}%d{i + 1}: %s{lines.[i]}" |]
                            |> String.concat "\n"

                        return
                            $"Error context at %s{file}:%d{line}\n\n%s{context}\n\nError: %s{error}\n\nUse debug_hint with the error message for suggestions."
            with ex ->
                return "trace_error error: " + ex.Message
        }

    [<TarsToolAttribute("explain_error", "Explains a .NET/F# error code. Input: error code (e.g., FS0001, CS1002)")>]
    let explainError (args: string) =
        task {
            let errorCode = ToolHelpers.parseStringArg args "code"
            let code = errorCode.ToUpper().Trim()
            printfn $"📖 EXPLAINING: %s{code}"

            let explanation =
                match code with
                | "FS0001" -> "Type constraint mismatch - the inferred type doesn't match the expected type"
                | "FS0003" -> "This value is not a function - attempting to call something that isn't callable"
                | "FS0010" -> "Unexpected token in type expression - syntax error in type annotation"
                | "FS0025" -> "Incomplete pattern match - not all cases are covered"
                | "FS0039" -> "Undefined value - the name is not in scope"
                | "FS0041" -> "Unique overload could not be determined - ambiguous function call"
                | "FS0072" -> "Lookup on object of indeterminate type - need type annotation"
                | "FS0588" -> "The block ending with this expression has the wrong type"
                | "FS3350" -> "Feature not supported by target runtime"
                | "FS3373" -> "Invalid string interpolation - escape issues with curly braces"
                | _ ->
                    if code.StartsWith("FS") then
                        $"F# error %s{code} - check https://docs.microsoft.com/en-us/dotnet/fsharp/language-reference/compiler-messages/%s{code.ToLower()}"
                    elif code.StartsWith("CS") then
                        $"C# error %s{code} - check https://docs.microsoft.com/en-us/dotnet/csharp/misc/%s{code.ToLower()}"
                    else
                        $"Unknown error code: %s{code}. Try searching for it online."

            return $"Error Code: %s{code}\n\n%s{explanation}"
        }

module DocTools =

    [<TarsToolAttribute("generate_docs",
                        "Generates documentation for a function or type. Input JSON: { \"code\": \"let foo x = ...\", \"style\": \"xml|markdown\" }")>]
    let generateDocs (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let code = root.GetProperty("code").GetString()

                let mutable styleProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let style =
                    if root.TryGetProperty("style", &styleProp) then
                        styleProp.GetString()
                    else
                        "xml"

                printfn $"📝 GENERATING DOCS (style: %s{style})"

                // Simple function name extraction
                let funcName =
                    if code.Contains("let ") then
                        let start = code.IndexOf("let ") + 4
                        let endIdx = code.IndexOfAny([| ' '; '(' |], start)

                        if endIdx > start then
                            code.Substring(start, endIdx - start)
                        else
                            "function"
                    else
                        "item"

                let docs =
                    if style.ToLower() = "markdown" then
                        $"## `%s{funcName}`\n\n**Description**: [Add description]\n\n**Parameters**:\n- TODO: List parameters\n\n**Returns**: TODO\n\n**Example**:\n```fsharp\n%s{code}\n```"
                    else
                        $"/// <summary>\n/// [Add description for %s{funcName}]\n/// </summary>\n/// <param name=\"TODO\">Parameter description</param>\n/// <returns>Return value description</returns>"

                return
                    $"Generated documentation for '%s{funcName}':\n\n%s{docs}\n\nUse write_code or patch_code to add this documentation to your code."
            with ex ->
                return "generate_docs error: " + ex.Message
        }

    [<TarsToolAttribute("update_readme",
                        "Suggests README updates based on code changes. Input: description of changes made")>]
    let updateReadme (args: string) =
        task {
            let changes = ToolHelpers.parseStringArg args "changes"
            printfn "📄 SUGGESTING README UPDATES"

            let suggestion =
                "README Update Suggestions:\n\n"
                + "Based on changes: "
                + changes
                + "\n\n"
                + "Consider updating these sections:\n"
                + "  - [ ] Features list (if new features added)\n"
                + "  - [ ] Installation (if dependencies changed)\n"
                + "  - [ ] Usage examples (if API changed)\n"
                + "  - [ ] Configuration (if settings added)\n"
                + "  - [ ] Changelog/version notes\n\n"
                + "Use read_code with 'README.md' to view current README, then patch_code to update."

            return suggestion
        }

module MemoryTools =

    /// In-memory notes storage (persists for session)
    let private notes = Dictionary<string, string>()

    [<TarsToolAttribute("save_note",
                        "Saves a note for later recall. Input JSON: { \"key\": \"note_name\", \"content\": \"note content\" }")>]
    let saveNote (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let key = root.GetProperty("key").GetString()
                let content = root.GetProperty("content").GetString()

                printfn $"📌 SAVING NOTE: %s{key}"
                notes.[key] <- content

                return $"Saved note '%s{key}' (%d{content.Length} characters). Use recall_note to retrieve it."
            with ex ->
                return "save_note error: " + ex.Message
        }

    [<TarsToolAttribute("recall_note", "Recalls a previously saved note. Input: note key/name")>]
    let recallNote (args: string) =
        task {
            let key = ToolHelpers.parseStringArg args "key"
            let keyTrimmed = key.Trim()
            printfn $"📖 RECALLING NOTE: %s{keyTrimmed}"

            if notes.ContainsKey(keyTrimmed) then
                return $"Note '%s{keyTrimmed}':\n\n%s{notes.[keyTrimmed]}"
            else
                let available = String.concat ", " (notes.Keys |> Seq.map (sprintf "'%s'"))

                if notes.Count = 0 then
                    return $"Note '%s{keyTrimmed}' not found. No notes saved yet."
                else
                    return $"Note '%s{keyTrimmed}' not found. Available notes: %s{available}"
        }

    [<TarsToolAttribute("list_notes", "Lists all saved notes. No input required.")>]
    let listNotes (_: string) =
        task {
            printfn "📋 LISTING NOTES"

            if notes.Count = 0 then
                return "No notes saved yet. Use save_note to save notes during your work."
            else
                let noteList =
                    notes
                    |> Seq.map (fun kvp ->
                        let preview =
                            if kvp.Value.Length > 40 then
                                kvp.Value.Substring(0, 40) + "..."
                            else
                                kvp.Value

                        $"  - %s{kvp.Key}: %s{preview}")
                    |> String.concat "\n"

                return $"Saved notes (%d{notes.Count} total):\n%s{noteList}"
        }
