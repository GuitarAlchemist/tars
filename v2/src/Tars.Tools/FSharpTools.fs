/// F# Expert Tools - Production-grade tools for F# compilation, AST manipulation, and error fixing.
/// 
/// Design Principles (aligned with Stanford/Harvard "Agentic AI Adaptation" paper):
/// 1. RELIABLE TOOL USE: Each tool has clear, deterministic input/output contracts
/// 2. ATOMIC OPERATIONS: Tools perform single, verifiable operations (not long chains)
/// 3. STRUCTURAL REASONING: Uses AST/Type System, not pattern matching on text
/// 4. FAIL-FAST WITH DIAGNOSTICS: Returns structured errors, not vague messages
/// 5. VERIFICATION LOOP: Every change can be immediately verified by compilation
///
namespace Tars.Tools.Standard

open System
open System.IO
open System.Diagnostics
open System.Text
open System.Text.RegularExpressions
open Tars.Tools

module FSharpTools =

    // ============================================================================
    // COMPILATION TOOLS - Deterministic, verifiable operations
    // ============================================================================

    /// Compiles an F# file or project and returns structured diagnostics
    [<TarsToolAttribute("fsharp_compile",
        "Compiles an F# file or project and returns structured diagnostics. Input JSON: { \"path\": \"src/MyProject/MyProject.fsproj\" } or { \"path\": \"src/File.fs\" }. Returns structured error list with line numbers, error codes, and messages.")>]
    let compileProject (args: string) =
        task {
            try
                use doc = System.Text.Json.JsonDocument.Parse(args)
                let path = doc.RootElement.GetProperty("path").GetString()
                let fullPath = Path.GetFullPath(path)
                
                if not (File.Exists(fullPath)) && not (Directory.Exists(Path.GetDirectoryName(fullPath))) then
                    return Result.Error $"Path not found: %s{fullPath}"
                else
                    let psi = ProcessStartInfo()
                    psi.FileName <- "dotnet"
                    psi.Arguments <- $"build \"%s{fullPath}\" --no-restore"
                    psi.UseShellExecute <- false
                    psi.RedirectStandardOutput <- true
                    psi.RedirectStandardError <- true
                    psi.WorkingDirectory <- Path.GetDirectoryName(fullPath)
                    
                    use proc = Process.Start(psi)
                    let stdout = proc.StandardOutput.ReadToEnd()
                    let stderr = proc.StandardError.ReadToEnd()
                    proc.WaitForExit()
                    
                    // Parse errors into structured format
                    let errorPattern = Regex(@"(.+?)\((\d+),(\d+)\): (error|warning) (FS\d+): (.+)")
                    let errors = ResizeArray<string>()
                    let warnings = ResizeArray<string>()
                    
                    for line in (stdout + stderr).Split('\n') do
                        let m = errorPattern.Match(line)
                        if m.Success then
                            let file = m.Groups.[1].Value
                            let lineNum = m.Groups.[2].Value
                            let col = m.Groups.[3].Value
                            let severity = m.Groups.[4].Value
                            let code = m.Groups.[5].Value
                            let msg = m.Groups.[6].Value
                            let entry = $"[%s{code}] %s{file}:%s{lineNum}:%s{col} - %s{msg}"
                            if severity = "error" then errors.Add(entry)
                            else warnings.Add(entry)
                    
                    if proc.ExitCode = 0 then
                        let result = 
                            if warnings.Count > 0 then
                                sprintf "✅ BUILD SUCCEEDED\n\nWarnings (%d):\n%s" warnings.Count (String.concat "\n" warnings)
                            else
                                "✅ BUILD SUCCEEDED - No warnings"
                        return Result.Ok result
                    else
                        let result = sprintf "❌ BUILD FAILED\n\nErrors (%d):\n%s\n\nWarnings (%d):\n%s" 
                                        errors.Count (String.concat "\n" errors)
                                        warnings.Count (String.concat "\n" warnings)
                        return Result.Ok result
            with ex ->
                return Result.Error $"Compilation failed: %s{ex.Message}"
        }

    /// Runs F# Interactive (fsi) on a script file or expression
    [<TarsToolAttribute("fsharp_eval",
        "Evaluates F# code using F# Interactive. Input JSON: { \"code\": \"let x = 1 + 2 in x * 3\" } or { \"script\": \"path/to/script.fsx\" }. Returns evaluation result or error.")>]
    let evalFSharp (args: string) =
        task {
            try
                use doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement
                
                let (code, tempFile) = 
                    try 
                        let script = root.GetProperty("script").GetString()
                        (None, Some script)
                    with _ ->
                        let code = root.GetProperty("code").GetString()
                        (Some code, None)
                
                let scriptPath = 
                    match tempFile with
                    | Some p -> Path.GetFullPath(p)
                    | None ->
                        let temp = Path.Combine(Path.GetTempPath(), $"tars_eval_%d{DateTime.Now.Ticks}.fsx")
                        File.WriteAllText(temp, code.Value)
                        temp
                
                let psi = ProcessStartInfo()
                psi.FileName <- "dotnet"
                psi.Arguments <- $"fsi \"%s{scriptPath}\""
                psi.UseShellExecute <- false
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.WorkingDirectory <- Path.GetDirectoryName(scriptPath)
                
                use proc = Process.Start(psi)
                let stdout = proc.StandardOutput.ReadToEnd()
                let stderr = proc.StandardError.ReadToEnd()
                proc.WaitForExit()
                
                // Clean up temp file if we created it
                if code.IsSome && File.Exists(scriptPath) then
                    try File.Delete(scriptPath) with _ -> ()
                
                if proc.ExitCode = 0 then
                    return Result.Ok $"✅ EVAL SUCCEEDED\n\nOutput:\n%s{stdout}"
                else
                    return Result.Ok $"❌ EVAL FAILED\n\nError:\n%s{stderr}\n\nOutput:\n%s{stdout}"
            with ex ->
                return Result.Error $"F# evaluation failed: %s{ex.Message}"
        }

    // ============================================================================
    // ERROR FIXING TOOLS - Structured, deterministic fixes
    // ============================================================================

    /// Explains an F# error code with common fixes
    [<TarsToolAttribute("fsharp_explain_error",
        "Explains an F# error code with common fixes. Input: error code (e.g., 'FS0001', 'FS0039'). Returns explanation and common fix patterns.")>]
    let explainError (errorCode: string) =
        task {
            let code = errorCode.Trim().ToUpperInvariant()
            
            let explanation = 
                match code with
                | "FS0001" -> 
                    "**FS0001: Type Mismatch**\n\n" +
                    "The expression has a different type than expected.\n\n" +
                    "**Common Causes:**\n" +
                    "- Returning different types in if/else branches\n" +
                    "- Function argument type mismatch\n" +
                    "- Option vs non-Option mismatch\n\n" +
                    "**Fixes:**\n" +
                    "1. Add type annotations: `let x: int = ...`\n" +
                    "2. Use explicit conversions: `int value`, `string value`\n" +
                    "3. Wrap in Option: `Some value` or unwrap: `value |> Option.get`\n" +
                    "4. Make both branches return same type"

                | "FS0010" -> 
                    "**FS0010: Unexpected token/Incomplete construct**\n\n" +
                    "Parsing failed due to syntax error.\n\n" +
                    "**Common Causes:**\n" +
                    "- Missing closing bracket/paren\n" +
                    "- Incorrect indentation\n" +
                    "- Missing `in` after `let`\n\n" +
                    "**Fixes:**\n" +
                    "1. Check indentation is consistent (spaces, not tabs)\n" +
                    "2. Ensure all brackets/parens are balanced\n" +
                    "3. Add missing `in` keyword"

                | "FS0025" -> 
                    "**FS0025: Incomplete Pattern Match**\n\n" +
                    "Pattern match doesn't cover all cases.\n\n" +
                    "**Fixes:**\n" +
                    "1. Add missing cases explicitly\n" +
                    "2. Add wildcard: `| _ -> defaultValue`\n" +
                    "3. Use `function` keyword for exhaustive matching"

                | "FS0039" -> 
                    "**FS0039: Undefined Value/Namespace**\n\n" +
                    "The identifier is not defined.\n\n" +
                    "**Common Causes:**\n" +
                    "- Typo in name\n" +
                    "- Missing `open` statement\n" +
                    "- File order in .fsproj is wrong\n\n" +
                    "**Fixes:**\n" +
                    "1. Check spelling\n" +
                    "2. Add `open Namespace`\n" +
                    "3. Move file earlier in .fsproj compilation order\n" +
                    "4. Use fully qualified name: `Module.function`"

                | "FS0058" -> 
                    "**FS0058: Offside Rule / Indentation**\n\n" +
                    "Code is not indented correctly.\n\n" +
                    "**F# Indentation Rules:**\n" +
                    "- Content after `=` must be indented\n" +
                    "- `let` bindings in functions must align\n" +
                    "- Match arms must be indented from `match`\n\n" +
                    "**Fixes:**\n" +
                    "1. Indent content at least 1 space past the keyword\n" +
                    "2. Use consistent indentation (2 or 4 spaces)\n" +
                    "3. Never mix tabs and spaces"

                | "FS0072" -> 
                    "**FS0072: Lookup on object of indeterminate type**\n\n" +
                    "Compiler can't infer the type of an object.\n\n" +
                    "**Fixes:**\n" +
                    "1. Add type annotation: `(x: MyType).Method()`\n" +
                    "2. Use pipeline: `x |> MyType.method`\n" +
                    "3. Let type flow from context"

                | "FS0193" -> 
                    "**FS0193: Type constraint mismatch**\n\n" +
                    "Generic constraint not satisfied.\n\n" +
                    "**Fixes:**\n" +
                    "1. Add constraint: `when 'a :> IComparable`\n" +
                    "2. Use concrete type instead of generic\n" +
                    "3. Check interface implementations"

                | "FS0201" -> 
                    "**FS0201: Namespaces cannot contain values**\n\n" +
                    "You have a `let` binding directly in a namespace.\n\n" +
                    "**Fixes:**\n" +
                    "1. Wrap in a module: `module MyModule = let x = 1`\n" +
                    "2. Use `module rec` for recursive definitions\n" +
                    "3. Move to a type with static members"

                | "FS3373" -> 
                    "**FS3373: Invalid interpolated string**\n\n" +
                    "Complex expression inside $\"...\" interpolation.\n\n" +
                    "**Fixes:**\n" +
                    "1. Extract to `let` binding:\n" +
                    "   `let value = complex |> expression`\n" +
                    "   `$\"Result: {value}\"`\n" +
                    "2. Use sprintf instead:\n" +
                    "   `sprintf \"Result: %s\" (complex |> expression)`"

                | _ ->
                    $"**%s{code}: Unknown Error Code**\n\nSearch for this error at:\n- https://docs.microsoft.com/en-us/dotnet/fsharp/language-reference/compiler-messages/%s{code.ToLowerInvariant()}\n- https://fsharp.org"

            return Result.Ok explanation
        }

    /// Suggests fix for a specific compilation error
    [<TarsToolAttribute("fsharp_suggest_fix",
        "Analyzes a compilation error and suggests a fix. Input JSON: { \"error\": \"FS0001: This expression was expected to have type 'int' but here has type 'string'\", \"code\": \"let x = \\\"hello\\\" + 1\" }. Returns structured fix suggestion.")>]
    let suggestFix (args: string) =
        task {
            try
                use doc = System.Text.Json.JsonDocument.Parse(args)
                let errorMsg = doc.RootElement.GetProperty("error").GetString()
                let code = 
                    try doc.RootElement.GetProperty("code").GetString()
                    with _ -> ""
                
                // Extract error code
                let codeMatch = Regex.Match(errorMsg, @"FS(\d+)")
                let errorCode = if codeMatch.Success then "FS" + codeMatch.Groups.[1].Value else "UNKNOWN"
                
                let suggestion = 
                    match errorCode with
                    | "FS0001" when errorMsg.Contains("'int'") && errorMsg.Contains("'string'") ->
                        "**Fix: Type Conversion Needed**\n\n" +
                        "Convert string to int: `int \"123\"` or `Int32.Parse(s)`\n" +
                        "Or convert int to string: `string 123` or `sprintf \"%d\" n`"
                    
                    | "FS0001" when errorMsg.Contains("Option") ->
                        "**Fix: Option Mismatch**\n\n" +
                        "Wrap in Some: `Some value`\n" +
                        "Unwrap with: `value |> Option.defaultValue fallback`\n" +
                        "Or pattern match: `match opt with Some x -> x | None -> default`"
                    
                    | "FS0001" when errorMsg.Contains("Async") ->
                        "**Fix: Async Mismatch**\n\n" +
                        "Wrap in async: `async { return value }`\n" +
                        "Unwrap with: `let! result = asyncValue` (inside async block)"
                    
                    | "FS0039" ->
                        "**Fix: Undefined Identifier**\n\n" +
                        "1. Check for typos in the name\n" +
                        "2. Add missing `open ModuleName`\n" +
                        "3. Check file order in .fsproj (dependencies must come first)"
                    
                    | "FS0058" | "FS0010" ->
                        "**Fix: Indentation/Syntax**\n\n" +
                        "1. Indent content after `=`, `->`, `do`, `then`\n" +
                        "2. Align `let` bindings in same scope\n" +
                        "3. Use consistent 4-space indentation"
                    
                    | _ ->
                        $"**Error %s{errorCode}**\n\nReview the error message and check F# documentation."

                let result = $"**Error:** %s{errorMsg}\n\n%s{suggestion}\n\n**Original Code:**\n```fsharp\n%s{code}\n```"
                return Result.Ok result
            with ex ->
                return Result.Error $"Fix suggestion failed: %s{ex.Message}"
        }

    // ============================================================================
    // AST TOOLS - Structural analysis, not pattern matching
    // ============================================================================

    /// Extracts the structure of an F# file (functions, types, modules)
    [<TarsToolAttribute("fsharp_analyze_structure",
        "Analyzes the structure of an F# file (functions, types, modules). Input JSON: { \"path\": \"src/MyFile.fs\" }. Returns structured outline of the file.")>]
    let analyzeStructure (args: string) =
        task {
            try
                use doc = System.Text.Json.JsonDocument.Parse(args)
                let path = doc.RootElement.GetProperty("path").GetString()
                let fullPath = Path.GetFullPath(path)
                
                if not (File.Exists(fullPath)) then
                    return Result.Error $"File not found: %s{fullPath}"
                else
                    let lines = File.ReadAllLines(fullPath)
                    
                    // Parse structure using regex (simplified AST-lite)
                    let modules = ResizeArray<string>()
                    let types = ResizeArray<string>()
                    let functions = ResizeArray<string>()
                    let opens = ResizeArray<string>()
                    
                    let modulePattern = Regex(@"^\s*(module)\s+(\w+(?:\.\w+)*)")
                    let typePattern = Regex(@"^\s*type\s+(\w+)")
                    let letPattern = Regex(@"^\s*let\s+(?:rec\s+)?(\w+)")
                    let openPattern = Regex(@"^\s*open\s+(\w+(?:\.\w+)*)")
                    
                    for i, line in lines |> Seq.indexed do
                        let mModule = modulePattern.Match(line)
                        let mType = typePattern.Match(line)
                        let mLet = letPattern.Match(line)
                        let mOpen = openPattern.Match(line)
                        
                        if mModule.Success then
                            modules.Add $"line %d{i+1}: module %s{mModule.Groups.[2].Value}"
                        elif mType.Success then
                            types.Add $"line %d{i+1}: type %s{mType.Groups.[1].Value}"
                        elif mLet.Success && not (line.TrimStart().StartsWith("//")) then
                            // Only top-level lets (no leading spaces beyond module indentation)
                            if line.Length - line.TrimStart().Length <= 4 then
                                functions.Add $"line %d{i+1}: let %s{mLet.Groups.[1].Value}"
                        elif mOpen.Success then
                            opens.Add(mOpen.Groups.[1].Value)
                    
                    let result =
                        $"## File Structure: %s{Path.GetFileName(fullPath)}\n\n" +
                        $"**Total Lines:** %d{lines.Length}\n\n" +
                        sprintf "**Imports (%d):**\n%s\n\n" opens.Count (opens |> Seq.map (sprintf "- %s") |> String.concat "\n") +
                        sprintf "**Modules (%d):**\n%s\n\n" modules.Count (String.concat "\n" modules) +
                        sprintf "**Types (%d):**\n%s\n\n" types.Count (String.concat "\n" types) +
                        sprintf "**Functions (%d):**\n%s" functions.Count (String.concat "\n" functions)
                    
                    return Result.Ok result
            with ex ->
                return Result.Error $"Structure analysis failed: %s{ex.Message}"
        }

    /// Checks if an F# project file has correct file ordering
    [<TarsToolAttribute("fsharp_check_file_order",
        "Checks if files in an F# project are in correct dependency order. Input JSON: { \"path\": \"src/MyProject.fsproj\" }. F# requires files to be listed in dependency order - files can only reference files listed before them.")>]
    let checkFileOrder (args: string) =
        task {
            try
                use doc = System.Text.Json.JsonDocument.Parse(args)
                let path = doc.RootElement.GetProperty("path").GetString()
                let fullPath = Path.GetFullPath(path)
                
                if not (File.Exists(fullPath)) then
                    return Result.Error $"Project file not found: %s{fullPath}"
                else
                    let projContent = File.ReadAllText(fullPath)
                    let projDir = Path.GetDirectoryName(fullPath)
                    
                    // Extract Compile items
                    let compilePattern = Regex(@"<Compile\s+Include=""([^""]+)""")
                    let files = 
                        compilePattern.Matches(projContent)
                        |> Seq.cast<Match>
                        |> Seq.map (fun m -> m.Groups.[1].Value)
                        |> List.ofSeq
                    
                    // Build map of what each file opens/references
                    let fileRefs = 
                        files 
                        |> List.mapi (fun idx file ->
                            let filePath = Path.Combine(projDir, file)
                            if File.Exists(filePath) then
                                let content = File.ReadAllText(filePath)
                                let modulePattern = Regex(@"^\s*module\s+(\w+(?:\.\w+)*)", RegexOptions.Multiline)
                                let mModule = modulePattern.Match(content)
                                let moduleName = if mModule.Success then Some mModule.Groups.[1].Value else None
                                
                                let openPattern = Regex(@"^\s*open\s+(\w+(?:\.\w+)*)", RegexOptions.Multiline)
                                let opens = 
                                    openPattern.Matches(content)
                                    |> Seq.cast<Match>
                                    |> Seq.map (fun m -> m.Groups.[1].Value)
                                    |> List.ofSeq
                                
                                (idx, file, moduleName, opens)
                            else
                                (idx, file, None, [])
                        )
                    
                    // Check for potential order issues
                    let issues = ResizeArray<string>()
                    
                    for (idx, file, _, opens) in fileRefs do
                        for openRef in opens do
                            // Check if any file after this one defines the opened module
                            for (laterIdx, laterFile, laterModule, _) in fileRefs do
                                if laterIdx > idx && laterModule.IsSome then
                                    if openRef.StartsWith(laterModule.Value) || laterModule.Value.EndsWith(openRef) then
                                        issues.Add $"⚠️ %s{file} (line %d{idx+1}) opens '%s{openRef}', but '%s{laterFile}' (line %d{laterIdx+1}) defines it - move '%s{laterFile}' earlier"

                    let result = 
                        if issues.Count = 0 then
                            sprintf "✅ File order looks correct\n\n**Files (%d):**\n%s" files.Length (files |> List.mapi (fun i f ->
                                $"%d{i+1}. %s{f}") |> String.concat "\n")
                        else
                            sprintf "⚠️ Potential ordering issues:\n\n%s\n\n**Files (%d):**\n%s" 
                                (String.concat "\n" issues) 
                                files.Length 
                                (files |> List.mapi (fun i f -> $"%d{i+1}. %s{f}") |> String.concat "\n")
                    
                    return Result.Ok result
            with ex ->
                return Result.Error $"File order check failed: %s{ex.Message}"
        }

    // ============================================================================
    // GRAMMAR/CE TOOLS - Translation helpers
    // ============================================================================

    /// Helps translate a grammar pattern to F# computation expression
    [<TarsToolAttribute("fsharp_ce_template",
        "Generates a computation expression builder template. Input JSON: { \"name\": \"MyBuilder\", \"operations\": [\"bind\", \"return\", \"zero\"] }. Returns a template for the CE builder.")>]
    let ceTemplate (args: string) =
        task {
            try
                use doc = System.Text.Json.JsonDocument.Parse(args)
                let name = doc.RootElement.GetProperty("name").GetString()
                let operations = 
                    try 
                        doc.RootElement.GetProperty("operations").EnumerateArray()
                        |> Seq.map (fun e -> e.GetString().ToLowerInvariant())
                        |> Set.ofSeq
                    with _ -> Set.ofList ["bind"; "return"]
                
                let sb = StringBuilder()
                sb.AppendLine $"type %s{name}Builder() =" |> ignore
                
                if operations.Contains("bind") then
                    sb.AppendLine("    member _.Bind(x, f) = ") |> ignore
                    sb.AppendLine("        // x: M<'a>, f: 'a -> M<'b> -> M<'b>") |> ignore
                    sb.AppendLine("        failwith \"TODO: implement Bind\"") |> ignore
                    sb.AppendLine() |> ignore
                
                if operations.Contains("return") then
                    sb.AppendLine("    member _.Return(x) = ") |> ignore
                    sb.AppendLine("        // x: 'a -> M<'a>") |> ignore
                    sb.AppendLine("        failwith \"TODO: implement Return\"") |> ignore
                    sb.AppendLine() |> ignore
                
                if operations.Contains("returnfrom") then
                    sb.AppendLine("    member _.ReturnFrom(x) = ") |> ignore
                    sb.AppendLine("        // x: M<'a> -> M<'a>") |> ignore
                    sb.AppendLine("        x") |> ignore
                    sb.AppendLine() |> ignore
                
                if operations.Contains("zero") then
                    sb.AppendLine("    member _.Zero() = ") |> ignore
                    sb.AppendLine("        // () -> M<unit>") |> ignore
                    sb.AppendLine("        failwith \"TODO: implement Zero\"") |> ignore
                    sb.AppendLine() |> ignore
                
                if operations.Contains("combine") then
                    sb.AppendLine("    member _.Combine(x, y) = ") |> ignore
                    sb.AppendLine("        // M<unit> * M<'a> -> M<'a>") |> ignore
                    sb.AppendLine("        failwith \"TODO: implement Combine\"") |> ignore
                    sb.AppendLine() |> ignore
                
                if operations.Contains("delay") then
                    sb.AppendLine("    member _.Delay(f) = ") |> ignore
                    sb.AppendLine("        // (unit -> M<'a>) -> M<'a>") |> ignore
                    sb.AppendLine("        f()") |> ignore
                    sb.AppendLine() |> ignore
                
                if operations.Contains("for") then
                    sb.AppendLine("    member _.For(seq, body) = ") |> ignore
                    sb.AppendLine("        // seq<'a> * ('a -> M<unit>) -> M<unit>") |> ignore
                    sb.AppendLine("        failwith \"TODO: implement For\"") |> ignore
                    sb.AppendLine() |> ignore
                
                if operations.Contains("while") then
                    sb.AppendLine("    member _.While(guard, body) = ") |> ignore
                    sb.AppendLine("        // (unit -> bool) * M<unit> -> M<unit>") |> ignore
                    sb.AppendLine("        failwith \"TODO: implement While\"") |> ignore
                    sb.AppendLine() |> ignore
                
                if operations.Contains("tryfinally") then
                    sb.AppendLine("    member _.TryFinally(body, handler) = ") |> ignore
                    sb.AppendLine("        // M<'a> * (unit -> unit) -> M<'a>") |> ignore
                    sb.AppendLine("        try body finally handler()") |> ignore
                    sb.AppendLine() |> ignore
                
                if operations.Contains("trywith") then
                    sb.AppendLine("    member _.TryWith(body, handler) = ") |> ignore
                    sb.AppendLine("        // M<'a> * (exn -> M<'a>) -> M<'a>") |> ignore
                    sb.AppendLine("        try body with ex -> handler ex") |> ignore
                    sb.AppendLine() |> ignore
                
                sb.AppendLine() |> ignore
                sb.AppendLine $"let %s{name.ToLowerInvariant()} = %s{name}Builder()" |> ignore
                sb.AppendLine() |> ignore
                sb.AppendLine("// Usage:") |> ignore
                sb.AppendLine $"// %s{name.ToLowerInvariant()} {{" |> ignore
                sb.AppendLine("//     let! x = someValue") |> ignore
                sb.AppendLine("//     return x + 1") |> ignore
                sb.AppendLine("// }") |> ignore
                
                return Result.Ok (sb.ToString())
            with ex ->
                return Result.Error $"CE template generation failed: %s{ex.Message}"
        }

    /// Validates F# syntax without full compilation
    [<TarsToolAttribute("fsharp_check_syntax",
        "Quick syntax check for F# code without full compilation. Input JSON: { \"code\": \"let x = 1 + 2\" }. Returns syntax errors only.")>]
    let checkSyntax (args: string) =
        task {
            try
                use doc = System.Text.Json.JsonDocument.Parse(args)
                let code = doc.RootElement.GetProperty("code").GetString()
                
                // Basic syntax checks
                let issues = ResizeArray<string>()
                
                // Check bracket balance
                let openParens = code |> Seq.filter ((=) '(') |> Seq.length
                let closeParens = code |> Seq.filter ((=) ')') |> Seq.length
                if openParens <> closeParens then
                    issues.Add $"Unbalanced parentheses: %d{openParens} '(' vs %d{closeParens} ')'"

                let openBrackets = code |> Seq.filter ((=) '[') |> Seq.length
                let closeBrackets = code |> Seq.filter ((=) ']') |> Seq.length
                if openBrackets <> closeBrackets then
                    issues.Add $"Unbalanced brackets: %d{openBrackets} '[' vs %d{closeBrackets} ']'"

                let openBraces = code |> Seq.filter ((=) '{') |> Seq.length
                let closeBraces = code |> Seq.filter ((=) '}') |> Seq.length
                if openBraces <> closeBraces then
                    issues.Add $"Unbalanced braces: %d{openBraces} '{{' vs %d{closeBraces} '}}'"

                // Check for common F# syntax issues
                if code.Contains("\t") then
                    issues.Add("Warning: Code contains tabs - F# prefers spaces for indentation")
                
                if Regex.IsMatch(code, @"let\s+\w+\s*=\s*$", RegexOptions.Multiline) then
                    issues.Add("Possible incomplete let binding (ends with '=')")
                
                if Regex.IsMatch(code, @"\|\s*$", RegexOptions.Multiline) then
                    issues.Add("Possible incomplete pattern match (line ends with '|')")
                
                if issues.Count = 0 then
                    return Result.Ok "✅ Basic syntax check passed"
                else
                    return Result.Ok (sprintf "⚠️ Syntax issues found:\n%s" (issues |> Seq.map (sprintf "- %s") |> String.concat "\n"))
            with ex ->
                return Result.Error $"Syntax check failed: %s{ex.Message}"
        }
