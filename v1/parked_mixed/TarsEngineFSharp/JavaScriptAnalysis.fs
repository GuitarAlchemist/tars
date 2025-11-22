namespace TarsEngineFSharp

module JavaScriptAnalysis =
    open System
    open System.IO
    open System.Text.RegularExpressions

    // Simple types for code issues and fixes
    type CodeIssueType =
        | MissingNullCheck
        | IneffectiveLoop
        | UnusedVariable
        | MissingErrorHandling
        | IneffectiveStringConcatenation
        | PerformanceIssue
        | SecurityVulnerability
        | AccessibilityIssue

    type CodeIssue = {
        Type: CodeIssueType
        Location: string
        Description: string
    }

    type CodeFix = {
        Original: string
        Replacement: string
        Description: string
    }

    type AnalysisResult = {
        FilePath: string
        Issues: CodeIssue list
        Fixes: CodeFix list
    }

    // Helper function to get line number from position in text
    let getLineNumber (text: string) (position: int) =
        let mutable lineCount = 1
        for i in 0 .. min position (text.Length - 1) do
            if text.[i] = '\n' then lineCount <- lineCount + 1
        lineCount

    // Pattern detection functions using regex for JavaScript
    let detectMissingNullChecks (code: string) =
        // Find function parameters that might need null checks
        let functionRegex = Regex(@"function\s+(\w+)\s*\(([^)]*)\)", RegexOptions.Compiled)
        let matches = functionRegex.Matches(code)
        
        let issues = ResizeArray<CodeIssue * CodeFix>()
        
        for m in matches do
            let functionName = m.Groups.[1].Value
            let parameters = m.Groups.[2].Value.Split(',') |> Array.map (fun p -> p.Trim()) |> Array.filter (fun p -> p <> "")
            
            // Check if there are null checks for these parameters
            for param in parameters do
                // Simple check for null checks
                let nullCheckRegex = Regex(sprintf "if\\s*\\(\\s*%s\\s*===\\s*null|if\\s*\\(\\s*%s\\s*==\\s*null|if\\s*\\(!\\s*%s\\s*\\)" param param param)
                if not (nullCheckRegex.IsMatch(code)) then
                    let lineNumber = getLineNumber code m.Index
                    
                    let issue = {
                        Type = MissingNullCheck
                        Location = sprintf "Line %d" lineNumber
                        Description = sprintf "Parameter '%s' in function '%s' is not checked for null/undefined" param functionName
                    }
                    
                    // Create a fix
                    let functionBodyStart = code.IndexOf('{', m.Index) + 1
                    let indentation = "    " // Assume 4-space indentation
                    let nullCheck = sprintf "%sif (%s === null || %s === undefined) { throw new Error('%s is null or undefined'); }\n" indentation param param param
                    
                    let original = code.Substring(m.Index, functionBodyStart - m.Index + 1)
                    let replacement = original + "\n" + nullCheck
                    
                    let fix = {
                        Original = original
                        Replacement = replacement
                        Description = sprintf "Add null check for parameter '%s'" param
                    }
                    
                    issues.Add((issue, fix))
        
        issues |> Seq.toList

    let detectIneffectiveLoops (code: string) =
        // Find for loops that could be replaced with array methods
        let forLoopRegex = Regex(@"for\s*\(\s*(?:var|let|const)?\s*(\w+)\s*=\s*0\s*;\s*\1\s*<\s*(\w+)(?:\.length)?\s*;\s*\1\s*\+\+\s*\)\s*{([^}]*)}", RegexOptions.Compiled)
        let matches = forLoopRegex.Matches(code)
        
        let issues = ResizeArray<CodeIssue * CodeFix>()
        
        for m in matches do
            let indexVar = m.Groups.[1].Value
            let arrayName = m.Groups.[2].Value
            let loopBody = m.Groups.[3].Value
            
            // Check if this is a map, filter, or reduce operation
            if loopBody.Contains("+=") || loopBody.Contains("=") then
                let lineNumber = getLineNumber code m.Index
                
                let issue = {
                    Type = IneffectiveLoop
                    Location = sprintf "Line %d" lineNumber
                    Description = "For loop could be replaced with array method (map, filter, reduce)"
                }
                
                // Determine which array method to use
                let arrayMethod, newBody =
                    if loopBody.Contains("+=") then
                        // Likely a reduce operation
                        "reduce", sprintf "%s.reduce((acc, item) => acc + item, 0)" arrayName
                    elif loopBody.Contains("push") then
                        // Likely a map or filter operation
                        if loopBody.Contains("if") then
                            "filter", sprintf "%s.filter(item => %s)" arrayName (loopBody.Replace(indexVar, "item"))
                        else
                            "map", sprintf "%s.map(item => %s)" arrayName (loopBody.Replace(indexVar, "item"))
                    else
                        // Default to forEach
                        "forEach", sprintf "%s.forEach(item => %s)" arrayName (loopBody.Replace(indexVar, "item"))
                
                let fix = {
                    Original = m.Value
                    Replacement = newBody
                    Description = sprintf "Replace for loop with array.%s()" arrayMethod
                }
                
                issues.Add((issue, fix))
        
        issues |> Seq.toList

    let detectUnusedVariables (code: string) =
        // Find variable declarations
        let varRegex = Regex(@"(?:var|let|const)\s+(\w+)\s*=", RegexOptions.Compiled)
        let matches = varRegex.Matches(code)
        
        let issues = ResizeArray<CodeIssue * CodeFix>()
        
        for m in matches do
            let varName = m.Groups.[1].Value
            
            // Check if the variable is used elsewhere (simple approach)
            let varUsageRegex = Regex(sprintf "[^\\w](%s)[^\\w=]" varName)
            let usageMatches = varUsageRegex.Matches(code)
            
            // If only one match (the declaration), it's unused
            if usageMatches.Count <= 1 then
                let lineNumber = getLineNumber code m.Index
                
                let issue = {
                    Type = UnusedVariable
                    Location = sprintf "Line %d" lineNumber
                    Description = sprintf "Variable '%s' is declared but never used" varName
                }
                
                // Find the whole declaration statement
                let statementEnd = code.IndexOf(';', m.Index)
                let original = code.Substring(m.Index, statementEnd - m.Index + 1)
                
                let fix = {
                    Original = original
                    Replacement = sprintf "// Removed unused variable: %s" original
                    Description = sprintf "Remove unused variable '%s'" varName
                }
                
                issues.Add((issue, fix))
        
        issues |> Seq.toList

    let detectMissingErrorHandling (code: string) =
        // Find async functions or promises without catch
        let asyncRegex = Regex(@"(async\s+function|\w+\s*\.\s*then)\s*\(", RegexOptions.Compiled)
        let matches = asyncRegex.Matches(code)
        
        let issues = ResizeArray<CodeIssue * CodeFix>()
        
        for m in matches do
            // Check if there's a catch block
            let catchRegex = Regex(@"\.catch\s*\(")
            let hasCatch = catchRegex.IsMatch(code.Substring(m.Index))
            
            if not hasCatch then
                let lineNumber = getLineNumber code m.Index
                
                let issue = {
                    Type = MissingErrorHandling
                    Location = sprintf "Line %d" lineNumber
                    Description = "Async operation without error handling"
                }
                
                // Find the end of the then block
                let blockStart = code.IndexOf('{', m.Index)
                let mutable blockEnd = -1
                let mutable depth = 1
                let mutable pos = blockStart + 1
                
                while depth > 0 && pos < code.Length do
                    match code.[pos] with
                    | '{' -> depth <- depth + 1
                    | '}' -> depth <- depth - 1
                    | _ -> ()
                    pos <- pos + 1
                
                if blockEnd = -1 then pos <- code.IndexOf('}', m.Index)
                
                let original = code.Substring(m.Index, pos - m.Index)
                let replacement = original + ".catch(error => { console.error('Error:', error); })"
                
                let fix = {
                    Original = original
                    Replacement = replacement
                    Description = "Add error handling with .catch()"
                }
                
                issues.Add((issue, fix))
        
        issues |> Seq.toList

    let detectIneffectiveStringConcatenation (code: string) =
        // Find string concatenation with +
        let concatRegex = Regex(@"""[^""]*""\s*\+\s*(?:""[^""]*""|\w+)", RegexOptions.Compiled)
        let matches = concatRegex.Matches(code)
        
        let issues = ResizeArray<CodeIssue * CodeFix>()
        
        for m in matches do
            let lineNumber = getLineNumber code m.Index
            
            let issue = {
                Type = IneffectiveStringConcatenation
                Location = sprintf "Line %d" lineNumber
                Description = "String concatenation could be replaced with template literals"
            }
            
            // Convert to template literal
            let original = m.Value
            let parts = original.Split([|'+' |], StringSplitOptions.RemoveEmptyEntries)
                        |> Array.map (fun p -> p.Trim())
            
            let templateLiteral =
                let sb = System.Text.StringBuilder("`")
                for part in parts do
                    if part.StartsWith("\"") && part.EndsWith("\"") then
                        // String literal - remove quotes
                        sb.Append(part.Substring(1, part.Length - 2)) |> ignore
                    else
                        // Variable - add ${} syntax
                        sb.Append("${").Append(part).Append("}") |> ignore
                sb.Append("`").ToString()
            
            let fix = {
                Original = original
                Replacement = templateLiteral
                Description = "Replace string concatenation with template literal"
            }
            
            issues.Add((issue, fix))
        
        issues |> Seq.toList

    // Main analysis function
    let analyzeFile (filePath: string) : AnalysisResult =
        try
            // Read the file
            let code = File.ReadAllText(filePath)
            
            // Detect issues
            let nullCheckIssues = detectMissingNullChecks code
            let loopIssues = detectIneffectiveLoops code
            let unusedVarIssues = detectUnusedVariables code
            let errorHandlingIssues = detectMissingErrorHandling code
            let stringConcatIssues = detectIneffectiveStringConcatenation code
            
            // Combine all issues and fixes
            {
                FilePath = filePath
                Issues =
                    (nullCheckIssues |> List.map fst) @
                    (loopIssues |> List.map fst) @
                    (unusedVarIssues |> List.map fst) @
                    (errorHandlingIssues |> List.map fst) @
                    (stringConcatIssues |> List.map fst)
                Fixes =
                    (nullCheckIssues |> List.map snd) @
                    (loopIssues |> List.map snd) @
                    (unusedVarIssues |> List.map snd) @
                    (errorHandlingIssues |> List.map snd) @
                    (stringConcatIssues |> List.map snd)
            }
        with
        | ex ->
            printfn "Error analyzing file %s: %s" filePath ex.Message
            { FilePath = filePath; Issues = []; Fixes = [] }

    // Analyze code string directly
    let analyzeCode (code: string) : AnalysisResult =
        try
            // Detect issues
            let nullCheckIssues = detectMissingNullChecks code
            let loopIssues = detectIneffectiveLoops code
            let unusedVarIssues = detectUnusedVariables code
            let errorHandlingIssues = detectMissingErrorHandling code
            let stringConcatIssues = detectIneffectiveStringConcatenation code
            
            // Combine all issues and fixes
            {
                FilePath = "<memory>"
                Issues =
                    (nullCheckIssues |> List.map fst) @
                    (loopIssues |> List.map fst) @
                    (unusedVarIssues |> List.map fst) @
                    (errorHandlingIssues |> List.map fst) @
                    (stringConcatIssues |> List.map fst)
                Fixes =
                    (nullCheckIssues |> List.map snd) @
                    (loopIssues |> List.map snd) @
                    (unusedVarIssues |> List.map snd) @
                    (errorHandlingIssues |> List.map snd) @
                    (stringConcatIssues |> List.map snd)
            }
        with
        | ex ->
            printfn "Error analyzing code: %s" ex.Message
            { FilePath = "<memory>"; Issues = []; Fixes = [] }

    // Apply fixes to a code string
    let applyFixes (code: string) (fixes: CodeFix list) : string =
        try
            // Apply each fix
            // Note: This is a simplified approach that doesn't handle overlapping fixes
            let transformedCode =
                fixes |> List.fold (fun (acc: string) (fix: CodeFix) ->
                    acc.Replace(fix.Original, fix.Replacement)) code

            transformedCode
        with
        | ex ->
            printfn "Error applying fixes to code: %s" ex.Message
            code
