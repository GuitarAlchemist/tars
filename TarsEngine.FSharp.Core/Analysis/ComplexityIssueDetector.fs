namespace TarsEngine.FSharp.Core.Analysis

open System
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging

/// <summary>
/// Implementation of IComplexityIssueDetector for C# and F# languages.
/// </summary>
type ComplexityIssueDetector(logger: ILogger<ComplexityIssueDetector>) =
    
    /// <summary>
    /// Gets the language supported by this detector.
    /// </summary>
    member _.Language = "csharp"
    
    /// <summary>
    /// Detects issues in the provided content.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectIssues(content: string) =
        try
            logger.LogInformation("Detecting complexity issues")
            
            // Extract structures (simplified for this example)
            let structures = []
            
            // Detect methods with high cyclomatic complexity
            let highComplexityIssues = this.DetectHighCyclomaticComplexity(content, structures)
            
            // Detect methods with too many parameters
            let tooManyParametersIssues = this.DetectTooManyParameters(content)
            
            // Detect deeply nested code
            let deepNestingIssues = this.DetectDeepNesting(content)
            
            // Detect long methods
            let longMethodsIssues = this.DetectLongMethods(content, structures)
            
            // Combine all issues
            List.concat [
                highComplexityIssues
                tooManyParametersIssues
                deepNestingIssues
                longMethodsIssues
            ]
        with
        | ex ->
            logger.LogError(ex, "Error detecting complexity issues")
            []
    
    /// <summary>
    /// Detects issues in a file.
    /// </summary>
    /// <param name="filePath">The path to the file.</param>
    /// <returns>A list of detected issues.</returns>
    member this.DetectIssuesInFile(filePath: string) =
        try
            logger.LogInformation("Detecting complexity issues in file: {FilePath}", filePath)
            
            // Read the file content
            let content = System.IO.File.ReadAllText(filePath)
            
            // Detect issues in the content
            let issues = this.DetectIssues(content)
            
            // Update issues with file path
            issues |> List.map (fun issue -> { issue with FilePath = Some filePath })
        with
        | ex ->
            logger.LogError(ex, "Error detecting complexity issues in file: {FilePath}", filePath)
            []
    
    /// <summary>
    /// Gets the supported issue types.
    /// </summary>
    /// <returns>A list of supported issue types.</returns>
    member _.GetSupportedIssueTypes() =
        [
            CodeIssueType.Complexity
        ]
    
    /// <summary>
    /// Detects methods with high cyclomatic complexity.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <param name="structures">The extracted code structures.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectHighCyclomaticComplexity(content: string, structures: CodeStructure list) =
        try
            // Define patterns for control flow statements that increase cyclomatic complexity
            let patterns = [
                @"if\s*\(", "if statement"
                @"else\s+if\s*\(", "else if statement"
                @"else", "else statement"
                @"switch\s*\(", "switch statement"
                @"case\s+[^:]+:", "case statement"
                @"for\s*\(", "for loop"
                @"foreach\s*\(", "foreach loop"
                @"while\s*\(", "while loop"
                @"do\s*\{", "do-while loop"
                @"catch\s*\(", "catch block"
                @"\?\s*[^:]+\s*:", "conditional operator"
                @"\|\|", "logical OR operator"
                @"&&", "logical AND operator"
            ]
            
            // Find method blocks in the content
            let methodPattern = @"(public|private|protected|internal)?\s*(static|virtual|abstract|override|sealed)?\s*[a-zA-Z0-9_<>]+\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*\{([^}]*)\}"
            let methodMatches = Regex.Matches(content, methodPattern, RegexOptions.Singleline)
            
            // Analyze each method
            methodMatches
            |> Seq.cast<Match>
            |> Seq.map (fun m ->
                let methodName = m.Groups.[3].Value
                let methodBody = m.Groups.[5].Value
                let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                
                // Count complexity by counting control flow statements
                let complexity = 
                    patterns
                    |> List.sumBy (fun (pattern, _) ->
                        Regex.Matches(methodBody, pattern, RegexOptions.Singleline).Count
                    )
                    |> (+) 1 // Base complexity is 1
                
                // Only report methods with high complexity
                if complexity > 10 then
                    Some {
                        IssueType = CodeIssueType.Complexity
                        Severity = IssueSeverity.Warning
                        Message = $"Method '{methodName}' has high cyclomatic complexity ({complexity})"
                        LineNumber = Some lineNumber
                        ColumnNumber = None
                        FilePath = None
                        CodeSnippet = Some (m.Value.Substring(0, Math.Min(200, m.Value.Length)) + "...")
                        SuggestedFix = Some "Refactor the method into smaller, more focused methods"
                        AdditionalInfo = Map.empty
                    }
                else
                    None
            )
            |> Seq.choose id
            |> Seq.toList
        with
        | ex ->
            logger.LogError(ex, "Error detecting high cyclomatic complexity")
            []
    
    /// <summary>
    /// Detects methods with too many parameters.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectTooManyParameters(content: string) =
        try
            // Define patterns for method declarations
            let patterns = [
                // C# patterns
                @"(public|private|protected|internal)?\s*(static|virtual|abstract|override|sealed)?\s*[a-zA-Z0-9_<>]+\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)", "C# method"
                @"(public|private|protected|internal)?\s*(static|virtual|abstract|override|sealed)?\s*[a-zA-Z0-9_<>]+\s+([a-zA-Z0-9_]+)\s*<[^>]*>\s*\(([^)]*)\)", "C# generic method"
                
                // F# patterns
                @"let\s+([a-zA-Z0-9_]+)\s*([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*)\s*=", "F# function"
                @"member\s+(?:this|self|_)\.([a-zA-Z0-9_]+)\s*([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*)\s*=", "F# member"
                @"static\s+member\s+([a-zA-Z0-9_]+)\s*([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*)\s*=", "F# static member"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, methodType) ->
                Regex.Matches(content, pattern, RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    
                    let methodName = 
                        if methodType.StartsWith("C#") then
                            m.Groups.[3].Value
                        else
                            m.Groups.[1].Value
                    
                    let parameters = 
                        if methodType.StartsWith("C#") then
                            let paramString = m.Groups.[4].Value
                            if String.IsNullOrWhiteSpace(paramString) then
                                []
                            else
                                paramString.Split(',') |> Array.toList
                        else
                            let paramString = m.Groups.[2].Value
                            if String.IsNullOrWhiteSpace(paramString) then
                                []
                            else
                                paramString.Split(' ') |> Array.toList
                    
                    // Only report methods with too many parameters
                    if parameters.Length > 7 then
                        Some {
                            IssueType = CodeIssueType.Complexity
                            Severity = IssueSeverity.Warning
                            Message = $"{methodType} '{methodName}' has too many parameters ({parameters.Length})"
                            LineNumber = Some lineNumber
                            ColumnNumber = Some columnNumber
                            FilePath = None
                            CodeSnippet = Some codeSnippet
                            SuggestedFix = Some "Consider using a parameter object or breaking the method into smaller methods"
                            AdditionalInfo = Map.empty
                        }
                    else
                        None
                )
                |> Seq.choose id
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting methods with too many parameters")
            []
    
    /// <summary>
    /// Detects deeply nested code.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectDeepNesting(content: string) =
        try
            // Find blocks with deep nesting
            let lines = content.Split('\n')
            let issues = ResizeArray<CodeIssue>()
            
            // Track nesting level for each line
            let mutable nestingLevel = 0
            let mutable maxNestingLevel = 0
            let mutable maxNestingLineNumber = 0
            
            for i = 0 to lines.Length - 1 do
                let line = lines.[i]
                
                // Count opening braces
                let openingBraces = line.Count(fun c -> c = '{')
                
                // Count closing braces
                let closingBraces = line.Count(fun c -> c = '}')
                
                // Update nesting level
                nestingLevel <- nestingLevel + openingBraces - closingBraces
                
                // Track maximum nesting level
                if nestingLevel > maxNestingLevel then
                    maxNestingLevel <- nestingLevel
                    maxNestingLineNumber <- i + 1
            
            // Only report deep nesting
            if maxNestingLevel > 4 then
                issues.Add({
                    IssueType = CodeIssueType.Complexity
                    Severity = IssueSeverity.Warning
                    Message = $"Code has deep nesting (level {maxNestingLevel})"
                    LineNumber = Some maxNestingLineNumber
                    ColumnNumber = None
                    FilePath = None
                    CodeSnippet = Some (if maxNestingLineNumber < lines.Length then lines.[maxNestingLineNumber - 1] else "")
                    SuggestedFix = Some "Refactor the code to reduce nesting, possibly by extracting methods or using early returns"
                    AdditionalInfo = Map.empty
                })
            
            issues |> Seq.toList
        with
        | ex ->
            logger.LogError(ex, "Error detecting deeply nested code")
            []
    
    /// <summary>
    /// Detects long methods.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <param name="structures">The extracted code structures.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectLongMethods(content: string, structures: CodeStructure list) =
        try
            // Define patterns for method declarations
            let patterns = [
                // C# patterns
                @"(public|private|protected|internal)?\s*(static|virtual|abstract|override|sealed)?\s*[a-zA-Z0-9_<>]+\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*\{([^}]*)\}", "C# method"
                
                // F# patterns
                @"let\s+([a-zA-Z0-9_]+)(?:\s+[a-zA-Z0-9_]+)*\s*=\s*([^i]*?in\s*)?([^l]*)", "F# function"
                @"member\s+(?:this|self|_)\.([a-zA-Z0-9_]+)(?:\s+[a-zA-Z0-9_]+)*\s*=\s*([^i]*?in\s*)?([^l]*)", "F# member"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, methodType) ->
                Regex.Matches(content, pattern, RegexOptions.Singleline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    
                    let methodName = 
                        if methodType.StartsWith("C#") then
                            m.Groups.[3].Value
                        else
                            m.Groups.[1].Value
                    
                    let methodBody = 
                        if methodType.StartsWith("C#") then
                            m.Groups.[5].Value
                        else
                            m.Groups.[3].Value
                    
                    let lineCount = methodBody.Split('\n').Length
                    
                    // Only report long methods
                    if lineCount > 50 then
                        Some {
                            IssueType = CodeIssueType.Complexity
                            Severity = IssueSeverity.Warning
                            Message = $"{methodType} '{methodName}' is too long ({lineCount} lines)"
                            LineNumber = Some lineNumber
                            ColumnNumber = Some columnNumber
                            FilePath = None
                            CodeSnippet = Some (m.Value.Substring(0, Math.Min(200, m.Value.Length)) + "...")
                            SuggestedFix = Some "Break the method into smaller, more focused methods"
                            AdditionalInfo = Map.empty
                        }
                    else
                        None
                )
                |> Seq.choose id
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting long methods")
            []
    
    interface IIssueDetector with
        member this.Language = this.Language
        member this.DetectIssues(content) = this.DetectIssues(content)
        member this.DetectIssuesInFile(filePath) = this.DetectIssuesInFile(filePath)
        member this.GetSupportedIssueTypes() = this.GetSupportedIssueTypes()
    
    interface IComplexityIssueDetector with
        member this.DetectHighCyclomaticComplexity(content, structures) = this.DetectHighCyclomaticComplexity(content, structures)
        member this.DetectTooManyParameters(content) = this.DetectTooManyParameters(content)
        member this.DetectDeepNesting(content) = this.DetectDeepNesting(content)
        member this.DetectLongMethods(content, structures) = this.DetectLongMethods(content, structures)
