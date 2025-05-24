namespace TarsEngine.FSharp.Core.Analysis

open System
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging

/// <summary>
/// Implementation of IStyleIssueDetector for C# and F# languages.
/// </summary>
type StyleIssueDetector(logger: ILogger<StyleIssueDetector>) =
    
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
            logger.LogInformation("Detecting style issues")
            
            // Detect naming convention violations
            let namingConventionIssues = this.DetectNamingConventionViolations(content)
            
            // Detect formatting issues
            let formattingIssues = this.DetectFormattingIssues(content)
            
            // Detect code style inconsistencies
            let codeStyleIssues = this.DetectCodeStyleInconsistencies(content)
            
            // Detect comment style issues
            let commentStyleIssues = this.DetectCommentStyleIssues(content)
            
            // Combine all issues
            List.concat [
                namingConventionIssues
                formattingIssues
                codeStyleIssues
                commentStyleIssues
            ]
        with
        | ex ->
            logger.LogError(ex, "Error detecting style issues")
            []
    
    /// <summary>
    /// Detects issues in a file.
    /// </summary>
    /// <param name="filePath">The path to the file.</param>
    /// <returns>A list of detected issues.</returns>
    member this.DetectIssuesInFile(filePath: string) =
        try
            logger.LogInformation("Detecting style issues in file: {FilePath}", filePath)
            
            // Read the file content
            let content = System.IO.File.ReadAllText(filePath)
            
            // Detect issues in the content
            let issues = this.DetectIssues(content)
            
            // Update issues with file path
            issues |> List.map (fun issue -> { issue with FilePath = Some filePath })
        with
        | ex ->
            logger.LogError(ex, "Error detecting style issues in file: {FilePath}", filePath)
            []
    
    /// <summary>
    /// Gets the supported issue types.
    /// </summary>
    /// <returns>A list of supported issue types.</returns>
    member _.GetSupportedIssueTypes() =
        [
            CodeIssueType.Style
        ]
    
    /// <summary>
    /// Detects naming convention violations.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectNamingConventionViolations(content: string) =
        try
            // Define patterns for naming conventions
            let patterns = [
                // C# patterns
                @"class\s+([a-z][a-zA-Z0-9_]*)", "Class name should start with an uppercase letter"
                @"interface\s+([^I][a-zA-Z0-9_]*)", "Interface name should start with 'I'"
                @"enum\s+([a-z][a-zA-Z0-9_]*)", "Enum name should start with an uppercase letter"
                @"(public|private|protected|internal)\s+[a-zA-Z0-9_<>]+\s+([A-Z][a-zA-Z0-9_]*)\s*\(", "Method name should start with a lowercase letter"
                @"(public|private|protected|internal)\s+[a-zA-Z0-9_<>]+\s+([A-Z][a-zA-Z0-9_]*)\s*{", "Property name should start with a lowercase letter"
                @"(public|private|protected|internal)\s+[a-zA-Z0-9_<>]+\s+([A-Z][a-zA-Z0-9_]*)\s*;", "Field name should start with a lowercase letter"
                @"(public|private|protected|internal)\s+const\s+[a-zA-Z0-9_<>]+\s+([a-z][a-zA-Z0-9_]*)", "Constant name should be all uppercase"
                
                // F# patterns
                @"type\s+([a-z][a-zA-Z0-9_]*)", "Type name should start with an uppercase letter"
                @"type\s+([^I][a-zA-Z0-9_]*)\s*=\s*interface", "Interface name should start with 'I'"
                @"let\s+([A-Z][a-zA-Z0-9_]*)\s*=", "Value name should start with a lowercase letter"
                @"let\s+([A-Z][a-zA-Z0-9_]*)\s+[a-zA-Z0-9_]+", "Function name should start with a lowercase letter"
                @"member\s+(?:this|self|_)\.([A-Z][a-zA-Z0-9_]*)", "Member name should start with a lowercase letter"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, message) ->
                Regex.Matches(content, pattern, RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    let name = m.Groups.[m.Groups.Count - 1].Value
                    
                    {
                        IssueType = CodeIssueType.Style
                        Severity = IssueSeverity.Info
                        Message = $"Naming convention violation: {message} ('{name}')"
                        LineNumber = Some lineNumber
                        ColumnNumber = Some columnNumber
                        FilePath = None
                        CodeSnippet = Some codeSnippet
                        SuggestedFix = Some "Rename according to the naming convention"
                        AdditionalInfo = Map.empty
                    }
                )
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting naming convention violations")
            []
    
    /// <summary>
    /// Detects formatting issues.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectFormattingIssues(content: string) =
        try
            // Define patterns for formatting issues
            let patterns = [
                // C# patterns
                @"if\s*\([^)]+\)[a-zA-Z0-9_]+", "Missing space after if statement"
                @"for\s*\([^)]+\)[a-zA-Z0-9_]+", "Missing space after for statement"
                @"while\s*\([^)]+\)[a-zA-Z0-9_]+", "Missing space after while statement"
                @"foreach\s*\([^)]+\)[a-zA-Z0-9_]+", "Missing space after foreach statement"
                @"if\s*\([^)]+\)\s*\{\s*\}", "Empty if block"
                @"for\s*\([^)]+\)\s*\{\s*\}", "Empty for block"
                @"while\s*\([^)]+\)\s*\{\s*\}", "Empty while block"
                @"foreach\s*\([^)]+\)\s*\{\s*\}", "Empty foreach block"
                @"catch\s*\([^)]+\)\s*\{\s*\}", "Empty catch block"
                @"finally\s*\{\s*\}", "Empty finally block"
                @"[ \t]+$", "Trailing whitespace"
                
                // F# patterns
                @"if\s+[^t]+then[a-zA-Z0-9_]+", "Missing space after then keyword"
                @"else[a-zA-Z0-9_]+", "Missing space after else keyword"
                @"for\s+[^i]+in[a-zA-Z0-9_]+", "Missing space after in keyword"
                @"while\s+[^d]+do[a-zA-Z0-9_]+", "Missing space after do keyword"
                @"match\s+[^w]+with[a-zA-Z0-9_]+", "Missing space after with keyword"
                @"[ \t]+$", "Trailing whitespace"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, message) ->
                Regex.Matches(content, pattern, RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    
                    {
                        IssueType = CodeIssueType.Style
                        Severity = IssueSeverity.Info
                        Message = $"Formatting issue: {message}"
                        LineNumber = Some lineNumber
                        ColumnNumber = Some columnNumber
                        FilePath = None
                        CodeSnippet = Some codeSnippet
                        SuggestedFix = Some "Fix the formatting according to the style guide"
                        AdditionalInfo = Map.empty
                    }
                )
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting formatting issues")
            []
    
    /// <summary>
    /// Detects code style inconsistencies.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectCodeStyleInconsistencies(content: string) =
        try
            // Define patterns for code style inconsistencies
            let patterns = [
                // C# patterns
                @"var\s+[a-zA-Z0-9_]+\s*=\s*new\s+([a-zA-Z0-9_<>]+)", "Explicit type can be inferred"
                @"([a-zA-Z0-9_<>]+)\s+[a-zA-Z0-9_]+\s*=\s*new\s+\1", "Type can be inferred using var"
                @"if\s*\([^)]+\)\s*return\s+[^;]+;\s*else\s*return\s+[^;]+;", "Can be simplified using conditional operator"
                @"if\s*\([^)]+\)\s*{\s*return\s+[^;]+;\s*}\s*else\s*{\s*return\s+[^;]+;\s*}", "Can be simplified using conditional operator"
                @"if\s*\([^)]+\)\s*{\s*[^}]+\s*}\s*else\s*{\s*[^}]+\s*}", "Consider using guard clauses"
                
                // F# patterns
                @"if\s+[^t]+then\s+[^e]+else\s+[^i]+", "Consider using pattern matching instead of if-then-else"
                @"match\s+[^w]+with\s*\|\s*[^-]+->\s*[^|]+\|\s*_\s*->\s*[^e]+", "Consider handling all cases explicitly instead of using wildcard pattern"
                @"let\s+[a-zA-Z0-9_]+\s*=\s*[^i]+\s+in\s+[^l]+", "Consider using let binding without in keyword"
                @"let\s+mutable\s+[a-zA-Z0-9_]+", "Consider using immutable values instead of mutable ones"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, message) ->
                Regex.Matches(content, pattern, RegexOptions.Singleline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    
                    {
                        IssueType = CodeIssueType.Style
                        Severity = IssueSeverity.Info
                        Message = $"Code style inconsistency: {message}"
                        LineNumber = Some lineNumber
                        ColumnNumber = Some columnNumber
                        FilePath = None
                        CodeSnippet = Some codeSnippet
                        SuggestedFix = Some "Refactor the code according to the style guide"
                        AdditionalInfo = Map.empty
                    }
                )
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting code style inconsistencies")
            []
    
    /// <summary>
    /// Detects comment style issues.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectCommentStyleIssues(content: string) =
        try
            // Define patterns for comment style issues
            let patterns = [
                // C# patterns
                @"//[a-zA-Z0-9]", "Missing space after comment delimiter"
                @"//\s*[a-z]", "Comment should start with an uppercase letter"
                @"//\s*[A-Z][^.!?]*[^.!?]$", "Comment should end with a period"
                @"/\*[a-zA-Z0-9]", "Missing space after comment delimiter"
                @"/\*\s*[a-z]", "Comment should start with an uppercase letter"
                @"[a-zA-Z0-9]\*/", "Missing space before comment delimiter"
                
                // F# patterns
                @"//[a-zA-Z0-9]", "Missing space after comment delimiter"
                @"//\s*[a-z]", "Comment should start with an uppercase letter"
                @"//\s*[A-Z][^.!?]*[^.!?]$", "Comment should end with a period"
                @"\(\*[a-zA-Z0-9]", "Missing space after comment delimiter"
                @"\(\*\s*[a-z]", "Comment should start with an uppercase letter"
                @"[a-zA-Z0-9]\*\)", "Missing space before comment delimiter"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, message) ->
                Regex.Matches(content, pattern, RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    
                    {
                        IssueType = CodeIssueType.Style
                        Severity = IssueSeverity.Info
                        Message = $"Comment style issue: {message}"
                        LineNumber = Some lineNumber
                        ColumnNumber = Some columnNumber
                        FilePath = None
                        CodeSnippet = Some codeSnippet
                        SuggestedFix = Some "Fix the comment according to the style guide"
                        AdditionalInfo = Map.empty
                    }
                )
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting comment style issues")
            []
    
    interface IIssueDetector with
        member this.Language = this.Language
        member this.DetectIssues(content) = this.DetectIssues(content)
        member this.DetectIssuesInFile(filePath) = this.DetectIssuesInFile(filePath)
        member this.GetSupportedIssueTypes() = this.GetSupportedIssueTypes()
    
    interface IStyleIssueDetector with
        member this.DetectNamingConventionViolations(content) = this.DetectNamingConventionViolations(content)
        member this.DetectFormattingIssues(content) = this.DetectFormattingIssues(content)
        member this.DetectCodeStyleInconsistencies(content) = this.DetectCodeStyleInconsistencies(content)
        member this.DetectCommentStyleIssues(content) = this.DetectCommentStyleIssues(content)
