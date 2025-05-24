namespace TarsEngine.FSharp.Core.Analysis

open System
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging

/// <summary>
/// Implementation of IPerformanceIssueDetector for C# and F# languages.
/// </summary>
type PerformanceIssueDetector(logger: ILogger<PerformanceIssueDetector>) =
    
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
            logger.LogInformation("Detecting performance issues")
            
            // Detect inefficient loops
            let inefficientLoopIssues = this.DetectInefficientLoops(content)
            
            // Detect large object creation
            let largeObjectCreationIssues = this.DetectLargeObjectCreation(content)
            
            // Detect excessive memory usage
            let excessiveMemoryUsageIssues = this.DetectExcessiveMemoryUsage(content)
            
            // Detect inefficient string operations
            let inefficientStringOperationsIssues = this.DetectInefficientStringOperations(content)
            
            // Combine all issues
            List.concat [
                inefficientLoopIssues
                largeObjectCreationIssues
                excessiveMemoryUsageIssues
                inefficientStringOperationsIssues
            ]
        with
        | ex ->
            logger.LogError(ex, "Error detecting performance issues")
            []
    
    /// <summary>
    /// Detects issues in a file.
    /// </summary>
    /// <param name="filePath">The path to the file.</param>
    /// <returns>A list of detected issues.</returns>
    member this.DetectIssuesInFile(filePath: string) =
        try
            logger.LogInformation("Detecting performance issues in file: {FilePath}", filePath)
            
            // Read the file content
            let content = System.IO.File.ReadAllText(filePath)
            
            // Detect issues in the content
            let issues = this.DetectIssues(content)
            
            // Update issues with file path
            issues |> List.map (fun issue -> { issue with FilePath = Some filePath })
        with
        | ex ->
            logger.LogError(ex, "Error detecting performance issues in file: {FilePath}", filePath)
            []
    
    /// <summary>
    /// Gets the supported issue types.
    /// </summary>
    /// <returns>A list of supported issue types.</returns>
    member _.GetSupportedIssueTypes() =
        [
            CodeIssueType.Performance
        ]
    
    /// <summary>
    /// Detects inefficient loops.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectInefficientLoops(content: string) =
        try
            // Define patterns for inefficient loops
            let patterns = [
                // C# patterns
                @"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*([a-zA-Z0-9_\.]+)\.Count\s*;\s*i\s*\+\+\s*\)", "Inefficient loop using Count property in each iteration"
                @"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*([a-zA-Z0-9_\.]+)\.Length\s*;\s*i\s*\+\+\s*\)", "Inefficient loop using Length property in each iteration"
                @"foreach\s*\(\s*var\s+[a-zA-Z0-9_]+\s+in\s+([a-zA-Z0-9_\.]+)\.Where\s*\(", "Inefficient use of LINQ Where in foreach loop"
                @"foreach\s*\(\s*var\s+[a-zA-Z0-9_]+\s+in\s+([a-zA-Z0-9_\.]+)\.Select\s*\(", "Inefficient use of LINQ Select in foreach loop"
                @"foreach\s*\(\s*var\s+[a-zA-Z0-9_]+\s+in\s+([a-zA-Z0-9_\.]+)\.OrderBy\s*\(", "Inefficient use of LINQ OrderBy in foreach loop"
                @"foreach\s*\(\s*var\s+[a-zA-Z0-9_]+\s+in\s+([a-zA-Z0-9_\.]+)\.OrderByDescending\s*\(", "Inefficient use of LINQ OrderByDescending in foreach loop"
                
                // F# patterns
                @"for\s+i\s*=\s*0\s+to\s+([a-zA-Z0-9_\.]+)\.Length\s*-\s*1\s+do", "Inefficient loop using Length property in each iteration"
                @"for\s+i\s*=\s*0\s+to\s+([a-zA-Z0-9_\.]+)\.Count\s*-\s*1\s+do", "Inefficient loop using Count property in each iteration"
                @"for\s+[a-zA-Z0-9_]+\s+in\s+([a-zA-Z0-9_\.]+)\s+\|\>\s+Seq\.filter", "Inefficient use of Seq.filter in for loop"
                @"for\s+[a-zA-Z0-9_]+\s+in\s+([a-zA-Z0-9_\.]+)\s+\|\>\s+Seq\.map", "Inefficient use of Seq.map in for loop"
                @"for\s+[a-zA-Z0-9_]+\s+in\s+([a-zA-Z0-9_\.]+)\s+\|\>\s+Seq\.sortBy", "Inefficient use of Seq.sortBy in for loop"
                @"for\s+[a-zA-Z0-9_]+\s+in\s+([a-zA-Z0-9_\.]+)\s+\|\>\s+Seq\.sortByDescending", "Inefficient use of Seq.sortByDescending in for loop"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, message) ->
                Regex.Matches(content, pattern, RegexOptions.IgnoreCase ||| RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    let collectionName = if m.Groups.Count > 1 then m.Groups.[1].Value else ""
                    
                    let suggestedFix = 
                        if message.Contains("Count") || message.Contains("Length") then
                            "Cache the collection length/count before the loop"
                        elif message.Contains("LINQ") || message.Contains("Seq.") then
                            "Materialize the collection before the loop using ToList() or ToArray()"
                        else
                            "Optimize the loop for better performance"
                    
                    {
                        IssueType = CodeIssueType.Performance
                        Severity = IssueSeverity.Warning
                        Message = $"Potential performance issue: {message} for collection '{collectionName}'"
                        LineNumber = Some lineNumber
                        ColumnNumber = Some columnNumber
                        FilePath = None
                        CodeSnippet = Some codeSnippet
                        SuggestedFix = Some suggestedFix
                        AdditionalInfo = Map.empty
                    }
                )
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting inefficient loops")
            []
    
    /// <summary>
    /// Detects large object creation.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectLargeObjectCreation(content: string) =
        try
            // Define patterns for large object creation
            let patterns = [
                // C# patterns
                @"new\s+([a-zA-Z0-9_]+)\s*\[\s*(\d+)\s*\]", "Large array creation"
                @"new\s+List<[a-zA-Z0-9_]+>\s*\(\s*(\d+)\s*\)", "Large list creation"
                @"new\s+Dictionary<[a-zA-Z0-9_,\s]+>\s*\(\s*(\d+)\s*\)", "Large dictionary creation"
                @"new\s+HashSet<[a-zA-Z0-9_]+>\s*\(\s*(\d+)\s*\)", "Large hash set creation"
                @"new\s+StringBuilder\s*\(\s*(\d+)\s*\)", "Large string builder creation"
                
                // F# patterns
                @"Array\.create\s+(\d+)", "Large array creation"
                @"Array\.zeroCreate\s+(\d+)", "Large array creation"
                @"Array\.init\s+(\d+)", "Large array creation"
                @"List\.init\s+(\d+)", "Large list creation"
                @"Array\.replicate\s+(\d+)", "Large array creation"
                @"List\.replicate\s+(\d+)", "Large list creation"
                @"Dictionary\(\s*(\d+)\s*\)", "Large dictionary creation"
                @"HashSet\(\s*(\d+)\s*\)", "Large hash set creation"
                @"StringBuilder\(\s*(\d+)\s*\)", "Large string builder creation"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, message) ->
                Regex.Matches(content, pattern, RegexOptions.IgnoreCase ||| RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    let sizeStr = if m.Groups.Count > 1 then m.Groups.[1].Value else ""
                    
                    // Parse the size
                    let size = 
                        match Int32.TryParse(sizeStr) with
                        | true, value -> value
                        | _ -> 0
                    
                    // Only report issues for large objects
                    if size > 1000000 then
                        Some {
                            IssueType = CodeIssueType.Performance
                            Severity = IssueSeverity.Warning
                            Message = $"Potential performance issue: {message} with size {size:N0}"
                            LineNumber = Some lineNumber
                            ColumnNumber = Some columnNumber
                            FilePath = None
                            CodeSnippet = Some codeSnippet
                            SuggestedFix = Some "Consider using a smaller initial size or a different data structure"
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
            logger.LogError(ex, "Error detecting large object creation")
            []
    
    /// <summary>
    /// Detects excessive memory usage.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectExcessiveMemoryUsage(content: string) =
        try
            // Define patterns for excessive memory usage
            let patterns = [
                // C# patterns
                @"\.ToList\(\)\.ToArray\(\)", "Unnecessary conversion between collections"
                @"\.ToArray\(\)\.ToList\(\)", "Unnecessary conversion between collections"
                @"\.AsEnumerable\(\)\.ToList\(\)", "Unnecessary conversion between collections"
                @"\.AsEnumerable\(\)\.ToArray\(\)", "Unnecessary conversion between collections"
                @"\.Select\([^)]+\)\.Select\([^)]+\)", "Chained LINQ operations without materialization"
                @"\.Where\([^)]+\)\.Where\([^)]+\)", "Chained LINQ operations without materialization"
                @"\.OrderBy\([^)]+\)\.OrderBy\([^)]+\)", "Chained LINQ operations without materialization"
                
                // F# patterns
                @"\|\>\s+Seq\.toList\s+\|\>\s+Seq\.toArray", "Unnecessary conversion between collections"
                @"\|\>\s+Seq\.toArray\s+\|\>\s+Seq\.toList", "Unnecessary conversion between collections"
                @"\|\>\s+Seq\.map\s+[^|]+\|\>\s+Seq\.map", "Chained sequence operations without materialization"
                @"\|\>\s+Seq\.filter\s+[^|]+\|\>\s+Seq\.filter", "Chained sequence operations without materialization"
                @"\|\>\s+Seq\.sortBy\s+[^|]+\|\>\s+Seq\.sortBy", "Chained sequence operations without materialization"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, message) ->
                Regex.Matches(content, pattern, RegexOptions.IgnoreCase ||| RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    
                    let suggestedFix = 
                        if message.Contains("conversion") then
                            "Use a single collection type instead of converting between types"
                        elif message.Contains("Chained") then
                            "Materialize intermediate results using ToList() or ToArray()"
                        else
                            "Optimize memory usage"
                    
                    {
                        IssueType = CodeIssueType.Performance
                        Severity = IssueSeverity.Warning
                        Message = $"Potential performance issue: {message}"
                        LineNumber = Some lineNumber
                        ColumnNumber = Some columnNumber
                        FilePath = None
                        CodeSnippet = Some codeSnippet
                        SuggestedFix = Some suggestedFix
                        AdditionalInfo = Map.empty
                    }
                )
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting excessive memory usage")
            []
    
    /// <summary>
    /// Detects inefficient string operations.
    /// </summary>
    /// <param name="content">The source code content.</param>
    /// <returns>A list of detected issues.</returns>
    member _.DetectInefficientStringOperations(content: string) =
        try
            // Define patterns for inefficient string operations
            let patterns = [
                // C# patterns
                @"string\.Concat\s*\(\s*[^)]+\s*\+\s*[^)]+\s*\)", "Inefficient string concatenation in String.Concat"
                @"for\s*\([^)]+\)\s*\{[^}]*\+=[^}]*\}", "String concatenation in a loop"
                @"while\s*\([^)]+\)\s*\{[^}]*\+=[^}]*\}", "String concatenation in a loop"
                @"foreach\s*\([^)]+\)\s*\{[^}]*\+=[^}]*\}", "String concatenation in a loop"
                @"do\s*\{[^}]*\+=[^}]*\}\s*while", "String concatenation in a loop"
                
                // F# patterns
                @"String\.Concat\s*\(\s*[^)]+\s*\+\s*[^)]+\s*\)", "Inefficient string concatenation in String.Concat"
                @"for\s+[^d]+do[^d]*\+[^d]*done", "String concatenation in a loop"
                @"while\s+[^d]+do[^d]*\+[^d]*done", "String concatenation in a loop"
            ]
            
            // Detect issues using patterns
            patterns
            |> List.collect (fun (pattern, message) ->
                Regex.Matches(content, pattern, RegexOptions.IgnoreCase ||| RegexOptions.Multiline)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let lineNumber = content.Substring(0, m.Index).Split('\n').Length
                    let columnNumber = m.Index - content.Substring(0, m.Index).LastIndexOf('\n') - 1
                    let codeSnippet = m.Value
                    
                    let suggestedFix = 
                        if message.Contains("loop") then
                            "Use StringBuilder instead of string concatenation in loops"
                        else
                            "Use StringBuilder or string.Format instead of string concatenation"
                    
                    {
                        IssueType = CodeIssueType.Performance
                        Severity = IssueSeverity.Warning
                        Message = $"Potential performance issue: {message}"
                        LineNumber = Some lineNumber
                        ColumnNumber = Some columnNumber
                        FilePath = None
                        CodeSnippet = Some codeSnippet
                        SuggestedFix = Some suggestedFix
                        AdditionalInfo = Map.empty
                    }
                )
                |> Seq.toList
            )
        with
        | ex ->
            logger.LogError(ex, "Error detecting inefficient string operations")
            []
    
    interface IIssueDetector with
        member this.Language = this.Language
        member this.DetectIssues(content) = this.DetectIssues(content)
        member this.DetectIssuesInFile(filePath) = this.DetectIssuesInFile(filePath)
        member this.GetSupportedIssueTypes() = this.GetSupportedIssueTypes()
    
    interface IPerformanceIssueDetector with
        member this.DetectInefficientLoops(content) = this.DetectInefficientLoops(content)
        member this.DetectLargeObjectCreation(content) = this.DetectLargeObjectCreation(content)
        member this.DetectExcessiveMemoryUsage(content) = this.DetectExcessiveMemoryUsage(content)
        member this.DetectInefficientStringOperations(content) = this.DetectInefficientStringOperations(content)
