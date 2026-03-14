namespace TarsEngine.TreeOfThought

open System
open System.IO
open System.Text.RegularExpressions

/// Represents different types of code issues
type CodeIssueType =
    | SyntaxError
    | TypeMismatch
    | MissingReference
    | UnusedVariable
    | DeadCode
    | PerformanceIssue
    | SecurityVulnerability

/// Represents a code issue found during analysis
type CodeIssue = {
    Type: CodeIssueType
    Message: string
    File: string
    Line: int
    Column: int
    Severity: string
    Suggestion: string option
}

/// Represents the result of code analysis
type AnalysisResult = {
    Issues: CodeIssue list
    FilesAnalyzed: int
    LinesAnalyzed: int
    AnalysisTime: TimeSpan
}

/// Code analysis functionality for Tree of Thought reasoning
module CodeAnalysis =
    
    /// Create a new code issue
    let createIssue issueType message file line column severity suggestion =
        {
            Type = issueType
            Message = message
            File = file
            Line = line
            Column = column
            Severity = severity
            Suggestion = suggestion
        }
    
    /// Analyze a single file for common issues
    let analyzeFile (filePath: string) : CodeIssue list =
        if not (File.Exists(filePath)) then
            [createIssue MissingReference $"File not found: {filePath}" filePath 0 0 "Error" None]
        else
            try
                let content = File.ReadAllText(filePath)
                let lines = content.Split([|'\n'|], StringSplitOptions.None)
                
                let mutable issues = []
                
                // Check for common F# issues
                for i in 0 .. lines.Length - 1 do
                    let line = lines.[i]
                    let lineNumber = i + 1
                    
                    // Check for unused variables (simple heuristic)
                    let unusedVarPattern = @"let\s+(\w+)\s*="
                    let matches = Regex.Matches(line, unusedVarPattern)
                    for m in matches do
                        let varName = m.Groups.[1].Value
                        if not (content.Contains(varName + " ") && content.IndexOf(varName + " ") > content.IndexOf(line)) then
                            issues <- createIssue UnusedVariable $"Variable '{varName}' appears to be unused" filePath lineNumber m.Index "Warning" (Some "Consider removing unused variable") :: issues
                    
                    // Check for TODO comments
                    if line.Contains("TODO") || line.Contains("FIXME") || line.Contains("HACK") then
                        issues <- createIssue DeadCode "TODO/FIXME/HACK comment found" filePath lineNumber (line.IndexOf("TODO") + line.IndexOf("FIXME") + line.IndexOf("HACK") + 3) "Info" (Some "Address the TODO item") :: issues
                    
                    // Check for long lines
                    if line.Length > 120 then
                        issues <- createIssue PerformanceIssue "Line is too long (>120 characters)" filePath lineNumber 0 "Warning" (Some "Consider breaking the line") :: issues
                    
                    // Check for potential null reference issues
                    if line.Contains(".Value") && not (line.Contains("Option.") || line.Contains("Some")) then
                        issues <- createIssue SecurityVulnerability "Potential null reference access" filePath lineNumber (line.IndexOf(".Value")) "Warning" (Some "Use pattern matching or Option.defaultValue") :: issues
                
                issues
            with
            | ex ->
                [createIssue SyntaxError $"Error analyzing file: {ex.Message}" filePath 0 0 "Error" None]
    
    /// Analyze multiple files
    let analyzeFiles (filePaths: string list) : AnalysisResult =
        let startTime = DateTime.Now
        
        let allIssues = 
            filePaths
            |> List.collect analyzeFile
        
        let totalLines = 
            filePaths
            |> List.sumBy (fun path ->
                if File.Exists(path) then
                    File.ReadAllLines(path).Length
                else 0
            )
        
        let endTime = DateTime.Now
        
        {
            Issues = allIssues
            FilesAnalyzed = filePaths.Length
            LinesAnalyzed = totalLines
            AnalysisTime = endTime - startTime
        }
    
    /// Analyze a directory recursively
    let analyzeDirectory (directoryPath: string) (pattern: string) : AnalysisResult =
        if not (Directory.Exists(directoryPath)) then
            {
                Issues = [createIssue MissingReference $"Directory not found: {directoryPath}" directoryPath 0 0 "Error" None]
                FilesAnalyzed = 0
                LinesAnalyzed = 0
                AnalysisTime = TimeSpan.Zero
            }
        else
            let files = Directory.GetFiles(directoryPath, pattern, SearchOption.AllDirectories) |> Array.toList
            analyzeFiles files
    
    /// Filter issues by severity
    let filterBySeverity (severity: string) (result: AnalysisResult) : AnalysisResult =
        { result with Issues = result.Issues |> List.filter (fun issue -> issue.Severity = severity) }
    
    /// Group issues by type
    let groupByType (result: AnalysisResult) : Map<CodeIssueType, CodeIssue list> =
        result.Issues
        |> List.groupBy (fun issue -> issue.Type)
        |> Map.ofList
    
    /// Get summary statistics
    let getSummary (result: AnalysisResult) : string =
        let issuesByType = groupByType result
        let typeStats = 
            issuesByType
            |> Map.toList
            |> List.map (fun (issueType, issues) -> $"{issueType}: {issues.Length}")
            |> String.concat ", "
        
        $"Analysis completed in {result.AnalysisTime.TotalMilliseconds:F0}ms. " +
        $"Files: {result.FilesAnalyzed}, Lines: {result.LinesAnalyzed}, Issues: {result.Issues.Length}. " +
        $"Breakdown: {typeStats}"
    
    /// Apply automatic fixes where possible
    let applyAutomaticFixes (result: AnalysisResult) : string list =
        result.Issues
        |> List.choose (fun issue ->
            match issue.Suggestion with
            | Some suggestion when issue.Type = UnusedVariable ->
                Some $"Remove unused variable '{issue.Message}' at {issue.File}:{issue.Line}"
            | Some suggestion when issue.Type = DeadCode ->
                Some $"Address TODO at {issue.File}:{issue.Line}"
            | _ -> None
        )
