namespace TarsEngine.TreeOfThought

open System
open System.IO
open System.Text.RegularExpressions

/// Represents a potential fix for a code issue
type CodeFix = {
    Id: string
    Description: string
    TargetFile: string
    TargetLine: int
    TargetColumn: int
    OriginalText: string
    ReplacementText: string
    Confidence: float
    Category: string
}

/// Represents the result of fix generation
type FixGenerationResult = {
    Fixes: CodeFix list
    GenerationTime: TimeSpan
    SuccessRate: float
}

/// Fix generation functionality for Tree of Thought reasoning
module FixGeneration =
    
    /// Create a new code fix
    let createFix id description targetFile targetLine targetColumn originalText replacementText confidence category =
        {
            Id = id
            Description = description
            TargetFile = targetFile
            TargetLine = targetLine
            TargetColumn = targetColumn
            OriginalText = originalText
            ReplacementText = replacementText
            Confidence = confidence
            Category = category
        }
    
    /// Generate fixes for syntax errors
    let generateSyntaxFixes (issue: CodeIssue) : CodeFix list =
        match issue.Type with
        | SyntaxError ->
            let fixes = []
            
            // Common F# syntax fixes
            let fixes = 
                if issue.Message.Contains("missing") && issue.Message.Contains("'('") then
                    createFix 
                        (Guid.NewGuid().ToString()) 
                        "Add missing opening parenthesis" 
                        issue.File 
                        issue.Line 
                        issue.Column 
                        "" 
                        "(" 
                        0.8 
                        "Syntax" :: fixes
                else fixes
            
            let fixes = 
                if issue.Message.Contains("missing") && issue.Message.Contains("')'") then
                    createFix 
                        (Guid.NewGuid().ToString()) 
                        "Add missing closing parenthesis" 
                        issue.File 
                        issue.Line 
                        issue.Column 
                        "" 
                        ")" 
                        0.8 
                        "Syntax" :: fixes
                else fixes
            
            let fixes = 
                if issue.Message.Contains("unexpected") && issue.Message.Contains("';'") then
                    createFix 
                        (Guid.NewGuid().ToString()) 
                        "Remove unexpected semicolon" 
                        issue.File 
                        issue.Line 
                        issue.Column 
                        ";" 
                        "" 
                        0.9 
                        "Syntax" :: fixes
                else fixes
            
            fixes
        | _ -> []
    
    /// Generate fixes for type mismatches
    let generateTypeFixes (issue: CodeIssue) : CodeFix list =
        match issue.Type with
        | TypeMismatch ->
            let fixes = []
            
            // Common type conversion fixes
            let fixes = 
                if issue.Message.Contains("string") && issue.Message.Contains("int") then
                    createFix 
                        (Guid.NewGuid().ToString()) 
                        "Convert string to int using int()" 
                        issue.File 
                        issue.Line 
                        issue.Column 
                        "" 
                        "int(" 
                        0.7 
                        "Type" :: fixes
                else fixes
            
            let fixes = 
                if issue.Message.Contains("option") && issue.Message.Contains("value") then
                    createFix 
                        (Guid.NewGuid().ToString()) 
                        "Unwrap option using Option.defaultValue" 
                        issue.File 
                        issue.Line 
                        issue.Column 
                        "" 
                        "Option.defaultValue " 
                        0.6 
                        "Type" :: fixes
                else fixes
            
            fixes
        | _ -> []
    
    /// Generate fixes for missing references
    let generateReferenceFixes (issue: CodeIssue) : CodeFix list =
        match issue.Type with
        | MissingReference ->
            let fixes = []
            
            // Common reference fixes
            let fixes = 
                if issue.Message.Contains("namespace") || issue.Message.Contains("module") then
                    createFix 
                        (Guid.NewGuid().ToString()) 
                        "Add open statement" 
                        issue.File 
                        1 
                        0 
                        "" 
                        "open System\n" 
                        0.5 
                        "Reference" :: fixes
                else fixes
            
            fixes
        | _ -> []
    
    /// Generate fixes for unused variables
    let generateUnusedVariableFixes (issue: CodeIssue) : CodeFix list =
        match issue.Type with
        | UnusedVariable ->
            let variableName = 
                let pattern = @"Variable '(\w+)'"
                let m = Regex.Match(issue.Message, pattern)
                if m.Success then m.Groups.[1].Value else ""
            
            if not (String.IsNullOrEmpty(variableName)) then
                [
                    createFix 
                        (Guid.NewGuid().ToString()) 
                        $"Remove unused variable '{variableName}'" 
                        issue.File 
                        issue.Line 
                        issue.Column 
                        $"let {variableName} =" 
                        "// Removed unused variable" 
                        0.9 
                        "Cleanup"
                    
                    createFix 
                        (Guid.NewGuid().ToString()) 
                        $"Prefix variable with underscore to indicate intentional non-use" 
                        issue.File 
                        issue.Line 
                        issue.Column 
                        $"let {variableName}" 
                        $"let _{variableName}" 
                        0.8 
                        "Cleanup"
                ]
            else []
        | _ -> []
    
    /// Generate all possible fixes for an issue
    let generateFixesForIssue (issue: CodeIssue) : CodeFix list =
        let syntaxFixes = generateSyntaxFixes issue
        let typeFixes = generateTypeFixes issue
        let referenceFixes = generateReferenceFixes issue
        let unusedVarFixes = generateUnusedVariableFixes issue
        
        syntaxFixes @ typeFixes @ referenceFixes @ unusedVarFixes
    
    /// Generate fixes for multiple issues
    let generateFixes (issues: CodeIssue list) : FixGenerationResult =
        let startTime = DateTime.Now
        
        let allFixes = 
            issues
            |> List.collect generateFixesForIssue
        
        let endTime = DateTime.Now
        
        // Calculate success rate (percentage of issues that have at least one fix)
        let issuesWithFixes = issues |> List.filter (fun issue -> generateFixesForIssue issue |> List.isEmpty |> not) |> List.length
        let successRate = if issues.Length > 0 then float issuesWithFixes / float issues.Length else 0.0
        
        {
            Fixes = allFixes
            GenerationTime = endTime - startTime
            SuccessRate = successRate
        }
    
    /// Filter fixes by confidence threshold
    let filterByConfidence (threshold: float) (result: FixGenerationResult) : FixGenerationResult =
        { result with Fixes = result.Fixes |> List.filter (fun fix -> fix.Confidence >= threshold) }
    
    /// Group fixes by category
    let groupByCategory (result: FixGenerationResult) : Map<string, CodeFix list> =
        result.Fixes
        |> List.groupBy (fun fix -> fix.Category)
        |> Map.ofList
    
    /// Sort fixes by confidence (highest first)
    let sortByConfidence (result: FixGenerationResult) : FixGenerationResult =
        { result with Fixes = result.Fixes |> List.sortByDescending (fun fix -> fix.Confidence) }
    
    /// Get fix summary
    let getFixSummary (result: FixGenerationResult) : string =
        let fixesByCategory = groupByCategory result
        let categoryStats = 
            fixesByCategory
            |> Map.toList
            |> List.map (fun (category, fixes) -> $"{category}: {fixes.Length}")
            |> String.concat ", "
        
        $"Fix generation completed in {result.GenerationTime.TotalMilliseconds:F0}ms. " +
        $"Generated {result.Fixes.Length} fixes with {result.SuccessRate:P1} success rate. " +
        $"Breakdown: {categoryStats}"
