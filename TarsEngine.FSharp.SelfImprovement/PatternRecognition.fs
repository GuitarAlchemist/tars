namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Text.RegularExpressions
open ImprovementTypes

/// Advanced pattern recognition for code improvement
module PatternRecognition =

    /// Pattern with ID for matching
    type PatternWithId = {
        Id: string
        Name: string
        Description: string
        PatternType: PatternType
        Severity: Severity
        Example: string option
        Recommendation: string
    }

    /// Common patterns with IDs (for Ollama integration)
    let commonPatterns = [
        { Id = "CS003"; Name = "String concatenation in loop"; Description = "String concatenation using + in loops"; PatternType = PatternType.Performance; Severity = Severity.Medium; Example = Some "for(...) { str += item; }"; Recommendation = "Use StringBuilder for better performance" }
        { Id = "CS004"; Name = "LINQ in tight loops"; Description = "LINQ operations in performance-critical loops"; PatternType = PatternType.Performance; Severity = Severity.Medium; Example = Some "for(...) { var x = list.Where(...); }"; Recommendation = "Use traditional loops for better performance" }
        { Id = "CS005"; Name = "Collection without capacity"; Description = "Collections initialized without capacity"; PatternType = PatternType.Performance; Severity = Severity.Low; Example = Some "var list = new List<int>();"; Recommendation = "Initialize with estimated capacity" }
        { Id = "CS006"; Name = "Async void"; Description = "Async void methods"; PatternType = PatternType.Maintainability; Severity = Severity.High; Example = Some "async void Method()"; Recommendation = "Use async Task instead" }
        { Id = "CS007"; Name = "Missing using"; Description = "IDisposable without using"; PatternType = PatternType.Maintainability; Severity = Severity.High; Example = Some "var stream = new FileStream(...)"; Recommendation = "Use using declaration" }
        { Id = "FS003"; Name = "Non-tail recursion"; Description = "Recursive function without tail call"; PatternType = PatternType.Performance; Severity = Severity.Medium; Example = Some "let rec factorial n = if n = 0 then 1 else n * factorial (n-1)"; Recommendation = "Use tail recursion with accumulator" }
        { Id = "FS004"; Name = "List concatenation in loop"; Description = "List concatenation using @ in loops"; PatternType = PatternType.Performance; Severity = Severity.Medium; Example = Some "for x in xs do list <- list @ [x]"; Recommendation = "Use list comprehension" }
        { Id = "FS005"; Name = "Missing type annotation"; Description = "Function without type annotation"; PatternType = PatternType.Documentation; Severity = Severity.Low; Example = Some "let func x = x + 1"; Recommendation = "Add type annotation for clarity" }
        { Id = "GEN003"; Name = "Commented code"; Description = "Commented out code"; PatternType = PatternType.Maintainability; Severity = Severity.Low; Example = Some "// oldFunction()"; Recommendation = "Remove commented code" }
        { Id = "GEN004"; Name = "Magic string"; Description = "Hardcoded string values"; PatternType = PatternType.Maintainability; Severity = Severity.Medium; Example = Some "var x = \"hardcoded_value\""; Recommendation = "Extract to named constant" }
        { Id = "GEN005"; Name = "Complex condition"; Description = "Complex conditional expression"; PatternType = PatternType.Maintainability; Severity = Severity.Medium; Example = Some "if (a && b || c && d)"; Recommendation = "Break down into named boolean" }
    ]

    /// Built-in improvement patterns for F# and C#
    let private builtInPatterns = [
        // F# specific patterns
        {
            Name = "Unused open statement"
            Description = "Open statement that is not used in the file"
            PatternType = PatternType.Maintainability
            Severity = Severity.Low
            Example = Some "open System.Collections.Generic // Not used"
            Recommendation = "Remove unused open statements to reduce namespace pollution"
        }
        
        {
            Name = "Long function"
            Description = "Function with too many lines (>50)"
            PatternType = PatternType.Maintainability
            Severity = Severity.Medium
            Example = Some "let longFunction() = // 60+ lines of code"
            Recommendation = "Break down into smaller, focused functions"
        }
        
        {
            Name = "Magic numbers"
            Description = "Hardcoded numeric values without explanation"
            PatternType = PatternType.Maintainability
            Severity = Severity.Medium
            Example = Some "if x > 42 then // What is 42?"
            Recommendation = "Replace with named constants or configuration"
        }
        
        // C# specific patterns
        {
            Name = "Missing async/await"
            Description = "Synchronous calls to async methods"
            PatternType = PatternType.Performance
            Severity = Severity.High
            Example = Some "var result = asyncMethod().Result;"
            Recommendation = "Use await instead of .Result to avoid deadlocks"
        }
        
        {
            Name = "String concatenation in loop"
            Description = "String concatenation using + in loops"
            PatternType = PatternType.Performance
            Severity = Severity.Medium
            Example = Some "for(...) { str += item; }"
            Recommendation = "Use StringBuilder for better performance"
        }
        
        // General patterns
        {
            Name = "TODO comments"
            Description = "TODO, FIXME, HACK comments indicating incomplete work"
            PatternType = PatternType.Documentation
            Severity = Severity.Low
            Example = Some "// TODO: Implement this properly"
            Recommendation = "Address TODO items or create proper issues"
        }
        
        {
            Name = "Empty catch blocks"
            Description = "Exception handling with empty catch blocks"
            PatternType = PatternType.Security
            Severity = Severity.High
            Example = Some "try { ... } catch { }"
            Recommendation = "Add proper error handling and logging"
        }
    ]
    
    /// Analyze file content for improvement patterns
    let analyzeFile (filePath: string) (content: string) : AnalysisResult =
        let extension = Path.GetExtension(filePath).ToLower()
        let lines = content.Split([|'\n'|], StringSplitOptions.None)
        
        let detectedIssues = 
            builtInPatterns
            |> List.filter (fun pattern ->
                match pattern.Name with
                | "Unused open statement" when extension = ".fs" ->
                    // Simple heuristic: find open statements and check if they're used
                    let openRegex = Regex(@"open\s+([A-Za-z\.]+)")
                    let opens = openRegex.Matches(content)
                    opens.Count > 0 // Simplified - would need more sophisticated analysis
                    
                | "Long function" ->
                    // Check for functions with >50 lines
                    let functionRegex = 
                        if extension = ".fs" then Regex(@"let\s+\w+.*=")
                        else Regex(@"(public|private|protected|internal).*\w+\s*\([^)]*\)\s*{")
                    
                    functionRegex.Matches(content).Count > 0 // Simplified
                    
                | "Magic numbers" ->
                    // Look for hardcoded numbers (excluding 0, 1, -1)
                    let magicNumberRegex = Regex(@"\b(?!0\b|1\b|-1\b)\d{2,}\b")
                    magicNumberRegex.Matches(content).Count > 0
                    
                | "Missing async/await" when extension = ".cs" ->
                    content.Contains(".Result") || content.Contains(".Wait()")
                    
                | "String concatenation in loop" ->
                    let loopConcatRegex = Regex(@"(for|while).*\{[^}]*\+=.*string", RegexOptions.Singleline)
                    loopConcatRegex.IsMatch(content)
                    
                | "TODO comments" ->
                    let todoRegex = Regex(@"//\s*(TODO|FIXME|HACK|XXX)", RegexOptions.IgnoreCase)
                    todoRegex.IsMatch(content)
                    
                | "Empty catch blocks" ->
                    let emptyCatchRegex = Regex(@"catch[^{]*\{\s*\}")
                    emptyCatchRegex.IsMatch(content)
                    
                | _ -> false
            )
        
        // Calculate overall score (0-10, higher is better)
        let issueCount = detectedIssues.Length
        let criticalIssues = detectedIssues |> List.filter (fun i -> i.Severity = Severity.Critical) |> List.length
        let highIssues = detectedIssues |> List.filter (fun i -> i.Severity = Severity.High) |> List.length
        
        let overallScore = 
            10.0 - (float criticalIssues * 3.0) - (float highIssues * 2.0) - (float issueCount * 0.5)
            |> max 0.0
        
        let recommendations = 
            detectedIssues 
            |> List.map (fun issue -> issue.Recommendation)
            |> List.distinct
        
        {
            FilePath = filePath
            Issues = detectedIssues
            OverallScore = overallScore
            Recommendations = recommendations
            AnalyzedAt = DateTime.UtcNow
        }
    
    /// Get all available patterns
    let getAllPatterns() = builtInPatterns
    
    /// Add custom pattern
    let addCustomPattern (pattern: ImprovementPattern) =
        // In a real implementation, this would persist to configuration
        pattern :: builtInPatterns
