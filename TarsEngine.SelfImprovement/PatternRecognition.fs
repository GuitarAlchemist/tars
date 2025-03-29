namespace TarsEngine.SelfImprovement

open System
open System.Text.RegularExpressions
open System.Collections.Generic

/// <summary>
/// Represents a code pattern that can be recognized in source code
/// </summary>
type CodePattern =
    { Id: string
      Name: string
      Description: string
      Language: string
      Pattern: string
      IsRegex: bool
      Severity: int
      Recommendation: string
      Examples: string list
      Tags: string list }

/// <summary>
/// Represents a pattern match found in source code
/// </summary>
type PatternMatch =
    { PatternId: string
      LineNumber: int
      ColumnNumber: int
      MatchedText: string
      Context: string
      Recommendation: string }

/// <summary>
/// Functions for recognizing patterns in source code
/// </summary>
module PatternRecognition =
    /// <summary>
    /// Common code patterns to look for
    /// </summary>
    let commonPatterns =
        [
            // C# patterns
            { Id = "CS001"
              Name = "Empty catch block"
              Description = "Empty catch blocks suppress exceptions without handling them"
              Language = "csharp"
              Pattern = "catch\\s*\\([^)]*\\)\\s*{\\s*}"
              IsRegex = true
              Severity = 2
              Recommendation = "Handle the exception or log it instead of suppressing it"
              Examples = ["try { DoSomething(); } catch (Exception) { }"]
              Tags = ["csharp"; "exception-handling"; "code-quality"] }
            
            { Id = "CS002"
              Name = "Magic number"
              Description = "Unnamed numeric literals make code harder to understand"
              Language = "csharp"
              Pattern = "(?<![\\w.])[0-9]+(?![\\w.])"
              IsRegex = true
              Severity = 1
              Recommendation = "Replace magic numbers with named constants"
              Examples = ["int timeout = 300;"; "if (count > 1000) { ... }"]
              Tags = ["csharp"; "readability"; "maintainability"] }
            
            { Id = "CS003"
              Name = "String concatenation in loop"
              Description = "String concatenation in loops is inefficient"
              Language = "csharp"
              Pattern = "for\\s*\\([^)]*\\)\\s*{[^}]*\\+=[^}]*}"
              IsRegex = true
              Severity = 2
              Recommendation = "Use StringBuilder instead of string concatenation in loops"
              Examples = ["for (int i = 0; i < 100; i++) { result += i.ToString(); }"]
              Tags = ["csharp"; "performance"; "best-practice"] }
            
            // F# patterns
            { Id = "FS001"
              Name = "Mutable variable"
              Description = "Mutable variables should be avoided when possible"
              Language = "fsharp"
              Pattern = "let mutable"
              IsRegex = false
              Severity = 1
              Recommendation = "Consider using immutable values and functional transformations"
              Examples = ["let mutable count = 0"]
              Tags = ["fsharp"; "functional-programming"; "immutability"] }
            
            { Id = "FS002"
              Name = "Imperative loop"
              Description = "Imperative loops should be replaced with functional alternatives"
              Language = "fsharp"
              Pattern = "for\\s+.*\\s+do"
              IsRegex = true
              Severity = 1
              Recommendation = "Consider using List.map, List.iter, or other higher-order functions"
              Examples = ["for i in 1..10 do printfn \"%d\" i"]
              Tags = ["fsharp"; "functional-programming"; "higher-order-functions"] }
            
            // General patterns
            { Id = "GEN001"
              Name = "TODO comment"
              Description = "TODO comments indicate incomplete work"
              Language = "any"
              Pattern = "//\\s*TODO"
              IsRegex = true
              Severity = 1
              Recommendation = "Complete the TODO item or create a task to track it"
              Examples = ["// TODO: Implement error handling"]
              Tags = ["general"; "comments"; "technical-debt"] }
            
            { Id = "GEN002"
              Name = "Long method"
              Description = "Long methods are harder to understand and maintain"
              Language = "any"
              Pattern = ""
              IsRegex = false
              Severity = 2
              Recommendation = "Break down long methods into smaller, focused methods"
              Examples = []
              Tags = ["general"; "method-length"; "maintainability"] }
        ]
    
    /// <summary>
    /// Recognizes patterns in source code
    /// </summary>
    let recognizePatterns (content: string) (language: string) =
        let matches = new List<PatternMatch>()
        let lines = content.Split('\n')
        
        // Apply regex patterns
        for pattern in commonPatterns do
            if pattern.Language = language || pattern.Language = "any" then
                if pattern.IsRegex then
                    let regex = new Regex(pattern.Pattern)
                    let matchCollection = regex.Matches(content)
                    
                    for m in matchCollection do
                        // Find line and column number
                        let beforeMatch = content.Substring(0, m.Index)
                        let lineNumber = beforeMatch.Split('\n').Length
                        let lastNewline = beforeMatch.LastIndexOf('\n')
                        let columnNumber = if lastNewline >= 0 then m.Index - lastNewline else m.Index + 1
                        
                        // Get context (the line containing the match)
                        let contextLine = lines.[lineNumber - 1]
                        
                        matches.Add(
                            { PatternId = pattern.Id
                              LineNumber = lineNumber
                              ColumnNumber = columnNumber
                              MatchedText = m.Value
                              Context = contextLine.Trim()
                              Recommendation = pattern.Recommendation })
                else if pattern.Pattern <> "" then
                    // Simple string search for non-regex patterns
                    let mutable index = content.IndexOf(pattern.Pattern)
                    while index >= 0 do
                        // Find line and column number
                        let beforeMatch = content.Substring(0, index)
                        let lineNumber = beforeMatch.Split('\n').Length
                        let lastNewline = beforeMatch.LastIndexOf('\n')
                        let columnNumber = if lastNewline >= 0 then index - lastNewline else index + 1
                        
                        // Get context (the line containing the match)
                        let contextLine = lines.[lineNumber - 1]
                        
                        matches.Add(
                            { PatternId = pattern.Id
                              LineNumber = lineNumber
                              ColumnNumber = columnNumber
                              MatchedText = pattern.Pattern
                              Context = contextLine.Trim()
                              Recommendation = pattern.Recommendation })
                        
                        index <- content.IndexOf(pattern.Pattern, index + pattern.Pattern.Length)
        
        // Special case for long methods
        let longMethodPattern = commonPatterns |> List.find (fun p -> p.Id = "GEN002")
        let methodRegex = 
            if language = "csharp" then
                new Regex(@"(public|private|protected|internal|static)?\s+\w+\s+\w+\s*\([^)]*\)\s*{")
            else if language = "fsharp" then
                new Regex(@"let\s+\w+(?:\s+\w+)*\s*=")
            else
                new Regex(@"function\s+\w+\s*\([^)]*\)")
        
        let methodMatches = methodRegex.Matches(content)
        let mutable lastMethodStart = 0
        let mutable lastMethodLine = 0
        
        for m in methodMatches do
            if lastMethodStart > 0 then
                let methodLength = m.Index - lastMethodStart
                let methodContent = content.Substring(lastMethodStart, methodLength)
                let lineCount = methodContent.Split('\n').Length
                
                // Consider methods with more than 30 lines as "long"
                if lineCount > 30 then
                    matches.Add(
                        { PatternId = "GEN002"
                          LineNumber = lastMethodLine
                          ColumnNumber = 1
                          MatchedText = "Long method"
                          Context = $"Method with {lineCount} lines"
                          Recommendation = longMethodPattern.Recommendation })
            
            lastMethodStart <- m.Index
            let beforeMatch = content.Substring(0, m.Index)
            lastMethodLine <- beforeMatch.Split('\n').Length
        
        matches |> Seq.toList
    
    /// <summary>
    /// Gets recommendations based on pattern matches
    /// </summary>
    let getRecommendations (matches: PatternMatch list) =
        matches
        |> List.map (fun m -> m.Recommendation)
        |> List.distinct
    
    /// <summary>
    /// Gets issues based on pattern matches
    /// </summary>
    let getIssues (matches: PatternMatch list) =
        matches
        |> List.map (fun m -> 
            let pattern = commonPatterns |> List.find (fun p -> p.Id = m.PatternId)
            $"Line {m.LineNumber}: {pattern.Name} - {pattern.Description}")
        |> List.distinct
