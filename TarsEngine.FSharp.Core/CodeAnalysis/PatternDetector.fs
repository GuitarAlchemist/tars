namespace TarsEngine.FSharp.Core.CodeAnalysis

/// Module for detecting patterns in code
module PatternDetector =
    open System
    open System.IO
    open System.Text.RegularExpressions
    open Types
    
    /// Detects patterns in a single line of code
    let detectPatternsInLine (patterns: Pattern list) (filePath: string) (lineNumber: int) (line: string) : Match list =
        patterns
        |> List.collect (fun pattern ->
            let regex = Regex(pattern.Regex)
            regex.Matches(line)
            |> Seq.cast<System.Text.RegularExpressions.Match>
            |> Seq.map (fun m ->
                { Pattern = pattern
                  Text = m.Value
                  LineNumber = lineNumber
                  ColumnNumber = m.Index
                  FilePath = filePath })
            |> Seq.toList)
    
    /// Detects patterns in a file
    let detectPatternsInFile (patterns: Pattern list) (filePath: string) : Match list =
        try
            let lines = File.ReadAllLines(filePath)
            lines
            |> Array.mapi (fun i line -> detectPatternsInLine patterns filePath (i + 1) line)
            |> Array.toList
            |> List.concat
        with
        | ex -> 
            printfn "Error analyzing file %s: %s" filePath ex.Message
            []
    
    /// Detects patterns in a directory
    let detectPatternsInDirectory (config: Configuration) (directoryPath: string) : Match list =
        let isExcluded (path: string) =
            config.ExcludeDirectories
            |> List.exists (fun exclude -> path.Contains(exclude))
            || config.ExcludeFiles
               |> List.exists (fun exclude -> Path.GetFileName(path) = exclude)
        
        let isIncluded (path: string) =
            config.FileExtensions
            |> List.exists (fun ext -> Path.GetExtension(path).ToLower() = ext.ToLower())
        
        Directory.GetFiles(directoryPath, "*.*", SearchOption.AllDirectories)
        |> Array.filter (fun path -> not (isExcluded path) && isIncluded path)
        |> Array.map (fun path -> detectPatternsInFile config.Patterns path)
        |> Array.toList
        |> List.concat
    
    /// Filters matches by language
    let filterMatchesByLanguage (language: string) (matches: Match list) : Match list =
        matches
        |> List.filter (fun m -> m.Pattern.Language = language)
    
    /// Filters matches by category
    let filterMatchesByCategory (category: string) (matches: Match list) : Match list =
        matches
        |> List.filter (fun m -> m.Pattern.Category = category)
    
    /// Filters matches by severity
    let filterMatchesBySeverity (minSeverity: float) (matches: Match list) : Match list =
        matches
        |> List.filter (fun m -> m.Pattern.Severity >= minSeverity)
    
    /// Sorts matches by severity
    let sortMatchesBySeverity (matches: Match list) : Match list =
        matches
        |> List.sortByDescending (fun m -> m.Pattern.Severity)
    
    /// Sorts matches by file path
    let sortMatchesByFilePath (matches: Match list) : Match list =
        matches
        |> List.sortBy (fun m -> m.FilePath)
    
    /// Sorts matches by line number
    let sortMatchesByLineNumber (matches: Match list) : Match list =
        matches
        |> List.sortBy (fun m -> m.LineNumber)
    
    /// Groups matches by pattern
    let groupMatchesByPattern (matches: Match list) : (Pattern * Match list) list =
        matches
        |> List.groupBy (fun m -> m.Pattern)
        |> List.sortByDescending (fun (pattern, _) -> pattern.Severity)
    
    /// Groups matches by file
    let groupMatchesByFile (matches: Match list) : (string * Match list) list =
        matches
        |> List.groupBy (fun m -> m.FilePath)
        |> List.sortBy (fun (filePath, _) -> filePath)
    
    /// Calculates the overall score based on matches
    let calculateScore (matches: Match list) : float =
        if List.isEmpty matches then
            1.0
        else
            let totalSeverity = 
                matches
                |> List.sumBy (fun m -> m.Pattern.Severity)
            
            let maxSeverity = 
                matches
                |> List.length
                |> float
            
            1.0 - (totalSeverity / maxSeverity)
    
    /// Creates a summary of the matches
    let createSummary (matches: Match list) : string =
        let patternGroups = groupMatchesByPattern matches
        let fileGroups = groupMatchesByFile matches
        
        let patternSummary =
            patternGroups
            |> List.map (fun (pattern, matches) ->
                sprintf "Pattern '%s' (%s): %d matches" pattern.Name pattern.Category matches.Length)
            |> String.concat "\n"
        
        let fileSummary =
            fileGroups
            |> List.map (fun (filePath, matches) ->
                sprintf "File '%s': %d matches" filePath matches.Length)
            |> String.concat "\n"
        
        sprintf "Analysis Summary:\n\nPatterns:\n%s\n\nFiles:\n%s" patternSummary fileSummary
    
    /// Creates a report from matches
    let createReport (matches: Match list) (transformations: (Transformation * string) list) : Report =
        { Matches = matches
          Transformations = transformations
          Summary = createSummary matches
          Score = calculateScore matches }
