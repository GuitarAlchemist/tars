namespace TarsEngine.DSL

open System
open System.Text.RegularExpressions
open System.Collections.Generic

/// <summary>
/// Module for handling warning suppression.
/// </summary>
module WarningSuppression =
    /// <summary>
    /// Registry of suppressed warnings.
    /// </summary>
    let private suppressedWarnings = Dictionary<int, HashSet<WarningCode>>()
    
    /// <summary>
    /// Clear all suppressed warnings.
    /// </summary>
    let clearSuppressedWarnings() =
        suppressedWarnings.Clear()
    
    /// <summary>
    /// Parse warning suppression comments from code.
    /// </summary>
    /// <param name="code">The code to parse.</param>
    let parseWarningSuppressionComments (code: string) =
        // Clear existing suppressions
        clearSuppressedWarnings()
        
        // Parse suppression comments
        // Format: // @suppress-warning: WarningCode [line]
        let suppressionPattern = @"//\s*@suppress-warning:\s*(\w+)(?:\s+(\d+))?"
        let matches = Regex.Matches(code, suppressionPattern)
        
        let lines = code.Split([|'\n'|], StringSplitOptions.None)
        
        for m in matches do
            if m.Groups.Count >= 2 then
                let warningCodeStr = m.Groups.[1].Value
                let lineNumberStr = if m.Groups.Count >= 3 then m.Groups.[2].Value else ""
                
                // Parse warning code
                let warningCode = 
                    match Enum.TryParse<WarningCode>(warningCodeStr) with
                    | true, code -> Some code
                    | false, _ -> None
                
                // Parse line number
                let lineNumber = 
                    if String.IsNullOrEmpty(lineNumberStr) then
                        // If no line number is specified, use the line number of the comment
                        let commentLineNumber = 
                            let commentStartIndex = m.Index
                            let lineStartIndices = 
                                lines 
                                |> Array.scan (fun acc line -> acc + line.Length + 1) 0 
                                |> Array.take lines.Length
                            
                            lineStartIndices 
                            |> Array.findIndex (fun startIndex -> startIndex > commentStartIndex) 
                        
                        commentLineNumber + 1 // Convert to 1-based line number
                    else
                        match Int32.TryParse(lineNumberStr) with
                        | true, num -> num
                        | false, _ -> 0
                
                // Add suppression
                match warningCode with
                | Some code ->
                    if not (suppressedWarnings.ContainsKey(lineNumber)) then
                        suppressedWarnings.Add(lineNumber, HashSet<WarningCode>())
                    
                    suppressedWarnings.[lineNumber].Add(code) |> ignore
                | None -> ()
    
    /// <summary>
    /// Check if a warning is suppressed for a specific line.
    /// </summary>
    /// <param name="warningCode">The warning code.</param>
    /// <param name="line">The line number.</param>
    /// <returns>True if the warning is suppressed, false otherwise.</returns>
    let isWarningSupressed (warningCode: WarningCode) (line: int) =
        match suppressedWarnings.TryGetValue(line) with
        | true, codes -> codes.Contains(warningCode)
        | false, _ -> false
    
    /// <summary>
    /// Suppress a warning for a specific line.
    /// </summary>
    /// <param name="warningCode">The warning code to suppress.</param>
    /// <param name="line">The line number.</param>
    let suppressWarning (warningCode: WarningCode) (line: int) =
        if not (suppressedWarnings.ContainsKey(line)) then
            suppressedWarnings.Add(line, HashSet<WarningCode>())
        
        suppressedWarnings.[line].Add(warningCode) |> ignore
    
    /// <summary>
    /// Unsuppress a warning for a specific line.
    /// </summary>
    /// <param name="warningCode">The warning code to unsuppress.</param>
    /// <param name="line">The line number.</param>
    let unsuppressWarning (warningCode: WarningCode) (line: int) =
        if suppressedWarnings.ContainsKey(line) then
            suppressedWarnings.[line].Remove(warningCode) |> ignore
    
    /// <summary>
    /// Get all suppressed warnings.
    /// </summary>
    /// <returns>A list of tuples containing (line, warningCode).</returns>
    let getAllSuppressedWarnings() =
        suppressedWarnings
        |> Seq.collect (fun kvp -> 
            kvp.Value 
            |> Seq.map (fun code -> 
                (kvp.Key, code)))
        |> Seq.toList
    
    /// <summary>
    /// Filter diagnostics based on suppressed warnings.
    /// </summary>
    /// <param name="diagnostics">The list of diagnostics to filter.</param>
    /// <returns>A list of diagnostics with suppressed warnings removed.</returns>
    let filterSuppressedWarnings (diagnostics: Diagnostic list) =
        diagnostics
        |> List.filter (fun diagnostic -> 
            not (isWarningSupressed diagnostic.Code diagnostic.Line))
