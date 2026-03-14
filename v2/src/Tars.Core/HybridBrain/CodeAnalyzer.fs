namespace Tars.Core.HybridBrain

open System
open System.IO
open System.Text.RegularExpressions

/// Analyzes code and creates refactoring plans
module CodeAnalyzer =

    /// A detected issue in the code
    type CodeIssue =
        | LongFunction of name: string * lineCount: int * startLine: int * endLine: int
        | MissingDocumentation of name: string * lineNumber: int
        | DuplicateCode of block1Start: int * block1End: int * block2Start: int * block2End: int
        | DeadCode of startLine: int * endLine: int * reason: string
        | ComplexCondition of lineNumber: int * complexity: int

    /// Result of code analysis
    type AnalysisResult =
        { FilePath: string
          TotalLines: int
          Functions: (string * int * int) list
          Issues: CodeIssue list
          SuggestedActions: Tars.Core.HybridBrain.Action list }

    /// Configuration for analysis thresholds
    type AnalysisConfig =
        { MaxFunctionLines: int
          RequireDocumentation: bool
          DetectDuplicates: bool
          MinDuplicateLines: int }

    let defaultConfig =
        { MaxFunctionLines = 30
          RequireDocumentation = true
          DetectDuplicates = true
          MinDuplicateLines = 5 }

    /// Parse F# file and extract function locations
    let private extractFunctions (lines: string array) : (string * int * int) list =
        let mutable functions = []
        let mutable currentFunc = None
        let mutable braceDepth = 0

        for i in 0 .. lines.Length - 1 do
            let trimmed = lines.[i].TrimStart()

            // Detect function start (must start at column 0 in F# for top-level)
            if
                lines.[i].StartsWith("let ")
                && trimmed.Contains("=")
                && not (trimmed.StartsWith("let mutable"))
            then
                let nameMatch =
                    Regex.Match(trimmed, @"let\s+(?:(?:private|rec|inline|mutable)\s+)*(\w+)")

                if nameMatch.Success then
                    currentFunc <- Some(nameMatch.Groups.[1].Value, i + 1)

            // Track indentation to find function end
            match currentFunc with
            | Some(name, startLine) ->
                let rawLine = lines.[i]

                if
                    i > startLine - 1
                    && rawLine.Length > 0
                    && not (rawLine.StartsWith(" "))
                    && not (rawLine.StartsWith("\t"))
                    && not (rawLine.StartsWith("///"))
                then
                    // Found end of function (back to column 0)
                    functions <- (name, startLine, i) :: functions
                    currentFunc <- None
            | None -> ()

        // Handle last function
        match currentFunc with
        | Some(name, startLine) -> functions <- (name, startLine, lines.Length) :: functions
        | None -> ()

        functions |> List.rev

    /// Check if a line has documentation (XML comment)
    let private hasDocumentation (lines: string array) (lineNumber: int) : bool =
        if lineNumber <= 1 then
            false
        else
            let prevLine = lines.[lineNumber - 2].Trim()
            prevLine.StartsWith("///") || prevLine.StartsWith("(*")

    /// Find duplicate code blocks using simple hash comparison
    let private findDuplicates (lines: string array) (minLines: int) : (int * int * int * int) list =
        let normalizedLines =
            lines
            |> Array.map (fun l -> l.Trim().ToLowerInvariant())
            |> Array.mapi (fun i l -> (i + 1, l))

        let mutable duplicates = []

        // Simple n-gram comparison (could be improved with more sophisticated algorithms)
        for i in 0 .. lines.Length - minLines - 1 do
            let block1 = normalizedLines.[i .. i + minLines - 1] |> Array.map snd
            let block1Hash = String.Join("\n", block1)

            for j in i + minLines .. lines.Length - minLines - 1 do
                let block2 = normalizedLines.[j .. j + minLines - 1] |> Array.map snd
                let block2Hash = String.Join("\n", block2)

                if block1Hash = block2Hash && not (block1Hash.Trim() = "") then
                    duplicates <- (i + 1, i + minLines, j + 1, j + minLines) :: duplicates

        duplicates |> List.distinctBy (fun (a, b, c, d) -> (min a c, max a c))

    /// Analyze an F# file for refactoring opportunities
    let analyzeFile (config: AnalysisConfig) (filePath: string) : AnalysisResult =
        let lines = File.ReadAllLines(filePath)
        let functions = extractFunctions lines

        let mutable issues: CodeIssue list = []
        let mutable actions: Tars.Core.HybridBrain.Action list = []

        // Check for long functions
        for (name, startLine, endLine) in functions do
            let lineCount = endLine - startLine + 1

            if lineCount > config.MaxFunctionLines then
                issues <- LongFunction(name, lineCount, startLine, endLine) :: issues
                // Suggest extracting middle portion
                let midStart = startLine + 5
                let midEnd = endLine - 5

                if midEnd > midStart then
                    actions <-
                        Tars.Core.HybridBrain.Action.ExtractFunction($"{name}_helper", midStart, midEnd)
                        :: actions

        // Check for missing documentation
        if config.RequireDocumentation then
            for (name, startLine, _) in functions do
                if not (hasDocumentation lines startLine) then
                    issues <- MissingDocumentation(name, startLine) :: issues

                    actions <-
                        Tars.Core.HybridBrain.Action.AddDocumentation($"TODO: Document {name}", startLine)
                        :: actions

        // Check for duplicate code
        if config.DetectDuplicates then
            let duplicates = findDuplicates lines config.MinDuplicateLines

            for (b1s, b1e, b2s, b2e) in duplicates do
                issues <- DuplicateCode(b1s, b1e, b2s, b2e) :: issues

                actions <-
                    Tars.Core.HybridBrain.Action.ExtractFunction($"extracted_{b1s}", b1s, b1e)
                    :: actions

        // Filter out overlapping actions
        let filterOverlappingActions (actions: Tars.Core.HybridBrain.Action list) : Tars.Core.HybridBrain.Action list =
            let getRange action =
                match action with
                | Tars.Core.HybridBrain.Action.ExtractFunction(_, s, e) -> Some(s, e)
                | Tars.Core.HybridBrain.Action.RemoveDeadCode(s, e) -> Some(s, e)
                | _ -> None

            let rec filterNonOverlapping
                (acc: Tars.Core.HybridBrain.Action list)
                (remaining: Tars.Core.HybridBrain.Action list)
                =
                match remaining with
                | [] -> acc |> List.rev
                | action :: rest ->
                    match getRange action with
                    | None -> filterNonOverlapping (action :: acc) rest
                    | Some(s1, e1) ->
                        let overlaps =
                            acc
                            |> List.exists (fun a ->
                                match getRange a with
                                | None -> false
                                | Some(s2, e2) -> not (e1 < s2 || e2 < s1) // Ranges overlap
                            )

                        if overlaps then
                            filterNonOverlapping acc rest
                        else
                            filterNonOverlapping (action :: acc) rest

            filterNonOverlapping [] actions

        let filteredActions =
            filterOverlappingActions (actions |> List.rev) |> List.truncate 5

        { FilePath = filePath
          TotalLines = lines.Length
          Functions = functions
          Issues = issues |> List.rev
          SuggestedActions = filteredActions }

    /// Create a refactoring plan from analysis results
    let createPlanFromAnalysis (result: AnalysisResult) : Plan<Draft> =
        let basePlan =
            StateTransitions.createDraft
                $"Refactor {Path.GetFileName(result.FilePath)}"
                $"Address {result.Issues.Length} issues found in analysis"

        let steps =
            result.SuggestedActions
            |> List.mapi (fun i action ->
                { Id = i + 1
                  Name = $"Step {i + 1}: {action.GetType().Name}"
                  Description = $"{action}"
                  Action = action
                  Preconditions = []
                  Postconditions = []
                  EvidenceRequired = false
                  Timeout = None
                  RetryCount = 0 }
                : Step)

        { basePlan with Steps = steps }

    /// Generate a human-readable report
    let generateReport (result: AnalysisResult) : string =
        let sb = System.Text.StringBuilder()

        sb.AppendLine($"═══════════════════════════════════════════════════════════════")
        |> ignore

        sb.AppendLine($"                   CODE ANALYSIS REPORT                        ")
        |> ignore

        sb.AppendLine($"═══════════════════════════════════════════════════════════════")
        |> ignore

        sb.AppendLine() |> ignore
        sb.AppendLine($"File: {result.FilePath}") |> ignore
        sb.AppendLine($"Lines: {result.TotalLines}") |> ignore
        sb.AppendLine($"Functions: {result.Functions.Length}") |> ignore
        sb.AppendLine($"Issues Found: {result.Issues.Length}") |> ignore
        sb.AppendLine() |> ignore

        if result.Issues.Length > 0 then
            sb.AppendLine("ISSUES:") |> ignore

            for issue in result.Issues do
                match issue with
                | LongFunction(name, count, s, e) ->
                    sb.AppendLine($"  ⚠ Long function '{name}' ({count} lines) at {s}-{e}")
                    |> ignore
                | MissingDocumentation(name, line) ->
                    sb.AppendLine($"  📝 Missing docs for '{name}' at line {line}") |> ignore
                | DuplicateCode(b1s, b1e, b2s, b2e) ->
                    sb.AppendLine($"  🔁 Duplicate code: lines {b1s}-{b1e} ≈ {b2s}-{b2e}") |> ignore
                | DeadCode(s, e, reason) -> sb.AppendLine($"  💀 Dead code at {s}-{e}: {reason}") |> ignore
                | ComplexCondition(line, complexity) ->
                    sb.AppendLine($"  🔀 Complex condition at line {line} (score: {complexity})")
                    |> ignore

            sb.AppendLine() |> ignore

        if result.SuggestedActions.Length > 0 then
            sb.AppendLine("SUGGESTED ACTIONS:") |> ignore

            for i, action in result.SuggestedActions |> List.indexed do
                sb.AppendLine($"  {i + 1}. {action}") |> ignore

        sb.AppendLine() |> ignore

        sb.AppendLine($"═══════════════════════════════════════════════════════════════")
        |> ignore

        sb.ToString()
