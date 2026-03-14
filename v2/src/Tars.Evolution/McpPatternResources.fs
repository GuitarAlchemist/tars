namespace Tars.Evolution

open System
open System.Text.Json
open Tars.Llm

/// <summary>
/// MCP-exposed pattern library resources.
/// Provides tools that MCP clients (like Claude Code) can use to query,
/// search, and inspect the reasoning pattern library.
/// </summary>
module McpPatternResources =

    // =========================================================================
    // Response Types
    // =========================================================================

    type PatternSummary =
        { Name: string
          Description: string
          Goal: string
          Score: float
          CreatedFromRunId: string option }

    type LibraryStats =
        { TotalPatterns: int
          AvgFitness: float
          TopPatterns: PatternSummary list
          DiversityScore: float
          GoalCoverage: string list }

    type PatternSuggestion =
        { PatternName: string
          Score: float
          Description: string
          Rationale: string }

    // =========================================================================
    // Tool Implementations
    // =========================================================================

    /// List all available reasoning patterns with names, descriptions, and fitness scores.
    let listPatterns (_input: string) : Async<Result<string, string>> =
        async {
            try
                let patterns = PatternLibrary.loadAll ()

                let summaries =
                    patterns
                    |> List.sortByDescending (fun p -> p.Score)
                    |> List.map (fun p ->
                        { Name = p.Name
                          Description = p.Description
                          Goal = p.Goal
                          Score = p.Score
                          CreatedFromRunId = p.CreatedFromRunId |> Option.map (fun g -> g.ToString()) })

                let options = JsonSerializerOptions(WriteIndented = true)
                options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
                let json = JsonSerializer.Serialize(summaries, options)
                return Ok json
            with ex ->
                return Error (sprintf "Failed to list patterns: %s" ex.Message)
        }

    /// Get a specific pattern's full definition including WoT template.
    let getPattern (input: string) : Async<Result<string, string>> =
        async {
            try
                let patternName = input.Trim()
                let patterns = PatternLibrary.loadAll ()

                match patterns |> List.tryFind (fun p -> p.Name = patternName) with
                | Some pattern ->
                    let options = JsonSerializerOptions(WriteIndented = true)
                    options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
                    let json = JsonSerializer.Serialize(pattern, options)
                    return Ok json
                | None ->
                    // Try partial match
                    let partialMatches =
                        patterns
                        |> List.filter (fun p ->
                            p.Name.Contains(patternName, StringComparison.OrdinalIgnoreCase))

                    match partialMatches with
                    | [ single ] ->
                        let options = JsonSerializerOptions(WriteIndented = true)
                        options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
                        let json = JsonSerializer.Serialize(single, options)
                        return Ok json
                    | multiple when multiple.Length > 0 ->
                        let names = multiple |> List.map (fun p -> p.Name) |> String.concat ", "
                        return Error (sprintf "Multiple patterns match '%s': %s. Be more specific." patternName names)
                    | _ ->
                        return Error (sprintf "Pattern '%s' not found" patternName)
            with ex ->
                return Error (sprintf "Failed to get pattern: %s" ex.Message)
        }

    /// Given a goal description, suggest the best matching pattern.
    /// Uses LLM-based matching via PatternLibrary.findMatch.
    let suggestPatternWithLlm (llm: ILlmService) (input: string) : Async<Result<string, string>> =
        async {
            try
                let goal = input.Trim()
                let patterns = PatternLibrary.loadAll ()

                if patterns.IsEmpty then
                    return Ok """{"suggestion": "No patterns in library yet. Run some retroaction cycles to build up the pattern library."}"""
                else
                    let! bestMatch = PatternLibrary.findMatch llm goal patterns

                    match bestMatch with
                    | Some pattern ->
                        let suggestion =
                            { PatternName = pattern.Name
                              Score = pattern.Score
                              Description = pattern.Description
                              Rationale = sprintf "Best match for goal '%s' based on description similarity" goal }

                        let options = JsonSerializerOptions(WriteIndented = true)
                        options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
                        let json = JsonSerializer.Serialize(suggestion, options)
                        return Ok json
                    | None ->
                        return Ok (sprintf """{"suggestion": "No suitable pattern found for goal: '%s'. Consider running the retroaction loop to generate patterns for this domain."}""" goal)
            with ex ->
                return Error (sprintf "Failed to suggest pattern: %s" ex.Message)
        }

    /// Simple heuristic pattern suggestion (no LLM needed).
    let suggestPattern (input: string) : Async<Result<string, string>> =
        async {
            try
                let goal = input.Trim().ToLowerInvariant()
                let patterns = PatternLibrary.loadAll ()

                if patterns.IsEmpty then
                    return Ok """{"suggestion": "No patterns in library yet."}"""
                else
                    // Simple keyword matching + score ranking
                    let goalWords = goal.Split([| ' '; ','; '.'; '!'; '?' |], StringSplitOptions.RemoveEmptyEntries)

                    let scored =
                        patterns
                        |> List.map (fun p ->
                            let descLower = p.Description.ToLowerInvariant()
                            let goalLower = p.Goal.ToLowerInvariant()
                            let keywordHits =
                                goalWords
                                |> Array.filter (fun w -> descLower.Contains(w) || goalLower.Contains(w))
                                |> Array.length
                            let keywordScore = float keywordHits / float (max 1 goalWords.Length)
                            let combinedScore = 0.6 * keywordScore + 0.4 * p.Score
                            (p, combinedScore))
                        |> List.sortByDescending snd

                    match scored with
                    | (bestPattern, bestScore) :: _ when bestScore > 0.1 ->
                        let suggestion =
                            { PatternName = bestPattern.Name
                              Score = bestPattern.Score
                              Description = bestPattern.Description
                              Rationale = sprintf "Best keyword match (relevance=%.2f) for goal" bestScore }

                        let options = JsonSerializerOptions(WriteIndented = true)
                        options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
                        let json = JsonSerializer.Serialize(suggestion, options)
                        return Ok json
                    | _ ->
                        return Ok """{"suggestion": "No suitable pattern found. Consider running retroaction cycles."}"""
            with ex ->
                return Error (sprintf "Failed to suggest pattern: %s" ex.Message)
        }

    /// Get library metrics: total patterns, average fitness, diversity, top patterns.
    let getLibraryStats (_input: string) : Async<Result<string, string>> =
        async {
            try
                let patterns = PatternLibrary.loadAll ()

                let avgFitness =
                    if patterns.IsEmpty then 0.0
                    else patterns |> List.averageBy (fun p -> p.Score)

                let goals = patterns |> List.map (fun p -> p.Goal) |> List.distinct
                let diversity =
                    if patterns.IsEmpty then 0.0
                    else float goals.Length / float patterns.Length

                let topPatterns =
                    patterns
                    |> List.sortByDescending (fun p -> p.Score)
                    |> List.truncate 5
                    |> List.map (fun p ->
                        { Name = p.Name
                          Description = p.Description
                          Goal = p.Goal
                          Score = p.Score
                          CreatedFromRunId = p.CreatedFromRunId |> Option.map (fun g -> g.ToString()) })

                let stats =
                    { TotalPatterns = patterns.Length
                      AvgFitness = avgFitness
                      TopPatterns = topPatterns
                      DiversityScore = diversity
                      GoalCoverage = goals }

                let options = JsonSerializerOptions(WriteIndented = true)
                options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
                let json = JsonSerializer.Serialize(stats, options)
                return Ok json
            with ex ->
                return Error (sprintf "Failed to get library stats: %s" ex.Message)
        }

    // =========================================================================
    // Promotion Pipeline Resources
    // =========================================================================

    /// Get promotion pipeline status: recurrence records, level distribution, lineage.
    let getPromotionStatus (_input: string) : Async<Result<string, string>> =
        async {
            try
                let records = PromotionPipeline.getRecurrenceRecords ()
                let lineage = PromotionPipeline.getLineageRecords ()

                let recurrenceOutputs = records |> List.map StructuredOutput.fromRecurrence
                let byLevel =
                    records
                    |> List.groupBy (fun r -> r.CurrentLevel)
                    |> List.map (fun (level, group) ->
                        PromotionLevel.label level, group.Length)
                    |> Map.ofList

                let status =
                    {| RecurrenceRecords = recurrenceOutputs
                       LevelDistribution = byLevel
                       TotalPatterns = records.Length
                       PromotedCount = records |> List.filter (fun r -> r.CurrentLevel <> Implementation) |> List.length
                       LineageCount = lineage.Length
                       AvgScore =
                           if records.IsEmpty then 0.0
                           else records |> List.averageBy (fun r -> r.AverageScore) |> fun x -> Math.Round(x, 3) |}

                let options = JsonSerializerOptions(WriteIndented = true)
                options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
                let json = JsonSerializer.Serialize(status, options)
                return Ok json
            with ex ->
                return Error (sprintf "Failed to get promotion status: %s" ex.Message)
        }

    /// Get promotion lineage (governance decisions and history).
    let getPromotionLineage (_input: string) : Async<Result<string, string>> =
        async {
            try
                let lineage = PromotionPipeline.getLineageRecords ()

                let outputs =
                    lineage
                    |> List.sortByDescending (fun r -> r.PromotedAt)
                    |> List.map (fun r ->
                        let decisionStr, reason =
                            match r.Decision with
                            | Approve reason -> "approve", reason
                            | Reject reason -> "reject", reason
                            | Defer reason -> "defer", reason
                        {| Id = r.Id
                           PatternId = r.PatternId
                           FromLevel = PromotionLevel.label r.FromLevel
                           ToLevel = PromotionLevel.label r.ToLevel
                           Decision = decisionStr
                           Reason = reason
                           Confidence = r.Confidence
                           PromotedAt = r.PromotedAt.ToString("o")
                           PromotedBy = r.PromotedBy |})

                let options = JsonSerializerOptions(WriteIndented = true)
                options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
                let json = JsonSerializer.Serialize(outputs, options)
                return Ok json
            with ex ->
                return Error (sprintf "Failed to get promotion lineage: %s" ex.Message)
        }

    /// Run the promotion pipeline and return structured JSON results.
    let runPromotionPipeline (input: string) : Async<Result<string, string>> =
        async {
            try
                let minOccurrences =
                    match Int32.TryParse(input.Trim()) with
                    | true, v when v > 0 -> v
                    | _ -> 3

                // Use existing recurrence records if available, otherwise create test data
                let existingRecords = PromotionPipeline.getRecurrenceRecords ()
                let artifacts =
                    if existingRecords.IsEmpty then
                        // No data yet — return empty run
                        []
                    else
                        // Create artifacts from existing records to re-evaluate promotions
                        existingRecords
                        |> List.collect (fun r ->
                            r.TaskIds
                            |> List.map (fun taskId ->
                                { PromotionPipeline.TraceArtifact.TaskId = taskId
                                  PatternName = r.PatternName
                                  PatternTemplate = r.PatternName
                                  Context = r.Contexts |> List.tryHead |> Option.defaultValue "unknown"
                                  Score = r.AverageScore
                                  Timestamp = DateTime.UtcNow
                                  RollbackExpansion = None } : PromotionPipeline.TraceArtifact))

                let results = PromotionPipeline.run minOccurrences artifacts
                let json = StructuredOutput.pipelineRunToJson results artifacts.Length
                return Ok json
            with ex ->
                return Error (sprintf "Failed to run promotion pipeline: %s" ex.Message)
        }

    // =========================================================================
    // Tool Registration
    // =========================================================================

    /// Create Tool definitions for MCP registration.
    let createTools (llm: ILlmService option) : Tars.Core.Tool list =
        [ { Name = "list_patterns"
            Description = "List all available reasoning patterns with names, descriptions, and fitness scores"
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = listPatterns }

          { Name = "get_pattern"
            Description = "Get a specific reasoning pattern's full definition including WoT template. Input: pattern name."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = getPattern }

          { Name = "suggest_pattern"
            Description = "Suggest the best reasoning pattern for a given goal. Input: goal description."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute =
                match llm with
                | Some llmService -> suggestPatternWithLlm llmService
                | None -> suggestPattern }

          { Name = "pattern_library_stats"
            Description = "Get pattern library statistics: total patterns, average fitness, diversity score, and top patterns."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = getLibraryStats }

          { Name = "promotion_status"
            Description = "Get promotion pipeline status: recurrence records, level distribution, and pattern counts at each promotion level."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = getPromotionStatus }

          { Name = "promotion_lineage"
            Description = "Get promotion lineage: governance decisions, promotion history, and confidence scores for all evaluated patterns."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = getPromotionLineage }

          { Name = "run_promotion_pipeline"
            Description = "Run the 7-step promotion pipeline (Inspect>Extract>Classify>Propose>Validate>Persist>Govern). Input: minimum occurrences (default 3). Returns structured JSON."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = runPromotionPipeline } ]
