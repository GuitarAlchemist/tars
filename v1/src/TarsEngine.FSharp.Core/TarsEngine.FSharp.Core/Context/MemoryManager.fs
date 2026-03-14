namespace TarsEngine.FSharp.Core.Context

open System
open System.IO
open System.Threading.Tasks
open System.Text.Json
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Context.Types

/// Configuration for tiered memory system
type MemoryConfig = {
    EphemeralPath: string
    WorkingSetPath: string
    LongTermPath: string
    EphemeralMaxSpans: int
    WorkingSetMaxSpans: int
    SalienceDecayRate: float
    PromotionThreshold: float
    ConsolidationFrequency: string
}

/// File-based tiered memory implementation
type TieredMemoryManager(config: MemoryConfig, logger: ILogger<TieredMemoryManager>) =
    
    let tokenEstimator (text: string) = Math.Max(1, text.Length / 4)
    
    /// Serialize context span to JSON
    let serializeSpan (span: ContextSpan) =
        JsonSerializer.Serialize(span, JsonSerializerOptions(WriteIndented = false))
    
    /// Deserialize context span from JSON
    let deserializeSpan (json: string) =
        try
            JsonSerializer.Deserialize<ContextSpan>(json) |> Some
        with
        | ex ->
            logger.LogError(ex, "Failed to deserialize context span: {Json}", json)
            None
    
    /// Load spans from file
    let loadSpansFromFile (filePath: string) =
        task {
            try
                if File.Exists(filePath) then
                    let! lines = File.ReadAllLinesAsync(filePath)
                    let spans = 
                        lines
                        |> Array.choose deserializeSpan
                        |> Array.toList
                    
                    logger.LogDebug("Loaded {SpanCount} spans from {FilePath}", spans.Length, filePath)
                    return spans
                else
                    logger.LogDebug("Memory file not found: {FilePath}", filePath)
                    return []
            with
            | ex ->
                logger.LogError(ex, "Failed to load spans from {FilePath}", filePath)
                return []
        }
    
    /// Save spans to file
    let saveSpansToFile (filePath: string) (spans: ContextSpan list) =
        task {
            try
                // Ensure directory exists
                let directory = Path.GetDirectoryName(filePath)
                if not (String.IsNullOrEmpty(directory)) then
                    Directory.CreateDirectory(directory) |> ignore
                
                let lines = spans |> List.map serializeSpan |> List.toArray
                do! File.WriteAllLinesAsync(filePath, lines)
                
                logger.LogDebug("Saved {SpanCount} spans to {FilePath}", spans.Length, filePath)
            with
            | ex ->
                logger.LogError(ex, "Failed to save spans to {FilePath}", filePath)
        }
    
    /// Append spans to file
    let appendSpansToFile (filePath: string) (spans: ContextSpan list) =
        task {
            try
                // Ensure directory exists
                let directory = Path.GetDirectoryName(filePath)
                if not (String.IsNullOrEmpty(directory)) then
                    Directory.CreateDirectory(directory) |> ignore
                
                let lines = spans |> List.map serializeSpan |> List.toArray
                do! File.AppendAllLinesAsync(filePath, lines)
                
                logger.LogDebug("Appended {SpanCount} spans to {FilePath}", spans.Length, filePath)
            with
            | ex ->
                logger.LogError(ex, "Failed to append spans to {FilePath}", filePath)
        }
    
    /// Apply salience decay to spans
    let applySalienceDecay (spans: ContextSpan list) =
        let now = DateTime.UtcNow
        spans
        |> List.map (fun span ->
            let ageHours = (now - span.Timestamp).TotalHours
            let decayFactor = Math.Pow(config.SalienceDecayRate, ageHours / 24.0) // Daily decay
            let newSalience = span.Salience * decayFactor
            { span with Salience = Math.Max(0.01, newSalience) })
    
    /// Filter spans by salience threshold
    let filterBySalience (threshold: float) (spans: ContextSpan list) =
        spans |> List.filter (fun span -> span.Salience >= threshold)
    
    /// Limit spans by count, keeping highest salience
    let limitSpanCount (maxCount: int) (spans: ContextSpan list) =
        spans
        |> List.sortByDescending (fun span -> span.Salience)
        |> List.truncate maxCount
    
    /// Detect conflicts between spans
    let detectConflicts (spans: ContextSpan list) =
        // Simple conflict detection based on contradictory content
        let conflicts = ResizeArray<string>()
        
        for i in 0 .. spans.Length - 1 do
            for j in i + 1 .. spans.Length - 1 do
                let span1 = spans.[i]
                let span2 = spans.[j]
                
                // Check for contradictory statements (simplified)
                if span1.Text.Contains("not") && span2.Text.Contains(span1.Text.Replace("not", "").Trim()) ||
                   span2.Text.Contains("not") && span1.Text.Contains(span2.Text.Replace("not", "").Trim()) then
                    conflicts.Add($"Conflict between {span1.Id} and {span2.Id}: contradictory statements")
        
        conflicts |> List.ofSeq
    
    /// Summarize spans for consolidation
    let summarizeSpans (spans: ContextSpan list) =
        let groupedBySource = spans |> List.groupBy (fun s -> s.Source)
        
        groupedBySource
        |> List.map (fun (source, sourceSpans) ->
            let topSpan = sourceSpans |> List.maxBy (fun s -> s.Salience)
            let combinedText = 
                sourceSpans 
                |> List.map (fun s -> s.Text) 
                |> List.distinct
                |> String.concat " | "
            
            let summary = 
                if combinedText.Length > 500 then
                    combinedText.Substring(0, 497) + "..."
                else
                    combinedText
            
            {
                Id = $"summary-{source}-{DateTime.UtcNow:yyyyMMdd-HHmmss}"
                Text = summary
                Tokens = tokenEstimator summary
                Salience = sourceSpans |> List.map (fun s -> s.Salience) |> List.average
                Source = $"consolidated-{source}"
                Timestamp = DateTime.UtcNow
                Intent = topSpan.Intent
                Metadata = Map.ofList [("original_count", sourceSpans.Length.ToString())]
            })
    
    interface IContextMemory with
        
        member _.LoadEphemeralAsync() =
            task {
                let! spans = loadSpansFromFile config.EphemeralPath
                let decayedSpans = applySalienceDecay spans
                return limitSpanCount config.EphemeralMaxSpans decayedSpans
            }
        
        member _.LoadWorkingSetAsync() =
            task {
                let! spans = loadSpansFromFile config.WorkingSetPath
                let decayedSpans = applySalienceDecay spans
                return limitSpanCount config.WorkingSetMaxSpans decayedSpans
            }
        
        member _.LoadLongTermAsync() =
            loadSpansFromFile config.LongTermPath
        
        member _.StoreEphemeralAsync(spans) =
            task {
                logger.LogInformation("Storing {SpanCount} spans in ephemeral memory", spans.Length)
                
                // Load existing spans
                let! existingSpans = loadSpansFromFile config.EphemeralPath
                
                // Combine and deduplicate
                let allSpans = (existingSpans @ spans) |> List.distinctBy (fun s -> s.Id)
                
                // Apply limits
                let limitedSpans = limitSpanCount config.EphemeralMaxSpans allSpans
                
                // Save back
                do! saveSpansToFile config.EphemeralPath limitedSpans
            }
        
        member _.PromoteToWorkingSetAsync(spans) =
            task {
                logger.LogInformation("Promoting {SpanCount} spans to working set", spans.Length)
                
                // Filter by promotion threshold
                let promotableSpans = filterBySalience config.PromotionThreshold spans
                
                if not promotableSpans.IsEmpty then
                    // Load existing working set
                    let! existingSpans = loadSpansFromFile config.WorkingSetPath
                    
                    // Combine and deduplicate
                    let allSpans = (existingSpans @ promotableSpans) |> List.distinctBy (fun s -> s.Id)
                    
                    // Apply decay and limits
                    let decayedSpans = applySalienceDecay allSpans
                    let limitedSpans = limitSpanCount config.WorkingSetMaxSpans decayedSpans
                    
                    // Save back
                    do! saveSpansToFile config.WorkingSetPath limitedSpans
                    
                    logger.LogInformation("Promoted {PromotedCount} spans to working set", promotableSpans.Length)
                else
                    logger.LogInformation("No spans met promotion threshold of {Threshold}", config.PromotionThreshold)
            }
        
        member _.ConsolidateAsync(runId) =
            task {
                logger.LogInformation("Starting memory consolidation for run {RunId}", runId)
                
                try
                    // Load working set
                    let! workingSetSpans = loadSpansFromFile config.WorkingSetPath
                    
                    // Apply decay
                    let decayedSpans = applySalienceDecay workingSetSpans
                    
                    // Detect conflicts
                    let conflicts = detectConflicts decayedSpans
                    
                    // Summarize high-salience spans for long-term storage
                    let highSalienceSpans = filterBySalience 0.5 decayedSpans
                    let summarizedSpans = summarizeSpans highSalienceSpans
                    
                    // Append to long-term memory
                    if not summarizedSpans.IsEmpty then
                        do! appendSpansToFile config.LongTermPath summarizedSpans
                    
                    // Keep only medium-salience spans in working set
                    let remainingSpans = filterBySalience 0.3 decayedSpans |> filterBySalience 0.5 >> not
                    do! saveSpansToFile config.WorkingSetPath remainingSpans
                    
                    let result = {
                        RunId = runId
                        SpansProcessed = workingSetSpans.Length
                        SpansPromoted = summarizedSpans.Length
                        SpansArchived = workingSetSpans.Length - remainingSpans.Length
                        ConflictsDetected = conflicts
                        Summary = $"Consolidated {summarizedSpans.Length} spans to long-term, {conflicts.Length} conflicts detected"
                    }
                    
                    logger.LogInformation("Memory consolidation completed: {Summary}", result.Summary)
                    return result
                    
                with
                | ex ->
                    logger.LogError(ex, "Memory consolidation failed for run {RunId}", runId)
                    return {
                        RunId = runId
                        SpansProcessed = 0
                        SpansPromoted = 0
                        SpansArchived = 0
                        ConflictsDetected = [ex.Message]
                        Summary = $"Consolidation failed: {ex.Message}"
                    }
            }
        
        member _.ClearEphemeralAsync() =
            task {
                logger.LogInformation("Clearing ephemeral memory")
                
                try
                    if File.Exists(config.EphemeralPath) then
                        File.Delete(config.EphemeralPath)
                    logger.LogInformation("Ephemeral memory cleared")
                with
                | ex ->
                    logger.LogError(ex, "Failed to clear ephemeral memory")
            }
