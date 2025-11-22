// ================================================
// 🔄 TARS TRSX Diff Engine
// ================================================
// Semantic diff computation for metascript versions
// Based on ChatGPT-Leveraging Primes for TARS document

namespace TarsEngine.FSharp.Core

open System
open System.IO
open System.Text
open Microsoft.Extensions.Logging

/// Represents a line change in a diff
[<Struct>]
type LineChange = {
    LineNumber: int
    ChangeType: string // "added", "removed", "modified"
    OldContent: string option
    NewContent: string option
    Context: string list // surrounding lines for context
}

/// Represents a section change in a metascript
[<Struct>]
type SectionChange = {
    SectionName: string
    SectionType: string // "reasoning", "code", "metadata", etc.
    Changes: LineChange list
    Significance: float // 0.0 to 1.0 importance score
}

/// Represents a complete diff between two metascript versions
type TrsxDiff = {
    SourceVersion: string
    TargetVersion: string
    Timestamp: DateTime
    SectionChanges: SectionChange list
    OverallSignificance: float
    ChangeVector: float array // 16D semantic embedding
}

/// Result type for diff operations
type DiffResult<'T> = 
    | Success of 'T
    | Error of string

/// Performance metrics for diff computation
type DiffPerformance = {
    LinesProcessed: int
    SectionsAnalyzed: int
    ElapsedMs: int64
    LinesPerSecond: float
}

module TarsRsxDiff =

    /// Extract sections from a metascript content
    let extractSections (content: string) : (string * string * string list) list =
        let lines = content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        let mutable sections = []
        let mutable currentSection = "default"
        let mutable currentType = "text"
        let mutable currentLines = []
        
        for line in lines do
            let trimmed = line.Trim()
            if trimmed.StartsWith("##") then
                // Save previous section
                if currentLines.Length > 0 then
                    sections <- (currentSection, currentType, List.rev currentLines) :: sections
                // Start new section
                currentSection <- trimmed.Substring(2).Trim()
                currentType <- 
                    if currentSection.ToLower().Contains("reasoning") then "reasoning"
                    elif currentSection.ToLower().Contains("code") then "code"
                    elif currentSection.ToLower().Contains("metadata") then "metadata"
                    else "text"
                currentLines <- []
            else
                currentLines <- line :: currentLines
        
        // Add final section
        if currentLines.Length > 0 then
            sections <- (currentSection, currentType, List.rev currentLines) :: sections
        
        List.rev sections

    /// Compute line-level diff between two text blocks
    let computeLineDiff (oldLines: string list) (newLines: string list) : LineChange list =
        let mutable changes = []
        let mutable oldIndex = 0
        let mutable newIndex = 0
        
        while oldIndex < oldLines.Length || newIndex < newLines.Length do
            if oldIndex >= oldLines.Length then
                // Remaining lines are additions
                changes <- {
                    LineNumber = newIndex + 1
                    ChangeType = "added"
                    OldContent = None
                    NewContent = Some newLines.[newIndex]
                    Context = []
                } :: changes
                newIndex <- newIndex + 1
            elif newIndex >= newLines.Length then
                // Remaining lines are deletions
                changes <- {
                    LineNumber = oldIndex + 1
                    ChangeType = "removed"
                    OldContent = Some oldLines.[oldIndex]
                    NewContent = None
                    Context = []
                } :: changes
                oldIndex <- oldIndex + 1
            elif oldLines.[oldIndex] = newLines.[newIndex] then
                // Lines are identical, skip
                oldIndex <- oldIndex + 1
                newIndex <- newIndex + 1
            else
                // Lines are different, mark as modified
                changes <- {
                    LineNumber = oldIndex + 1
                    ChangeType = "modified"
                    OldContent = Some oldLines.[oldIndex]
                    NewContent = Some newLines.[newIndex]
                    Context = []
                } :: changes
                oldIndex <- oldIndex + 1
                newIndex <- newIndex + 1
        
        List.rev changes

    /// Calculate significance score for a section change
    let calculateSectionSignificance (sectionType: string) (changes: LineChange list) : float =
        let baseWeight = 
            match sectionType with
            | "reasoning" -> 0.8 // High importance for reasoning changes
            | "code" -> 0.9 // Very high importance for code changes
            | "metadata" -> 0.3 // Lower importance for metadata
            | _ -> 0.5 // Default importance
        
        let changeCount = float changes.Length
        let maxChanges = 50.0 // Normalize against reasonable maximum
        let changeRatio = min 1.0 (changeCount / maxChanges)
        
        baseWeight * changeRatio

    /// Generate 16D semantic embedding vector for changes
    let generateChangeVector (sectionChanges: SectionChange list) : float array =
        let vector = Array.zeroCreate 16
        
        for i, sectionChange in sectionChanges |> List.indexed do
            let index = i % 16
            let weight = sectionChange.Significance
            let changeCount = float sectionChange.Changes.Length
            
            // Distribute change information across vector dimensions
            vector.[index] <- vector.[index] + weight * changeCount
            
            // Add type-specific weighting
            let typeWeight = 
                match sectionChange.SectionType with
                | "reasoning" -> 2.0
                | "code" -> 3.0
                | "metadata" -> 1.0
                | _ -> 1.5
            
            let secondIndex = (index + 8) % 16
            vector.[secondIndex] <- vector.[secondIndex] + typeWeight
        
        // Normalize vector
        let magnitude = vector |> Array.map (fun x -> x * x) |> Array.sum |> sqrt
        if magnitude > 0.0 then
            vector |> Array.map (fun x -> x / magnitude)
        else
            vector

    /// Compute comprehensive diff between two metascript versions
    let computeTrsxDiff (sourceContent: string) (targetContent: string) (sourceVersion: string) (targetVersion: string) (logger: ILogger) : DiffResult<TrsxDiff> =
        try
            let stopwatch = System.Diagnostics.Stopwatch.StartNew()
            
            logger.LogInformation($"🔄 Computing TRSX diff: {sourceVersion} -> {targetVersion}")
            
            // Extract sections from both versions
            let sourceSections = extractSections sourceContent
            let targetSections = extractSections targetContent
            
            logger.LogInformation($"📊 Source sections: {sourceSections.Length}, Target sections: {targetSections.Length}")
            
            // Compute section-level changes
            let mutable sectionChanges = []
            let mutable totalLinesProcessed = 0
            
            // Create a map of target sections for efficient lookup
            let targetSectionMap = 
                targetSections 
                |> List.map (fun (name, sType, lines) -> (name, (sType, lines)))
                |> Map.ofList
            
            // Process each source section
            for (sourceName, sourceType, sourceLines) in sourceSections do
                totalLinesProcessed <- totalLinesProcessed + sourceLines.Length
                
                match targetSectionMap.TryFind(sourceName) with
                | Some (targetType, targetLines) ->
                    // Section exists in both versions, compute diff
                    let lineChanges = computeLineDiff sourceLines targetLines
                    if lineChanges.Length > 0 then
                        let significance = calculateSectionSignificance sourceType lineChanges
                        let sectionChange = {
                            SectionName = sourceName
                            SectionType = sourceType
                            Changes = lineChanges
                            Significance = significance
                        }
                        sectionChanges <- sectionChange :: sectionChanges
                | None ->
                    // Section was removed
                    let lineChanges = sourceLines |> List.mapi (fun i line -> {
                        LineNumber = i + 1
                        ChangeType = "removed"
                        OldContent = Some line
                        NewContent = None
                        Context = []
                    })
                    let significance = calculateSectionSignificance sourceType lineChanges
                    let sectionChange = {
                        SectionName = sourceName
                        SectionType = sourceType
                        Changes = lineChanges
                        Significance = significance
                    }
                    sectionChanges <- sectionChange :: sectionChanges
            
            // Process new sections in target
            for (targetName, targetType, targetLines) in targetSections do
                if not (sourceSections |> List.exists (fun (name, _, _) -> name = targetName)) then
                    totalLinesProcessed <- totalLinesProcessed + targetLines.Length
                    
                    // Section was added
                    let lineChanges = targetLines |> List.mapi (fun i line -> {
                        LineNumber = i + 1
                        ChangeType = "added"
                        OldContent = None
                        NewContent = Some line
                        Context = []
                    })
                    let significance = calculateSectionSignificance targetType lineChanges
                    let sectionChange = {
                        SectionName = targetName
                        SectionType = targetType
                        Changes = lineChanges
                        Significance = significance
                    }
                    sectionChanges <- sectionChange :: sectionChanges
            
            let finalSectionChanges = List.rev sectionChanges
            
            // Calculate overall significance
            let overallSignificance = 
                if finalSectionChanges.Length > 0 then
                    finalSectionChanges 
                    |> List.map (fun sc -> sc.Significance)
                    |> List.average
                else 0.0
            
            // Generate semantic embedding vector
            let changeVector = generateChangeVector finalSectionChanges
            
            stopwatch.Stop()
            let elapsedMs = stopwatch.ElapsedMilliseconds
            let linesPerSec = if elapsedMs > 0L then (float totalLinesProcessed * 1000.0) / (float elapsedMs) else 0.0
            
            logger.LogInformation($"✅ Diff computed: {finalSectionChanges.Length} section changes")
            logger.LogInformation($"📈 Performance: {linesPerSec:F0} lines/second, significance: {overallSignificance:F3}")
            
            let diff = {
                SourceVersion = sourceVersion
                TargetVersion = targetVersion
                Timestamp = DateTime.UtcNow
                SectionChanges = finalSectionChanges
                OverallSignificance = overallSignificance
                ChangeVector = changeVector
            }
            
            Success diff
            
        with
        | ex ->
            logger.LogError($"❌ TRSX diff computation failed: {ex.Message}")
            Error ex.Message

    /// Format diff for display
    let formatDiff (diff: TrsxDiff) : string =
        let sb = StringBuilder()
        sb.AppendLine($"TRSX Diff: {diff.SourceVersion} -> {diff.TargetVersion}") |> ignore
        let timestampStr = diff.Timestamp.ToString("yyyy-MM-dd HH:mm:ss")
        sb.AppendLine($"Timestamp: {timestampStr}") |> ignore
        sb.AppendLine($"Overall Significance: {diff.OverallSignificance:F3}") |> ignore
        sb.AppendLine($"Section Changes: {diff.SectionChanges.Length}") |> ignore
        sb.AppendLine() |> ignore
        
        for sectionChange in diff.SectionChanges do
            sb.AppendLine($"## Section: {sectionChange.SectionName} ({sectionChange.SectionType})") |> ignore
            sb.AppendLine($"   Significance: {sectionChange.Significance:F3}") |> ignore
            sb.AppendLine($"   Changes: {sectionChange.Changes.Length}") |> ignore
            
            for change in sectionChange.Changes |> List.take (min 3 sectionChange.Changes.Length) do
                match change.ChangeType with
                | "added" -> sb.AppendLine($"   + {change.NewContent.Value}") |> ignore
                | "removed" -> sb.AppendLine($"   - {change.OldContent.Value}") |> ignore
                | "modified" -> 
                    sb.AppendLine($"   - {change.OldContent.Value}") |> ignore
                    sb.AppendLine($"   + {change.NewContent.Value}") |> ignore
                | _ -> ()
            
            if sectionChange.Changes.Length > 3 then
                let remainingChanges = sectionChange.Changes.Length - 3
                sb.AppendLine($"   ... and {remainingChanges} more changes") |> ignore
            
            sb.AppendLine() |> ignore
        
        sb.ToString()

    /// Test TRSX diff computation
    let testTrsxDiff (logger: ILogger) : bool =
        try
            logger.LogInformation("🧪 Testing TRSX diff computation")
            
            let sourceContent = """## Reasoning
This is the original reasoning section.
It has multiple lines.

## Code
let x = 1
let y = 2

## Metadata
version: 1.0
author: test"""

            let targetContent = """## Reasoning
This is the modified reasoning section.
It has multiple lines and new content.
Added a new line here.

## Code
let x = 1
let y = 3
let z = x + y

## Metadata
version: 1.1
author: test
date: 2025-06-17"""

            match computeTrsxDiff sourceContent targetContent "v1.0" "v1.1" logger with
            | Success diff ->
                logger.LogInformation($"✅ Diff computed successfully")
                logger.LogInformation($"   Sections changed: {diff.SectionChanges.Length}")
                logger.LogInformation($"   Overall significance: {diff.OverallSignificance:F3}")
                logger.LogInformation($"   Vector dimensions: {diff.ChangeVector.Length}")
                
                let formatted = formatDiff diff
                let previewLength = min 200 formatted.Length
                let preview = formatted.Substring(0, previewLength)
                logger.LogInformation($"📄 Formatted diff preview:\n{preview}...")
                
                true
            | Error err ->
                logger.LogError($"❌ Diff computation failed: {err}")
                false
                
        with
        | ex ->
            logger.LogError($"❌ TRSX diff test failed: {ex.Message}")
            false
