namespace TarsEngine.FSharp.Main.Intelligence.Measurement

open System
open System.Collections.Generic
open Microsoft.Extensions.Logging

/// <summary>
/// Analyzer for code modifications.
/// </summary>
type ModificationAnalyzer(logger: ILogger<ModificationAnalyzer>) =
    /// <summary>
    /// Analyzes code modifications over a period of time.
    /// </summary>
    /// <param name="startTime">The start time of the analysis period.</param>
    /// <param name="endTime">The end time of the analysis period.</param>
    /// <param name="modifications">The code modifications to analyze.</param>
    /// <returns>A modification analysis.</returns>
    member this.AnalyzeModifications(startTime: DateTime, endTime: DateTime, 
                                    modifications: CodeModification list) : ModificationAnalysis =
        logger.LogInformation("Analyzing {ModificationCount} modifications from {StartTime} to {EndTime}", 
                             modifications.Length, startTime, endTime)
        
        if modifications.IsEmpty then
            logger.LogWarning("No modifications to analyze.")
            {
                StartTime = startTime
                EndTime = endTime
                TotalModifications = 0
                TotalLinesAdded = 0
                TotalLinesRemoved = 0
                TotalLinesModified = 0
                AverageComplexityChange = 0.0
                AverageReadabilityChange = 0.0
                AveragePerformanceImpact = 0.0
                ModificationsByType = Map.empty
                ModificationsByFile = Map.empty
                ModificationTrend = ModificationTrend.Stable
            }
        else
            // Filter modifications within the time period
            let filteredModifications = 
                modifications 
                |> List.filter (fun m -> m.Timestamp >= startTime && m.Timestamp <= endTime)
            
            if filteredModifications.IsEmpty then
                logger.LogWarning("No modifications within the specified time period.")
                {
                    StartTime = startTime
                    EndTime = endTime
                    TotalModifications = 0
                    TotalLinesAdded = 0
                    TotalLinesRemoved = 0
                    TotalLinesModified = 0
                    AverageComplexityChange = 0.0
                    AverageReadabilityChange = 0.0
                    AveragePerformanceImpact = 0.0
                    ModificationsByType = Map.empty
                    ModificationsByFile = Map.empty
                    ModificationTrend = ModificationTrend.Stable
                }
            else
                // Calculate total metrics
                let totalLinesAdded = filteredModifications |> List.sumBy (fun m -> m.LinesAdded)
                let totalLinesRemoved = filteredModifications |> List.sumBy (fun m -> m.LinesRemoved)
                let totalLinesModified = filteredModifications |> List.sumBy (fun m -> m.LinesModified)
                
                let avgComplexityChange = 
                    filteredModifications 
                    |> List.averageBy (fun m -> m.ComplexityChange)
                
                let avgReadabilityChange = 
                    filteredModifications 
                    |> List.averageBy (fun m -> m.ReadabilityChange)
                
                let avgPerformanceImpact = 
                    filteredModifications 
                    |> List.averageBy (fun m -> m.PerformanceImpact)
                
                // Group modifications by type
                let modsByType = 
                    filteredModifications 
                    |> List.groupBy (fun m -> m.ImprovementType)
                    |> List.map (fun (t, mods) -> (t, mods.Length))
                    |> Map.ofList
                
                // Group modifications by file
                let modsByFile = 
                    filteredModifications 
                    |> List.groupBy (fun m -> m.FilePath)
                    |> List.map (fun (f, mods) -> (f, mods.Length))
                    |> Map.ofList
                
                // Determine modification trend
                let trend = 
                    // Sort modifications by timestamp
                    let sortedMods = filteredModifications |> List.sortBy (fun m -> m.Timestamp)
                    
                    // Split into two halves
                    let halfIndex = sortedMods.Length / 2
                    let firstHalf = sortedMods.[0..halfIndex-1]
                    let secondHalf = sortedMods.[halfIndex..]
                    
                    // Calculate average complexity change for each half
                    let firstHalfAvgComplexity = 
                        if firstHalf.IsEmpty then 0.0
                        else firstHalf |> List.averageBy (fun m -> m.ComplexityChange)
                    
                    let secondHalfAvgComplexity = 
                        if secondHalf.IsEmpty then 0.0
                        else secondHalf |> List.averageBy (fun m -> m.ComplexityChange)
                    
                    // Calculate trend based on change in complexity
                    let complexityDiff = secondHalfAvgComplexity - firstHalfAvgComplexity
                    
                    if complexityDiff < -0.1 then ModificationTrend.Decreasing
                    elif complexityDiff < -0.05 then ModificationTrend.SlightlyDecreasing
                    elif complexityDiff > 0.1 then ModificationTrend.Increasing
                    elif complexityDiff > 0.05 then ModificationTrend.SlightlyIncreasing
                    else ModificationTrend.Stable
                
                // Return the analysis
                {
                    StartTime = startTime
                    EndTime = endTime
                    TotalModifications = filteredModifications.Length
                    TotalLinesAdded = totalLinesAdded
                    TotalLinesRemoved = totalLinesRemoved
                    TotalLinesModified = totalLinesModified
                    AverageComplexityChange = avgComplexityChange
                    AverageReadabilityChange = avgReadabilityChange
                    AveragePerformanceImpact = avgPerformanceImpact
                    ModificationsByType = modsByType
                    ModificationsByFile = modsByFile
                    ModificationTrend = trend
                }
    
    /// <summary>
    /// Calculates the complexity of a code snippet.
    /// </summary>
    /// <param name="code">The code snippet.</param>
    /// <returns>A complexity score.</returns>
    member this.CalculateComplexity(code: string) : float =
        if String.IsNullOrEmpty(code) then 0.0
        else
            // This is a simplified complexity calculation
            // In a real implementation, we would use a more sophisticated algorithm
            
            // Count lines
            let lines = code.Split([|"\r\n"; "\n"|], StringSplitOptions.None)
            let lineCount = float lines.Length
            
            // Count control structures (if, for, while, etc.)
            let controlStructureCount = 
                lines 
                |> Array.filter (fun line -> 
                    line.Contains("if ") || 
                    line.Contains("for ") || 
                    line.Contains("while ") || 
                    line.Contains("switch ") || 
                    line.Contains("case ") ||
                    line.Contains("match "))
                |> Array.length
                |> float
            
            // Count nesting levels
            let nestingLevels = 
                lines
                |> Array.fold (fun (maxLevel, currentLevel) line ->
                    let openBraces = line.Count(fun c -> c = '{')
                    let closeBraces = line.Count(fun c -> c = '}')
                    let newLevel = currentLevel + openBraces - closeBraces
                    (Math.Max(maxLevel, newLevel), newLevel)) (0, 0)
                |> fst
                |> float
            
            // Calculate complexity score
            let complexityScore = lineCount * 0.1 + controlStructureCount * 0.5 + nestingLevels * 1.0
            
            complexityScore
    
    /// <summary>
    /// Calculates the readability of a code snippet.
    /// </summary>
    /// <param name="code">The code snippet.</param>
    /// <returns>A readability score (higher is better).</returns>
    member this.CalculateReadability(code: string) : float =
        if String.IsNullOrEmpty(code) then 0.0
        else
            // This is a simplified readability calculation
            // In a real implementation, we would use a more sophisticated algorithm
            
            // Count lines
            let lines = code.Split([|"\r\n"; "\n"|], StringSplitOptions.None)
            let lineCount = float lines.Length
            
            // Calculate average line length
            let avgLineLength = 
                lines
                |> Array.averageBy (fun line -> float line.Length)
            
            // Count comments
            let commentCount = 
                lines 
                |> Array.filter (fun line -> 
                    line.Contains("//") || 
                    line.Contains("/*") || 
                    line.Contains("*/") ||
                    line.Contains("'''") ||
                    line.Contains("\"\"\""))
                |> Array.length
                |> float
            
            // Calculate readability score
            let readabilityScore = 
                10.0 - (avgLineLength * 0.1) + (commentCount / lineCount) * 5.0
            
            // Clamp between 0 and 10
            Math.Max(0.0, Math.Min(10.0, readabilityScore))
    
    /// <summary>
    /// Calculates the performance impact of a code modification.
    /// </summary>
    /// <param name="originalCode">The original code.</param>
    /// <param name="modifiedCode">The modified code.</param>
    /// <returns>A performance impact score (positive means improvement).</returns>
    member this.CalculatePerformanceImpact(originalCode: string, modifiedCode: string) : float =
        if String.IsNullOrEmpty(originalCode) || String.IsNullOrEmpty(modifiedCode) then 0.0
        else
            // This is a simplified performance impact calculation
            // In a real implementation, we would use a more sophisticated algorithm
            
            // Calculate complexity of both versions
            let originalComplexity = this.CalculateComplexity(originalCode)
            let modifiedComplexity = this.CalculateComplexity(modifiedCode)
            
            // Calculate performance impact based on complexity change
            // Assuming lower complexity generally means better performance
            let complexityImpact = originalComplexity - modifiedComplexity
            
            // Check for performance-related keywords
            let performanceKeywords = 
                [|"performance"; "optimize"; "efficient"; "speed"; "fast"; "cache"; "memory"|]
            
            let keywordImpact = 
                if modifiedCode.ToLowerInvariant().Split([|' '; '\t'; '\r'; '\n'|], StringSplitOptions.RemoveEmptyEntries)
                   |> Array.exists (fun word -> Array.contains word performanceKeywords)
                then 1.0
                else 0.0
            
            // Calculate overall performance impact
            let performanceImpact = complexityImpact * 0.5 + keywordImpact
            
            // Clamp between -5.0 and 5.0
            Math.Max(-5.0, Math.Min(5.0, performanceImpact))
    
    /// <summary>
    /// Identifies the most significant modifications based on various metrics.
    /// </summary>
    /// <param name="modifications">The code modifications to analyze.</param>
    /// <param name="count">The number of significant modifications to return.</param>
    /// <returns>A list of the most significant modifications.</returns>
    member this.IdentifySignificantModifications(modifications: CodeModification list, count: int) 
        : CodeModification list =
        
        logger.LogInformation("Identifying {Count} significant modifications from {ModificationCount} total", 
                             count, modifications.Length)
        
        if modifications.IsEmpty then
            logger.LogWarning("No modifications to analyze for significance.")
            []
        else
            // Calculate significance score for each modification
            let modsWithScore = 
                modifications
                |> List.map (fun m ->
                    let complexityScore = Math.Abs(m.ComplexityChange) * 2.0
                    let readabilityScore = m.ReadabilityChange * 1.5
                    let performanceScore = m.PerformanceImpact * 3.0
                    let sizeScore = float (m.LinesAdded + m.LinesRemoved + m.LinesModified) * 0.1
                    
                    let totalScore = complexityScore + readabilityScore + performanceScore + sizeScore
                    (m, totalScore))
            
            // Sort by significance score (descending) and take the top 'count'
            modsWithScore
            |> List.sortByDescending snd
            |> List.take (Math.Min(count, modsWithScore.Length))
            |> List.map fst
