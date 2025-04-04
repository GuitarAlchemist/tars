namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Collections.Generic
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Represents a pattern that can be applied during code improvement
/// </summary>
type ImprovementPattern =
    { Id: string
      Name: string
      Description: string
      Pattern: string
      Replacement: string
      Context: string
      Score: float
      SuccessCount: int
      FailureCount: int
      UsageCount: int
      CreatedAt: DateTime
      LastUsed: DateTime option
      Tags: string list
      Metadata: Dictionary<string, string> }

/// <summary>
/// Represents the result of applying a pattern
/// </summary>
type PatternApplicationResult =
    { PatternId: string
      Success: bool
      Context: string
      BeforeCode: string
      AfterCode: string
      Timestamp: DateTime
      Metrics: Dictionary<string, float> }

/// <summary>
/// Represents a retroaction event
/// </summary>
type RetroactionEvent =
    { Id: string
      PatternId: string
      EventType: string // "Feedback", "TestResult", "RuntimeMetric", "UserEvaluation"
      Value: float // Normalized score between -1.0 and 1.0
      Context: string
      Timestamp: DateTime
      Metadata: Dictionary<string, string> }

/// <summary>
/// Represents the retroaction loop state
/// </summary>
type RetroactionState =
    { Patterns: ImprovementPattern list
      ApplicationResults: PatternApplicationResult list
      Events: RetroactionEvent list
      LastUpdated: DateTime }

/// <summary>
/// Provides functionality for the retroaction loop
/// </summary>
module RetroactionLoop =
    let private retroactionFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "retroaction.json")
    
    /// <summary>
    /// Ensures the data directory exists
    /// </summary>
    let private ensureDataDirectory() =
        let dataDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data")
        if not (Directory.Exists(dataDir)) then
            Directory.CreateDirectory(dataDir) |> ignore
    
    /// <summary>
    /// Creates a new retroaction state
    /// </summary>
    let createState() =
        { Patterns = []
          ApplicationResults = []
          Events = []
          LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Loads the retroaction state from disk
    /// </summary>
    let loadState() =
        async {
            ensureDataDirectory()
            
            if File.Exists(retroactionFilePath) then
                try
                    let! json = File.ReadAllTextAsync(retroactionFilePath) |> Async.AwaitTask
                    let options = JsonSerializerOptions()
                    options.PropertyNameCaseInsensitive <- true
                    return JsonSerializer.Deserialize<RetroactionState>(json, options)
                with ex ->
                    // If there's an error loading the state, create a new one
                    return createState()
            else
                return createState()
        }
    
    /// <summary>
    /// Saves the retroaction state to disk
    /// </summary>
    let saveState (state: RetroactionState) =
        async {
            ensureDataDirectory()
            
            try
                let options = JsonSerializerOptions()
                options.WriteIndented <- true
                let json = JsonSerializer.Serialize(state, options)
                do! File.WriteAllTextAsync(retroactionFilePath, json) |> Async.AwaitTask
                return true
            with ex ->
                return false
        }
    
    /// <summary>
    /// Creates a new improvement pattern
    /// </summary>
    let createPattern name description pattern replacement context =
        { Id = Guid.NewGuid().ToString()
          Name = name
          Description = description
          Pattern = pattern
          Replacement = replacement
          Context = context
          Score = 0.5 // Start with a neutral score
          SuccessCount = 0
          FailureCount = 0
          UsageCount = 0
          CreatedAt = DateTime.UtcNow
          LastUsed = None
          Tags = []
          Metadata = new Dictionary<string, string>() }
    
    /// <summary>
    /// Adds a pattern to the retroaction state
    /// </summary>
    let addPattern (state: RetroactionState) (pattern: ImprovementPattern) =
        { state with
            Patterns = state.Patterns @ [pattern]
            LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Records the result of applying a pattern
    /// </summary>
    let recordPatternApplication (state: RetroactionState) (result: PatternApplicationResult) =
        // Update the pattern's usage statistics
        let updatedPatterns =
            state.Patterns
            |> List.map (fun p ->
                if p.Id = result.PatternId then
                    let updatedPattern =
                        { p with
                            UsageCount = p.UsageCount + 1
                            LastUsed = Some DateTime.UtcNow
                            SuccessCount = if result.Success then p.SuccessCount + 1 else p.SuccessCount
                            FailureCount = if not result.Success then p.FailureCount + 1 else p.FailureCount }
                    updatedPattern
                else
                    p)
        
        // Add the result to the state
        { state with
            Patterns = updatedPatterns
            ApplicationResults = state.ApplicationResults @ [result]
            LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Records a retroaction event
    /// </summary>
    let recordEvent (state: RetroactionState) (event: RetroactionEvent) =
        // Update the pattern's score based on the event
        let updatedPatterns =
            state.Patterns
            |> List.map (fun p ->
                if p.Id = event.PatternId then
                    // Adjust the score based on the event value
                    // Use a learning rate to control how quickly the score changes
                    let learningRate = 0.1
                    let newScore = p.Score + (event.Value * learningRate)
                    // Clamp the score between 0.0 and 1.0
                    let clampedScore = Math.Max(0.0, Math.Min(1.0, newScore))
                    { p with Score = clampedScore }
                else
                    p)
        
        // Add the event to the state
        { state with
            Patterns = updatedPatterns
            Events = state.Events @ [event]
            LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Creates a retroaction event
    /// </summary>
    let createEvent patternId eventType value context =
        { Id = Guid.NewGuid().ToString()
          PatternId = patternId
          EventType = eventType
          Value = value
          Context = context
          Timestamp = DateTime.UtcNow
          Metadata = new Dictionary<string, string>() }
    
    /// <summary>
    /// Gets the top patterns for a given context
    /// </summary>
    let getTopPatterns (state: RetroactionState) (context: string) (count: int) =
        state.Patterns
        |> List.filter (fun p -> p.Context = context)
        |> List.sortByDescending (fun p -> p.Score)
        |> List.truncate count
    
    /// <summary>
    /// Gets patterns that match a given code snippet
    /// </summary>
    let getMatchingPatterns (state: RetroactionState) (context: string) (code: string) =
        state.Patterns
        |> List.filter (fun p -> 
            p.Context = context && 
            System.Text.RegularExpressions.Regex.IsMatch(code, p.Pattern))
    
    /// <summary>
    /// Applies a pattern to a code snippet
    /// </summary>
    let applyPattern (pattern: ImprovementPattern) (code: string) =
        try
            let result = System.Text.RegularExpressions.Regex.Replace(code, pattern.Pattern, pattern.Replacement)
            (true, result)
        with ex ->
            (false, code)
    
    /// <summary>
    /// Applies all matching patterns to a code snippet
    /// </summary>
    let applyPatterns (state: RetroactionState) (context: string) (code: string) =
        // Get patterns that match the code and have a score above a threshold
        let matchingPatterns =
            state.Patterns
            |> List.filter (fun p -> 
                p.Context = context && 
                p.Score >= 0.6 && // Only apply patterns with a good score
                System.Text.RegularExpressions.Regex.IsMatch(code, p.Pattern))
            |> List.sortByDescending (fun p -> p.Score)
        
        // Apply each pattern in sequence
        let mutable currentCode = code
        let mutable results = []
        
        for pattern in matchingPatterns do
            let (success, newCode) = applyPattern pattern currentCode
            
            // Create a result record
            let result =
                { PatternId = pattern.Id
                  Success = success
                  Context = context
                  BeforeCode = currentCode
                  AfterCode = newCode
                  Timestamp = DateTime.UtcNow
                  Metrics = new Dictionary<string, float>() }
            
            results <- results @ [result]
            
            // Update the current code if the pattern was applied successfully
            if success && currentCode <> newCode then
                currentCode <- newCode
        
        (currentCode, results)
    
    /// <summary>
    /// Analyzes pattern performance and adjusts scores
    /// </summary>
    let analyzePatternPerformance (state: RetroactionState) =
        // Group application results by pattern
        let resultsByPattern =
            state.ApplicationResults
            |> List.groupBy (fun r -> r.PatternId)
            |> Map.ofList
        
        // Update pattern scores based on success rate
        let updatedPatterns =
            state.Patterns
            |> List.map (fun p ->
                match Map.tryFind p.Id resultsByPattern with
                | Some results ->
                    let totalCount = results.Length
                    if totalCount > 0 then
                        let successCount = results |> List.filter (fun r -> r.Success) |> List.length
                        let successRate = float successCount / float totalCount
                        // Adjust score based on success rate
                        let newScore = (p.Score * 0.7) + (successRate * 0.3)
                        { p with Score = newScore }
                    else
                        p
                | None -> p)
        
        { state with
            Patterns = updatedPatterns
            LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Generates new patterns based on successful improvements
    /// </summary>
    let generateNewPatterns (state: RetroactionState) (logger: ILogger) =
        // Find successful pattern applications with significant improvements
        let significantImprovements =
            state.ApplicationResults
            |> List.filter (fun r -> 
                r.Success && 
                r.BeforeCode <> r.AfterCode &&
                r.BeforeCode.Length > 10) // Avoid trivial changes
        
        // Group by context to generate context-specific patterns
        let improvementsByContext =
            significantImprovements
            |> List.groupBy (fun r -> r.Context)
        
        let mutable newPatterns = []
        
        // For each context, try to identify common patterns
        for (context, improvements) in improvementsByContext do
            // This is a simplified approach - in a real system, you would use more sophisticated
            // pattern recognition techniques, possibly involving machine learning
            
            // For now, we'll just look for common string replacements
            let commonReplacements =
                improvements
                |> List.map (fun r -> (r.BeforeCode, r.AfterCode))
                |> List.distinctBy (fun (before, after) -> (before, after))
                |> List.truncate 5 // Limit to 5 patterns per context
            
            for (before, after) in commonReplacements do
                try
                    // Create a simple pattern based on the before/after code
                    // This is very simplified - real pattern generation would be more sophisticated
                    let escapedBefore = System.Text.RegularExpressions.Regex.Escape(before)
                    let pattern = createPattern
                                    (sprintf "Auto-generated pattern for %s" context)
                                    (sprintf "Pattern generated from successful improvement in %s" context)
                                    escapedBefore
                                    after
                                    context
                    
                    newPatterns <- newPatterns @ [pattern]
                    logger.LogInformation(sprintf "Generated new pattern for context %s" context)
                with ex ->
                    logger.LogError(ex, sprintf "Error generating pattern for context %s" context)
        
        // Add the new patterns to the state
        let updatedState =
            newPatterns
            |> List.fold addPattern state
        
        updatedState
    
    /// <summary>
    /// Runs the retroaction loop
    /// </summary>
    let runRetroactionLoop (logger: ILogger) =
        async {
            logger.LogInformation("Running retroaction loop")
            
            // Load the current state
            let! state = loadState()
            
            // Analyze pattern performance
            let updatedState = analyzePatternPerformance state
            
            // Generate new patterns
            let stateWithNewPatterns = generateNewPatterns updatedState logger
            
            // Save the updated state
            let! saveResult = saveState stateWithNewPatterns
            
            if saveResult then
                logger.LogInformation("Retroaction loop completed successfully")
            else
                logger.LogError("Error saving retroaction state")
            
            return stateWithNewPatterns
        }
    
    /// <summary>
    /// Gets statistics about the retroaction loop
    /// </summary>
    let getStatistics (state: RetroactionState) =
        let totalPatterns = state.Patterns.Length
        let activePatterns = state.Patterns |> List.filter (fun p -> p.Score >= 0.6) |> List.length
        let totalApplications = state.ApplicationResults.Length
        let successfulApplications = state.ApplicationResults |> List.filter (fun r -> r.Success) |> List.length
        let successRate = 
            if totalApplications > 0 then
                float successfulApplications / float totalApplications
            else
                0.0
        
        let patternsByContext =
            state.Patterns
            |> List.groupBy (fun p -> p.Context)
            |> List.map (fun (context, patterns) -> (context, patterns.Length))
            |> Map.ofList
        
        let eventsByType =
            state.Events
            |> List.groupBy (fun e -> e.EventType)
            |> List.map (fun (eventType, events) -> (eventType, events.Length))
            |> Map.ofList
        
        let averagePatternScore =
            if totalPatterns > 0 then
                state.Patterns |> List.averageBy (fun p -> p.Score)
            else
                0.0
        
        {| TotalPatterns = totalPatterns
           ActivePatterns = activePatterns
           TotalApplications = totalApplications
           SuccessfulApplications = successfulApplications
           SuccessRate = successRate
           PatternsByContext = patternsByContext
           EventsByType = eventsByType
           AveragePatternScore = averagePatternScore
           LastUpdated = state.LastUpdated |}
