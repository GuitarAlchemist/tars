namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.CuriosityDrive

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence
open TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

/// <summary>
/// Base implementation of the curiosity drive capabilities.
/// </summary>
type CuriosityDriveBase(logger: ILogger<CuriosityDriveBase>) =
    // State variables
    let mutable isInitialized = false
    let mutable isActive = false
    let mutable curiosityLevel = 0.5 // Starting with moderate curiosity
    let mutable noveltySeekingLevel = 0.6 // Starting with moderate novelty seeking
    let mutable questionGenerationLevel = 0.5 // Starting with moderate question generation
    let mutable explorationLevel = 0.4 // Starting with moderate exploration
    let mutable questions = List.empty<CuriosityQuestion>
    let mutable explorations = List.empty<CuriosityExploration>
    let mutable informationGaps = Map.empty<string, InformationGap>
    let random = System.Random()
    let mutable lastQuestionTime = DateTime.MinValue
    let mutable lastExplorationTime = DateTime.MinValue
    
    /// <summary>
    /// Gets the curiosity level (0.0 to 1.0).
    /// </summary>
    member _.CuriosityLevel = curiosityLevel
    
    /// <summary>
    /// Gets the novelty seeking level (0.0 to 1.0).
    /// </summary>
    member _.NoveltySeekingLevel = noveltySeekingLevel
    
    /// <summary>
    /// Gets the question generation level (0.0 to 1.0).
    /// </summary>
    member _.QuestionGenerationLevel = questionGenerationLevel
    
    /// <summary>
    /// Gets the exploration level (0.0 to 1.0).
    /// </summary>
    member _.ExplorationLevel = explorationLevel
    
    /// <summary>
    /// Gets the questions.
    /// </summary>
    member _.Questions = questions
    
    /// <summary>
    /// Gets the explorations.
    /// </summary>
    member _.Explorations = explorations
    
    /// <summary>
    /// Gets the information gaps.
    /// </summary>
    member _.InformationGaps = informationGaps
    
    /// <summary>
    /// Initializes the curiosity drive.
    /// </summary>
    /// <returns>True if initialization was successful.</returns>
    member _.InitializeAsync() =
        task {
            try
                logger.LogInformation("Initializing curiosity drive")
                
                // Initialize state
                isInitialized <- true
                
                logger.LogInformation("Curiosity drive initialized successfully")
                return true
            with
            | ex ->
                logger.LogError(ex, "Error initializing curiosity drive")
                return false
        }
    
    /// <summary>
    /// Activates the curiosity drive.
    /// </summary>
    /// <returns>True if activation was successful.</returns>
    member _.ActivateAsync() =
        task {
            if not isInitialized then
                logger.LogWarning("Cannot activate curiosity drive: not initialized")
                return false
            
            if isActive then
                logger.LogInformation("Curiosity drive is already active")
                return true
            
            try
                logger.LogInformation("Activating curiosity drive")
                
                // Activate state
                isActive <- true
                
                logger.LogInformation("Curiosity drive activated successfully")
                return true
            with
            | ex ->
                logger.LogError(ex, "Error activating curiosity drive")
                return false
        }
    
    /// <summary>
    /// Deactivates the curiosity drive.
    /// </summary>
    /// <returns>True if deactivation was successful.</returns>
    member _.DeactivateAsync() =
        task {
            if not isActive then
                logger.LogInformation("Curiosity drive is already inactive")
                return true
            
            try
                logger.LogInformation("Deactivating curiosity drive")
                
                // Deactivate state
                isActive <- false
                
                logger.LogInformation("Curiosity drive deactivated successfully")
                return true
            with
            | ex ->
                logger.LogError(ex, "Error deactivating curiosity drive")
                return false
        }
    
    /// <summary>
    /// Updates the curiosity drive.
    /// </summary>
    /// <returns>True if update was successful.</returns>
    member _.UpdateAsync() =
        task {
            if not isInitialized then
                logger.LogWarning("Cannot update curiosity drive: not initialized")
                return false
            
            try
                // Gradually increase curiosity levels over time (very slowly)
                if curiosityLevel < 0.95 then
                    curiosityLevel <- curiosityLevel + 0.0001 * random.NextDouble()
                    curiosityLevel <- Math.Min(curiosityLevel, 1.0)
                
                if noveltySeekingLevel < 0.95 then
                    noveltySeekingLevel <- noveltySeekingLevel + 0.0001 * random.NextDouble()
                    noveltySeekingLevel <- Math.Min(noveltySeekingLevel, 1.0)
                
                if questionGenerationLevel < 0.95 then
                    questionGenerationLevel <- questionGenerationLevel + 0.0001 * random.NextDouble()
                    questionGenerationLevel <- Math.Min(questionGenerationLevel, 1.0)
                
                if explorationLevel < 0.95 then
                    explorationLevel <- explorationLevel + 0.0001 * random.NextDouble()
                    explorationLevel <- Math.Min(explorationLevel, 1.0)
                
                return true
            with
            | ex ->
                logger.LogError(ex, "Error updating curiosity drive")
                return false
        }
    
    /// <summary>
    /// Gets recent questions.
    /// </summary>
    /// <param name="count">The number of questions to get.</param>
    /// <returns>The recent questions.</returns>
    member _.GetRecentQuestions(count: int) =
        questions
        |> List.sortByDescending (fun question -> question.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets questions by method.
    /// </summary>
    /// <param name="method">The question generation method.</param>
    /// <param name="count">The number of questions to get.</param>
    /// <returns>The questions generated by the specified method.</returns>
    member _.GetQuestionsByMethod(method: QuestionGenerationMethod, count: int) =
        questions
        |> List.filter (fun question -> question.Method = method)
        |> List.sortByDescending (fun question -> question.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets questions by domain.
    /// </summary>
    /// <param name="domain">The domain.</param>
    /// <param name="count">The number of questions to get.</param>
    /// <returns>The questions in the domain.</returns>
    member _.GetQuestionsByDomain(domain: string, count: int) =
        questions
        |> List.filter (fun question -> question.Domain.Contains(domain, StringComparison.OrdinalIgnoreCase))
        |> List.sortByDescending (fun question -> question.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets the most important questions.
    /// </summary>
    /// <param name="count">The number of questions to get.</param>
    /// <returns>The most important questions.</returns>
    member _.GetMostImportantQuestions(count: int) =
        questions
        |> List.sortByDescending (fun question -> question.Importance)
        |> List.truncate count
    
    /// <summary>
    /// Gets recent explorations.
    /// </summary>
    /// <param name="count">The number of explorations to get.</param>
    /// <returns>The recent explorations.</returns>
    member _.GetRecentExplorations(count: int) =
        explorations
        |> List.sortByDescending (fun exploration -> exploration.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets explorations by strategy.
    /// </summary>
    /// <param name="strategy">The exploration strategy.</param>
    /// <param name="count">The number of explorations to get.</param>
    /// <returns>The explorations using the specified strategy.</returns>
    member _.GetExplorationsByStrategy(strategy: ExplorationStrategy, count: int) =
        explorations
        |> List.filter (fun exploration -> exploration.Strategy = strategy)
        |> List.sortByDescending (fun exploration -> exploration.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets explorations by topic.
    /// </summary>
    /// <param name="topic">The topic.</param>
    /// <param name="count">The number of explorations to get.</param>
    /// <returns>The explorations on the topic.</returns>
    member _.GetExplorationsByTopic(topic: string, count: int) =
        explorations
        |> List.filter (fun exploration -> exploration.Topic.Contains(topic, StringComparison.OrdinalIgnoreCase))
        |> List.sortByDescending (fun exploration -> exploration.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets the most satisfying explorations.
    /// </summary>
    /// <param name="count">The number of explorations to get.</param>
    /// <returns>The most satisfying explorations.</returns>
    member _.GetMostSatisfyingExplorations(count: int) =
        explorations
        |> List.sortByDescending (fun exploration -> exploration.Satisfaction)
        |> List.truncate count
    
    /// <summary>
    /// Adds a question.
    /// </summary>
    /// <param name="question">The question to add.</param>
    member _.AddQuestion(question: CuriosityQuestion) =
        questions <- question :: questions
        lastQuestionTime <- DateTime.UtcNow
    
    /// <summary>
    /// Adds an exploration.
    /// </summary>
    /// <param name="exploration">The exploration to add.</param>
    member _.AddExploration(exploration: CuriosityExploration) =
        explorations <- exploration :: explorations
        lastExplorationTime <- DateTime.UtcNow
    
    /// <summary>
    /// Adds an information gap.
    /// </summary>
    /// <param name="gap">The information gap to add.</param>
    member _.AddInformationGap(gap: InformationGap) =
        informationGaps <- Map.add gap.Id gap informationGaps
    
    /// <summary>
    /// Gets whether the curiosity drive is initialized.
    /// </summary>
    member _.IsInitialized = isInitialized
    
    /// <summary>
    /// Gets whether the curiosity drive is active.
    /// </summary>
    member _.IsActive = isActive
