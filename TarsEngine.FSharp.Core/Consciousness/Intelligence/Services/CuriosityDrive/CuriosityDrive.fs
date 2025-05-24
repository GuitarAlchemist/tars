namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence
open TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.CuriosityDrive

/// <summary>
/// Implementation of the curiosity drive capabilities.
/// </summary>
type CuriosityDrive(logger: ILogger<CuriosityDrive>) =
    inherit CuriosityDriveBase(logger)
    
    let random = System.Random()
    let mutable lastQuestionTime = DateTime.MinValue
    let mutable lastExplorationTime = DateTime.MinValue
    
    /// <summary>
    /// Generates a curiosity question.
    /// </summary>
    /// <returns>The generated question.</returns>
    member this.GenerateCuriosityQuestionAsync() =
        task {
            if not this.IsInitialized || not this.IsActive then
                return None
            
            // Only generate questions periodically
            if (DateTime.UtcNow - lastQuestionTime).TotalSeconds < 30 then
                return None
            
            try
                logger.LogDebug("Generating curiosity question")
                
                // Choose a question generation method based on current levels
                let method = QuestionGeneration.chooseQuestionMethod 
                              this.NoveltySeekingLevel 
                              this.QuestionGenerationLevel 
                              random
                
                // Generate question based on method
                let question = QuestionGeneration.generateQuestionByMethod 
                                method 
                                this.Explorations 
                                this.NoveltySeekingLevel 
                                this.QuestionGenerationLevel 
                                random
                
                // Add to questions list
                this.AddQuestion(question)
                
                lastQuestionTime <- DateTime.UtcNow
                
                logger.LogInformation("Generated curiosity question: {Question} (Method: {Method}, Importance: {Importance:F2})",
                                     question.Question, question.Method, question.Importance)
                
                return Some question
            with
            | ex ->
                logger.LogError(ex, "Error generating curiosity question")
                return None
        }
    
    /// <summary>
    /// Generates a question by a specific method.
    /// </summary>
    /// <param name="method">The question generation method.</param>
    /// <returns>The generated question.</returns>
    member this.GenerateQuestionByMethodAsync(method: QuestionGenerationMethod) =
        task {
            if not this.IsInitialized || not this.IsActive then
                logger.LogWarning("Cannot generate question: curiosity drive not initialized or active")
                return None
            
            try
                logger.LogInformation("Generating question using method: {Method}", method)
                
                // Generate question based on method
                let question = QuestionGeneration.generateQuestionByMethod 
                                method 
                                this.Explorations 
                                this.NoveltySeekingLevel 
                                this.QuestionGenerationLevel 
                                random
                
                // Add to questions list
                this.AddQuestion(question)
                
                lastQuestionTime <- DateTime.UtcNow
                
                logger.LogInformation("Generated question: {Question} (Method: {Method}, Importance: {Importance:F2})",
                                     question.Question, question.Method, question.Importance)
                
                return Some question
            with
            | ex ->
                logger.LogError(ex, "Error generating question by method")
                return None
        }
    
    /// <summary>
    /// Explores a curiosity topic.
    /// </summary>
    /// <param name="topic">The topic.</param>
    /// <returns>The exploration.</returns>
    member this.ExploreCuriosityTopicAsync(topic: string) =
        task {
            if not this.IsInitialized || not this.IsActive then
                logger.LogWarning("Cannot explore topic: curiosity drive not initialized or active")
                return None
            
            try
                logger.LogInformation("Exploring curiosity topic: {Topic}", topic)
                
                // Choose an exploration strategy based on current levels
                let strategy = ExplorationMethods.chooseExplorationStrategy 
                                this.NoveltySeekingLevel 
                                this.ExplorationLevel 
                                random
                
                // Explore the topic
                let exploration = ExplorationMethods.exploreTopic 
                                   topic 
                                   strategy 
                                   this.ExplorationLevel 
                                   random
                
                // Add to explorations list
                this.AddExploration(exploration)
                
                lastExplorationTime <- DateTime.UtcNow
                
                // Check if this exploration is related to any questions
                let relatedQuestions = 
                    this.Questions
                    |> List.filter (fun q -> 
                        q.ExplorationId.IsNone && 
                        (q.Question.Contains(topic, StringComparison.OrdinalIgnoreCase) ||
                         q.Domain.Contains(topic, StringComparison.OrdinalIgnoreCase)))
                
                // Update related questions with this exploration
                relatedQuestions
                |> List.iter (fun question ->
                    // Create updated question
                    let updatedQuestion = { 
                        question with 
                            ExplorationId = Some exploration.Id 
                    }
                    
                    // Replace the question in the list using reflection
                    let baseType = this.GetType().BaseType
                    let field = baseType.GetField("questions", System.Reflection.BindingFlags.NonPublic ||| System.Reflection.BindingFlags.Instance)
                    let currentQuestions = field.GetValue(this) :?> CuriosityQuestion list
                    let updatedQuestions = 
                        currentQuestions 
                        |> List.map (fun q -> if q.Id = question.Id then updatedQuestion else q)
                    field.SetValue(this, updatedQuestions))
                
                logger.LogInformation("Explored topic: {Topic} (Strategy: {Strategy}, Satisfaction: {Satisfaction:F2})",
                                     topic, strategy, exploration.Satisfaction)
                
                return Some exploration
            with
            | ex ->
                logger.LogError(ex, "Error exploring curiosity topic")
                return None
        }
    
    /// <summary>
    /// Answers a curiosity question.
    /// </summary>
    /// <param name="questionId">The question ID.</param>
    /// <param name="answer">The answer.</param>
    /// <param name="satisfaction">The satisfaction (0.0 to 1.0).</param>
    /// <returns>The updated question.</returns>
    member this.AnswerQuestionAsync(questionId: string, answer: string, satisfaction: float) =
        task {
            try
                logger.LogInformation("Answering question with ID: {QuestionId}", questionId)
                
                // Find the question
                let questionOption = 
                    this.Questions 
                    |> List.tryFind (fun q -> q.Id = questionId)
                
                match questionOption with
                | Some question ->
                    // Update the question
                    let updatedQuestion = {
                        question with
                            Answer = Some answer
                            AnswerTimestamp = Some DateTime.UtcNow
                            AnswerSatisfaction = satisfaction
                    }
                    
                    // Replace the question in the list using reflection
                    let baseType = this.GetType().BaseType
                    let field = baseType.GetField("questions", System.Reflection.BindingFlags.NonPublic ||| System.Reflection.BindingFlags.Instance)
                    let currentQuestions = field.GetValue(this) :?> CuriosityQuestion list
                    let updatedQuestions = 
                        currentQuestions 
                        |> List.map (fun q -> if q.Id = questionId then updatedQuestion else q)
                    field.SetValue(this, updatedQuestions)
                    
                    logger.LogInformation("Answered question with ID: {QuestionId}", questionId)
                    
                    // If satisfaction is low, create an information gap
                    if satisfaction < 0.5 then
                        let gap = {
                            Id = Guid.NewGuid().ToString()
                            Domain = question.Domain
                            Description = sprintf "Incomplete understanding of: %s" question.Question
                            GapSize = 1.0 - satisfaction
                            Importance = question.Importance
                            CreationTimestamp = DateTime.UtcNow
                            LastExploredTimestamp = None
                            ExplorationCount = 0
                            RelatedQuestionIds = [questionId]
                            RelatedExplorationIds = []
                        }
                        
                        this.AddInformationGap(gap)
                        
                        logger.LogInformation("Created information gap for question with ID: {QuestionId}", questionId)
                    
                    return Some updatedQuestion
                | None ->
                    logger.LogWarning("Question with ID {QuestionId} not found", questionId)
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error answering question")
                return None
        }
    
    /// <summary>
    /// Adds an information gap.
    /// </summary>
    /// <param name="domain">The domain.</param>
    /// <param name="description">The description.</param>
    /// <param name="gapSize">The gap size (0.0 to 1.0).</param>
    /// <param name="importance">The importance (0.0 to 1.0).</param>
    /// <returns>The added information gap.</returns>
    member this.AddInformationGapAsync(domain: string, description: string, gapSize: float, importance: float) =
        task {
            try
                logger.LogInformation("Adding information gap in domain: {Domain}", domain)
                
                // Create a new information gap
                let gap = {
                    Id = Guid.NewGuid().ToString()
                    Domain = domain
                    Description = description
                    GapSize = Math.Max(0.0, Math.Min(1.0, gapSize)) // Ensure between 0 and 1
                    Importance = Math.Max(0.0, Math.Min(1.0, importance)) // Ensure between 0 and 1
                    CreationTimestamp = DateTime.UtcNow
                    LastExploredTimestamp = None
                    ExplorationCount = 0
                    RelatedQuestionIds = []
                    RelatedExplorationIds = []
                }
                
                // Add the gap
                this.AddInformationGap(gap)
                
                logger.LogInformation("Added information gap with ID: {Id} in domain: {Domain}", gap.Id, domain)
                
                return gap
            with
            | ex ->
                logger.LogError(ex, "Error adding information gap")
                return raise ex
        }
    
    interface ICuriosityDrive with
        member this.CuriosityLevel = this.CuriosityLevel
        member this.NoveltySeekingLevel = this.NoveltySeekingLevel
        member this.QuestionGenerationLevel = this.QuestionGenerationLevel
        member this.ExplorationLevel = this.ExplorationLevel
        member this.Questions = this.Questions
        member this.Explorations = this.Explorations
        member this.InformationGaps = this.InformationGaps
        
        member this.InitializeAsync() = this.InitializeAsync()
        member this.ActivateAsync() = this.ActivateAsync()
        member this.DeactivateAsync() = this.DeactivateAsync()
        member this.UpdateAsync() = this.UpdateAsync()
        
        member this.GenerateCuriosityQuestionAsync() = this.GenerateCuriosityQuestionAsync()
        member this.GenerateQuestionByMethodAsync(method) = this.GenerateQuestionByMethodAsync(method)
        member this.ExploreCuriosityTopicAsync(topic) = this.ExploreCuriosityTopicAsync(topic)
        member this.AnswerQuestionAsync(questionId, answer, satisfaction) = this.AnswerQuestionAsync(questionId, answer, satisfaction)
        
        member this.GetRecentQuestions(count) = this.GetRecentQuestions(count)
        member this.GetQuestionsByMethod(method, count) = this.GetQuestionsByMethod(method, count)
        member this.GetQuestionsByDomain(domain, count) = this.GetQuestionsByDomain(domain, count)
        member this.GetMostImportantQuestions(count) = this.GetMostImportantQuestions(count)
        
        member this.GetRecentExplorations(count) = this.GetRecentExplorations(count)
        member this.GetExplorationsByStrategy(strategy, count) = this.GetExplorationsByStrategy(strategy, count)
        member this.GetExplorationsByTopic(topic, count) = this.GetExplorationsByTopic(topic, count)
        member this.GetMostSatisfyingExplorations(count) = this.GetMostSatisfyingExplorations(count)
        
        member this.AddInformationGapAsync(domain, description, gapSize, importance) = this.AddInformationGapAsync(domain, description, gapSize, importance)
