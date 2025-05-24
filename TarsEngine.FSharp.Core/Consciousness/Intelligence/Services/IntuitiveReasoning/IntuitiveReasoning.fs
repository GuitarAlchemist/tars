namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence
open TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.IntuitiveReasoning

/// <summary>
/// Implementation of the intuitive reasoning capabilities.
/// </summary>
type IntuitiveReasoning(logger: ILogger<IntuitiveReasoning>) =
    inherit IntuitiveReasoningBase(logger)
    
    let random = System.Random()
    let mutable lastIntuitionTime = DateTime.MinValue
    
    /// <summary>
    /// Generates an intuition.
    /// </summary>
    /// <returns>The generated intuition.</returns>
    member this.GenerateIntuitionAsync() =
        task {
            if not this.IsInitialized || not this.IsActive then
                return None
            
            // Only generate intuitions periodically
            if (DateTime.UtcNow - lastIntuitionTime).TotalSeconds < 30 then
                return None
            
            try
                logger.LogDebug("Generating intuition")
                
                // Choose an intuition type based on current levels
                let intuitionType = IntuitionGeneration.chooseIntuitionType 
                                     this.PatternRecognitionLevel 
                                     this.HeuristicReasoningLevel 
                                     this.GutFeelingLevel 
                                     random
                
                // Generate intuition based on type
                let intuition = IntuitionGeneration.generateIntuitionByType 
                                 intuitionType 
                                 this.PatternRecognitionLevel 
                                 this.HeuristicReasoningLevel 
                                 this.GutFeelingLevel 
                                 random
                
                // Add to intuitions list
                this.AddIntuition(intuition)
                
                lastIntuitionTime <- DateTime.UtcNow
                
                logger.LogInformation("Generated intuition: {Description} (Confidence: {Confidence:F2}, Type: {Type})",
                                     intuition.Description, intuition.Confidence, intuition.Type)
                
                return Some intuition
            with
            | ex ->
                logger.LogError(ex, "Error generating intuition")
                return None
        }
    
    /// <summary>
    /// Makes an intuitive decision.
    /// </summary>
    /// <param name="options">The options.</param>
    /// <param name="intuitionType">The intuition type.</param>
    /// <returns>The selected option and the intuition.</returns>
    member this.MakeIntuitiveDecisionAsync(options: string list, intuitionType: IntuitionType option) =
        task {
            if not this.IsInitialized || not this.IsActive then
                logger.LogWarning("Cannot make intuitive decision: intuitive reasoning not initialized or active")
                // Return a random option as fallback
                let randomOption = options.[random.Next(options.Length)]
                let fallbackIntuition = {
                    Id = Guid.NewGuid().ToString()
                    Description = "Fallback intuition (system not initialized)"
                    Type = defaultArg intuitionType IntuitionType.GutFeeling
                    Confidence = 0.1
                    Timestamp = DateTime.UtcNow
                    Context = Map.empty
                    Tags = ["fallback"; "uninitalized"]
                    Source = "System Fallback"
                    VerificationStatus = VerificationStatus.Unverified
                    VerificationTimestamp = None
                    VerificationNotes = ""
                    Accuracy = None
                    Impact = 0.1
                    Explanation = "System not initialized, random selection made"
                    Decision = "Fallback Selection"
                    SelectedOption = randomOption
                    Options = options
                }
                return (randomOption, fallbackIntuition)
            
            try
                logger.LogInformation("Making intuitive decision among {Count} options", options.Length)
                
                // Make the decision
                let (selectedOption, intuition) = IntuitiveDecisionMaking.makeIntuitiveDecision 
                                                   options 
                                                   intuitionType 
                                                   this.PatternRecognitionLevel 
                                                   this.HeuristicReasoningLevel 
                                                   this.GutFeelingLevel 
                                                   random
                
                // Add to intuitions list
                this.AddIntuition(intuition)
                
                logger.LogInformation("Made intuitive decision: selected '{Option}' with confidence {Confidence:F2}",
                                     selectedOption, intuition.Confidence)
                
                return (selectedOption, intuition)
            with
            | ex ->
                logger.LogError(ex, "Error making intuitive decision")
                // Return a random option as fallback
                let randomOption = options.[random.Next(options.Length)]
                let fallbackIntuition = {
                    Id = Guid.NewGuid().ToString()
                    Description = "Error fallback intuition"
                    Type = defaultArg intuitionType IntuitionType.GutFeeling
                    Confidence = 0.1
                    Timestamp = DateTime.UtcNow
                    Context = Map.empty
                    Tags = ["fallback"; "error"]
                    Source = "Error Fallback"
                    VerificationStatus = VerificationStatus.Unverified
                    VerificationTimestamp = None
                    VerificationNotes = ""
                    Accuracy = None
                    Impact = 0.1
                    Explanation = "Error occurred, random selection made"
                    Decision = "Error Fallback Selection"
                    SelectedOption = randomOption
                    Options = options
                }
                return (randomOption, fallbackIntuition)
        }
    
    /// <summary>
    /// Evaluates options intuitively.
    /// </summary>
    /// <param name="options">The options.</param>
    /// <param name="intuitionType">The intuition type.</param>
    /// <returns>The option scores.</returns>
    member this.EvaluateOptionsIntuitivelyAsync(options: string list, intuitionType: IntuitionType option) =
        task {
            if not this.IsInitialized || not this.IsActive then
                logger.LogWarning("Cannot evaluate options intuitively: intuitive reasoning not initialized or active")
                // Return equal scores as fallback
                return options |> List.map (fun o -> (o, 0.5)) |> Map.ofList
            
            try
                logger.LogInformation("Evaluating {Count} options intuitively", options.Length)
                
                // Evaluate the options
                let scores = IntuitiveDecisionMaking.evaluateOptionsIntuitively 
                              options 
                              intuitionType 
                              this.PatternRecognitionLevel 
                              this.HeuristicReasoningLevel 
                              this.GutFeelingLevel 
                              random
                
                logger.LogInformation("Evaluated options intuitively")
                
                return scores
            with
            | ex ->
                logger.LogError(ex, "Error evaluating options intuitively")
                // Return equal scores as fallback
                return options |> List.map (fun o -> (o, 0.5)) |> Map.ofList
        }
    
    /// <summary>
    /// Verifies an intuition.
    /// </summary>
    /// <param name="intuitionId">The intuition ID.</param>
    /// <param name="isCorrect">Whether the intuition is correct.</param>
    /// <param name="accuracy">The accuracy.</param>
    /// <param name="notes">The verification notes.</param>
    /// <returns>The updated intuition.</returns>
    member this.VerifyIntuitionAsync(intuitionId: string, isCorrect: bool, accuracy: float, notes: string) =
        task {
            try
                logger.LogInformation("Verifying intuition with ID: {IntuitionId}", intuitionId)
                
                // Find the intuition
                let intuitionOption = 
                    this.Intuitions 
                    |> List.tryFind (fun i -> i.Id = intuitionId)
                
                match intuitionOption with
                | Some intuition ->
                    // Update the intuition
                    let updatedIntuition = {
                        intuition with
                            VerificationStatus = if isCorrect then VerificationStatus.Verified else VerificationStatus.Falsified
                            VerificationTimestamp = Some DateTime.UtcNow
                            VerificationNotes = notes
                            Accuracy = Some accuracy
                    }
                    
                    // Replace the intuition in the list
                    let updatedIntuitions = 
                        this.Intuitions 
                        |> List.map (fun i -> if i.Id = intuitionId then updatedIntuition else i)
                    
                    // Update the intuitions list using reflection (since it's a mutable field in the base class)
                    let baseType = this.GetType().BaseType
                    let field = baseType.GetField("intuitions", System.Reflection.BindingFlags.NonPublic ||| System.Reflection.BindingFlags.Instance)
                    field.SetValue(this, updatedIntuitions)
                    
                    logger.LogInformation("Verified intuition with ID: {IntuitionId}", intuitionId)
                    
                    return Some updatedIntuition
                | None ->
                    logger.LogWarning("Intuition with ID {IntuitionId} not found", intuitionId)
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error verifying intuition")
                return None
        }
    
    interface IIntuitiveReasoning with
        member this.IntuitionLevel = this.IntuitionLevel
        member this.PatternRecognitionLevel = this.PatternRecognitionLevel
        member this.HeuristicReasoningLevel = this.HeuristicReasoningLevel
        member this.GutFeelingLevel = this.GutFeelingLevel
        member this.Intuitions = this.Intuitions
        
        member this.InitializeAsync() = this.InitializeAsync()
        member this.ActivateAsync() = this.ActivateAsync()
        member this.DeactivateAsync() = this.DeactivateAsync()
        member this.UpdateAsync() = this.UpdateAsync()
        
        member this.GenerateIntuitionAsync() = this.GenerateIntuitionAsync()
        member this.MakeIntuitiveDecisionAsync(options, intuitionType) = this.MakeIntuitiveDecisionAsync(options, intuitionType)
        member this.EvaluateOptionsIntuitivelyAsync(options, intuitionType) = this.EvaluateOptionsIntuitivelyAsync(options, intuitionType)
        
        member this.GetRecentIntuitions(count) = this.GetRecentIntuitions(count)
        member this.GetIntuitionsByType(intuitionType, count) = this.GetIntuitionsByType(intuitionType, count)
        member this.GetIntuitionsByTag(tag, count) = this.GetIntuitionsByTag(tag, count)
        member this.GetMostConfidentIntuitions(count) = this.GetMostConfidentIntuitions(count)
        member this.GetMostAccurateIntuitions(count) = this.GetMostAccurateIntuitions(count)
        
        member this.VerifyIntuitionAsync(intuitionId, isCorrect, accuracy, notes) = this.VerifyIntuitionAsync(intuitionId, isCorrect, accuracy, notes)
