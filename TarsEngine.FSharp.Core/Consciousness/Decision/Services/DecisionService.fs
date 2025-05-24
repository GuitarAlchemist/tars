namespace TarsEngine.FSharp.Core.Consciousness.Decision.Services

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Decision

/// <summary>
/// Implementation of IDecisionService.
/// </summary>
type DecisionService(logger: ILogger<DecisionService>) =
    // In-memory storage for decisions and options
    let decisions = System.Collections.Generic.Dictionary<Guid, Decision>()
    let options = System.Collections.Generic.Dictionary<Guid, DecisionOption>()
    
    // Random number generator for simulating intuitive decisions
    let random = System.Random()
    
    // Intuitive decision level (0.0 to 1.0)
    let mutable intuitiveDecisionLevel = 0.5
    
    // Decision history
    let decisionHistory = System.Collections.Generic.List<DecisionRecord>()
    
    /// <summary>
    /// Creates a new decision.
    /// </summary>
    /// <param name="description">The description of the decision.</param>
    /// <param name="options">The options available for the decision.</param>
    /// <param name="domain">The domain of the decision (optional).</param>
    /// <returns>The created decision.</returns>
    member _.CreateDecision(description: string, options: string list, ?domain: string) =
        task {
            try
                logger.LogInformation("Creating decision: {Description}", description)
                
                // Create a new decision
                let id = Guid.NewGuid()
                let decision = {
                    Id = id
                    Description = description
                    Options = options
                    SelectedOption = None
                    Confidence = 0.0
                    Timestamp = DateTime.UtcNow
                    Domain = domain
                    Explanation = None
                    Data = Map.empty
                }
                
                // Add the decision to the dictionary
                decisions.Add(id, decision)
                
                logger.LogInformation("Created decision with ID: {Id}", id)
                
                return decision
            with
            | ex ->
                logger.LogError(ex, "Error creating decision")
                return raise ex
        }
    
    /// <summary>
    /// Gets a decision by ID.
    /// </summary>
    /// <param name="id">The ID of the decision.</param>
    /// <returns>The decision, or None if not found.</returns>
    member _.GetDecision(id: Guid) =
        task {
            try
                logger.LogInformation("Getting decision with ID: {Id}", id)
                
                // Try to get the decision from the dictionary
                let success, decision = decisions.TryGetValue(id)
                
                if success then
                    logger.LogInformation("Found decision with ID: {Id}", id)
                    return Some decision
                else
                    logger.LogWarning("Decision with ID {Id} not found", id)
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error getting decision")
                return None
        }
    
    /// <summary>
    /// Gets all decisions.
    /// </summary>
    /// <returns>The list of all decisions.</returns>
    member _.GetAllDecisions() =
        task {
            try
                logger.LogInformation("Getting all decisions")
                
                // Convert the dictionary values to a list
                let decisionList = decisions.Values |> Seq.toList
                
                logger.LogInformation("Found {Count} decisions", decisionList.Length)
                
                return decisionList
            with
            | ex ->
                logger.LogError(ex, "Error getting all decisions")
                return []
        }
    
    /// <summary>
    /// Updates a decision.
    /// </summary>
    /// <param name="decision">The updated decision.</param>
    /// <returns>The updated decision.</returns>
    member _.UpdateDecision(decision: Decision) =
        task {
            try
                logger.LogInformation("Updating decision with ID: {Id}", decision.Id)
                
                // Check if the decision exists
                if decisions.ContainsKey(decision.Id) then
                    // Update the decision
                    decisions.[decision.Id] <- decision
                    
                    logger.LogInformation("Updated decision with ID: {Id}", decision.Id)
                    
                    return decision
                else
                    logger.LogWarning("Decision with ID {Id} not found", decision.Id)
                    return raise (KeyNotFoundException($"Decision with ID {decision.Id} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error updating decision")
                return raise ex
        }
    
    /// <summary>
    /// Deletes a decision.
    /// </summary>
    /// <param name="id">The ID of the decision to delete.</param>
    /// <returns>True if the decision was deleted, false otherwise.</returns>
    member _.DeleteDecision(id: Guid) =
        task {
            try
                logger.LogInformation("Deleting decision with ID: {Id}", id)
                
                // Try to remove the decision from the dictionary
                let success = decisions.Remove(id)
                
                if success then
                    logger.LogInformation("Deleted decision with ID: {Id}", id)
                    
                    // Also remove any options associated with this decision
                    let optionsToRemove = 
                        options.Values 
                        |> Seq.filter (fun o -> o.DecisionId = id) 
                        |> Seq.map (fun o -> o.Id)
                        |> Seq.toArray
                    
                    for optionId in optionsToRemove do
                        options.Remove(optionId) |> ignore
                    
                    return true
                else
                    logger.LogWarning("Decision with ID {Id} not found", id)
                    return false
            with
            | ex ->
                logger.LogError(ex, "Error deleting decision")
                return false
        }
    
    /// <summary>
    /// Adds an option to a decision.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="description">The description of the option.</param>
    /// <returns>The added option.</returns>
    member _.AddOption(decisionId: Guid, description: string) =
        task {
            try
                logger.LogInformation("Adding option to decision with ID: {Id}", decisionId)
                
                // Check if the decision exists
                if decisions.ContainsKey(decisionId) then
                    // Create a new option
                    let id = Guid.NewGuid()
                    let option = {
                        Id = id
                        DecisionId = decisionId
                        Description = description
                        Score = 0.0
                        Data = Map.empty
                    }
                    
                    // Add the option to the dictionary
                    options.Add(id, option)
                    
                    // Update the decision's options list
                    let decision = decisions.[decisionId]
                    let updatedOptions = decision.Options @ [description]
                    decisions.[decisionId] <- { decision with Options = updatedOptions }
                    
                    logger.LogInformation("Added option with ID: {Id}", id)
                    
                    return option
                else
                    logger.LogWarning("Decision with ID {Id} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error adding option")
                return raise ex
        }
    
    /// <summary>
    /// Removes an option from a decision.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="optionId">The ID of the option to remove.</param>
    /// <returns>True if the option was removed, false otherwise.</returns>
    member _.RemoveOption(decisionId: Guid, optionId: Guid) =
        task {
            try
                logger.LogInformation("Removing option with ID: {OptionId} from decision with ID: {DecisionId}", optionId, decisionId)
                
                // Check if the decision exists
                if decisions.ContainsKey(decisionId) then
                    // Check if the option exists
                    if options.ContainsKey(optionId) then
                        let option = options.[optionId]
                        
                        // Check if the option belongs to the decision
                        if option.DecisionId = decisionId then
                            // Remove the option from the dictionary
                            options.Remove(optionId) |> ignore
                            
                            // Update the decision's options list
                            let decision = decisions.[decisionId]
                            let updatedOptions = decision.Options |> List.filter (fun o -> o <> option.Description)
                            decisions.[decisionId] <- { decision with Options = updatedOptions }
                            
                            logger.LogInformation("Removed option with ID: {OptionId}", optionId)
                            
                            return true
                        else
                            logger.LogWarning("Option with ID {OptionId} does not belong to decision with ID {DecisionId}", optionId, decisionId)
                            return false
                    else
                        logger.LogWarning("Option with ID {OptionId} not found", optionId)
                        return false
                else
                    logger.LogWarning("Decision with ID {DecisionId} not found", decisionId)
                    return false
            with
            | ex ->
                logger.LogError(ex, "Error removing option")
                return false
        }
    
    /// <summary>
    /// Makes a decision.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="selectedOptionId">The ID of the selected option.</param>
    /// <param name="confidence">The confidence of the decision (0.0 to 1.0).</param>
    /// <param name="explanation">The explanation of the decision (optional).</param>
    /// <returns>The updated decision.</returns>
    member _.MakeDecision(decisionId: Guid, selectedOptionId: Guid, confidence: float, ?explanation: string) =
        task {
            try
                logger.LogInformation("Making decision with ID: {Id}", decisionId)
                
                // Check if the decision exists
                if decisions.ContainsKey(decisionId) then
                    // Check if the option exists
                    if options.ContainsKey(selectedOptionId) then
                        let option = options.[selectedOptionId]
                        
                        // Check if the option belongs to the decision
                        if option.DecisionId = decisionId then
                            // Update the decision
                            let decision = decisions.[decisionId]
                            let updatedDecision = { 
                                decision with 
                                    SelectedOption = Some option.Description
                                    Confidence = confidence
                                    Explanation = explanation
                                    Timestamp = DateTime.UtcNow
                            }
                            
                            decisions.[decisionId] <- updatedDecision
                            
                            logger.LogInformation("Made decision with ID: {Id}", decisionId)
                            
                            return updatedDecision
                        else
                            logger.LogWarning("Option with ID {OptionId} does not belong to decision with ID {DecisionId}", selectedOptionId, decisionId)
                            return raise (InvalidOperationException($"Option with ID {selectedOptionId} does not belong to decision with ID {decisionId}"))
                    else
                        logger.LogWarning("Option with ID {OptionId} not found", selectedOptionId)
                        return raise (KeyNotFoundException($"Option with ID {selectedOptionId} not found"))
                else
                    logger.LogWarning("Decision with ID {DecisionId} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error making decision")
                return raise ex
        }
    
    /// <summary>
    /// Makes an intuitive decision.
    /// </summary>
    /// <param name="decision">The decision description.</param>
    /// <param name="options">The options.</param>
    /// <param name="domain">The domain (optional).</param>
    /// <returns>The intuitive decision.</returns>
    member _.MakeIntuitiveDecision(decision: string, options: string list, ?domain: string) =
        task {
            try
                logger.LogInformation("Making intuitive decision: {Decision}", decision)
                
                if options.IsEmpty then
                    return raise (ArgumentException("No options provided for decision"))
                
                // Choose intuition type based on decision characteristics
                let intuitionType = 
                    match domain with
                    | Some d when d.Contains("design", StringComparison.OrdinalIgnoreCase) || d.Contains("architecture", StringComparison.OrdinalIgnoreCase) ->
                        IntuitionType.PatternRecognition
                    | Some d when d.Contains("development", StringComparison.OrdinalIgnoreCase) || d.Contains("coding", StringComparison.OrdinalIgnoreCase) ->
                        IntuitionType.HeuristicReasoning
                    | Some d when d.Contains("user", StringComparison.OrdinalIgnoreCase) || d.Contains("experience", StringComparison.OrdinalIgnoreCase) ->
                        IntuitionType.GutFeeling
                    | _ ->
                        // Choose randomly
                        let rand = random.NextDouble()
                        if rand < 0.4 then IntuitionType.PatternRecognition
                        elif rand < 0.8 then IntuitionType.HeuristicReasoning
                        else IntuitionType.GutFeeling
                
                logger.LogDebug("Chosen intuition type: {IntuitionType}", intuitionType)
                
                // Score each option
                let optionScores = 
                    options 
                    |> List.map (fun option -> 
                        // Calculate a score based on the intuition type
                        let score = 
                            match intuitionType with
                            | IntuitionType.PatternRecognition ->
                                // Simulate pattern recognition by favoring options with certain keywords
                                let baseScore = 0.3 + (random.NextDouble() * 0.4)
                                if option.Contains("pattern", StringComparison.OrdinalIgnoreCase) then baseScore + 0.2
                                elif option.Contains("design", StringComparison.OrdinalIgnoreCase) then baseScore + 0.15
                                else baseScore
                            | IntuitionType.HeuristicReasoning ->
                                // Simulate heuristic reasoning by favoring options with certain keywords
                                let baseScore = 0.4 + (random.NextDouble() * 0.3)
                                if option.Contains("logic", StringComparison.OrdinalIgnoreCase) then baseScore + 0.2
                                elif option.Contains("rule", StringComparison.OrdinalIgnoreCase) then baseScore + 0.15
                                else baseScore
                            | IntuitionType.GutFeeling ->
                                // Simulate gut feeling with more randomness
                                0.3 + (random.NextDouble() * 0.6)
                            | IntuitionType.Custom _ ->
                                // Default score for custom intuition types
                                0.5
                        
                        // Add some randomness
                        let finalScore = score + (0.1 * (random.NextDouble() - 0.5))
                        
                        // Ensure score is within bounds
                        let boundedScore = Math.Max(0.1, Math.Min(0.9, finalScore))
                        
                        (option, boundedScore)
                    )
                
                // Choose the option with the highest score
                let selectedOption, confidence = 
                    optionScores 
                    |> List.maxBy snd
                
                // Apply intuitive decision level to confidence
                let adjustedConfidence = confidence * intuitiveDecisionLevel
                
                // Generate explanation
                let explanation = 
                    match intuitionType with
                    | IntuitionType.PatternRecognition ->
                        $"This decision is based on recognizing patterns in the options. The selected option '{selectedOption}' matched known patterns with a confidence of {adjustedConfidence:F2}."
                    | IntuitionType.HeuristicReasoning ->
                        $"This decision is based on applying heuristic reasoning principles to the options. The selected option '{selectedOption}' aligned with these principles with a confidence of {adjustedConfidence:F2}."
                    | IntuitionType.GutFeeling ->
                        $"This decision is based on a gut feeling about the options. The selected option '{selectedOption}' felt right with a confidence of {adjustedConfidence:F2}."
                    | IntuitionType.Custom name ->
                        $"This decision is based on {name} intuition. The selected option '{selectedOption}' was chosen with a confidence of {adjustedConfidence:F2}."
                
                // Create intuitive decision
                let intuitiveDecision = {
                    Decision = decision
                    SelectedOption = selectedOption
                    Options = options
                    Confidence = adjustedConfidence
                    IntuitionType = intuitionType
                    Timestamp = DateTime.UtcNow
                    Explanation = explanation
                }
                
                // Record decision
                let record = {
                    Decision = decision
                    SelectedOption = selectedOption
                    Options = options
                    Confidence = adjustedConfidence
                    IntuitionType = intuitionType
                    Timestamp = DateTime.UtcNow
                    Explanation = explanation
                }
                
                decisionHistory.Add(record)
                
                logger.LogInformation("Made intuitive decision: {SelectedOption} for {Decision} (Confidence: {Confidence:F2})",
                    intuitiveDecision.SelectedOption, intuitiveDecision.Decision, intuitiveDecision.Confidence)
                
                return intuitiveDecision
            with
            | ex ->
                logger.LogError(ex, "Error making intuitive decision")
                
                // Return basic decision
                return {
                    Decision = decision
                    SelectedOption = options |> List.tryHead |> Option.defaultValue ""
                    Options = options
                    Confidence = 0.3
                    IntuitionType = IntuitionType.GutFeeling
                    Timestamp = DateTime.UtcNow
                    Explanation = "Decision made with low confidence due to an error in the decision-making process"
                }
        }
    
    /// <summary>
    /// Generates an intuition for a situation.
    /// </summary>
    /// <param name="situation">The situation description.</param>
    /// <param name="domain">The domain (optional).</param>
    /// <returns>The generated intuition.</returns>
    member _.GenerateIntuition(situation: string, ?domain: string) =
        task {
            try
                logger.LogInformation("Generating intuition for situation: {Situation}", situation)
                
                // Choose intuition type based on situation characteristics
                let intuitionType = 
                    match domain with
                    | Some d when d.Contains("design", StringComparison.OrdinalIgnoreCase) || d.Contains("architecture", StringComparison.OrdinalIgnoreCase) ->
                        IntuitionType.PatternRecognition
                    | Some d when d.Contains("development", StringComparison.OrdinalIgnoreCase) || d.Contains("coding", StringComparison.OrdinalIgnoreCase) ->
                        IntuitionType.HeuristicReasoning
                    | Some d when d.Contains("user", StringComparison.OrdinalIgnoreCase) || d.Contains("experience", StringComparison.OrdinalIgnoreCase) ->
                        IntuitionType.GutFeeling
                    | _ ->
                        // Choose randomly
                        let rand = random.NextDouble()
                        if rand < 0.4 then IntuitionType.PatternRecognition
                        elif rand < 0.8 then IntuitionType.HeuristicReasoning
                        else IntuitionType.GutFeeling
                
                // Generate intuition based on type
                let description, confidence = 
                    match intuitionType with
                    | IntuitionType.PatternRecognition ->
                        // Simulate pattern recognition
                        let baseConfidence = 0.6 + (random.NextDouble() * 0.3)
                        let description = $"I recognize a pattern in this situation that suggests a {if random.NextDouble() > 0.5 then "positive" else "cautious"} approach would be best."
                        description, baseConfidence
                    | IntuitionType.HeuristicReasoning ->
                        // Simulate heuristic reasoning
                        let baseConfidence = 0.7 + (random.NextDouble() * 0.2)
                        let description = $"Based on heuristic principles, this situation calls for a {if random.NextDouble() > 0.5 then "systematic" else "creative"} solution."
                        description, baseConfidence
                    | IntuitionType.GutFeeling ->
                        // Simulate gut feeling
                        let baseConfidence = 0.5 + (random.NextDouble() * 0.4)
                        let description = $"My gut feeling about this situation is that it {if random.NextDouble() > 0.5 then "presents an opportunity" else "requires caution"}."
                        description, baseConfidence
                    | IntuitionType.Custom name ->
                        // Default for custom intuition types
                        let baseConfidence = 0.5 + (random.NextDouble() * 0.3)
                        let description = $"Using {name} intuition, I sense that this situation {if random.NextDouble() > 0.5 then "has potential" else "needs careful handling"}."
                        description, baseConfidence
                
                // Apply intuitive decision level to confidence
                let adjustedConfidence = confidence * intuitiveDecisionLevel
                
                // Create intuition
                let intuition = {
                    Description = description
                    Confidence = adjustedConfidence
                    IntuitionType = intuitionType
                    Timestamp = DateTime.UtcNow
                    Data = Map.empty
                }
                
                logger.LogInformation("Generated intuition: {Description} (Confidence: {Confidence:F2})",
                    intuition.Description, intuition.Confidence)
                
                return intuition
            with
            | ex ->
                logger.LogError(ex, "Error generating intuition")
                
                // Return basic intuition
                return {
                    Description = "I have a vague feeling about this situation, but can't articulate it clearly."
                    Confidence = 0.3
                    IntuitionType = IntuitionType.GutFeeling
                    Timestamp = DateTime.UtcNow
                    Data = Map.empty
                }
        }
    
    interface IDecisionService with
        member this.CreateDecision(description, options, ?domain) = this.CreateDecision(description, options, ?domain=domain)
        member this.GetDecision(id) = this.GetDecision(id)
        member this.GetAllDecisions() = this.GetAllDecisions()
        member this.UpdateDecision(decision) = this.UpdateDecision(decision)
        member this.DeleteDecision(id) = this.DeleteDecision(id)
        member this.AddOption(decisionId, description) = this.AddOption(decisionId, description)
        member this.RemoveOption(decisionId, optionId) = this.RemoveOption(decisionId, optionId)
        member this.MakeDecision(decisionId, selectedOptionId, confidence, ?explanation) = this.MakeDecision(decisionId, selectedOptionId, confidence, ?explanation=explanation)
        member this.MakeIntuitiveDecision(decision, options, ?domain) = this.MakeIntuitiveDecision(decision, options, ?domain=domain)
        member this.GenerateIntuition(situation, ?domain) = this.GenerateIntuition(situation, ?domain=domain)
