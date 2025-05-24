namespace TarsEngine.FSharp.Core.Consciousness.Decision.Services

open System
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Core
open TarsEngine.FSharp.Core.Consciousness.Decision

/// <summary>
/// Implementation of decision-making and evaluation methods for IDecisionService.
/// </summary>
type DecisionMaking(logger: ILogger<DecisionServiceNew>, decisions: Map<Guid, Decision> ref, intuitiveDecisionLevel: float) =
    // Random number generator for simulating intuitive decisions
    let random = System.Random()
    
    /// <summary>
    /// Makes a decision by selecting the best option based on criteria.
    /// </summary>
    member _.MakeDecision(decisionId: Guid) =
        task {
            try
                logger.LogInformation("Making decision with ID: {DecisionId}", decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId !decisions with
                | Some decision ->
                    // Check if there are options
                    if List.isEmpty decision.Options then
                        logger.LogWarning("No options found for decision with ID {DecisionId}", decisionId)
                        return raise (InvalidOperationException($"No options found for decision with ID {decisionId}"))
                    else
                        // Calculate scores for each option based on criteria
                        let optionScores =
                            decision.Options
                            |> List.map (fun option ->
                                let criteriaScores =
                                    decision.Criteria
                                    |> List.choose (fun criterion ->
                                        match Map.tryFind option.Id criterion.Scores with
                                        | Some score -> Some (score * criterion.Weight)
                                        | None -> None
                                    )
                                
                                let totalScore =
                                    if List.isEmpty criteriaScores then
                                        // If no criteria scores, use option's score or default to 0.5
                                        defaultArg option.Score 0.5
                                    else
                                        // Calculate weighted average
                                        let totalWeight = decision.Criteria |> List.sumBy (fun c -> c.Weight)
                                        if totalWeight > 0.0 then
                                            List.sum criteriaScores / totalWeight
                                        else
                                            defaultArg option.Score 0.5
                                
                                (option.Id, totalScore)
                            )
                        
                        // Find the option with the highest score
                        let bestOption =
                            optionScores
                            |> List.maxBy snd
                            |> fst
                        
                        // Update the decision
                        let updatedDecision = {
                            decision with
                                Status = DecisionStatus.Completed
                                SelectedOption = Some bestOption
                                CompletionTime = Some DateTime.UtcNow
                                Justification = Some "Selected based on criteria scores"
                        }
                        
                        // Update the decision in the dictionary
                        decisions := Map.add decisionId updatedDecision !decisions
                        
                        logger.LogInformation("Made decision with ID: {DecisionId}, selected option with ID: {OptionId}",
                                            decisionId, bestOption)
                        
                        return updatedDecision
                | None ->
                    logger.LogWarning("Decision with ID {Id} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error making decision")
                return raise ex
        }
    
    /// <summary>
    /// Evaluates a decision.
    /// </summary>
    member _.EvaluateDecision(decisionId: Guid) =
        task {
            try
                logger.LogInformation("Evaluating decision with ID: {DecisionId}", decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId !decisions with
                | Some decision ->
                    // Check if a decision has been made
                    match decision.SelectedOption with
                    | Some selectedOptionId ->
                        // Find the selected option
                        let selectedOption = 
                            decision.Options 
                            |> List.tryFind (fun o -> o.Id = selectedOptionId)
                        
                        match selectedOption with
                        | Some option ->
                            // Calculate overall score
                            let criteriaScores =
                                decision.Criteria
                                |> List.choose (fun criterion ->
                                    match Map.tryFind option.Id criterion.Scores with
                                    | Some score -> Some (score * criterion.Weight)
                                    | None -> None
                                )
                            
                            let overallScore =
                                if List.isEmpty criteriaScores then
                                    // If no criteria scores, use option's score or default to 0.7
                                    defaultArg option.Score 0.7
                                else
                                    // Calculate weighted average
                                    let totalWeight = decision.Criteria |> List.sumBy (fun c -> c.Weight)
                                    if totalWeight > 0.0 then
                                        List.sum criteriaScores / totalWeight
                                    else
                                        defaultArg option.Score 0.7
                            
                            // Generate SWOT analysis
                            let strengths = [
                                if not (List.isEmpty option.Pros) then
                                    yield $"Strong pros: {String.Join(", ", option.Pros)}"
                                if overallScore > 0.7 then
                                    yield "High overall score"
                                if decision.Criteria |> List.exists (fun c -> 
                                    Map.tryFind option.Id c.Scores 
                                    |> Option.map (fun s -> s > 0.8) 
                                    |> Option.defaultValue false) then
                                    yield "Excellent performance on some criteria"
                            ]
                            
                            let weaknesses = [
                                if not (List.isEmpty option.Cons) then
                                    yield $"Notable cons: {String.Join(", ", option.Cons)}"
                                if overallScore < 0.5 then
                                    yield "Low overall score"
                                if decision.Criteria |> List.exists (fun c -> 
                                    Map.tryFind option.Id c.Scores 
                                    |> Option.map (fun s -> s < 0.3) 
                                    |> Option.defaultValue false) then
                                    yield "Poor performance on some criteria"
                            ]
                            
                            let opportunities = [
                                "Potential for improvement with more data"
                                "Could be refined with additional criteria"
                                "May lead to better future decisions"
                            ]
                            
                            let threats = [
                                "Changing circumstances may affect validity"
                                "Limited criteria may have missed important factors"
                                "Potential for unforeseen consequences"
                            ]
                            
                            // Create evaluation
                            let evaluation = {
                                Decision = decision
                                Score = overallScore
                                EvaluationTime = DateTime.UtcNow
                                Strengths = strengths
                                Weaknesses = weaknesses
                                Opportunities = opportunities
                                Threats = threats
                                Metadata = Map.empty
                            }
                            
                            logger.LogInformation("Evaluated decision with ID: {DecisionId}, score: {Score:F2}",
                                                decisionId, overallScore)
                            
                            return evaluation
                        | None ->
                            logger.LogWarning("Selected option with ID {OptionId} not found for decision with ID {DecisionId}",
                                             selectedOptionId, decisionId)
                            return raise (InvalidOperationException($"Selected option with ID {selectedOptionId} not found for decision with ID {decisionId}"))
                    | None ->
                        logger.LogWarning("No option selected for decision with ID {DecisionId}", decisionId)
                        return raise (InvalidOperationException($"No option selected for decision with ID {decisionId}"))
                | None ->
                    logger.LogWarning("Decision with ID {Id} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error evaluating decision")
                return raise ex
        }
    
    /// <summary>
    /// Adds an emotion to a decision.
    /// </summary>
    member _.AddEmotionToDecision(decisionId: Guid, emotion: Emotion) =
        task {
            try
                logger.LogInformation("Adding emotion to decision with ID: {DecisionId}", decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId !decisions with
                | Some decision ->
                    // Add the emotion to the decision
                    let updatedDecision = {
                        decision with
                            AssociatedEmotions = emotion :: decision.AssociatedEmotions
                    }
                    
                    // Update the decision in the dictionary
                    decisions := Map.add decisionId updatedDecision !decisions
                    
                    logger.LogInformation("Added emotion to decision with ID: {DecisionId}", decisionId)
                    
                    return updatedDecision
                | None ->
                    logger.LogWarning("Decision with ID {Id} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error adding emotion to decision")
                return raise ex
        }
    
    /// <summary>
    /// Finds decisions based on a query.
    /// </summary>
    member _.FindDecisions(query: DecisionQuery) =
        task {
            try
                logger.LogInformation("Finding decisions based on query")
                
                let startTime = DateTime.UtcNow
                
                // Filter decisions based on query
                let filteredDecisions =
                    !decisions
                    |> Map.values
                    |> Seq.toList
                    |> List.filter (fun decision ->
                        // Filter by name pattern
                        let nameMatches =
                            match query.NamePattern with
                            | Some pattern -> 
                                decision.Name.Contains(pattern, StringComparison.OrdinalIgnoreCase)
                            | None -> true
                        
                        // Filter by types
                        let typeMatches =
                            match query.Types with
                            | Some types -> List.contains decision.Type types
                            | None -> true
                        
                        // Filter by statuses
                        let statusMatches =
                            match query.Statuses with
                            | Some statuses -> List.contains decision.Status statuses
                            | None -> true
                        
                        // Filter by priorities
                        let priorityMatches =
                            match query.Priorities with
                            | Some priorities -> List.contains decision.Priority priorities
                            | None -> true
                        
                        // Filter by creation time
                        let creationTimeMatches =
                            let minTimeMatches =
                                match query.MinimumCreationTime with
                                | Some minTime -> decision.CreationTime >= minTime
                                | None -> true
                            
                            let maxTimeMatches =
                                match query.MaximumCreationTime with
                                | Some maxTime -> decision.CreationTime <= maxTime
                                | None -> true
                            
                            minTimeMatches && maxTimeMatches
                        
                        // Combine all filters
                        nameMatches && typeMatches && statusMatches && priorityMatches && creationTimeMatches
                    )
                
                // Apply max results limit if specified
                let limitedDecisions =
                    match query.MaxResults with
                    | Some maxResults -> List.truncate maxResults filteredDecisions
                    | None -> filteredDecisions
                
                let endTime = DateTime.UtcNow
                let executionTime = endTime - startTime
                
                // Create query result
                let result = {
                    Query = query
                    Decisions = limitedDecisions
                    ExecutionTime = executionTime
                    Metadata = Map.empty
                }
                
                logger.LogInformation("Found {Count} decisions matching query", limitedDecisions.Length)
                
                return result
            with
            | ex ->
                logger.LogError(ex, "Error finding decisions")
                return raise ex
        }
