namespace TarsEngine.FSharp.Core.Consciousness.Decision.Services

open System
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Core
open TarsEngine.FSharp.Core.Consciousness.Decision

/// <summary>
/// Implementation of criterion management methods for IDecisionService.
/// </summary>
type CriterionManagement(logger: ILogger<DecisionServiceNew>, decisions: Map<Guid, Decision> ref) =
    /// <summary>
    /// Adds a criterion to a decision.
    /// </summary>
    member _.AddCriterion(decisionId: Guid, name: string, description: string, weight: float) =
        task {
            try
                logger.LogInformation("Adding criterion to decision with ID: {DecisionId}", decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId !decisions with
                | Some decision ->
                    // Create a new criterion
                    let criterion = {
                        Id = Guid.NewGuid()
                        Name = name
                        Description = description
                        Weight = weight
                        Scores = Map.empty
                        Metadata = Map.empty
                    }
                    
                    // Add the criterion to the decision
                    let updatedDecision = { 
                        decision with 
                            Criteria = criterion :: decision.Criteria 
                    }
                    
                    // Update the decision
                    decisions := Map.add decisionId updatedDecision !decisions
                    
                    logger.LogInformation("Added criterion with ID: {Id} to decision with ID: {DecisionId}", 
                                         criterion.Id, decisionId)
                    
                    return updatedDecision
                | None ->
                    logger.LogWarning("Decision with ID {Id} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error adding criterion to decision")
                return raise ex
        }
    
    /// <summary>
    /// Updates a criterion.
    /// </summary>
    member _.UpdateCriterion(decisionId: Guid, criterionId: Guid, ?name: string, ?description: string, ?weight: float) =
        task {
            try
                logger.LogInformation("Updating criterion with ID: {CriterionId} for decision with ID: {DecisionId}", 
                                     criterionId, decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId !decisions with
                | Some decision ->
                    // Find the criterion
                    let criterionIndex = decision.Criteria |> List.tryFindIndex (fun c -> c.Id = criterionId)
                    
                    match criterionIndex with
                    | Some index ->
                        let criterion = decision.Criteria.[index]
                        
                        // Update the criterion
                        let updatedCriterion = {
                            criterion with
                                Name = defaultArg name criterion.Name
                                Description = defaultArg description criterion.Description
                                Weight = defaultArg weight criterion.Weight
                        }
                        
                        // Update the criteria list
                        let updatedCriteria = 
                            decision.Criteria 
                            |> List.mapi (fun i c -> if i = index then updatedCriterion else c)
                        
                        // Update the decision
                        let updatedDecision = { decision with Criteria = updatedCriteria }
                        
                        // Update the decision in the dictionary
                        decisions := Map.add decisionId updatedDecision !decisions
                        
                        logger.LogInformation("Updated criterion with ID: {CriterionId} for decision with ID: {DecisionId}", 
                                            criterionId, decisionId)
                        
                        return updatedDecision
                    | None ->
                        logger.LogWarning("Criterion with ID {CriterionId} not found for decision with ID {DecisionId}", 
                                         criterionId, decisionId)
                        return raise (KeyNotFoundException($"Criterion with ID {criterionId} not found for decision with ID {decisionId}"))
                | None ->
                    logger.LogWarning("Decision with ID {DecisionId} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error updating criterion")
                return raise ex
        }
    
    /// <summary>
    /// Removes a criterion from a decision.
    /// </summary>
    member _.RemoveCriterion(decisionId: Guid, criterionId: Guid) =
        task {
            try
                logger.LogInformation("Removing criterion with ID: {CriterionId} from decision with ID: {DecisionId}", 
                                     criterionId, decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId !decisions with
                | Some decision ->
                    // Check if the criterion exists
                    let criterionExists = decision.Criteria |> List.exists (fun c -> c.Id = criterionId)
                    
                    if criterionExists then
                        // Remove the criterion
                        let updatedCriteria = decision.Criteria |> List.filter (fun c -> c.Id <> criterionId)
                        
                        // Update the decision
                        let updatedDecision = { decision with Criteria = updatedCriteria }
                        
                        // Update the decision in the dictionary
                        decisions := Map.add decisionId updatedDecision !decisions
                        
                        logger.LogInformation("Removed criterion with ID: {CriterionId} from decision with ID: {DecisionId}", 
                                            criterionId, decisionId)
                        
                        return updatedDecision
                    else
                        logger.LogWarning("Criterion with ID {CriterionId} not found for decision with ID {DecisionId}", 
                                         criterionId, decisionId)
                        return raise (KeyNotFoundException($"Criterion with ID {criterionId} not found for decision with ID {decisionId}"))
                | None ->
                    logger.LogWarning("Decision with ID {DecisionId} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error removing criterion")
                return raise ex
        }
    
    /// <summary>
    /// Scores an option for a criterion.
    /// </summary>
    member _.ScoreOption(decisionId: Guid, criterionId: Guid, optionId: Guid, score: float) =
        task {
            try
                logger.LogInformation("Scoring option with ID: {OptionId} for criterion with ID: {CriterionId} in decision with ID: {DecisionId}", 
                                     optionId, criterionId, decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId !decisions with
                | Some decision ->
                    // Find the criterion
                    let criterionIndex = decision.Criteria |> List.tryFindIndex (fun c -> c.Id = criterionId)
                    
                    match criterionIndex with
                    | Some cIndex ->
                        let criterion = decision.Criteria.[cIndex]
                        
                        // Check if the option exists
                        let optionExists = decision.Options |> List.exists (fun o -> o.Id = optionId)
                        
                        if optionExists then
                            // Update the criterion with the score
                            let updatedCriterion = {
                                criterion with
                                    Scores = Map.add optionId score criterion.Scores
                            }
                            
                            // Update the criteria list
                            let updatedCriteria = 
                                decision.Criteria 
                                |> List.mapi (fun i c -> if i = cIndex then updatedCriterion else c)
                            
                            // Update the decision
                            let updatedDecision = { decision with Criteria = updatedCriteria }
                            
                            // Update the decision in the dictionary
                            decisions := Map.add decisionId updatedDecision !decisions
                            
                            logger.LogInformation("Scored option with ID: {OptionId} for criterion with ID: {CriterionId} in decision with ID: {DecisionId}", 
                                                optionId, criterionId, decisionId)
                            
                            return updatedDecision
                        else
                            logger.LogWarning("Option with ID {OptionId} not found for decision with ID {DecisionId}", 
                                             optionId, decisionId)
                            return raise (KeyNotFoundException($"Option with ID {optionId} not found for decision with ID {decisionId}"))
                    | None ->
                        logger.LogWarning("Criterion with ID {CriterionId} not found for decision with ID {DecisionId}", 
                                         criterionId, decisionId)
                        return raise (KeyNotFoundException($"Criterion with ID {criterionId} not found for decision with ID {decisionId}"))
                | None ->
                    logger.LogWarning("Decision with ID {DecisionId} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error scoring option for criterion")
                return raise ex
        }
