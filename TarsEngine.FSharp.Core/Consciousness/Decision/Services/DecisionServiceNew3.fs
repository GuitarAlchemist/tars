namespace TarsEngine.FSharp.Core.Consciousness.Decision.Services

open System
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Core
open TarsEngine.FSharp.Core.Consciousness.Decision

/// <summary>
/// Implementation of constraint management methods for IDecisionService.
/// </summary>
type ConstraintManagement(logger: ILogger<DecisionServiceNew>, decisions: Map<Guid, Decision> ref) =
    /// <summary>
    /// Adds a constraint to a decision.
    /// </summary>
    member _.AddConstraint(decisionId: Guid, name: string, description: string, type': string, value: obj) =
        task {
            try
                logger.LogInformation("Adding constraint to decision with ID: {DecisionId}", decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId !decisions with
                | Some decision ->
                    // Create a new constraint
                    let constraint' = {
                        Id = Guid.NewGuid()
                        Name = name
                        Description = description
                        Type = type'
                        Value = value
                        IsSatisfied = false // Default to not satisfied
                        Metadata = Map.empty
                    }
                    
                    // Add the constraint to the decision
                    let updatedDecision = { 
                        decision with 
                            Constraints = constraint' :: decision.Constraints 
                    }
                    
                    // Update the decision
                    decisions := Map.add decisionId updatedDecision !decisions
                    
                    logger.LogInformation("Added constraint with ID: {Id} to decision with ID: {DecisionId}", 
                                         constraint'.Id, decisionId)
                    
                    return updatedDecision
                | None ->
                    logger.LogWarning("Decision with ID {Id} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error adding constraint to decision")
                return raise ex
        }
    
    /// <summary>
    /// Updates a constraint.
    /// </summary>
    member _.UpdateConstraint(decisionId: Guid, constraintId: Guid, ?name: string, ?description: string, 
                             ?type': string, ?value: obj, ?isSatisfied: bool) =
        task {
            try
                logger.LogInformation("Updating constraint with ID: {ConstraintId} for decision with ID: {DecisionId}", 
                                     constraintId, decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId !decisions with
                | Some decision ->
                    // Find the constraint
                    let constraintIndex = decision.Constraints |> List.tryFindIndex (fun c -> c.Id = constraintId)
                    
                    match constraintIndex with
                    | Some index ->
                        let constraint' = decision.Constraints.[index]
                        
                        // Update the constraint
                        let updatedConstraint = {
                            constraint' with
                                Name = defaultArg name constraint'.Name
                                Description = defaultArg description constraint'.Description
                                Type = defaultArg type' constraint'.Type
                                Value = defaultArg value constraint'.Value
                                IsSatisfied = defaultArg isSatisfied constraint'.IsSatisfied
                        }
                        
                        // Update the constraints list
                        let updatedConstraints = 
                            decision.Constraints 
                            |> List.mapi (fun i c -> if i = index then updatedConstraint else c)
                        
                        // Update the decision
                        let updatedDecision = { decision with Constraints = updatedConstraints }
                        
                        // Update the decision in the dictionary
                        decisions := Map.add decisionId updatedDecision !decisions
                        
                        logger.LogInformation("Updated constraint with ID: {ConstraintId} for decision with ID: {DecisionId}", 
                                            constraintId, decisionId)
                        
                        return updatedDecision
                    | None ->
                        logger.LogWarning("Constraint with ID {ConstraintId} not found for decision with ID {DecisionId}", 
                                         constraintId, decisionId)
                        return raise (KeyNotFoundException($"Constraint with ID {constraintId} not found for decision with ID {decisionId}"))
                | None ->
                    logger.LogWarning("Decision with ID {DecisionId} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error updating constraint")
                return raise ex
        }
    
    /// <summary>
    /// Removes a constraint from a decision.
    /// </summary>
    member _.RemoveConstraint(decisionId: Guid, constraintId: Guid) =
        task {
            try
                logger.LogInformation("Removing constraint with ID: {ConstraintId} from decision with ID: {DecisionId}", 
                                     constraintId, decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId !decisions with
                | Some decision ->
                    // Check if the constraint exists
                    let constraintExists = decision.Constraints |> List.exists (fun c -> c.Id = constraintId)
                    
                    if constraintExists then
                        // Remove the constraint
                        let updatedConstraints = decision.Constraints |> List.filter (fun c -> c.Id <> constraintId)
                        
                        // Update the decision
                        let updatedDecision = { decision with Constraints = updatedConstraints }
                        
                        // Update the decision in the dictionary
                        decisions := Map.add decisionId updatedDecision !decisions
                        
                        logger.LogInformation("Removed constraint with ID: {ConstraintId} from decision with ID: {DecisionId}", 
                                            constraintId, decisionId)
                        
                        return updatedDecision
                    else
                        logger.LogWarning("Constraint with ID {ConstraintId} not found for decision with ID {DecisionId}", 
                                         constraintId, decisionId)
                        return raise (KeyNotFoundException($"Constraint with ID {constraintId} not found for decision with ID {decisionId}"))
                | None ->
                    logger.LogWarning("Decision with ID {DecisionId} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error removing constraint")
                return raise ex
        }
