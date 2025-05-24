namespace TarsEngine.FSharp.Core.Consciousness.Decision.Services

open System
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Core
open TarsEngine.FSharp.Core.Consciousness.Decision

/// <summary>
/// Complete implementation of IDecisionService.
/// </summary>
type DecisionServiceComplete(logger: ILogger<DecisionServiceComplete>) =
    // In-memory storage for decisions
    let mutable decisions = Map.empty<Guid, Decision>
    
    // Reference to decisions for use in helper classes
    let decisionsRef = ref decisions
    
    // Random number generator for simulating intuitive decisions
    let random = System.Random()
    
    // Intuitive decision level (0.0 to 1.0)
    let mutable intuitiveDecisionLevel = 0.5 // Starting with moderate intuition
    
    // Helper classes for different aspects of decision management
    let criterionManagement = CriterionManagement(logger, decisionsRef)
    let constraintManagement = ConstraintManagement(logger, decisionsRef)
    let decisionMaking = DecisionMaking(logger, decisionsRef, intuitiveDecisionLevel)
    
    // Update the reference when decisions change
    do
        // Create a property changed event handler
        let updateRef () = decisionsRef := decisions
        
        // Call it initially
        updateRef()
    
    /// <summary>
    /// Gets the intuitive decision level (0.0 to 1.0).
    /// </summary>
    member _.IntuitiveDecisionLevel = intuitiveDecisionLevel
    
    /// <summary>
    /// Creates a new decision.
    /// </summary>
    member _.CreateDecision(name: string, description: string, type': DecisionType, 
                           ?priority: DecisionPriority, ?options: DecisionOption list, 
                           ?criteria: DecisionCriterion list, ?constraints: DecisionConstraint list, 
                           ?deadline: DateTime, ?context: string) =
        task {
            try
                logger.LogInformation("Creating decision: {Name}", name)
                
                // Create a new decision
                let decision = {
                    Id = Guid.NewGuid()
                    Name = name
                    Description = description
                    Type = type'
                    Status = DecisionStatus.Pending
                    Priority = defaultArg priority DecisionPriority.Medium
                    Options = defaultArg options []
                    Criteria = defaultArg criteria []
                    Constraints = defaultArg constraints []
                    SelectedOption = None
                    CreationTime = DateTime.UtcNow
                    Deadline = deadline
                    CompletionTime = None
                    AssociatedEmotions = []
                    Context = Option.map (fun c -> c) context
                    Justification = None
                    Metadata = Map.empty
                }
                
                // Add the decision to the dictionary
                decisions <- Map.add decision.Id decision decisions
                decisionsRef := decisions
                
                logger.LogInformation("Created decision with ID: {Id}", decision.Id)
                
                return decision
            with
            | ex ->
                logger.LogError(ex, "Error creating decision")
                return raise ex
        }
    
    /// <summary>
    /// Gets a decision by ID.
    /// </summary>
    member _.GetDecision(id: Guid) =
        task {
            try
                logger.LogInformation("Getting decision with ID: {Id}", id)
                
                // Try to get the decision from the dictionary
                return Map.tryFind id decisions
            with
            | ex ->
                logger.LogError(ex, "Error getting decision")
                return None
        }
    
    /// <summary>
    /// Gets all decisions.
    /// </summary>
    member _.GetAllDecisions() =
        task {
            try
                logger.LogInformation("Getting all decisions")
                
                // Convert the dictionary values to a list
                let decisionList = decisions |> Map.values |> Seq.toList
                
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
    member _.UpdateDecision(id: Guid, ?description: string, ?status: DecisionStatus, 
                           ?priority: DecisionPriority, ?selectedOption: Guid, 
                           ?deadline: DateTime, ?context: string, ?justification: string) =
        task {
            try
                logger.LogInformation("Updating decision with ID: {Id}", id)
                
                // Check if the decision exists
                match Map.tryFind id decisions with
                | Some decision ->
                    // Update the decision
                    let updatedDecision = { 
                        decision with 
                            Description = defaultArg description decision.Description
                            Status = defaultArg status decision.Status
                            Priority = defaultArg priority decision.Priority
                            SelectedOption = 
                                match selectedOption with
                                | Some optId -> Some optId
                                | None -> decision.SelectedOption
                            Deadline = 
                                match deadline with
                                | Some dl -> Some dl
                                | None -> decision.Deadline
                            Context = 
                                match context with
                                | Some ctx -> Some ctx
                                | None -> decision.Context
                            Justification = 
                                match justification with
                                | Some just -> Some just
                                | None -> decision.Justification
                            CompletionTime = 
                                if defaultArg status decision.Status = DecisionStatus.Completed && decision.CompletionTime.IsNone then
                                    Some DateTime.UtcNow
                                else
                                    decision.CompletionTime
                    }
                    
                    // Update the decision in the dictionary
                    decisions <- Map.add id updatedDecision decisions
                    decisionsRef := decisions
                    
                    logger.LogInformation("Updated decision with ID: {Id}", id)
                    
                    return updatedDecision
                | None ->
                    logger.LogWarning("Decision with ID {Id} not found", id)
                    return raise (KeyNotFoundException($"Decision with ID {id} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error updating decision")
                return raise ex
        }
    
    /// <summary>
    /// Deletes a decision.
    /// </summary>
    member _.DeleteDecision(id: Guid) =
        task {
            try
                logger.LogInformation("Deleting decision with ID: {Id}", id)
                
                // Check if the decision exists
                if Map.containsKey id decisions then
                    // Remove the decision
                    decisions <- Map.remove id decisions
                    decisionsRef := decisions
                    
                    logger.LogInformation("Deleted decision with ID: {Id}", id)
                    
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
    member _.AddOption(decisionId: Guid, name: string, description: string, 
                      ?pros: string list, ?cons: string list) =
        task {
            try
                logger.LogInformation("Adding option to decision with ID: {DecisionId}", decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId decisions with
                | Some decision ->
                    // Create a new option
                    let option = {
                        Id = Guid.NewGuid()
                        Name = name
                        Description = description
                        Pros = defaultArg pros []
                        Cons = defaultArg cons []
                        Score = None
                        Rank = None
                        Metadata = Map.empty
                    }
                    
                    // Add the option to the decision
                    let updatedDecision = { 
                        decision with 
                            Options = option :: decision.Options 
                    }
                    
                    // Update the decision
                    decisions <- Map.add decisionId updatedDecision decisions
                    decisionsRef := decisions
                    
                    logger.LogInformation("Added option with ID: {Id} to decision with ID: {DecisionId}", 
                                         option.Id, decisionId)
                    
                    return updatedDecision
                | None ->
                    logger.LogWarning("Decision with ID {Id} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error adding option to decision")
                return raise ex
        }
    
    /// <summary>
    /// Updates an option.
    /// </summary>
    member _.UpdateOption(decisionId: Guid, optionId: Guid, ?name: string, ?description: string, 
                         ?pros: string list, ?cons: string list, ?score: float, ?rank: int) =
        task {
            try
                logger.LogInformation("Updating option with ID: {OptionId} for decision with ID: {DecisionId}", 
                                     optionId, decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId decisions with
                | Some decision ->
                    // Find the option
                    let optionIndex = decision.Options |> List.tryFindIndex (fun o -> o.Id = optionId)
                    
                    match optionIndex with
                    | Some index ->
                        let option = decision.Options.[index]
                        
                        // Update the option
                        let updatedOption = {
                            option with
                                Name = defaultArg name option.Name
                                Description = defaultArg description option.Description
                                Pros = defaultArg pros option.Pros
                                Cons = defaultArg cons option.Cons
                                Score = 
                                    match score with
                                    | Some s -> Some s
                                    | None -> option.Score
                                Rank = 
                                    match rank with
                                    | Some r -> Some r
                                    | None -> option.Rank
                        }
                        
                        // Update the options list
                        let updatedOptions = 
                            decision.Options 
                            |> List.mapi (fun i o -> if i = index then updatedOption else o)
                        
                        // Update the decision
                        let updatedDecision = { decision with Options = updatedOptions }
                        
                        // Update the decision in the dictionary
                        decisions <- Map.add decisionId updatedDecision decisions
                        decisionsRef := decisions
                        
                        logger.LogInformation("Updated option with ID: {OptionId} for decision with ID: {DecisionId}", 
                                            optionId, decisionId)
                        
                        return updatedDecision
                    | None ->
                        logger.LogWarning("Option with ID {OptionId} not found for decision with ID {DecisionId}", 
                                         optionId, decisionId)
                        return raise (KeyNotFoundException($"Option with ID {optionId} not found for decision with ID {decisionId}"))
                | None ->
                    logger.LogWarning("Decision with ID {DecisionId} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error updating option")
                return raise ex
        }
    
    /// <summary>
    /// Removes an option from a decision.
    /// </summary>
    member _.RemoveOption(decisionId: Guid, optionId: Guid) =
        task {
            try
                logger.LogInformation("Removing option with ID: {OptionId} from decision with ID: {DecisionId}", 
                                     optionId, decisionId)
                
                // Check if the decision exists
                match Map.tryFind decisionId decisions with
                | Some decision ->
                    // Check if the option exists
                    let optionExists = decision.Options |> List.exists (fun o -> o.Id = optionId)
                    
                    if optionExists then
                        // Remove the option
                        let updatedOptions = decision.Options |> List.filter (fun o -> o.Id <> optionId)
                        
                        // Update the selected option if it was the one being removed
                        let updatedSelectedOption = 
                            match decision.SelectedOption with
                            | Some selectedId when selectedId = optionId -> None
                            | other -> other
                        
                        // Update the decision
                        let updatedDecision = { 
                            decision with 
                                Options = updatedOptions 
                                SelectedOption = updatedSelectedOption
                        }
                        
                        // Update the decision in the dictionary
                        decisions <- Map.add decisionId updatedDecision decisions
                        decisionsRef := decisions
                        
                        logger.LogInformation("Removed option with ID: {OptionId} from decision with ID: {DecisionId}", 
                                            optionId, decisionId)
                        
                        return updatedDecision
                    else
                        logger.LogWarning("Option with ID {OptionId} not found for decision with ID {DecisionId}", 
                                         optionId, decisionId)
                        return raise (KeyNotFoundException($"Option with ID {optionId} not found for decision with ID {decisionId}"))
                | None ->
                    logger.LogWarning("Decision with ID {DecisionId} not found", decisionId)
                    return raise (KeyNotFoundException($"Decision with ID {decisionId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error removing option")
                return raise ex
        }
    
    /// <summary>
    /// Updates the intuitive decision level.
    /// </summary>
    member _.UpdateIntuitiveDecisionLevel(level: float) =
        try
            // Ensure the level is between 0 and 1
            let adjustedLevel = Math.Max(0.0, Math.Min(1.0, level))
            
            // Update the level
            intuitiveDecisionLevel <- adjustedLevel
            
            logger.LogInformation("Updated intuitive decision level to: {Level:F2}", adjustedLevel)
            
            true
        with
        | ex ->
            logger.LogError(ex, "Error updating intuitive decision level")
            false
    
    interface IDecisionService with
        member this.CreateDecision(name, description, type', ?priority, ?options, ?criteria, ?constraints, ?deadline, ?context) =
            this.CreateDecision(name, description, type', ?priority = priority, ?options = options, 
                               ?criteria = criteria, ?constraints = constraints, ?deadline = deadline, ?context = context)
        
        member this.GetDecision(id) =
            this.GetDecision(id)
        
        member this.GetAllDecisions() =
            this.GetAllDecisions()
        
        member this.UpdateDecision(id, ?description, ?status, ?priority, ?selectedOption, ?deadline, ?context, ?justification) =
            this.UpdateDecision(id, ?description = description, ?status = status, ?priority = priority, 
                              ?selectedOption = selectedOption, ?deadline = deadline, ?context = context, ?justification = justification)
        
        member this.DeleteDecision(id) =
            this.DeleteDecision(id)
        
        member this.AddOption(decisionId, name, description, ?pros, ?cons) =
            this.AddOption(decisionId, name, description, ?pros = pros, ?cons = cons)
        
        member this.UpdateOption(decisionId, optionId, ?name, ?description, ?pros, ?cons, ?score, ?rank) =
            this.UpdateOption(decisionId, optionId, ?name = name, ?description = description, 
                            ?pros = pros, ?cons = cons, ?score = score, ?rank = rank)
        
        member this.RemoveOption(decisionId, optionId) =
            this.RemoveOption(decisionId, optionId)
        
        member this.AddCriterion(decisionId, name, description, weight) =
            criterionManagement.AddCriterion(decisionId, name, description, weight)
        
        member this.UpdateCriterion(decisionId, criterionId, ?name, ?description, ?weight) =
            criterionManagement.UpdateCriterion(decisionId, criterionId, ?name = name, ?description = description, ?weight = weight)
        
        member this.RemoveCriterion(decisionId, criterionId) =
            criterionManagement.RemoveCriterion(decisionId, criterionId)
        
        member this.ScoreOption(decisionId, criterionId, optionId, score) =
            criterionManagement.ScoreOption(decisionId, criterionId, optionId, score)
        
        member this.AddConstraint(decisionId, name, description, type', value) =
            constraintManagement.AddConstraint(decisionId, name, description, type', value)
        
        member this.UpdateConstraint(decisionId, constraintId, ?name, ?description, ?type', ?value, ?isSatisfied) =
            constraintManagement.UpdateConstraint(decisionId, constraintId, ?name = name, ?description = description, 
                                                ?type' = type', ?value = value, ?isSatisfied = isSatisfied)
        
        member this.RemoveConstraint(decisionId, constraintId) =
            constraintManagement.RemoveConstraint(decisionId, constraintId)
        
        member this.AddEmotionToDecision(decisionId, emotion) =
            decisionMaking.AddEmotionToDecision(decisionId, emotion)
        
        member this.EvaluateDecision(decisionId) =
            decisionMaking.EvaluateDecision(decisionId)
        
        member this.MakeDecision(decisionId) =
            decisionMaking.MakeDecision(decisionId)
        
        member this.FindDecisions(query) =
            decisionMaking.FindDecisions(query)

/// <summary>
/// Extension methods for IServiceCollection to register Decision services.
/// </summary>
module ServiceCollectionExtensions =
    open Microsoft.Extensions.DependencyInjection
    
    /// <summary>
    /// Adds TarsEngine.FSharp.Core.Consciousness.Decision services to the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let addTarsEngineFSharpDecision (services: IServiceCollection) =
        // Register decision services
        services.AddSingleton<IDecisionService, DecisionServiceComplete>() |> ignore
        
        // Return the service collection
        services
