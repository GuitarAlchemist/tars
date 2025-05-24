namespace TarsEngine.FSharp.Core.Consciousness.Decision.Services

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Core.Consciousness.Core
open TarsEngine.FSharp.Core.Consciousness.Decision

/// <summary>
/// Interface for decision services.
/// </summary>
type IDecisionService =
    /// <summary>
    /// Creates a decision.
    /// </summary>
    /// <param name="name">The name of the decision.</param>
    /// <param name="description">The description of the decision.</param>
    /// <param name="type">The type of the decision.</param>
    /// <param name="priority">The priority of the decision.</param>
    /// <param name="options">The options for the decision.</param>
    /// <param name="criteria">The criteria for the decision.</param>
    /// <param name="constraints">The constraints for the decision.</param>
    /// <param name="deadline">The deadline for the decision.</param>
    /// <param name="context">The context of the decision.</param>
    /// <returns>The created decision.</returns>
    abstract member CreateDecision : name:string * description:string * type':DecisionType * ?priority:DecisionPriority * ?options:DecisionOption list * ?criteria:DecisionCriterion list * ?constraints:DecisionConstraint list * ?deadline:DateTime * ?context:string -> Task<Decision>
    
    /// <summary>
    /// Gets a decision by ID.
    /// </summary>
    /// <param name="id">The ID of the decision.</param>
    /// <returns>The decision, if found.</returns>
    abstract member GetDecision : id:Guid -> Task<Decision option>
    
    /// <summary>
    /// Gets all decisions.
    /// </summary>
    /// <returns>The list of all decisions.</returns>
    abstract member GetAllDecisions : unit -> Task<Decision list>
    
    /// <summary>
    /// Updates a decision.
    /// </summary>
    /// <param name="id">The ID of the decision to update.</param>
    /// <param name="description">The new description of the decision.</param>
    /// <param name="status">The new status of the decision.</param>
    /// <param name="priority">The new priority of the decision.</param>
    /// <param name="selectedOption">The new selected option for the decision.</param>
    /// <param name="deadline">The new deadline for the decision.</param>
    /// <param name="context">The new context of the decision.</param>
    /// <param name="justification">The new justification for the decision.</param>
    /// <returns>The updated decision.</returns>
    abstract member UpdateDecision : id:Guid * ?description:string * ?status:DecisionStatus * ?priority:DecisionPriority * ?selectedOption:Guid * ?deadline:DateTime * ?context:string * ?justification:string -> Task<Decision>
    
    /// <summary>
    /// Deletes a decision.
    /// </summary>
    /// <param name="id">The ID of the decision to delete.</param>
    /// <returns>Whether the decision was deleted.</returns>
    abstract member DeleteDecision : id:Guid -> Task<bool>
    
    /// <summary>
    /// Adds an option to a decision.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="name">The name of the option.</param>
    /// <param name="description">The description of the option.</param>
    /// <param name="pros">The pros of the option.</param>
    /// <param name="cons">The cons of the option.</param>
    /// <returns>The updated decision.</returns>
    abstract member AddOption : decisionId:Guid * name:string * description:string * ?pros:string list * ?cons:string list -> Task<Decision>
    
    /// <summary>
    /// Updates an option.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="optionId">The ID of the option to update.</param>
    /// <param name="name">The new name of the option.</param>
    /// <param name="description">The new description of the option.</param>
    /// <param name="pros">The new pros of the option.</param>
    /// <param name="cons">The new cons of the option.</param>
    /// <param name="score">The new score of the option.</param>
    /// <param name="rank">The new rank of the option.</param>
    /// <returns>The updated decision.</returns>
    abstract member UpdateOption : decisionId:Guid * optionId:Guid * ?name:string * ?description:string * ?pros:string list * ?cons:string list * ?score:float * ?rank:int -> Task<Decision>
    
    /// <summary>
    /// Removes an option from a decision.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="optionId">The ID of the option to remove.</param>
    /// <returns>The updated decision.</returns>
    abstract member RemoveOption : decisionId:Guid * optionId:Guid -> Task<Decision>
    
    /// <summary>
    /// Adds a criterion to a decision.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="name">The name of the criterion.</param>
    /// <param name="description">The description of the criterion.</param>
    /// <param name="weight">The weight of the criterion.</param>
    /// <returns>The updated decision.</returns>
    abstract member AddCriterion : decisionId:Guid * name:string * description:string * weight:float -> Task<Decision>
    
    /// <summary>
    /// Updates a criterion.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="criterionId">The ID of the criterion to update.</param>
    /// <param name="name">The new name of the criterion.</param>
    /// <param name="description">The new description of the criterion.</param>
    /// <param name="weight">The new weight of the criterion.</param>
    /// <returns>The updated decision.</returns>
    abstract member UpdateCriterion : decisionId:Guid * criterionId:Guid * ?name:string * ?description:string * ?weight:float -> Task<Decision>
    
    /// <summary>
    /// Removes a criterion from a decision.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="criterionId">The ID of the criterion to remove.</param>
    /// <returns>The updated decision.</returns>
    abstract member RemoveCriterion : decisionId:Guid * criterionId:Guid -> Task<Decision>
    
    /// <summary>
    /// Scores an option for a criterion.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="criterionId">The ID of the criterion.</param>
    /// <param name="optionId">The ID of the option.</param>
    /// <param name="score">The score.</param>
    /// <returns>The updated decision.</returns>
    abstract member ScoreOption : decisionId:Guid * criterionId:Guid * optionId:Guid * score:float -> Task<Decision>
    
    /// <summary>
    /// Adds a constraint to a decision.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="name">The name of the constraint.</param>
    /// <param name="description">The description of the constraint.</param>
    /// <param name="type">The type of the constraint.</param>
    /// <param name="value">The value of the constraint.</param>
    /// <returns>The updated decision.</returns>
    abstract member AddConstraint : decisionId:Guid * name:string * description:string * type':string * value:obj -> Task<Decision>
    
    /// <summary>
    /// Updates a constraint.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="constraintId">The ID of the constraint to update.</param>
    /// <param name="name">The new name of the constraint.</param>
    /// <param name="description">The new description of the constraint.</param>
    /// <param name="type">The new type of the constraint.</param>
    /// <param name="value">The new value of the constraint.</param>
    /// <param name="isSatisfied">Whether the constraint is satisfied.</param>
    /// <returns>The updated decision.</returns>
    abstract member UpdateConstraint : decisionId:Guid * constraintId:Guid * ?name:string * ?description:string * ?type':string * ?value:obj * ?isSatisfied:bool -> Task<Decision>
    
    /// <summary>
    /// Removes a constraint from a decision.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="constraintId">The ID of the constraint to remove.</param>
    /// <returns>The updated decision.</returns>
    abstract member RemoveConstraint : decisionId:Guid * constraintId:Guid -> Task<Decision>
    
    /// <summary>
    /// Adds an emotion to a decision.
    /// </summary>
    /// <param name="decisionId">The ID of the decision.</param>
    /// <param name="emotion">The emotion to add.</param>
    /// <returns>The updated decision.</returns>
    abstract member AddEmotionToDecision : decisionId:Guid * emotion:Emotion -> Task<Decision>
    
    /// <summary>
    /// Evaluates a decision.
    /// </summary>
    /// <param name="decisionId">The ID of the decision to evaluate.</param>
    /// <returns>The decision evaluation.</returns>
    abstract member EvaluateDecision : decisionId:Guid -> Task<DecisionEvaluation>
    
    /// <summary>
    /// Makes a decision.
    /// </summary>
    /// <param name="decisionId">The ID of the decision to make.</param>
    /// <returns>The updated decision.</returns>
    abstract member MakeDecision : decisionId:Guid -> Task<Decision>
    
    /// <summary>
    /// Finds decisions.
    /// </summary>
    /// <param name="query">The decision query.</param>
    /// <returns>The decision query result.</returns>
    abstract member FindDecisions : query:DecisionQuery -> Task<DecisionQueryResult>
