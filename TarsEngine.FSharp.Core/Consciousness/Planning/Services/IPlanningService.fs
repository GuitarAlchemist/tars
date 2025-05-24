namespace TarsEngine.FSharp.Core.Consciousness.Planning.Services

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Core.Consciousness.Planning

/// <summary>
/// Interface for the planning service.
/// </summary>
type IPlanningService =
    /// <summary>
    /// Creates a new execution plan.
    /// </summary>
    /// <param name="name">The name of the execution plan.</param>
    /// <param name="description">The description of the execution plan.</param>
    /// <param name="improvementId">The ID of the improvement associated with the execution plan.</param>
    /// <param name="metascriptId">The ID of the metascript associated with the execution plan.</param>
    /// <returns>The created execution plan.</returns>
    abstract member CreateExecutionPlan: name:string * description:string * improvementId:string * metascriptId:string -> Task<ExecutionPlan>
    
    /// <summary>
    /// Gets an execution plan by ID.
    /// </summary>
    /// <param name="id">The ID of the execution plan.</param>
    /// <returns>The execution plan, or None if not found.</returns>
    abstract member GetExecutionPlan: id:string -> Task<ExecutionPlan option>
    
    /// <summary>
    /// Gets all execution plans.
    /// </summary>
    /// <returns>The list of all execution plans.</returns>
    abstract member GetAllExecutionPlans: unit -> Task<ExecutionPlan list>
    
    /// <summary>
    /// Updates an execution plan.
    /// </summary>
    /// <param name="plan">The updated execution plan.</param>
    /// <returns>The updated execution plan.</returns>
    abstract member UpdateExecutionPlan: plan:ExecutionPlan -> Task<ExecutionPlan>
    
    /// <summary>
    /// Deletes an execution plan.
    /// </summary>
    /// <param name="id">The ID of the execution plan to delete.</param>
    /// <returns>True if the execution plan was deleted, false otherwise.</returns>
    abstract member DeleteExecutionPlan: id:string -> Task<bool>
    
    /// <summary>
    /// Executes a step in an execution plan.
    /// </summary>
    /// <param name="planId">The ID of the execution plan.</param>
    /// <param name="stepId">The ID of the step to execute.</param>
    /// <returns>The result of the execution step.</returns>
    abstract member ExecuteStep: planId:string * stepId:string -> Task<ExecutionStepResult option>
    
    /// <summary>
    /// Executes an execution plan.
    /// </summary>
    /// <param name="id">The ID of the execution plan.</param>
    /// <returns>The result of the execution plan.</returns>
    abstract member ExecutePlan: id:string -> Task<ExecutionPlanResult option>
    
    /// <summary>
    /// Validates an execution plan.
    /// </summary>
    /// <param name="id">The ID of the execution plan.</param>
    /// <returns>True if the execution plan is valid, false otherwise.</returns>
    abstract member ValidatePlan: id:string -> Task<bool>
    
    /// <summary>
    /// Monitors an execution plan.
    /// </summary>
    /// <param name="id">The ID of the execution plan.</param>
    /// <returns>The current status and progress of the execution plan.</returns>
    abstract member MonitorPlan: id:string -> Task<(ExecutionPlanStatus * float) option>
    
    /// <summary>
    /// Adapts an execution plan based on feedback or changes.
    /// </summary>
    /// <param name="id">The ID of the execution plan.</param>
    /// <param name="adaptationReason">The reason for adaptation.</param>
    /// <returns>The adapted execution plan.</returns>
    abstract member AdaptPlan: id:string * adaptationReason:string -> Task<ExecutionPlan option>
    
    /// <summary>
    /// Rolls back a step in an execution plan.
    /// </summary>
    /// <param name="planId">The ID of the execution plan.</param>
    /// <param name="stepId">The ID of the step to roll back.</param>
    /// <returns>True if the step was rolled back, false otherwise.</returns>
    abstract member RollbackStep: planId:string * stepId:string -> Task<bool>
    
    /// <summary>
    /// Rolls back an entire execution plan.
    /// </summary>
    /// <param name="id">The ID of the execution plan.</param>
    /// <returns>True if the plan was rolled back, false otherwise.</returns>
    abstract member RollbackPlan: id:string -> Task<bool>
    
    /// <summary>
    /// Gets the execution context for an execution plan.
    /// </summary>
    /// <param name="planId">The ID of the execution plan.</param>
    /// <returns>The execution context, or None if not found.</returns>
    abstract member GetExecutionContext: planId:string -> Task<ExecutionContext option>
    
    /// <summary>
    /// Updates the execution context for an execution plan.
    /// </summary>
    /// <param name="planId">The ID of the execution plan.</param>
    /// <param name="context">The updated execution context.</param>
    /// <returns>The updated execution context.</returns>
    abstract member UpdateExecutionContext: planId:string * context:ExecutionContext -> Task<ExecutionContext>
