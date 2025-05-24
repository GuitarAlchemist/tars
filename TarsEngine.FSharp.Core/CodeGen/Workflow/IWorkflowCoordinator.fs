namespace TarsEngine.FSharp.Core.CodeGen.Workflow

open System
open System.Threading
open System.Threading.Tasks

/// <summary>
/// Represents the status of a workflow.
/// </summary>
type WorkflowStatus =
    | NotStarted
    | Running
    | Completed
    | Failed
    | Cancelled

/// <summary>
/// Represents a workflow step.
/// </summary>
type WorkflowStep = {
    /// <summary>
    /// The name of the step.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the step.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The action to execute for the step.
    /// </summary>
    Action: unit -> Task<obj>
    
    /// <summary>
    /// The status of the step.
    /// </summary>
    mutable Status: WorkflowStatus
    
    /// <summary>
    /// The result of the step, if any.
    /// </summary>
    mutable Result: obj option
    
    /// <summary>
    /// The error message, if any.
    /// </summary>
    mutable ErrorMessage: string option
}

/// <summary>
/// Represents a workflow.
/// </summary>
type Workflow = {
    /// <summary>
    /// The ID of the workflow.
    /// </summary>
    Id: Guid
    
    /// <summary>
    /// The name of the workflow.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the workflow.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The steps in the workflow.
    /// </summary>
    Steps: WorkflowStep list
    
    /// <summary>
    /// The status of the workflow.
    /// </summary>
    mutable Status: WorkflowStatus
    
    /// <summary>
    /// The start time of the workflow, if any.
    /// </summary>
    mutable StartTime: DateTime option
    
    /// <summary>
    /// The end time of the workflow, if any.
    /// </summary>
    mutable EndTime: DateTime option
    
    /// <summary>
    /// Additional information about the workflow.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Interface for coordinating workflows.
/// </summary>
type IWorkflowCoordinator =
    /// <summary>
    /// Creates a workflow.
    /// </summary>
    /// <param name="name">The name of the workflow.</param>
    /// <param name="description">The description of the workflow.</param>
    /// <param name="steps">The steps in the workflow.</param>
    /// <returns>The created workflow.</returns>
    abstract member CreateWorkflow : name:string * description:string * steps:(string * string * (unit -> Task<obj>)) list -> Workflow
    
    /// <summary>
    /// Executes a workflow.
    /// </summary>
    /// <param name="workflow">The workflow to execute.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The executed workflow.</returns>
    abstract member ExecuteWorkflowAsync : workflow:Workflow * ?cancellationToken:CancellationToken -> Task<Workflow>
    
    /// <summary>
    /// Gets a workflow by ID.
    /// </summary>
    /// <param name="id">The ID of the workflow.</param>
    /// <returns>The workflow, if found.</returns>
    abstract member GetWorkflow : id:Guid -> Workflow option
    
    /// <summary>
    /// Gets all workflows.
    /// </summary>
    /// <returns>The list of all workflows.</returns>
    abstract member GetAllWorkflows : unit -> Workflow list
    
    /// <summary>
    /// Gets the status of a workflow.
    /// </summary>
    /// <param name="id">The ID of the workflow.</param>
    /// <returns>The status of the workflow, if found.</returns>
    abstract member GetWorkflowStatus : id:Guid -> WorkflowStatus option
    
    /// <summary>
    /// Cancels a workflow.
    /// </summary>
    /// <param name="id">The ID of the workflow.</param>
    /// <returns>Whether the workflow was cancelled.</returns>
    abstract member CancelWorkflow : id:Guid -> bool
