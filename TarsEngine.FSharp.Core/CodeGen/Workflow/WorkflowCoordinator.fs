namespace TarsEngine.FSharp.Core.CodeGen.Workflow

open System
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Implementation of IWorkflowCoordinator.
/// </summary>
type WorkflowCoordinator(logger: ILogger<WorkflowCoordinator>) =
    
    // Dictionary of workflows by ID
    let workflows = Dictionary<Guid, Workflow>()
    
    // Dictionary of cancellation token sources by workflow ID
    let cancellationTokenSources = Dictionary<Guid, CancellationTokenSource>()
    
    /// <summary>
    /// Creates a workflow.
    /// </summary>
    /// <param name="name">The name of the workflow.</param>
    /// <param name="description">The description of the workflow.</param>
    /// <param name="steps">The steps in the workflow.</param>
    /// <returns>The created workflow.</returns>
    member _.CreateWorkflow(name: string, description: string, steps: (string * string * (unit -> Task<obj>)) list) =
        let workflowId = Guid.NewGuid()
        
        let workflowSteps = 
            steps
            |> List.map (fun (stepName, stepDescription, stepAction) ->
                {
                    Name = stepName
                    Description = stepDescription
                    Action = stepAction
                    Status = WorkflowStatus.NotStarted
                    Result = None
                    ErrorMessage = None
                }
            )
        
        let workflow = {
            Id = workflowId
            Name = name
            Description = description
            Steps = workflowSteps
            Status = WorkflowStatus.NotStarted
            StartTime = None
            EndTime = None
            AdditionalInfo = Map.empty
        }
        
        workflows.Add(workflowId, workflow)
        
        workflow
    
    /// <summary>
    /// Executes a workflow.
    /// </summary>
    /// <param name="workflow">The workflow to execute.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The executed workflow.</returns>
    member _.ExecuteWorkflowAsync(workflow: Workflow, ?cancellationToken: CancellationToken) =
        task {
            try
                logger.LogInformation("Executing workflow: {WorkflowName} ({WorkflowId})", workflow.Name, workflow.Id)
                
                // Create a cancellation token source
                let cts = new CancellationTokenSource()
                cancellationTokenSources.Add(workflow.Id, cts)
                
                // Link the provided cancellation token, if any
                match cancellationToken with
                | Some ct -> ct.Register(fun () -> cts.Cancel()) |> ignore
                | None -> ()
                
                // Update workflow status
                workflow.Status <- WorkflowStatus.Running
                workflow.StartTime <- Some DateTime.Now
                
                // Execute each step
                for step in workflow.Steps do
                    try
                        // Check if the workflow has been cancelled
                        if cts.Token.IsCancellationRequested then
                            workflow.Status <- WorkflowStatus.Cancelled
                            step.Status <- WorkflowStatus.Cancelled
                            break
                        
                        logger.LogInformation("Executing step: {StepName}", step.Name)
                        
                        // Update step status
                        step.Status <- WorkflowStatus.Running
                        
                        // Execute the step
                        let! result = step.Action()
                        
                        // Update step status and result
                        step.Status <- WorkflowStatus.Completed
                        step.Result <- Some result
                        
                        logger.LogInformation("Step completed: {StepName}", step.Name)
                    with
                    | ex ->
                        logger.LogError(ex, "Error executing step: {StepName}", step.Name)
                        
                        // Update step status and error message
                        step.Status <- WorkflowStatus.Failed
                        step.ErrorMessage <- Some ex.Message
                        
                        // Update workflow status
                        workflow.Status <- WorkflowStatus.Failed
                        
                        // Break the loop
                        break
                
                // Update workflow status if not already failed or cancelled
                if workflow.Status = WorkflowStatus.Running then
                    workflow.Status <- WorkflowStatus.Completed
                
                // Update workflow end time
                workflow.EndTime <- Some DateTime.Now
                
                // Remove the cancellation token source
                cancellationTokenSources.Remove(workflow.Id) |> ignore
                
                return workflow
            with
            | ex ->
                logger.LogError(ex, "Error executing workflow: {WorkflowName} ({WorkflowId})", workflow.Name, workflow.Id)
                
                // Update workflow status
                workflow.Status <- WorkflowStatus.Failed
                workflow.EndTime <- Some DateTime.Now
                
                // Remove the cancellation token source
                if cancellationTokenSources.ContainsKey(workflow.Id) then
                    cancellationTokenSources.Remove(workflow.Id) |> ignore
                
                return workflow
        }
    
    /// <summary>
    /// Gets a workflow by ID.
    /// </summary>
    /// <param name="id">The ID of the workflow.</param>
    /// <returns>The workflow, if found.</returns>
    member _.GetWorkflow(id: Guid) =
        if workflows.ContainsKey(id) then
            Some workflows.[id]
        else
            None
    
    /// <summary>
    /// Gets all workflows.
    /// </summary>
    /// <returns>The list of all workflows.</returns>
    member _.GetAllWorkflows() =
        workflows.Values |> Seq.toList
    
    /// <summary>
    /// Gets the status of a workflow.
    /// </summary>
    /// <param name="id">The ID of the workflow.</param>
    /// <returns>The status of the workflow, if found.</returns>
    member _.GetWorkflowStatus(id: Guid) =
        if workflows.ContainsKey(id) then
            Some workflows.[id].Status
        else
            None
    
    /// <summary>
    /// Cancels a workflow.
    /// </summary>
    /// <param name="id">The ID of the workflow.</param>
    /// <returns>Whether the workflow was cancelled.</returns>
    member _.CancelWorkflow(id: Guid) =
        if cancellationTokenSources.ContainsKey(id) then
            cancellationTokenSources.[id].Cancel()
            true
        else
            false
    
    interface IWorkflowCoordinator with
        member this.CreateWorkflow(name, description, steps) = this.CreateWorkflow(name, description, steps)
        member this.ExecuteWorkflowAsync(workflow, ?cancellationToken) = this.ExecuteWorkflowAsync(workflow, ?cancellationToken = cancellationToken)
        member this.GetWorkflow(id) = this.GetWorkflow(id)
        member this.GetAllWorkflows() = this.GetAllWorkflows()
        member this.GetWorkflowStatus(id) = this.GetWorkflowStatus(id)
        member this.CancelWorkflow(id) = this.CancelWorkflow(id)
