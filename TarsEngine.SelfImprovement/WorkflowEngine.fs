namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Module for executing workflows
/// </summary>
module WorkflowEngine =
    /// <summary>
    /// Executes a workflow step
    /// </summary>
    let executeStep (logger: ILogger) (handler: WorkflowState -> Task<StepResult>) (state: WorkflowState) (stepIndex: int) =
        task {
            // Start the step
            let updatedState = WorkflowState.startStep stepIndex state

            // Save the state
            let! _ = WorkflowState.save updatedState WorkflowState.defaultStatePath

            // Execute the step handler
            try
                let! result = handler updatedState

                match result with
                | Ok data ->
                    // Complete the step
                    let completedState = WorkflowState.completeStep stepIndex data updatedState
                    let! _ = WorkflowState.save completedState WorkflowState.defaultStatePath
                    return completedState
                | Error errorMessage ->
                    // Fail the step
                    logger.LogError("Step {StepName} failed: {ErrorMessage}",
                                   updatedState.Steps.[stepIndex].Name,
                                   errorMessage)
                    let failedState = WorkflowState.failStep stepIndex errorMessage updatedState
                    let! _ = WorkflowState.save failedState WorkflowState.defaultStatePath
                    return failedState
            with ex ->
                // Handle exceptions
                logger.LogError(ex, "Exception in step {StepName}",
                               updatedState.Steps.[stepIndex].Name)
                let failedState = WorkflowState.failStep stepIndex ex.Message updatedState
                let! _ = WorkflowState.save failedState WorkflowState.defaultStatePath
                return failedState
        }

    /// <summary>
    /// Executes a workflow
    /// </summary>
    let executeWorkflow (logger: ILogger) (handlers: (WorkflowState -> Task<StepResult>) list) (state: WorkflowState) =
        task {
            // Check if the workflow has already been completed or failed
            if state.Status = Completed || state.Status = Failed then
                logger.LogInformation("Workflow {WorkflowName} has already been {Status}",
                                     state.Name,
                                     state.Status)
                return state
            else

            // Initialize the workflow if it doesn't have any steps
            let initialState =
                if state.Steps.IsEmpty then
                    // Create steps based on the handlers
                    let steps =
                        handlers
                        |> List.mapi (fun i _ ->
                            WorkflowState.createStep (sprintf "Step %d" (i + 1)))

                    // Add steps to the workflow
                    let stateWithSteps =
                        steps
                        |> List.fold (fun s step -> WorkflowState.addStep step s) state

                    // Save the state
                    let task = WorkflowState.save stateWithSteps WorkflowState.defaultStatePath
                    Async.AwaitTask task |> Async.RunSynchronously |> ignore
                    stateWithSteps
                else
                    state

            // Get the current step index
            let currentStepIndex =
                match initialState.CurrentStepIndex with
                | Some index -> index
                | None -> 0

            // Execute the workflow steps recursively
            let rec executeSteps (currentState: WorkflowState) (index: int) =
                task {
                    // Check if we've reached the end of the steps
                    if index >= currentState.Steps.Length then
                        // Complete the workflow
                        let completedState = WorkflowState.complete currentState
                        let! _ = WorkflowState.save completedState WorkflowState.defaultStatePath
                        return completedState
                    elif WorkflowState.hasExceededMaxDuration currentState then
                        // Check if the workflow has exceeded its maximum duration
                        logger.LogWarning("Workflow {WorkflowName} has exceeded its maximum duration of {MaxDurationMinutes} minutes",
                                         currentState.Name,
                                         currentState.MaxDurationMinutes)
                        // Fail the workflow
                        let failedState = WorkflowState.fail currentState
                        let! _ = WorkflowState.save failedState WorkflowState.defaultStatePath
                        return failedState
                    else
                        // Execute the current step
                        let! updatedState = executeStep logger handlers.[index] currentState index

                        // Check if the step failed
                        if updatedState.Steps.[index].Status = Failed then
                            // Fail the workflow
                            let failedState = WorkflowState.fail updatedState
                            let! _ = WorkflowState.save failedState WorkflowState.defaultStatePath
                            return failedState
                        else
                            // Move to the next step
                            return! executeSteps updatedState (index + 1)
                }

            // Start executing the steps
            return! executeSteps initialState currentStepIndex
        }

    /// <summary>
    /// Creates a new workflow and executes it
    /// </summary>
    let createAndExecuteWorkflow (logger: ILogger) (name: string) (targetDirectories: string list) (maxDurationMinutes: int) (handlers: (WorkflowState -> Task<StepResult>) list) =
        task {
            // Create a new workflow state
            let state = WorkflowState.create name targetDirectories maxDurationMinutes

            // Save the state
            let! _ = WorkflowState.save state WorkflowState.defaultStatePath

            // Execute the workflow
            return! executeWorkflow logger handlers state
        }

    /// <summary>
    /// Resumes a workflow from its saved state
    /// </summary>
    let resumeWorkflow (logger: ILogger) (handlers: (WorkflowState -> Task<StepResult>) list) =
        task {
            // Try to load the workflow state
            let! stateOption = WorkflowState.tryLoad WorkflowState.defaultStatePath

            match stateOption with
            | Some state ->
                // Execute the workflow
                return! executeWorkflow logger handlers state
            | None ->
                // No saved state found
                logger.LogWarning("No saved workflow state found at {StatePath}",
                                 WorkflowState.defaultStatePath)
                return! Task.FromResult(WorkflowState.create "Default Workflow" [] 60)
        }
