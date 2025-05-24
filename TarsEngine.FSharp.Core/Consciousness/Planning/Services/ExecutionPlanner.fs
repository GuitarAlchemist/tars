namespace TarsEngine.FSharp.Core.Consciousness.Planning.Services

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Planning

/// <summary>
/// Implementation of IPlanningService.
/// </summary>
type ExecutionPlanner(logger: ILogger<ExecutionPlanner>) =
    // In-memory storage for execution plans
    let mutable executionPlans = Map.empty<string, ExecutionPlan>
    
    // In-memory storage for execution contexts
    let mutable executionContexts = Map.empty<string, ExecutionContext>
    
    /// <summary>
    /// Creates a new execution plan.
    /// </summary>
    /// <param name="name">The name of the execution plan.</param>
    /// <param name="description">The description of the execution plan.</param>
    /// <param name="improvementId">The ID of the improvement associated with the execution plan.</param>
    /// <param name="metascriptId">The ID of the metascript associated with the execution plan.</param>
    /// <returns>The created execution plan.</returns>
    member _.CreateExecutionPlan(name: string, description: string, improvementId: string, metascriptId: string) =
        task {
            try
                logger.LogInformation("Creating execution plan: {Name}", name)
                
                // Create a new execution plan
                let plan = ExecutionPlan.create name description improvementId metascriptId
                
                // Add the plan to the dictionary
                executionPlans <- Map.add plan.Id plan executionPlans
                
                logger.LogInformation("Created execution plan with ID: {Id}", plan.Id)
                
                return plan
            with
            | ex ->
                logger.LogError(ex, "Error creating execution plan")
                return raise ex
        }
    
    /// <summary>
    /// Gets an execution plan by ID.
    /// </summary>
    /// <param name="id">The ID of the execution plan.</param>
    /// <returns>The execution plan, or None if not found.</returns>
    member _.GetExecutionPlan(id: string) =
        task {
            try
                logger.LogInformation("Getting execution plan with ID: {Id}", id)
                
                // Try to get the plan from the dictionary
                match Map.tryFind id executionPlans with
                | Some plan ->
                    logger.LogInformation("Found execution plan with ID: {Id}", id)
                    return Some plan
                | None ->
                    logger.LogWarning("Execution plan with ID {Id} not found", id)
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error getting execution plan")
                return None
        }
    
    /// <summary>
    /// Gets all execution plans.
    /// </summary>
    /// <returns>The list of all execution plans.</returns>
    member _.GetAllExecutionPlans() =
        task {
            try
                logger.LogInformation("Getting all execution plans")
                
                // Convert the dictionary values to a list
                let planList = executionPlans |> Map.values |> Seq.toList
                
                logger.LogInformation("Found {Count} execution plans", planList.Length)
                
                return planList
            with
            | ex ->
                logger.LogError(ex, "Error getting all execution plans")
                return []
        }
    
    /// <summary>
    /// Updates an execution plan.
    /// </summary>
    /// <param name="plan">The updated execution plan.</param>
    /// <returns>The updated execution plan.</returns>
    member _.UpdateExecutionPlan(plan: ExecutionPlan) =
        task {
            try
                logger.LogInformation("Updating execution plan with ID: {Id}", plan.Id)
                
                // Check if the plan exists
                if Map.containsKey plan.Id executionPlans then
                    // Update the plan
                    let updatedPlan = { plan with UpdatedAt = Some DateTime.UtcNow }
                    executionPlans <- Map.add plan.Id updatedPlan executionPlans
                    
                    logger.LogInformation("Updated execution plan with ID: {Id}", plan.Id)
                    
                    return updatedPlan
                else
                    logger.LogWarning("Execution plan with ID {Id} not found", plan.Id)
                    return raise (KeyNotFoundException($"Execution plan with ID {plan.Id} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error updating execution plan")
                return raise ex
        }
    
    /// <summary>
    /// Deletes an execution plan.
    /// </summary>
    /// <param name="id">The ID of the execution plan to delete.</param>
    /// <returns>True if the execution plan was deleted, false otherwise.</returns>
    member _.DeleteExecutionPlan(id: string) =
        task {
            try
                logger.LogInformation("Deleting execution plan with ID: {Id}", id)
                
                // Check if the plan exists
                if Map.containsKey id executionPlans then
                    // Remove the plan
                    executionPlans <- Map.remove id executionPlans
                    
                    // Remove the context if it exists
                    if Map.containsKey id executionContexts then
                        executionContexts <- Map.remove id executionContexts
                    
                    logger.LogInformation("Deleted execution plan with ID: {Id}", id)
                    
                    return true
                else
                    logger.LogWarning("Execution plan with ID {Id} not found", id)
                    return false
            with
            | ex ->
                logger.LogError(ex, "Error deleting execution plan")
                return false
        }
    
    /// <summary>
    /// Executes a step in an execution plan.
    /// </summary>
    /// <param name="planId">The ID of the execution plan.</param>
    /// <param name="stepId">The ID of the step to execute.</param>
    /// <returns>The result of the execution step.</returns>
    member _.ExecuteStep(planId: string, stepId: string) =
        task {
            try
                logger.LogInformation("Executing step {StepId} in plan {PlanId}", stepId, planId)
                
                // Check if the plan exists
                match Map.tryFind planId executionPlans with
                | Some plan ->
                    // Find the step
                    let stepOption = plan.Steps |> List.tryFind (fun s -> s.Id = stepId)
                    
                    match stepOption with
                    | Some step ->
                        // Check if the step is ready to execute
                        let completedStepIds = 
                            plan.Steps 
                            |> List.filter ExecutionStep.isCompleted 
                            |> List.map (fun s -> s.Id) 
                            |> Set.ofList
                        
                        if ExecutionStep.isReadyToExecute completedStepIds step then
                            // Update step status to InProgress
                            let startTime = DateTime.UtcNow
                            let updatedStep = { step with Status = ExecutionStepStatus.InProgress; StartedAt = Some startTime }
                            
                            // Update the plan with the updated step
                            let updatedSteps = 
                                plan.Steps 
                                |> List.map (fun s -> if s.Id = stepId then updatedStep else s)
                            
                            let updatedPlan = { plan with Steps = updatedSteps; UpdatedAt = Some DateTime.UtcNow }
                            executionPlans <- Map.add planId updatedPlan executionPlans
                            
                            // Get or create execution context
                            let context = 
                                match Map.tryFind planId executionContexts with
                                | Some ctx -> ctx
                                | None -> 
                                    let newContext = ExecutionContext.create planId plan.ImprovementId plan.MetascriptId
                                    executionContexts <- Map.add planId newContext executionContexts
                                    newContext
                            
                            // Execute the step action (placeholder for actual execution)
                            // In a real implementation, this would dispatch to different execution methods based on step.Type
                            logger.LogInformation("Executing action: {Action}", step.Action)
                            
                            // Simulate execution
                            let isSuccessful = true // In a real implementation, this would be the result of the action
                            let output = $"Executed {step.Action} successfully" // In a real implementation, this would be the output of the action
                            
                            // Create step result
                            let completionTime = DateTime.UtcNow
                            let durationMs = (completionTime - startTime).TotalMilliseconds |> int64
                            
                            let stepResult = {
                                ExecutionStepId = stepId
                                Status = if isSuccessful then ExecutionStepStatus.Completed else ExecutionStepStatus.Failed
                                IsSuccessful = isSuccessful
                                StartedAt = Some startTime
                                CompletedAt = Some completionTime
                                DurationMs = Some durationMs
                                Output = output
                                Error = if isSuccessful then "" else "Error executing step"
                                Exception = None
                            }
                            
                            // Update step with result
                            let finalStep = { 
                                updatedStep with 
                                    Status = if isSuccessful then ExecutionStepStatus.Completed else ExecutionStepStatus.Failed
                                    Result = Some stepResult
                                    CompletedAt = Some completionTime
                                    DurationMs = Some durationMs
                            }
                            
                            // Update the plan with the final step
                            let finalSteps = 
                                updatedPlan.Steps 
                                |> List.map (fun s -> if s.Id = stepId then finalStep else s)
                            
                            let finalPlan = { updatedPlan with Steps = finalSteps; UpdatedAt = Some DateTime.UtcNow }
                            executionPlans <- Map.add planId finalPlan executionPlans
                            
                            logger.LogInformation("Executed step {StepId} in plan {PlanId}: {Status}", 
                                stepId, planId, if isSuccessful then "Success" else "Failure")
                            
                            return Some stepResult
                        else
                            logger.LogWarning("Step {StepId} in plan {PlanId} is not ready to execute", stepId, planId)
                            return None
                    | None ->
                        logger.LogWarning("Step {StepId} not found in plan {PlanId}", stepId, planId)
                        return None
                | None ->
                    logger.LogWarning("Execution plan with ID {PlanId} not found", planId)
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error executing step {StepId} in plan {PlanId}", stepId, planId)
                return None
        }
    
    /// <summary>
    /// Executes an execution plan.
    /// </summary>
    /// <param name="id">The ID of the execution plan.</param>
    /// <returns>The result of the execution plan.</returns>
    member this.ExecutePlan(id: string) =
        task {
            try
                logger.LogInformation("Executing plan with ID: {Id}", id)
                
                // Check if the plan exists
                match Map.tryFind id executionPlans with
                | Some plan ->
                    // Create or get execution context
                    let context = 
                        match Map.tryFind id executionContexts with
                        | Some ctx -> ctx
                        | None -> 
                            let newContext = ExecutionContext.create id plan.ImprovementId plan.MetascriptId
                            executionContexts <- Map.add id newContext executionContexts
                            newContext
                    
                    // Update plan status to InProgress
                    let startTime = DateTime.UtcNow
                    let updatedPlan = { 
                        plan with 
                            Status = ExecutionPlanStatus.InProgress
                            UpdatedAt = Some startTime
                            ExecutedAt = Some startTime
                            Context = Some context
                    }
                    
                    executionPlans <- Map.add id updatedPlan executionPlans
                    
                    // Execute steps in order
                    let mutable currentPlan = updatedPlan
                    let mutable isSuccessful = true
                    let mutable error = ""
                    let mutable stepResults = Map.empty<string, ExecutionStepResult>
                    
                    // Execute steps in dependency order
                    let rec executeSteps () =
                        task {
                            // Find steps that are ready to execute
                            let completedStepIds = 
                                currentPlan.Steps 
                                |> List.filter ExecutionStep.isCompleted 
                                |> List.map (fun s -> s.Id) 
                                |> Set.ofList
                            
                            let readySteps = 
                                currentPlan.Steps 
                                |> List.filter (fun s -> s.Status = ExecutionStepStatus.Pending && ExecutionStep.isReadyToExecute completedStepIds s)
                            
                            if not (List.isEmpty readySteps) then
                                // Execute the first ready step
                                let step = readySteps.[0]
                                
                                let! stepResultOption = this.ExecuteStep(id, step.Id)
                                
                                match stepResultOption with
                                | Some stepResult ->
                                    // Add step result to results map
                                    stepResults <- Map.add step.Id stepResult stepResults
                                    
                                    // Update current plan
                                    currentPlan <- Map.find id executionPlans
                                    
                                    // Check if step was successful
                                    if not stepResult.IsSuccessful then
                                        isSuccessful <- false
                                        error <- $"Step {step.Id} failed: {stepResult.Error}"
                                        return ()
                                    else
                                        // Continue executing steps
                                        return! executeSteps()
                                | None ->
                                    isSuccessful <- false
                                    error <- $"Failed to execute step {step.Id}"
                                    return ()
                            else
                                // Check if all steps are completed
                                let allStepsCompleted = 
                                    currentPlan.Steps 
                                    |> List.forall (fun s -> 
                                        match s.Status with
                                        | ExecutionStepStatus.Completed
                                        | ExecutionStepStatus.Failed
                                        | ExecutionStepStatus.Skipped -> true
                                        | _ -> false)
                                
                                if allStepsCompleted then
                                    return ()
                                else
                                    // Check if there are any steps that can't be executed
                                    let blockedSteps = 
                                        currentPlan.Steps 
                                        |> List.filter (fun s -> 
                                            s.Status = ExecutionStepStatus.Pending && 
                                            not (ExecutionStep.isReadyToExecute completedStepIds s))
                                    
                                    if not (List.isEmpty blockedSteps) then
                                        isSuccessful <- false
                                        error <- "Execution plan has blocked steps that cannot be executed"
                                    
                                    return ()
                        }
                    
                    // Start executing steps
                    do! executeSteps()
                    
                    // Update plan status
                    let completionTime = DateTime.UtcNow
                    let durationMs = (completionTime - startTime).TotalMilliseconds |> int64
                    
                    let planResult = {
                        ExecutionPlanId = id
                        Status = if isSuccessful then ExecutionPlanStatus.Completed else ExecutionPlanStatus.Failed
                        IsSuccessful = isSuccessful
                        StartedAt = Some startTime
                        CompletedAt = Some completionTime
                        DurationMs = Some durationMs
                        Output = if isSuccessful then "Execution plan completed successfully" else $"Execution plan failed: {error}"
                        Error = if isSuccessful then "" else error
                        Exception = None
                        StepResults = stepResults
                        Metrics = Map.empty
                    }
                    
                    // Update the plan with the result
                    let finalPlan = { 
                        currentPlan with 
                            Status = if isSuccessful then ExecutionPlanStatus.Completed else ExecutionPlanStatus.Failed
                            Result = Some planResult
                            UpdatedAt = Some completionTime
                    }
                    
                    executionPlans <- Map.add id finalPlan executionPlans
                    
                    logger.LogInformation("Executed plan {PlanId}: {Status}", 
                        id, if isSuccessful then "Success" else "Failure")
                    
                    return Some planResult
                | None ->
                    logger.LogWarning("Execution plan with ID {Id} not found", id)
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error executing plan {PlanId}", id)
                return None
        }
    
    /// <summary>
    /// Validates an execution plan.
    /// </summary>
    /// <param name="id">The ID of the execution plan.</param>
    /// <returns>True if the execution plan is valid, false otherwise.</returns>
    member _.ValidatePlan(id: string) =
        task {
            try
                logger.LogInformation("Validating plan with ID: {Id}", id)
                
                // Check if the plan exists
                match Map.tryFind id executionPlans with
                | Some plan ->
                    // Validate the plan
                    let isValid = ExecutionPlan.validate plan
                    
                    logger.LogInformation("Validated plan {PlanId}: {IsValid}", id, isValid)
                    
                    return isValid
                | None ->
                    logger.LogWarning("Execution plan with ID {Id} not found", id)
                    return false
            with
            | ex ->
                logger.LogError(ex, "Error validating plan {PlanId}", id)
                return false
        }
    
    /// <summary>
    /// Monitors an execution plan.
    /// </summary>
    /// <param name="id">The ID of the execution plan.</param>
    /// <returns>The current status and progress of the execution plan.</returns>
    member _.MonitorPlan(id: string) =
        task {
            try
                logger.LogInformation("Monitoring plan with ID: {Id}", id)
                
                // Check if the plan exists
                match Map.tryFind id executionPlans with
                | Some plan ->
                    // Get the current status and progress
                    let status = plan.Status
                    let progress = ExecutionPlan.progress plan
                    
                    logger.LogInformation("Monitored plan {PlanId}: Status={Status}, Progress={Progress:P2}", 
                        id, status, progress)
                    
                    return Some (status, progress)
                | None ->
                    logger.LogWarning("Execution plan with ID {Id} not found", id)
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error monitoring plan {PlanId}", id)
                return None
        }
    
    /// <summary>
    /// Adapts an execution plan based on feedback or changes.
    /// </summary>
    /// <param name="id">The ID of the execution plan.</param>
    /// <param name="adaptationReason">The reason for adaptation.</param>
    /// <returns>The adapted execution plan.</returns>
    member _.AdaptPlan(id: string, adaptationReason: string) =
        task {
            try
                logger.LogInformation("Adapting plan with ID: {Id}, Reason: {Reason}", id, adaptationReason)
                
                // Check if the plan exists
                match Map.tryFind id executionPlans with
                | Some plan ->
                    // In a real implementation, this would analyze the plan and make adaptations
                    // For now, we'll just add a metadata entry
                    let metadata = 
                        plan.Metadata 
                        |> Map.add "AdaptationReason" adaptationReason
                        |> Map.add "AdaptationTime" (DateTime.UtcNow.ToString("o"))
                    
                    // Create adapted plan
                    let adaptedPlan = { 
                        plan with 
                            Metadata = metadata
                            UpdatedAt = Some DateTime.UtcNow
                    }
                    
                    // Update the plan
                    executionPlans <- Map.add id adaptedPlan executionPlans
                    
                    logger.LogInformation("Adapted plan {PlanId}", id)
                    
                    return Some adaptedPlan
                | None ->
                    logger.LogWarning("Execution plan with ID {Id} not found", id)
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error adapting plan {PlanId}", id)
                return None
        }
    
    /// <summary>
    /// Rolls back a step in an execution plan.
    /// </summary>
    /// <param name="planId">The ID of the execution plan.</param>
    /// <param name="stepId">The ID of the step to roll back.</param>
    /// <returns>True if the step was rolled back, false otherwise.</returns>
    member _.RollbackStep(planId: string, stepId: string) =
        task {
            try
                logger.LogInformation("Rolling back step {StepId} in plan {PlanId}", stepId, planId)
                
                // Check if the plan exists
                match Map.tryFind planId executionPlans with
                | Some plan ->
                    // Find the step
                    let stepOption = plan.Steps |> List.tryFind (fun s -> s.Id = stepId)
                    
                    match stepOption with
                    | Some step ->
                        // Check if the step can be rolled back
                        if step.Status = ExecutionStepStatus.Completed || step.Status = ExecutionStepStatus.Failed then
                            // In a real implementation, this would execute the rollback action
                            logger.LogInformation("Executing rollback action: {Action}", step.RollbackAction)
                            
                            // Simulate rollback
                            let isSuccessful = true // In a real implementation, this would be the result of the rollback action
                            
                            if isSuccessful then
                                // Update step status to RolledBack
                                let updatedStep = { step with Status = ExecutionStepStatus.RolledBack }
                                
                                // Update the plan with the updated step
                                let updatedSteps = 
                                    plan.Steps 
                                    |> List.map (fun s -> if s.Id = stepId then updatedStep else s)
                                
                                let updatedPlan = { plan with Steps = updatedSteps; UpdatedAt = Some DateTime.UtcNow }
                                executionPlans <- Map.add planId updatedPlan executionPlans
                                
                                logger.LogInformation("Rolled back step {StepId} in plan {PlanId}", stepId, planId)
                                
                                return true
                            else
                                logger.LogWarning("Failed to roll back step {StepId} in plan {PlanId}", stepId, planId)
                                return false
                        else
                            logger.LogWarning("Step {StepId} in plan {PlanId} cannot be rolled back (Status: {Status})", 
                                stepId, planId, step.Status)
                            return false
                    | None ->
                        logger.LogWarning("Step {StepId} not found in plan {PlanId}", stepId, planId)
                        return false
                | None ->
                    logger.LogWarning("Execution plan with ID {PlanId} not found", planId)
                    return false
            with
            | ex ->
                logger.LogError(ex, "Error rolling back step {StepId} in plan {PlanId}", stepId, planId)
                return false
        }
    
    /// <summary>
    /// Rolls back an entire execution plan.
    /// </summary>
    /// <param name="id">The ID of the execution plan.</param>
    /// <returns>True if the plan was rolled back, false otherwise.</returns>
    member this.RollbackPlan(id: string) =
        task {
            try
                logger.LogInformation("Rolling back plan with ID: {Id}", id)
                
                // Check if the plan exists
                match Map.tryFind id executionPlans with
                | Some plan ->
                    // Get completed steps in reverse order
                    let completedSteps = 
                        plan.Steps 
                        |> List.filter (fun s -> 
                            s.Status = ExecutionStepStatus.Completed || 
                            s.Status = ExecutionStepStatus.Failed)
                        |> List.sortByDescending (fun s -> s.Order)
                    
                    // Roll back each step
                    let mutable allSuccessful = true
                    
                    for step in completedSteps do
                        let! stepRollbackSuccess = this.RollbackStep(id, step.Id)
                        
                        if not stepRollbackSuccess then
                            allSuccessful <- false
                    
                    // Update plan status
                    if allSuccessful then
                        let updatedPlan = { 
                            plan with 
                                Status = ExecutionPlanStatus.Cancelled
                                UpdatedAt = Some DateTime.UtcNow
                        }
                        
                        executionPlans <- Map.add id updatedPlan executionPlans
                        
                        logger.LogInformation("Rolled back plan {PlanId}", id)
                    else
                        logger.LogWarning("Partially rolled back plan {PlanId}", id)
                    
                    return allSuccessful
                | None ->
                    logger.LogWarning("Execution plan with ID {Id} not found", id)
                    return false
            with
            | ex ->
                logger.LogError(ex, "Error rolling back plan {PlanId}", id)
                return false
        }
    
    /// <summary>
    /// Gets the execution context for an execution plan.
    /// </summary>
    /// <param name="planId">The ID of the execution plan.</param>
    /// <returns>The execution context, or None if not found.</returns>
    member _.GetExecutionContext(planId: string) =
        task {
            try
                logger.LogInformation("Getting execution context for plan: {PlanId}", planId)
                
                // Check if the context exists
                match Map.tryFind planId executionContexts with
                | Some context ->
                    logger.LogInformation("Found execution context for plan: {PlanId}", planId)
                    return Some context
                | None ->
                    logger.LogWarning("Execution context for plan {PlanId} not found", planId)
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error getting execution context for plan {PlanId}", planId)
                return None
        }
    
    /// <summary>
    /// Updates the execution context for an execution plan.
    /// </summary>
    /// <param name="planId">The ID of the execution plan.</param>
    /// <param name="context">The updated execution context.</param>
    /// <returns>The updated execution context.</returns>
    member _.UpdateExecutionContext(planId: string, context: ExecutionContext) =
        task {
            try
                logger.LogInformation("Updating execution context for plan: {PlanId}", planId)
                
                // Check if the plan exists
                if Map.containsKey planId executionPlans then
                    // Update the context
                    let updatedContext = { context with UpdatedAt = Some DateTime.UtcNow }
                    executionContexts <- Map.add planId updatedContext executionContexts
                    
                    // Update the plan with the context
                    let plan = Map.find planId executionPlans
                    let updatedPlan = { plan with Context = Some updatedContext; UpdatedAt = Some DateTime.UtcNow }
                    executionPlans <- Map.add planId updatedPlan executionPlans
                    
                    logger.LogInformation("Updated execution context for plan: {PlanId}", planId)
                    
                    return updatedContext
                else
                    logger.LogWarning("Execution plan with ID {PlanId} not found", planId)
                    return raise (KeyNotFoundException($"Execution plan with ID {planId} not found"))
            with
            | ex ->
                logger.LogError(ex, "Error updating execution context for plan {PlanId}", planId)
                return raise ex
        }
    
    interface IPlanningService with
        member this.CreateExecutionPlan(name, description, improvementId, metascriptId) = 
            this.CreateExecutionPlan(name, description, improvementId, metascriptId)
        
        member this.GetExecutionPlan(id) = 
            this.GetExecutionPlan(id)
        
        member this.GetAllExecutionPlans() = 
            this.GetAllExecutionPlans()
        
        member this.UpdateExecutionPlan(plan) = 
            this.UpdateExecutionPlan(plan)
        
        member this.DeleteExecutionPlan(id) = 
            this.DeleteExecutionPlan(id)
        
        member this.ExecuteStep(planId, stepId) = 
            this.ExecuteStep(planId, stepId)
        
        member this.ExecutePlan(id) = 
            this.ExecutePlan(id)
        
        member this.ValidatePlan(id) = 
            this.ValidatePlan(id)
        
        member this.MonitorPlan(id) = 
            this.MonitorPlan(id)
        
        member this.AdaptPlan(id, adaptationReason) = 
            this.AdaptPlan(id, adaptationReason)
        
        member this.RollbackStep(planId, stepId) = 
            this.RollbackStep(planId, stepId)
        
        member this.RollbackPlan(id) = 
            this.RollbackPlan(id)
        
        member this.GetExecutionContext(planId) = 
            this.GetExecutionContext(planId)
        
        member this.UpdateExecutionContext(planId, context) = 
            this.UpdateExecutionContext(planId, context)
