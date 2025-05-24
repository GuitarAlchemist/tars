namespace TarsEngine.FSharp.Core.Consciousness.Planning

open System

/// <summary>
/// Represents the type of an execution step.
/// </summary>
type ExecutionStepType =
    | FileModification
    | FileCreation
    | FileDeletion
    | FileBackup
    | FileRestore
    | Validation
    | Compilation
    | TestExecution
    | CommandExecution
    | ApiCall
    | DatabaseOperation
    | Notification
    | Approval
    | Deployment
    | Rollback
    | Other

/// <summary>
/// Represents the status of an execution step.
/// </summary>
type ExecutionStepStatus =
    | Pending
    | InProgress
    | Completed
    | Failed
    | Skipped
    | Cancelled
    | RolledBack

/// <summary>
/// Represents the mode of execution.
/// </summary>
type ExecutionMode =
    | DryRun
    | Live

/// <summary>
/// Represents the environment for execution.
/// </summary>
type ExecutionEnvironment =
    | Sandbox
    | Development
    | Testing
    | Production

/// <summary>
/// Represents the status of an execution plan.
/// </summary>
type ExecutionPlanStatus =
    | Created
    | Scheduled
    | InProgress
    | Paused
    | Completed
    | Failed
    | Cancelled

/// <summary>
/// Represents the log level.
/// </summary>
type LogLevel =
    | Trace
    | Debug
    | Information
    | Warning
    | Error
    | Critical

/// <summary>
/// Represents a validation rule.
/// </summary>
type ValidationRule = {
    /// <summary>
    /// The name of the validation rule.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the validation rule.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The validation expression.
    /// </summary>
    Expression: string
    
    /// <summary>
    /// The error message to display if validation fails.
    /// </summary>
    ErrorMessage: string
    
    /// <summary>
    /// Whether the validation rule is required.
    /// </summary>
    IsRequired: bool
}

/// <summary>
/// Represents the result of an execution step.
/// </summary>
type ExecutionStepResult = {
    /// <summary>
    /// The ID of the execution step.
    /// </summary>
    ExecutionStepId: string
    
    /// <summary>
    /// The status of the execution step.
    /// </summary>
    Status: ExecutionStepStatus
    
    /// <summary>
    /// Whether the execution step was successful.
    /// </summary>
    IsSuccessful: bool
    
    /// <summary>
    /// The timestamp when the execution step was started.
    /// </summary>
    StartedAt: DateTime option
    
    /// <summary>
    /// The timestamp when the execution step was completed.
    /// </summary>
    CompletedAt: DateTime option
    
    /// <summary>
    /// The duration of the execution step in milliseconds.
    /// </summary>
    DurationMs: int64 option
    
    /// <summary>
    /// The output of the execution step.
    /// </summary>
    Output: string
    
    /// <summary>
    /// The error message of the execution step.
    /// </summary>
    Error: string
    
    /// <summary>
    /// The exception that caused the execution step to fail.
    /// </summary>
    Exception: exn option
}

/// <summary>
/// Represents a step in an execution plan.
/// </summary>
type ExecutionStep = {
    /// <summary>
    /// The ID of the execution step.
    /// </summary>
    Id: string
    
    /// <summary>
    /// The name of the execution step.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the execution step.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The type of the execution step.
    /// </summary>
    Type: ExecutionStepType
    
    /// <summary>
    /// The order of the execution step.
    /// </summary>
    Order: int
    
    /// <summary>
    /// The dependencies of the execution step.
    /// </summary>
    Dependencies: string list
    
    /// <summary>
    /// The status of the execution step.
    /// </summary>
    Status: ExecutionStepStatus
    
    /// <summary>
    /// The result of the execution step.
    /// </summary>
    Result: ExecutionStepResult option
    
    /// <summary>
    /// The timestamp when the execution step was started.
    /// </summary>
    StartedAt: DateTime option
    
    /// <summary>
    /// The timestamp when the execution step was completed.
    /// </summary>
    CompletedAt: DateTime option
    
    /// <summary>
    /// The duration of the execution step in milliseconds.
    /// </summary>
    DurationMs: int64 option
    
    /// <summary>
    /// The action to execute.
    /// </summary>
    Action: string
    
    /// <summary>
    /// The parameters for the action.
    /// </summary>
    Parameters: Map<string, string>
    
    /// <summary>
    /// The validation rules for the execution step.
    /// </summary>
    ValidationRules: ValidationRule list
    
    /// <summary>
    /// The rollback action.
    /// </summary>
    RollbackAction: string
    
    /// <summary>
    /// The parameters for the rollback action.
    /// </summary>
    RollbackParameters: Map<string, string>
    
    /// <summary>
    /// Additional metadata about the execution step.
    /// </summary>
    Metadata: Map<string, string>
}

/// <summary>
/// Represents a log entry in the execution context.
/// </summary>
type ExecutionLog = {
    /// <summary>
    /// The timestamp of the log entry.
    /// </summary>
    Timestamp: DateTime
    
    /// <summary>
    /// The level of the log entry.
    /// </summary>
    Level: LogLevel
    
    /// <summary>
    /// The message of the log entry.
    /// </summary>
    Message: string
    
    /// <summary>
    /// The source of the log entry.
    /// </summary>
    Source: string
}

/// <summary>
/// Represents an error in the execution context.
/// </summary>
type ExecutionError = {
    /// <summary>
    /// The timestamp of the error.
    /// </summary>
    Timestamp: DateTime
    
    /// <summary>
    /// The message of the error.
    /// </summary>
    Message: string
    
    /// <summary>
    /// The source of the error.
    /// </summary>
    Source: string
    
    /// <summary>
    /// The exception that caused the error.
    /// </summary>
    Exception: exn option
}

/// <summary>
/// Represents a warning in the execution context.
/// </summary>
type ExecutionWarning = {
    /// <summary>
    /// The timestamp of the warning.
    /// </summary>
    Timestamp: DateTime
    
    /// <summary>
    /// The message of the warning.
    /// </summary>
    Message: string
    
    /// <summary>
    /// The source of the warning.
    /// </summary>
    Source: string
}

/// <summary>
/// Represents the permissions for execution.
/// </summary>
type ExecutionPermissions = {
    /// <summary>
    /// Whether file system operations are allowed.
    /// </summary>
    AllowFileSystem: bool
    
    /// <summary>
    /// Whether network operations are allowed.
    /// </summary>
    AllowNetwork: bool
    
    /// <summary>
    /// Whether database operations are allowed.
    /// </summary>
    AllowDatabase: bool
    
    /// <summary>
    /// Whether process execution is allowed.
    /// </summary>
    AllowProcessExecution: bool
    
    /// <summary>
    /// Whether code compilation is allowed.
    /// </summary>
    AllowCompilation: bool
    
    /// <summary>
    /// Whether code execution is allowed.
    /// </summary>
    AllowCodeExecution: bool
    
    /// <summary>
    /// Whether notifications are allowed.
    /// </summary>
    AllowNotifications: bool
    
    /// <summary>
    /// Whether deployments are allowed.
    /// </summary>
    AllowDeployment: bool
}

/// <summary>
/// Represents the context for executing an improvement.
/// </summary>
type ExecutionContext = {
    /// <summary>
    /// The ID of the execution context.
    /// </summary>
    Id: string
    
    /// <summary>
    /// The ID of the execution plan.
    /// </summary>
    ExecutionPlanId: string
    
    /// <summary>
    /// The ID of the improvement.
    /// </summary>
    ImprovementId: string
    
    /// <summary>
    /// The ID of the metascript.
    /// </summary>
    MetascriptId: string
    
    /// <summary>
    /// The timestamp when the execution context was created.
    /// </summary>
    CreatedAt: DateTime
    
    /// <summary>
    /// The timestamp when the execution context was last updated.
    /// </summary>
    UpdatedAt: DateTime option
    
    /// <summary>
    /// The execution mode.
    /// </summary>
    Mode: ExecutionMode
    
    /// <summary>
    /// The execution environment.
    /// </summary>
    Environment: ExecutionEnvironment
    
    /// <summary>
    /// The execution timeout in milliseconds.
    /// </summary>
    TimeoutMs: int
    
    /// <summary>
    /// The execution variables.
    /// </summary>
    Variables: Map<string, obj>
    
    /// <summary>
    /// The execution state.
    /// </summary>
    State: Map<string, obj>
    
    /// <summary>
    /// The execution permissions.
    /// </summary>
    Permissions: ExecutionPermissions
    
    /// <summary>
    /// The execution options.
    /// </summary>
    Options: Map<string, string>
    
    /// <summary>
    /// The execution logs.
    /// </summary>
    Logs: ExecutionLog list
    
    /// <summary>
    /// The execution errors.
    /// </summary>
    Errors: ExecutionError list
    
    /// <summary>
    /// The execution warnings.
    /// </summary>
    Warnings: ExecutionWarning list
    
    /// <summary>
    /// The execution metrics.
    /// </summary>
    Metrics: Map<string, float>
    
    /// <summary>
    /// Additional metadata about the execution context.
    /// </summary>
    Metadata: Map<string, string>
    
    /// <summary>
    /// The working directory for the execution.
    /// </summary>
    WorkingDirectory: string
    
    /// <summary>
    /// The backup directory for the execution.
    /// </summary>
    BackupDirectory: string
    
    /// <summary>
    /// The output directory for the execution.
    /// </summary>
    OutputDirectory: string
    
    /// <summary>
    /// The list of files affected by the execution.
    /// </summary>
    AffectedFiles: string list
    
    /// <summary>
    /// The list of files backed up by the execution.
    /// </summary>
    BackedUpFiles: string list
    
    /// <summary>
    /// The list of files modified by the execution.
    /// </summary>
    ModifiedFiles: string list
    
    /// <summary>
    /// The list of files created by the execution.
    /// </summary>
    CreatedFiles: string list
    
    /// <summary>
    /// The list of files deleted by the execution.
    /// </summary>
    DeletedFiles: string list
}

/// <summary>
/// Represents the result of an execution plan.
/// </summary>
type ExecutionPlanResult = {
    /// <summary>
    /// The ID of the execution plan.
    /// </summary>
    ExecutionPlanId: string
    
    /// <summary>
    /// The status of the execution plan.
    /// </summary>
    Status: ExecutionPlanStatus
    
    /// <summary>
    /// Whether the execution plan was successful.
    /// </summary>
    IsSuccessful: bool
    
    /// <summary>
    /// The timestamp when the execution plan was started.
    /// </summary>
    StartedAt: DateTime option
    
    /// <summary>
    /// The timestamp when the execution plan was completed.
    /// </summary>
    CompletedAt: DateTime option
    
    /// <summary>
    /// The duration of the execution plan in milliseconds.
    /// </summary>
    DurationMs: int64 option
    
    /// <summary>
    /// The output of the execution plan.
    /// </summary>
    Output: string
    
    /// <summary>
    /// The error message of the execution plan.
    /// </summary>
    Error: string
    
    /// <summary>
    /// The exception that caused the execution plan to fail.
    /// </summary>
    Exception: exn option
    
    /// <summary>
    /// The results of the execution steps.
    /// </summary>
    StepResults: Map<string, ExecutionStepResult>
    
    /// <summary>
    /// The metrics of the execution plan.
    /// </summary>
    Metrics: Map<string, float>
}

/// <summary>
/// Represents a plan for executing an improvement.
/// </summary>
type ExecutionPlan = {
    /// <summary>
    /// The ID of the execution plan.
    /// </summary>
    Id: string
    
    /// <summary>
    /// The name of the execution plan.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the execution plan.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The ID of the improvement associated with the execution plan.
    /// </summary>
    ImprovementId: string
    
    /// <summary>
    /// The ID of the metascript associated with the execution plan.
    /// </summary>
    MetascriptId: string
    
    /// <summary>
    /// The steps in the execution plan.
    /// </summary>
    Steps: ExecutionStep list
    
    /// <summary>
    /// The timestamp when the execution plan was created.
    /// </summary>
    CreatedAt: DateTime
    
    /// <summary>
    /// The timestamp when the execution plan was last updated.
    /// </summary>
    UpdatedAt: DateTime option
    
    /// <summary>
    /// The timestamp when the execution plan was last executed.
    /// </summary>
    ExecutedAt: DateTime option
    
    /// <summary>
    /// The status of the execution plan.
    /// </summary>
    Status: ExecutionPlanStatus
    
    /// <summary>
    /// The result of the execution plan.
    /// </summary>
    Result: ExecutionPlanResult option
    
    /// <summary>
    /// The execution context.
    /// </summary>
    Context: ExecutionContext option
    
    /// <summary>
    /// Additional metadata about the execution plan.
    /// </summary>
    Metadata: Map<string, string>
}

/// <summary>
/// Functions for working with execution steps.
/// </summary>
module ExecutionStep =
    /// <summary>
    /// Creates a new execution step with default values.
    /// </summary>
    let create name description = {
        Id = Guid.NewGuid().ToString()
        Name = name
        Description = description
        Type = ExecutionStepType.Other
        Order = 0
        Dependencies = []
        Status = ExecutionStepStatus.Pending
        Result = None
        StartedAt = None
        CompletedAt = None
        DurationMs = None
        Action = ""
        Parameters = Map.empty
        ValidationRules = []
        RollbackAction = ""
        RollbackParameters = Map.empty
        Metadata = Map.empty
    }
    
    /// <summary>
    /// Gets whether the execution step is completed.
    /// </summary>
    let isCompleted step =
        match step.Status with
        | ExecutionStepStatus.Completed
        | ExecutionStepStatus.Failed
        | ExecutionStepStatus.Skipped -> true
        | _ -> false
    
    /// <summary>
    /// Gets whether the execution step is successful.
    /// </summary>
    let isSuccessful step =
        step.Status = ExecutionStepStatus.Completed &&
        (match step.Result with
         | Some result -> result.IsSuccessful
         | None -> false)
    
    /// <summary>
    /// Gets whether the execution step is ready to execute.
    /// </summary>
    let isReadyToExecute completedStepIds step =
        if step.Status <> ExecutionStepStatus.Pending then
            false
        else
            step.Dependencies |> List.forall (fun dependencyId -> Set.contains dependencyId completedStepIds)

/// <summary>
/// Functions for working with execution contexts.
/// </summary>
module ExecutionContext =
    /// <summary>
    /// Creates a new execution context with default values.
    /// </summary>
    let create executionPlanId improvementId metascriptId = {
        Id = Guid.NewGuid().ToString()
        ExecutionPlanId = executionPlanId
        ImprovementId = improvementId
        MetascriptId = metascriptId
        CreatedAt = DateTime.UtcNow
        UpdatedAt = None
        Mode = ExecutionMode.DryRun
        Environment = ExecutionEnvironment.Sandbox
        TimeoutMs = 30000
        Variables = Map.empty
        State = Map.empty
        Permissions = {
            AllowFileSystem = false
            AllowNetwork = false
            AllowDatabase = false
            AllowProcessExecution = false
            AllowCompilation = false
            AllowCodeExecution = false
            AllowNotifications = false
            AllowDeployment = false
        }
        Options = Map.empty
        Logs = []
        Errors = []
        Warnings = []
        Metrics = Map.empty
        Metadata = Map.empty
        WorkingDirectory = ""
        BackupDirectory = ""
        OutputDirectory = ""
        AffectedFiles = []
        BackedUpFiles = []
        ModifiedFiles = []
        CreatedFiles = []
        DeletedFiles = []
    }
    
    /// <summary>
    /// Adds a log entry to the execution context.
    /// </summary>
    let addLog level message source context =
        let log = {
            Timestamp = DateTime.UtcNow
            Level = level
            Message = message
            Source = source
        }
        { context with Logs = log :: context.Logs }
    
    /// <summary>
    /// Adds an error to the execution context.
    /// </summary>
    let addError message source exception context =
        let error = {
            Timestamp = DateTime.UtcNow
            Message = message
            Source = source
            Exception = exception
        }
        let contextWithError = { context with Errors = error :: context.Errors }
        addLog LogLevel.Error message source contextWithError
    
    /// <summary>
    /// Adds a warning to the execution context.
    /// </summary>
    let addWarning message source context =
        let warning = {
            Timestamp = DateTime.UtcNow
            Message = message
            Source = source
        }
        let contextWithWarning = { context with Warnings = warning :: context.Warnings }
        addLog LogLevel.Warning message source contextWithWarning
    
    /// <summary>
    /// Sets a variable in the execution context.
    /// </summary>
    let setVariable name value context =
        { context with Variables = Map.add name value context.Variables }
    
    /// <summary>
    /// Gets a variable from the execution context.
    /// </summary>
    let getVariable<'T> name defaultValue (context: ExecutionContext) =
        match Map.tryFind name context.Variables with
        | Some value when value :? 'T -> value :?> 'T
        | _ -> defaultValue
    
    /// <summary>
    /// Sets a state value in the execution context.
    /// </summary>
    let setState name value context =
        { context with 
            State = Map.add name value context.State
            UpdatedAt = Some DateTime.UtcNow }
    
    /// <summary>
    /// Gets a state value from the execution context.
    /// </summary>
    let getState<'T> name defaultValue (context: ExecutionContext) =
        match Map.tryFind name context.State with
        | Some value when value :? 'T -> value :?> 'T
        | _ -> defaultValue
    
    /// <summary>
    /// Sets a metric in the execution context.
    /// </summary>
    let setMetric name value context =
        { context with Metrics = Map.add name value context.Metrics }
    
    /// <summary>
    /// Gets a metric from the execution context.
    /// </summary>
    let getMetric name defaultValue context =
        Map.tryFind name context.Metrics |> Option.defaultValue defaultValue
    
    /// <summary>
    /// Adds a file to the list of affected files.
    /// </summary>
    let addAffectedFile filePath context =
        if List.contains filePath context.AffectedFiles then
            context
        else
            { context with AffectedFiles = filePath :: context.AffectedFiles }
    
    /// <summary>
    /// Adds a file to the list of backed up files.
    /// </summary>
    let addBackedUpFile filePath context =
        let contextWithBackedUpFile =
            if List.contains filePath context.BackedUpFiles then
                context
            else
                { context with BackedUpFiles = filePath :: context.BackedUpFiles }
        addAffectedFile filePath contextWithBackedUpFile
    
    /// <summary>
    /// Adds a file to the list of modified files.
    /// </summary>
    let addModifiedFile filePath context =
        let contextWithModifiedFile =
            if List.contains filePath context.ModifiedFiles then
                context
            else
                { context with ModifiedFiles = filePath :: context.ModifiedFiles }
        addAffectedFile filePath contextWithModifiedFile
    
    /// <summary>
    /// Adds a file to the list of created files.
    /// </summary>
    let addCreatedFile filePath context =
        let contextWithCreatedFile =
            if List.contains filePath context.CreatedFiles then
                context
            else
                { context with CreatedFiles = filePath :: context.CreatedFiles }
        addAffectedFile filePath contextWithCreatedFile
    
    /// <summary>
    /// Adds a file to the list of deleted files.
    /// </summary>
    let addDeletedFile filePath context =
        let contextWithDeletedFile =
            if List.contains filePath context.DeletedFiles then
                context
            else
                { context with DeletedFiles = filePath :: context.DeletedFiles }
        addAffectedFile filePath contextWithDeletedFile

/// <summary>
/// Functions for working with execution plans.
/// </summary>
module ExecutionPlan =
    /// <summary>
    /// Creates a new execution plan with default values.
    /// </summary>
    let create name description improvementId metascriptId = {
        Id = Guid.NewGuid().ToString()
        Name = name
        Description = description
        ImprovementId = improvementId
        MetascriptId = metascriptId
        Steps = []
        CreatedAt = DateTime.UtcNow
        UpdatedAt = None
        ExecutedAt = None
        Status = ExecutionPlanStatus.Created
        Result = None
        Context = None
        Metadata = Map.empty
    }
    
    /// <summary>
    /// Gets the total number of steps in the execution plan.
    /// </summary>
    let totalSteps plan =
        List.length plan.Steps
    
    /// <summary>
    /// Gets the number of completed steps in the execution plan.
    /// </summary>
    let completedSteps plan =
        plan.Steps
        |> List.filter (fun step -> 
            match step.Status with
            | ExecutionStepStatus.Completed -> true
            | _ -> false)
        |> List.length
    
    /// <summary>
    /// Gets the number of failed steps in the execution plan.
    /// </summary>
    let failedSteps plan =
        plan.Steps
        |> List.filter (fun step -> 
            match step.Status with
            | ExecutionStepStatus.Failed -> true
            | _ -> false)
        |> List.length
    
    /// <summary>
    /// Gets the progress of the execution plan (0.0 to 1.0).
    /// </summary>
    let progress plan =
        let total = totalSteps plan
        if total > 0 then
            float (completedSteps plan) / float total
        else
            0.0
    
    /// <summary>
    /// Gets whether the execution plan is completed.
    /// </summary>
    let isCompleted plan =
        match plan.Status with
        | ExecutionPlanStatus.Completed
        | ExecutionPlanStatus.Failed -> true
        | _ -> false
    
    /// <summary>
    /// Gets whether the execution plan is successful.
    /// </summary>
    let isSuccessful plan =
        plan.Status = ExecutionPlanStatus.Completed &&
        (match plan.Result with
         | Some result -> result.IsSuccessful
         | None -> false)
    
    /// <summary>
    /// Gets the next step to execute.
    /// </summary>
    let getNextStep plan =
        plan.Steps
        |> List.tryFind (fun step -> step.Status = ExecutionStepStatus.Pending)
    
    /// <summary>
    /// Gets the current step being executed.
    /// </summary>
    let getCurrentStep plan =
        plan.Steps
        |> List.tryFind (fun step -> step.Status = ExecutionStepStatus.InProgress)
    
    /// <summary>
    /// Gets the dependencies for a step.
    /// </summary>
    let getDependencies stepId plan =
        let step = plan.Steps |> List.tryFind (fun s -> s.Id = stepId)
        match step with
        | Some s ->
            plan.Steps
            |> List.filter (fun dep -> List.contains dep.Id s.Dependencies)
        | None -> []
    
    /// <summary>
    /// Gets the dependents for a step.
    /// </summary>
    let getDependents stepId plan =
        plan.Steps
        |> List.filter (fun s -> List.contains stepId s.Dependencies)
    
    /// <summary>
    /// Validates the execution plan.
    /// </summary>
    let validate plan =
        // Check if there are any steps
        if List.isEmpty plan.Steps then
            false
        else
            // Check if all dependencies exist
            let allStepIds = plan.Steps |> List.map (fun s -> s.Id) |> Set.ofList
            let allDependenciesExist =
                plan.Steps
                |> List.forall (fun step ->
                    step.Dependencies
                    |> List.forall (fun depId -> Set.contains depId allStepIds))
            
            if not allDependenciesExist then
                false
            else
                // Check for cycles
                let rec hasCycle visited path stepId =
                    if Set.contains stepId path then
                        true
                    elif Set.contains stepId visited then
                        false
                    else
                        let step = plan.Steps |> List.find (fun s -> s.Id = stepId)
                        let newVisited = Set.add stepId visited
                        let newPath = Set.add stepId path
                        step.Dependencies
                        |> List.exists (fun depId -> hasCycle newVisited newPath depId)
                
                not (plan.Steps |> List.exists (fun step -> hasCycle Set.empty Set.empty step.Id))
