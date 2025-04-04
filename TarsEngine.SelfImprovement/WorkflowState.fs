namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

/// <summary>
/// Represents the status of a workflow step
/// </summary>
type StepStatus =
    | NotStarted
    | InProgress
    | Completed
    | Failed
    | Skipped

/// <summary>
/// Represents a step in the autonomous improvement workflow
/// </summary>
type WorkflowStep = {
    /// <summary>
    /// The name of the step
    /// </summary>
    Name: string
    
    /// <summary>
    /// The status of the step
    /// </summary>
    Status: StepStatus
    
    /// <summary>
    /// The start time of the step
    /// </summary>
    StartTime: DateTime option
    
    /// <summary>
    /// The end time of the step
    /// </summary>
    EndTime: DateTime option
    
    /// <summary>
    /// The error message if the step failed
    /// </summary>
    ErrorMessage: string option
    
    /// <summary>
    /// Additional data associated with the step
    /// </summary>
    Data: Map<string, string>
}

/// <summary>
/// Represents the state of the autonomous improvement workflow
/// </summary>
type WorkflowState = {
    /// <summary>
    /// The ID of the workflow run
    /// </summary>
    Id: string
    
    /// <summary>
    /// The name of the workflow
    /// </summary>
    Name: string
    
    /// <summary>
    /// The start time of the workflow
    /// </summary>
    StartTime: DateTime
    
    /// <summary>
    /// The end time of the workflow
    /// </summary>
    EndTime: DateTime option
    
    /// <summary>
    /// The status of the workflow
    /// </summary>
    Status: StepStatus
    
    /// <summary>
    /// The steps in the workflow
    /// </summary>
    Steps: WorkflowStep list
    
    /// <summary>
    /// The current step index
    /// </summary>
    CurrentStepIndex: int option
    
    /// <summary>
    /// The target directories for the workflow
    /// </summary>
    TargetDirectories: string list
    
    /// <summary>
    /// The maximum duration of the workflow in minutes
    /// </summary>
    MaxDurationMinutes: int
    
    /// <summary>
    /// The time when the workflow was last updated
    /// </summary>
    LastUpdated: DateTime
}

/// <summary>
/// Module for managing workflow state
/// </summary>
module WorkflowState =
    /// <summary>
    /// The default file path for storing workflow state
    /// </summary>
    let defaultStatePath = "workflow_state.json"
    
    /// <summary>
    /// Creates a new workflow state
    /// </summary>
    let create name targetDirectories maxDurationMinutes =
        {
            Id = Guid.NewGuid().ToString()
            Name = name
            StartTime = DateTime.UtcNow
            EndTime = None
            Status = InProgress
            Steps = []
            CurrentStepIndex = None
            TargetDirectories = targetDirectories
            MaxDurationMinutes = maxDurationMinutes
            LastUpdated = DateTime.UtcNow
        }
    
    /// <summary>
    /// Creates a new workflow step
    /// </summary>
    let createStep name =
        {
            Name = name
            Status = NotStarted
            StartTime = None
            EndTime = None
            ErrorMessage = None
            Data = Map.empty
        }
    
    /// <summary>
    /// Adds a step to the workflow
    /// </summary>
    let addStep step state =
        { state with 
            Steps = state.Steps @ [step]
            LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Starts a step in the workflow
    /// </summary>
    let startStep stepIndex state =
        if stepIndex < 0 || stepIndex >= state.Steps.Length then
            state
        else
            let steps = state.Steps |> List.mapi (fun i step ->
                if i = stepIndex then
                    { step with 
                        Status = InProgress
                        StartTime = Some DateTime.UtcNow }
                else
                    step)
            
            { state with 
                Steps = steps
                CurrentStepIndex = Some stepIndex
                LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Completes a step in the workflow
    /// </summary>
    let completeStep stepIndex data state =
        if stepIndex < 0 || stepIndex >= state.Steps.Length then
            state
        else
            let steps = state.Steps |> List.mapi (fun i step ->
                if i = stepIndex then
                    { step with 
                        Status = Completed
                        EndTime = Some DateTime.UtcNow
                        Data = data }
                else
                    step)
            
            { state with 
                Steps = steps
                LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Fails a step in the workflow
    /// </summary>
    let failStep stepIndex errorMessage state =
        if stepIndex < 0 || stepIndex >= state.Steps.Length then
            state
        else
            let steps = state.Steps |> List.mapi (fun i step ->
                if i = stepIndex then
                    { step with 
                        Status = Failed
                        EndTime = Some DateTime.UtcNow
                        ErrorMessage = Some errorMessage }
                else
                    step)
            
            { state with 
                Steps = steps
                LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Skips a step in the workflow
    /// </summary>
    let skipStep stepIndex reason state =
        if stepIndex < 0 || stepIndex >= state.Steps.Length then
            state
        else
            let steps = state.Steps |> List.mapi (fun i step ->
                if i = stepIndex then
                    { step with 
                        Status = Skipped
                        EndTime = Some DateTime.UtcNow
                        Data = step.Data.Add("skip_reason", reason) }
                else
                    step)
            
            { state with 
                Steps = steps
                LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Completes the workflow
    /// </summary>
    let complete state =
        { state with 
            Status = Completed
            EndTime = Some DateTime.UtcNow
            LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Fails the workflow
    /// </summary>
    let fail state =
        { state with 
            Status = Failed
            EndTime = Some DateTime.UtcNow
            LastUpdated = DateTime.UtcNow }
    
    /// <summary>
    /// Checks if the workflow has exceeded its maximum duration
    /// </summary>
    let hasExceededMaxDuration state =
        let duration = DateTime.UtcNow - state.StartTime
        duration.TotalMinutes > float state.MaxDurationMinutes
    
    /// <summary>
    /// Gets the current step of the workflow
    /// </summary>
    let getCurrentStep state =
        match state.CurrentStepIndex with
        | Some index when index >= 0 && index < state.Steps.Length ->
            Some state.Steps.[index]
        | _ ->
            None
    
    /// <summary>
    /// Gets the next step of the workflow
    /// </summary>
    let getNextStep state =
        match state.CurrentStepIndex with
        | Some index when index >= 0 && index < state.Steps.Length - 1 ->
            Some (index + 1, state.Steps.[index + 1])
        | _ ->
            None
    
    /// <summary>
    /// Moves to the next step of the workflow
    /// </summary>
    let moveToNextStep state =
        match getNextStep state with
        | Some (nextIndex, _) ->
            startStep nextIndex state
        | None ->
            state
    
    /// <summary>
    /// Saves the workflow state to a file
    /// </summary>
    let save state (path: string) =
        task {
            let options = JsonSerializerOptions()
            options.WriteIndented <- true
            options.Converters.Add(JsonFSharpConverter())
            
            let json = JsonSerializer.Serialize(state, options)
            do! File.WriteAllTextAsync(path, json)
            return state
        }
    
    /// <summary>
    /// Loads the workflow state from a file
    /// </summary>
    let load (path: string) =
        task {
            if File.Exists(path) then
                let! json = File.ReadAllTextAsync(path)
                let options = JsonSerializerOptions()
                options.Converters.Add(JsonFSharpConverter())
                
                return JsonSerializer.Deserialize<WorkflowState>(json, options)
            else
                return! Task.FromException<WorkflowState>(FileNotFoundException(path))
        }
    
    /// <summary>
    /// Tries to load the workflow state from a file
    /// </summary>
    let tryLoad (path: string) =
        task {
            try
                if File.Exists(path) then
                    let! json = File.ReadAllTextAsync(path)
                    let options = JsonSerializerOptions()
                    options.Converters.Add(JsonFSharpConverter())
                    
                    return Some (JsonSerializer.Deserialize<WorkflowState>(json, options))
                else
                    return None
            with
            | _ -> return None
        }
