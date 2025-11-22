namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks
open FSharp.SystemTextJson

/// <summary>
/// Status of a workflow or step
/// </summary>
type StepStatus =
    | NotStarted
    | InProgress
    | Completed
    | Failed
    | Skipped

/// <summary>
/// Represents the result of a workflow step
/// </summary>
type StepResult = Result<Map<string, string>, string>

/// <summary>
/// Represents a step in the workflow
/// </summary>
type WorkflowStep = {
    Name: string
    Status: StepStatus
    StartTime: DateTime option
    EndTime: DateTime option
    ErrorMessage: string option
    Data: Map<string, string>
}

/// <summary>
/// Represents the state of a workflow
/// </summary>
type WorkflowState = {
    Id: string
    Name: string
    StartTime: DateTime
    EndTime: DateTime option
    Status: StepStatus
    Steps: WorkflowStep list
    CurrentStepIndex: int option
    TargetDirectories: string list
    MaxDurationMinutes: int
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
    /// Creates a new workflow step
    /// </summary>
    let createStep (name: string) =
        {
            Name = name
            Status = StepStatus.NotStarted
            StartTime = None
            EndTime = None
            ErrorMessage = None
            Data = Map.empty
        }

    /// <summary>
    /// Creates a new workflow state
    /// </summary>
    let create (name: string) (targetDirectories: string list) (maxDurationMinutes: int) =
        {
            Id = Guid.NewGuid().ToString()
            Name = name
            StartTime = DateTime.UtcNow
            EndTime = None
            Status = StepStatus.NotStarted
            Steps = []
            CurrentStepIndex = None
            TargetDirectories = targetDirectories
            MaxDurationMinutes = maxDurationMinutes
            LastUpdated = DateTime.UtcNow
        }

    /// <summary>
    /// Adds a step to the workflow
    /// </summary>
    let addStep (step: WorkflowStep) (state: WorkflowState) =
        { state with 
            Steps = state.Steps @ [step]
            LastUpdated = DateTime.UtcNow }

    /// <summary>
    /// Starts a step in the workflow
    /// </summary>
    let startStep (stepIndex: int) (state: WorkflowState) =
        if stepIndex < 0 || stepIndex >= state.Steps.Length then
            state
        else
            let steps = state.Steps |> List.mapi (fun i step ->
                if i = stepIndex then
                    { step with 
                        Status = StepStatus.InProgress
                        StartTime = Some DateTime.UtcNow }
                else
                    step)

            { state with 
                Steps = steps
                Status = StepStatus.InProgress
                CurrentStepIndex = Some stepIndex
                LastUpdated = DateTime.UtcNow }

    /// <summary>
    /// Completes a step in the workflow
    /// </summary>
    let completeStep (stepIndex: int) (data: Map<string, string>) (state: WorkflowState) =
        if stepIndex < 0 || stepIndex >= state.Steps.Length then
            state
        else
            let steps = state.Steps |> List.mapi (fun i step ->
                if i = stepIndex then
                    { step with 
                        Status = StepStatus.Completed
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
    let failStep (stepIndex: int) (errorMessage: string) (state: WorkflowState) =
        if stepIndex < 0 || stepIndex >= state.Steps.Length then
            state
        else
            let steps = state.Steps |> List.mapi (fun i step ->
                if i = stepIndex then
                    { step with 
                        Status = StepStatus.Failed
                        EndTime = Some DateTime.UtcNow
                        ErrorMessage = Some errorMessage }
                else
                    step)

            { state with 
                Steps = steps
                LastUpdated = DateTime.UtcNow }

    /// <summary>
    /// Completes the workflow
    /// </summary>
    let complete (state: WorkflowState) =
        { state with 
            Status = StepStatus.Completed
            EndTime = Some DateTime.UtcNow
            LastUpdated = DateTime.UtcNow }

    /// <summary>
    /// Fails the workflow
    /// </summary>
    let fail (state: WorkflowState) =
        { state with 
            Status = StepStatus.Failed
            EndTime = Some DateTime.UtcNow
            LastUpdated = DateTime.UtcNow }

    /// <summary>
    /// Checks if the workflow has exceeded its maximum duration
    /// </summary>
    let hasExceededMaxDuration (state: WorkflowState) =
        let duration = DateTime.UtcNow - state.StartTime
        duration.TotalMinutes > float state.MaxDurationMinutes

    /// <summary>
    /// Saves the workflow state to a file
    /// </summary>
    let save (state: WorkflowState) (filePath: string) =
        task {
            try
                // Create the directory if it doesn't exist
                let directory = Path.GetDirectoryName(filePath)
                if not (String.IsNullOrEmpty(directory)) && not (Directory.Exists(directory)) then
                    Directory.CreateDirectory(directory) |> ignore

                // Serialize the state to JSON
                let options = JsonSerializerOptions()
                options.WriteIndented <- true
                options.Converters.Add(JsonFSharpConverter())

                let json = JsonSerializer.Serialize(state, options)

                // Write the JSON to the file
                do! File.WriteAllTextAsync(filePath, json)

                return true
            with ex ->
                // Log the error
                Console.Error.WriteLine($"Error saving workflow state: {ex.Message}")
                return false
        }

    /// <summary>
    /// Loads the workflow state from a file
    /// </summary>
    let load (filePath: string) =
        task {
            try
                // Check if the file exists
                if not (File.Exists(filePath)) then
                    return None
                else
                    // Read the JSON from the file
                    let! json = File.ReadAllTextAsync(filePath)

                    // Deserialize the JSON to a workflow state
                    let options = JsonSerializerOptions()
                    options.Converters.Add(JsonFSharpConverter())

                    let state = JsonSerializer.Deserialize<WorkflowState>(json, options)

                    return Some state
            with ex ->
                // Log the error
                Console.Error.WriteLine($"Error loading workflow state: {ex.Message}")
                return None
        }

    /// <summary>
    /// Tries to load the workflow state from a file
    /// </summary>
    let tryLoad (filePath: string) =
        load filePath
