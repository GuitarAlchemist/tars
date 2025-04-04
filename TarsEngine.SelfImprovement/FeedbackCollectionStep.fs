namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open System.Text.Json
open System.Diagnostics

/// <summary>
/// Represents feedback on an applied improvement
/// </summary>
type ImprovementFeedback = {
    /// <summary>
    /// The file path
    /// </summary>
    FilePath: string

    /// <summary>
    /// The pattern ID that was applied
    /// </summary>
    PatternId: string

    /// <summary>
    /// Whether the improvement was successful
    /// </summary>
    IsSuccessful: bool

    /// <summary>
    /// The error message if the improvement was not successful
    /// </summary>
    ErrorMessage: string option

    /// <summary>
    /// The time when the feedback was collected
    /// </summary>
    CollectedAt: DateTime
}

/// <summary>
/// Module for the feedback collection step in the autonomous improvement workflow
/// </summary>
module FeedbackCollectionStep =
    /// <summary>
    /// The path to the applied improvements file
    /// </summary>
    let appliedImprovementsPath = "applied_improvements.json"

    /// <summary>
    /// The path to the feedback file
    /// </summary>
    let feedbackPath = "improvement_feedback.json"

    /// <summary>
    /// Loads the applied improvements
    /// </summary>
    let loadAppliedImprovements () =
        task {
            if File.Exists(appliedImprovementsPath) then
                let! json = File.ReadAllTextAsync(appliedImprovementsPath)
                return JsonSerializer.Deserialize<ImprovementApplicationStep.AppliedImprovement[]>(json)
            else
                return [||]
        }

    /// <summary>
    /// Validates an improvement by building the project
    /// </summary>
    let validateImprovement (logger: ILogger) (improvement: ImprovementApplicationStep.AppliedImprovement) =
        task {
            try
                // Get the project directory
                let projectDir = Path.GetDirectoryName(improvement.FilePath)

                // Find the nearest .csproj or .fsproj file
                let rec findProjectFile (dir: string) =
                    if String.IsNullOrEmpty(dir) || not (Directory.Exists(dir)) then
                        None
                    else
                        let projectFiles =
                            Directory.GetFiles(dir, "*.?sproj")
                            |> Array.filter (fun f ->
                                f.EndsWith(".csproj", StringComparison.OrdinalIgnoreCase) ||
                                f.EndsWith(".fsproj", StringComparison.OrdinalIgnoreCase))

                        if projectFiles.Length > 0 then
                            Some projectFiles.[0]
                        else
                            findProjectFile (Path.GetDirectoryName(dir))

                match findProjectFile projectDir with
                | Some projectFile ->
                    logger.LogInformation("Building project: {ProjectFile}", projectFile)

                    // Create the process start info
                    let startInfo = ProcessStartInfo()
                    startInfo.FileName <- "dotnet"
                    startInfo.Arguments <- $"build {projectFile}"
                    startInfo.UseShellExecute <- false
                    startInfo.RedirectStandardOutput <- true
                    startInfo.RedirectStandardError <- true
                    startInfo.CreateNoWindow <- true
                    startInfo.WorkingDirectory <- Path.GetDirectoryName(projectFile)

                    // Start the process
                    use proc = new Process()
                    proc.StartInfo <- startInfo
                    proc.EnableRaisingEvents <- true

                    // Create tasks for reading output and error
                    let outputTask = Task.Run(fun () ->
                        let output = System.Text.StringBuilder()
                        while not proc.StandardOutput.EndOfStream do
                            let line = proc.StandardOutput.ReadLine()
                            output.AppendLine(line) |> ignore
                        output.ToString())

                    let errorTask = Task.Run(fun () ->
                        let error = System.Text.StringBuilder()
                        while not proc.StandardError.EndOfStream do
                            let line = proc.StandardError.ReadLine()
                            error.AppendLine(line) |> ignore
                        error.ToString())

                    // Start the process
                    proc.Start() |> ignore

                    // Wait for the process to exit
                    let! _ = proc.WaitForExitAsync()

                    // Get the output and error
                    let! output = outputTask
                    let! error = errorTask

                    // Check if the build was successful
                    if proc.ExitCode = 0 then
                        logger.LogInformation("Build successful for project: {ProjectFile}", projectFile)

                        // Create the feedback
                        let feedback = {
                            FilePath = improvement.FilePath
                            PatternId = improvement.PatternId
                            IsSuccessful = true
                            ErrorMessage = None
                            CollectedAt = DateTime.UtcNow
                        }

                        return feedback
                    else
                        logger.LogError("Build failed for project: {ProjectFile}", projectFile)

                        // Create the feedback
                        let feedback = {
                            FilePath = improvement.FilePath
                            PatternId = improvement.PatternId
                            IsSuccessful = false
                            ErrorMessage = Some error
                            CollectedAt = DateTime.UtcNow
                        }

                        return feedback
                | None ->
                    logger.LogWarning("No project file found for file: {FilePath}", improvement.FilePath)

                    // Create the feedback (assume success if no project file is found)
                    let feedback = {
                        FilePath = improvement.FilePath
                        PatternId = improvement.PatternId
                        IsSuccessful = true
                        ErrorMessage = None
                        CollectedAt = DateTime.UtcNow
                    }

                    return feedback
            with ex ->
                logger.LogError(ex, "Error validating improvement for file: {FilePath}", improvement.FilePath)

                // Create the feedback
                let feedback = {
                    FilePath = improvement.FilePath
                    PatternId = improvement.PatternId
                    IsSuccessful = false
                    ErrorMessage = Some ex.Message
                    CollectedAt = DateTime.UtcNow
                }

                return feedback
        }

    /// <summary>
    /// Updates pattern scores based on feedback
    /// </summary>
    let updatePatternScores (logger: ILogger) (feedback: ImprovementFeedback list) =
        task {
            try
                // Load the RetroactionLoop state
                let! state = RetroactionLoop.loadState()

                // Update the pattern scores
                let updatedState =
                    feedback
                    |> List.fold (fun s f ->
                        // Find the pattern
                        let pattern =
                            s.Patterns
                            |> List.tryFind (fun p -> p.Id = f.PatternId)

                        match pattern with
                        | Some p ->
                            // Update the pattern
                            let updatedPattern =
                                if f.IsSuccessful then
                                    { p with
                                        Uses = p.Uses + 1
                                        Successes = p.Successes + 1
                                        Score = p.Score + 0.1 } // Increase score for successful improvements
                                else
                                    { p with
                                        Uses = p.Uses + 1
                                        Score = max 0.1 (p.Score - 0.2) } // Decrease score for failed improvements

                            // Replace the pattern in the state
                            let updatedPatterns =
                                s.Patterns
                                |> List.map (fun p -> if p.Id = f.PatternId then updatedPattern else p)

                            { s with Patterns = updatedPatterns }
                        | None ->
                            // Pattern not found
                            s) state

                // Save the updated state
                do! RetroactionLoop.saveState updatedState

                return updatedState
            with ex ->
                logger.LogError(ex, "Error updating pattern scores")
                return! RetroactionLoop.loadState()
        }

    /// <summary>
    /// Gets the feedback collection step handler
    /// </summary>
    let getHandler (logger: ILogger) : WorkflowEngine.StepHandler =
        fun state ->
            task {
                logger.LogInformation("Starting feedback collection step")

                // Load the applied improvements
                let! appliedImprovements = loadAppliedImprovements()

                if appliedImprovements.Length = 0 then
                    logger.LogInformation("No applied improvements found")
                    return Ok Map.empty

                // Validate each improvement
                let! feedback =
                    appliedImprovements
                    |> Array.map (validateImprovement logger)
                    |> Task.WhenAll

                let feedbackList = feedback |> Array.toList

                logger.LogInformation("Collected feedback for {ImprovementCount} improvements", feedbackList.Length)

                // Save the feedback to a file
                let json = JsonSerializer.Serialize(feedbackList, JsonSerializerOptions(WriteIndented = true))
                do! File.WriteAllTextAsync(feedbackPath, json)

                // Update pattern scores based on feedback
                let! updatedState = updatePatternScores logger feedbackList

                // Return the result data
                return Ok (Map.ofList [
                    "feedback_path", feedbackPath
                    "feedback_count", feedbackList.Length.ToString()
                    "successful_improvements", feedbackList |> List.filter (fun f -> f.IsSuccessful) |> List.length |> string
                    "failed_improvements", feedbackList |> List.filter (fun f -> not f.IsSuccessful) |> List.length |> string
                ])
            }
