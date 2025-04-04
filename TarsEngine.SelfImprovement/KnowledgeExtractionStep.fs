namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open System.Diagnostics

/// <summary>
/// Module for the knowledge extraction step in the autonomous improvement workflow
/// </summary>
module KnowledgeExtractionStep =
    /// <summary>
    /// The path to the knowledge extraction metascript
    /// </summary>
    let metascriptPath = Path.Combine("Examples", "metascripts", "documentation_knowledge_extraction.tars")

    /// <summary>
    /// The path to the knowledge base file
    /// </summary>
    let knowledgeBasePath = "knowledge_base.json"

    /// <summary>
    /// The path to the extraction state file
    /// </summary>
    let extractionStatePath = "extraction_state.json"

    /// <summary>
    /// Checks if the knowledge extraction metascript exists
    /// </summary>
    let metascriptExists () =
        File.Exists(metascriptPath)

    /// <summary>
    /// Runs the knowledge extraction metascript
    /// </summary>
    let runMetascript (logger: ILogger) (maxFiles: int) =
        task {
            try
                // Check if the metascript exists
                if not (metascriptExists()) then
                    return Error $"Metascript not found at {metascriptPath}"

                // Set environment variables for the metascript
                Environment.SetEnvironmentVariable("TARS_MAX_FILES_TO_PROCESS", maxFiles.ToString())

                // Create the process start info
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- "dotnet"
                startInfo.Arguments <- sprintf "run --project TarsCli/TarsCli.csproj -- dsl run %s" metascriptPath
                startInfo.UseShellExecute <- false
                startInfo.RedirectStandardOutput <- true
                startInfo.RedirectStandardError <- true
                startInfo.CreateNoWindow <- true

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
                        logger.LogInformation("Knowledge extraction: {Line}", line)
                    output.ToString())

                let errorTask = Task.Run(fun () ->
                    let error = System.Text.StringBuilder()
                    while not proc.StandardError.EndOfStream do
                        let line = proc.StandardError.ReadLine()
                        error.AppendLine(line) |> ignore
                        logger.LogError("Knowledge extraction error: {Line}", line)
                    error.ToString())

                // Start the process
                proc.Start() |> ignore

                // Wait for the process to exit
                let! _ = proc.WaitForExitAsync()

                // Get the output and error
                let! output = outputTask
                let! error = errorTask

                // Check if the process exited successfully
                if proc.ExitCode = 0 then
                    // Check if the knowledge base file was created
                    if File.Exists(knowledgeBasePath) then
                        return Ok Map.empty
                    else
                        return Error $"Knowledge base file not found at {knowledgeBasePath}"
                else
                    return Error $"Knowledge extraction failed with exit code {proc.ExitCode}: {error}"
            with ex ->
                logger.LogError(ex, "Error running knowledge extraction metascript")
                return Error ex.Message
        }

    /// <summary>
    /// Gets the knowledge extraction step handler
    /// </summary>
    let getHandler (logger: ILogger) (maxFiles: int) : WorkflowEngine.StepHandler =
        fun state ->
            task {
                logger.LogInformation("Starting knowledge extraction step")

                // Run the knowledge extraction metascript
                let! result = runMetascript logger maxFiles

                match result with
                | Ok _ ->
                    // Check if the knowledge base file exists
                    if File.Exists(knowledgeBasePath) then
                        // Get the file info
                        let fileInfo = FileInfo(knowledgeBasePath)

                        // Return the result data
                        return Ok (Map.ofList [
                            "knowledge_base_path", knowledgeBasePath
                            "knowledge_base_size", fileInfo.Length.ToString()
                            "extraction_time", DateTime.UtcNow.ToString("o")
                        ])
                    else
                        return Error $"Knowledge base file not found at {knowledgeBasePath}"
                | Error errorMessage ->
                    return Error errorMessage
            }
