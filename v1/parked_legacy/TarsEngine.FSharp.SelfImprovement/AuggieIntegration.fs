namespace TarsEngine.FSharp.SelfImprovement

open System
open System.Diagnostics
open System.Text
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Integration helpers for interacting with the Auggie CLI (https://www.augmentcode.com/changelog/auggie-cli).
module AuggieIntegration =

    /// Settings controlling how the Auggie CLI is invoked.
    type AuggieCliSettings =
        { ToolPath: string
          WorkingDirectory: string option
          ExtraArgs: string list
          Timeout: TimeSpan option }

    /// Default Auggie CLI settings (invokes `auggie` on PATH with no extra arguments).
    let defaultSettings =
        { ToolPath = "auggie"
          WorkingDirectory = None
          ExtraArgs = []
          Timeout = Some(TimeSpan.FromMinutes(5.0)) }

    /// A structured request that will be transformed into an Auggie CLI invocation.
    type AuggieDispatchRequest =
        { Title: string option
          Instruction: string
          Metadata: Map<string, string> }

    /// Result of executing the Auggie CLI.
    type AuggieDispatchResult =
        { Request: AuggieDispatchRequest
          Command: string
          ExitCode: int
          StartedAt: DateTime
          CompletedAt: DateTime
          Duration: TimeSpan
          StandardOutput: string
          StandardError: string
          Succeeded: bool
          ErrorMessage: string option }

    /// Build the argument list for the CLI invocation.
    let private buildArguments (settings: AuggieCliSettings) (request: AuggieDispatchRequest) =
        let builder = ResizeArray<string>()
        settings.ExtraArgs |> List.iter builder.Add
        builder.Add("--print")

        let titleBlock =
            request.Title
            |> Option.map (fun title -> $"# {title}")
            |> Option.defaultValue ""

        let metadataBlock =
            if request.Metadata.IsEmpty then
                ""
            else
                request.Metadata
                |> Seq.map (fun kvp -> $"@meta {kvp.Key}={kvp.Value}")
                |> String.concat Environment.NewLine

        let fullInstruction =
            [ titleBlock; metadataBlock; request.Instruction ]
            |> List.filter (String.IsNullOrWhiteSpace >> not)
            |> String.concat (Environment.NewLine + Environment.NewLine)

        builder.Add(fullInstruction)
        builder |> List.ofSeq

    /// Execute the Auggie CLI with the supplied instruction.
    let dispatchAsync (logger: ILogger) (settings: AuggieCliSettings) (request: AuggieDispatchRequest) =
        async {
            let arguments = buildArguments settings request
            let psi = ProcessStartInfo()
            psi.FileName <- settings.ToolPath
            psi.UseShellExecute <- false
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true

            arguments |> List.iter psi.ArgumentList.Add

            settings.WorkingDirectory
            |> Option.iter (fun dir -> psi.WorkingDirectory <- dir)

            let startedAt = DateTime.UtcNow
            let proc = new Process()
            proc.StartInfo <- psi

            let outputBuilder = StringBuilder()
            let errorBuilder = StringBuilder()

            let outputHandler (_sender: obj) (args: DataReceivedEventArgs) =
                if not (isNull args.Data) then
                    outputBuilder.AppendLine(args.Data) |> ignore

            let errorHandler (_sender: obj) (args: DataReceivedEventArgs) =
                if not (isNull args.Data) then
                    errorBuilder.AppendLine(args.Data) |> ignore

            proc.OutputDataReceived.AddHandler(DataReceivedEventHandler outputHandler)
            proc.ErrorDataReceived.AddHandler(DataReceivedEventHandler errorHandler)

            let joinedArgs = String.concat " " arguments
            let commandLine =
                if String.IsNullOrWhiteSpace joinedArgs then
                    psi.FileName
                else
                    $"{psi.FileName} {joinedArgs}"

            logger.LogInformation("Dispatching Auggie instruction via CLI: {CommandLine}", commandLine)

            let started =
                try
                    proc.Start()
                with ex ->
                    logger.LogError(ex, "Failed to start Auggie CLI process.")
                    false

            if not started then
                let completed = DateTime.UtcNow
                return
                    { Request = request
                      Command = commandLine
                      ExitCode = -1
                      StartedAt = startedAt
                      CompletedAt = completed
                      Duration = completed - startedAt
                      StandardOutput = ""
                      StandardError = ""
                      Succeeded = false
                      ErrorMessage = Some "Failed to start Auggie CLI process." }
            else
                proc.BeginOutputReadLine()
                proc.BeginErrorReadLine()

                let timeout =
                    settings.Timeout
                    |> Option.defaultValue (TimeSpan.FromMinutes(5.0))

                use cts = new CancellationTokenSource()
                let waitTask = proc.WaitForExitAsync(cts.Token)

                let exited =
                    try
                        waitTask.Wait(timeout)
                    with _ ->
                        false

                if not exited then
                    logger.LogWarning("Auggie CLI did not exit within timeout ({Timeout}). Killing process.", timeout)
                    try proc.Kill(entireProcessTree = true) with _ -> ()
                    proc.WaitForExit()

                let completedAt = DateTime.UtcNow
                let exitCode = proc.ExitCode
                let stdOut = outputBuilder.ToString()
                let stdErr = errorBuilder.ToString()

                let succeeded = exited && exitCode = 0
                let errorMessage =
                    if succeeded then None
                    elif not exited then Some "Auggie CLI timed out."
                    else
                        if String.IsNullOrWhiteSpace(stdErr) then
                            Some $"Auggie CLI exited with code {exitCode}."
                        else
                            Some (stdErr.Trim())

                return
                    { Request = request
                      Command = commandLine
                      ExitCode = exitCode
                      StartedAt = startedAt
                      CompletedAt = completedAt
                      Duration = completedAt - startedAt
                      StandardOutput = stdOut
                      StandardError = stdErr
                      Succeeded = succeeded
                      ErrorMessage = errorMessage }
        }
