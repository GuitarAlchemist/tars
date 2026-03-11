namespace Tars.Kernel

open System
open System.Diagnostics
open System.IO
open System.Text

type SafetyCheckResult =
    | Passed
    | Failed of reason: string

type ISafetyGate =
    abstract member CheckStatic: code: string -> Async<SafetyCheckResult>
    abstract member CheckTests: projectPath: string -> Async<SafetyCheckResult>
    abstract member CheckSandbox: code: string -> Async<SafetyCheckResult>

type SafetyGate() =
    interface ISafetyGate with
        member _.CheckStatic(code: string) =
            async {
                if String.IsNullOrWhiteSpace(code) then
                    return Failed "Code is empty"
                elif code.Contains("TODO") || code.Contains("FIXME") then
                    return Failed "Code contains TODO or FIXME markers"
                else
                    return Passed
            }

        member _.CheckTests(projectPath: string) =
            async {
                let resolvedPath = Path.GetFullPath(projectPath)

                if not (File.Exists(resolvedPath) || Directory.Exists(resolvedPath)) then
                    return Failed $"Project path not found: {resolvedPath}"
                else
                    let workingDir =
                        if File.Exists(resolvedPath) then
                            Path.GetDirectoryName(resolvedPath)
                        else
                            resolvedPath

                    let args =
                        if File.Exists(resolvedPath) then
                            $"test \"{resolvedPath}\" --nologo --verbosity minimal"
                        else
                            $"test \"{resolvedPath}\" --nologo --verbosity minimal"

                    let startInfo =
                        ProcessStartInfo(
                            FileName = "dotnet",
                            Arguments = args,
                            WorkingDirectory = workingDir,
                            RedirectStandardOutput = true,
                            RedirectStandardError = true,
                            UseShellExecute = false,
                            CreateNoWindow = true
                        )

                    use proc = new Process()
                    proc.StartInfo <- startInfo

                    let output = StringBuilder()
                    proc.OutputDataReceived.Add(fun e ->
                        if not (isNull e.Data) then
                            output.AppendLine(e.Data) |> ignore)

                    proc.ErrorDataReceived.Add(fun e ->
                        if not (isNull e.Data) then
                            output.AppendLine(e.Data) |> ignore)

                    if not (proc.Start()) then
                        return Failed "Failed to start dotnet test process."
                    else
                        proc.BeginOutputReadLine()
                        proc.BeginErrorReadLine()
                        do! proc.WaitForExitAsync() |> Async.AwaitTask

                        if proc.ExitCode = 0 then
                            return Passed
                        else
                            let details = output.ToString().Trim()
                            let message =
                                if String.IsNullOrWhiteSpace(details) then
                                    $"dotnet test failed with exit code {proc.ExitCode}"
                                else
                                    $"dotnet test failed (exit {proc.ExitCode}): {details}"

                            return Failed message
            }

        member _.CheckSandbox(code: string) =
            async {
                try
                    use client = Tars.Sandbox.DockerClient.createClient ()
                    // Escape single quotes for bash
                    let escapedCode = code.Replace("'", "'\\''")

                    let cmd =
                        [ "/bin/bash"; "-c"; $"echo '{escapedCode}' > script.py && python3 script.py" ]

                    // We use Task.AwaitTask to bridge the Task -> Async gap
                    let! result =
                        Tars.Sandbox.DockerClient.runContainer client "tars-sandbox:latest" cmd
                        |> Async.AwaitTask

                    match result with
                    | Result.Ok(stdout, stderr, exitCode) ->
                        if exitCode = 0L then
                            return Passed
                        else
                            return Failed $"Sandbox execution failed (Exit Code {exitCode}): {stderr}"
                    | Result.Error e -> return Failed $"Sandbox error: {e}"
                with ex ->
                    return Failed $"Sandbox exception: {ex.Message}"
            }
