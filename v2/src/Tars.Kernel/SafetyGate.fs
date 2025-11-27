namespace Tars.Kernel

open System
open System.Threading.Tasks
open Tars.Core

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
                // Placeholder: In a real scenario, this would run 'dotnet test'
                // For now, we assume if the path exists, it passes
                if System.IO.File.Exists(projectPath) || System.IO.Directory.Exists(projectPath) then
                    return Passed
                else
                    return Failed $"Project path not found: {projectPath}"
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
