namespace Tars.Tools.Standard

open System
open System.Threading.Tasks
open Tars.Tools
open Tars.Sandbox

module StandardTools =

    let private dockerClient = lazy (DockerClient.createClient ())

    [<TarsToolAttribute("run_command", "Executes a shell command in a secure sandbox. Input: command string.")>]
    let runCommand (command: string) =
        task {
            let client = dockerClient.Value
            // Use /bin/bash -c to execute the command string
            let cmdList = [ "/bin/bash"; "-c"; command ]

            // Use the tars-sandbox image
            let! result = DockerClient.runContainer client "tars-sandbox:latest" cmdList

            match result with
            | Ok(stdout, stderr, exitCode) ->
                if exitCode = 0L then
                    return stdout.Trim()
                else
                    return sprintf "Error (Exit Code %d): %s\nStdout: %s" exitCode (stderr.Trim()) (stdout.Trim())
            | Error e -> return sprintf "Sandbox Error: %s" e
        }
