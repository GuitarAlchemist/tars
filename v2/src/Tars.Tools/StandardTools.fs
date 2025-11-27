namespace Tars.Tools.Standard

open System
open System.Threading.Tasks
open System.Diagnostics
open Tars.Tools

type RunCommandTool() =
    interface ITool with
        member _.Name = "run_command"
        member _.Description = "Executes a shell command."

        member _.ExecuteAsync(args: Map<string, string>) =
            task {
                match args.TryFind "command" with
                | Some cmd ->
                    // TODO: Use Tars.Security.SandboxedProcess
                    // For now, just basic process execution for demo
                    let psi = ProcessStartInfo()
                    psi.FileName <- "cmd.exe"
                    psi.Arguments <- sprintf "/c %s" cmd
                    psi.RedirectStandardOutput <- true
                    psi.RedirectStandardError <- true
                    psi.UseShellExecute <- false
                    psi.CreateNoWindow <- true

                    use proc = new Process()
                    proc.StartInfo <- psi
                    proc.Start() |> ignore

                    let! stdout = proc.StandardOutput.ReadToEndAsync()
                    let! stderr = proc.StandardError.ReadToEndAsync()
                    do! proc.WaitForExitAsync()

                    if proc.ExitCode = 0 then
                        return stdout.Trim()
                    else
                        return sprintf "Error (Exit Code %d): %s" proc.ExitCode (stderr.Trim())
                | None -> return "Error: Missing 'command' argument."
            }
