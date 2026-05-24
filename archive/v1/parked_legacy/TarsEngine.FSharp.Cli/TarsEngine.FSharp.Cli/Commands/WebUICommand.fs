// REAL WEB UI COMMAND FOR TARS CLI
// Launches autonomous software engineering web interface with DeepSeek-R1

namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Commands
open TarsWebServer

/// <summary>
/// Command to launch the TARS autonomous software engineering web UI
/// </summary>
type WebUICommand() =
    interface ICommand with
        member _.Name = "web-ui"
        member _.Description = "Launch TARS autonomous software engineering web interface with DeepSeek-R1"
        member _.Usage = "tars web-ui [--port <port>]"
        member _.Examples = [ "tars web-ui"; "tars web-ui --port 9000" ]
        member _.ValidateOptions(_options: CommandOptions) = true
        
        member _.ExecuteAsync(options: CommandOptions) =
            Task.Run(fun () ->
                try
                    // Parse port option
                    let port =
                        match options.Options.TryFind("port") with
                        | Some portStr ->
                            match Int32.TryParse(portStr) with
                            | (true, p) -> p
                            | _ -> 8080
                        | None -> 8080
                    
                    printfn "🚀 TARS Autonomous Software Engineering Web UI"
                    printfn "=============================================="
                    printfn ""
                    printfn "🧠 Features:"
                    printfn "  • Real codebase analysis and problem detection"
                    printfn "  • DeepSeek-R1 powered autonomous reasoning"
                    printfn "  • Live problem solving and code generation"
                    printfn "  • Real-time integration with TARS CLI"
                    printfn ""
                    printfn $"🌐 Starting web server on port {port}..."
                    printfn $"   Open: http://localhost:{port}"
                    printfn ""
                    printfn "Press Ctrl+C to stop the server"
                    printfn ""
                    
                    // Start the web server (this will block)
                    startWebServer port |> Async.RunSynchronously
                    
                    {
                        Success = true
                        Message = "Web UI server stopped"
                        ExitCode = 0
                    }
                with
                | ex ->
                    {
                        Success = false
                        Message = $"Failed to start web UI: {ex.Message}"
                        ExitCode = 1
                    }
            )
