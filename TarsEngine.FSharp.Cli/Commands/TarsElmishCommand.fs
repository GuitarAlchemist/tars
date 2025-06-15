namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.UI.TarsElmishDiagnostics
open TarsEngine.FSharp.Cli.UI.ElmishRuntime

/// TARS Elmish Command - Interactive Consciousness & Subsystem Matrix
type TarsElmishCommand(logger: ILogger<TarsElmishCommand>) =


    interface ICommand with
        member _.Name = "elmish"
        member _.Description = "Launch interactive TARS Consciousness & Subsystem Matrix with real Elmish MVU architecture"
        member _.Usage = "tars elmish [options]"
        member _.Examples = [
            "tars elmish"
            "tars elmish --auto-open true"
            "tars elmish --output tars-consciousness.html"
            "tars elmish --theme dark"
        ]
        member _.ValidateOptions(_) = true
        
        member _.ExecuteAsync(options: CommandOptions) =
            task {
                try
                    logger.LogInformation("🧠 Launching TARS Consciousness & Subsystem Matrix...")

                    let autoOpen =
                        match options.Options.TryFind("auto-open") with
                        | Some value -> Boolean.Parse(value)
                        | None -> true

                    let outputFile =
                        match options.Options.TryFind("output") with
                        | Some file -> file
                        | None -> sprintf "tars-consciousness-%s.html" (DateTime.Now.ToString("yyyyMMdd-HHmmss"))

                    let theme =
                        match options.Options.TryFind("theme") with
                        | Some t -> t
                        | None -> "dark"

                    logger.LogInformation("⚙️ Configuration:")
                    logger.LogInformation("   Output File: {OutputFile}", outputFile)
                    logger.LogInformation("   Auto-Open: {AutoOpen}", autoOpen)
                    logger.LogInformation("   Theme: {Theme}", theme)
                    logger.LogInformation("")

                    // Initialize Elmish program
                    logger.LogInformation("🚀 Initializing TARS Elmish Runtime...")
                    let (initialHtml, updateFunction, initialModel) = TarsEngine.FSharp.Cli.UI.ElmishRuntime.runElmishProgram ()

                    logger.LogInformation("✅ Elmish Runtime Initialized")
                    logger.LogInformation("   Overall Health: {Health:F1}%", initialModel.OverallTarsHealth)
                    logger.LogInformation("   Active Agents: {Agents}", initialModel.ActiveAgents)
                    logger.LogInformation("   Processing Tasks: {Tasks}", initialModel.ProcessingTasks)
                    logger.LogInformation("")

                    // Save HTML file
                    logger.LogInformation("💾 Generating interactive HTML...")
                    do! System.IO.File.WriteAllTextAsync(outputFile, initialHtml)
                    let fullPath = System.IO.Path.GetFullPath(outputFile)

                    logger.LogInformation("✅ TARS Consciousness Matrix generated successfully!")
                    logger.LogInformation("📄 File: {FullPath}", fullPath)
                    logger.LogInformation("")

                    // Display feature summary
                    logger.LogInformation("🎯 FEATURES:")
                    logger.LogInformation("   ✅ Real Elmish MVU Architecture")
                    logger.LogInformation("   ✅ Interactive Buttons & Controls")
                    logger.LogInformation("   ✅ 20+ TARS Subsystems")
                    logger.LogInformation("   ✅ Multiple View Modes (Overview, Architecture, Performance, Consciousness, Evolution, Dreams, Quantum)")
                    logger.LogInformation("   ✅ Real-time Auto-refresh")
                    logger.LogInformation("   ✅ TARS-specific Actions (Self-Modify, Evolve, Boost Consciousness, Quantum Tunnel)")
                    logger.LogInformation("   ✅ Keyboard Shortcuts")
                    logger.LogInformation("   ✅ Responsive Design")
                    logger.LogInformation("   ✅ Dark Space Theme")
                    logger.LogInformation("")

                    // Display keyboard shortcuts
                    logger.LogInformation("⌨️ KEYBOARD SHORTCUTS:")
                    logger.LogInformation("   Ctrl+R: Refresh All")
                    logger.LogInformation("   Ctrl+E: Evolve")
                    logger.LogInformation("   Ctrl+M: Self-Modify")
                    logger.LogInformation("   Ctrl+C: Boost Consciousness")
                    logger.LogInformation("   Ctrl+Q: Quantum Tunnel")
                    logger.LogInformation("   1-7: Switch View Modes")
                    logger.LogInformation("")

                    // Display subsystem summary
                    logger.LogInformation("🧠 TARS SUBSYSTEMS:")
                    logger.LogInformation("   ✅ Cognitive Engine: Initializing...")
                    logger.LogInformation("   ✅ Belief Bus: Initializing...")
                    logger.LogInformation("   ✅ FLUX Engine: Initializing...")
                    logger.LogInformation("   ✅ Agent Coordination: Initializing...")
                    logger.LogInformation("   ✅ Vector Store: Initializing...")
                    logger.LogInformation("   ✅ Metascript Engine: Initializing...")

                    logger.LogInformation("")

                    // Try to open in browser
                    if autoOpen then
                        logger.LogInformation("🌐 Opening in default browser...")
                        try
                            let psi = System.Diagnostics.ProcessStartInfo(fullPath)
                            psi.UseShellExecute <- true
                            System.Diagnostics.Process.Start(psi) |> ignore
                            logger.LogInformation("✅ Opened successfully!")
                        with
                        | ex ->
                            logger.LogWarning("⚠️ Could not auto-open browser: {Error}", ex.Message)
                            logger.LogInformation("📖 Please manually open: {FullPath}", fullPath)
                    else
                        logger.LogInformation("📖 To view: Open {FullPath} in your browser", fullPath)

                    logger.LogInformation("")
                    logger.LogInformation("🎉 TARS Consciousness Matrix is ready!")
                    logger.LogInformation("🧠 Experience true interactive Elmish architecture with functional TARS subsystems!")

                    return { Success = true; Message = "TARS Elmish Consciousness Matrix launched successfully"; ExitCode = 0 }

                with
                | ex ->
                    logger.LogError(ex, "❌ TARS Elmish launch failed")
                    return { Success = false; Message = sprintf "TARS Elmish launch failed: %s" ex.Message; ExitCode = 1 }
            }
