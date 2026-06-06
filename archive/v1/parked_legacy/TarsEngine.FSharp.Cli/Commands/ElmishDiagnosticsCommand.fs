namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Diagnostics.TarsRealDiagnostics
open TarsEngine.FSharp.Cli.UI.ElmishRuntime

/// COMPREHENSIVE TARS ELMISH DIAGNOSTICS - Uses Full TarsElmishDiagnostics Module

type ElmishDiagnosticsCommand(logger: ILogger<ElmishDiagnosticsCommand>) =

    interface ICommand with
        member _.Name = "elmish-diagnostics"
        member _.Description = "Pure Elmish MVU diagnostics with functional reactive programming"
        member _.Usage = "tars elmish-diagnostics [options]"
        member _.Examples = [
            "tars elmish-diagnostics"
            "tars elmish-diagnostics --auto-refresh false"
            "tars elmish-diagnostics --view gpu"
        ]
        member _.ValidateOptions(_) = true
        
        member _.ExecuteAsync(options: CommandOptions) =
            task {
                try
                    logger.LogInformation("🚀 Starting Pure Elmish Diagnostics...")
                    logger.LogInformation("⚡ Initializing MVU (Model-View-Update) architecture")
                    
                    // Generate COMPREHENSIVE TARS DIAGNOSTICS with enhanced styling
                    logger.LogInformation("🧠 Loading ALL comprehensive TARS subsystems with enhanced UI...")
                    let (_, updateFunction, tarsModel) = runElmishProgram ()
                    logger.LogInformation("📊 Model initialized with {SubsystemCount} TARS subsystems", tarsModel.AllSubsystems.Length)

                    // Load real diagnostics data (for additional system info)
                    logger.LogInformation("🔍 Loading real system diagnostics...")
                    let! diagnostics = getComprehensiveDiagnostics Environment.CurrentDirectory
                    logger.LogInformation("✅ Diagnostics loaded into Elmish model")

                    // Generate COMPLETE HTML with CSS, JavaScript, and dark mode
                    logger.LogInformation("🎨 Generating complete TARS HTML with CSS, JavaScript, and dark mode...")
                    let htmlContent = generateCompleteHtml tarsModel

                    // Save comprehensive HTML with enhanced styling
                    let tempPath = Path.Combine(Path.GetTempPath(), "tars-comprehensive-enhanced-diagnostics.html")
                    do! File.WriteAllTextAsync(tempPath, htmlContent)

                    logger.LogInformation("✅ COMPREHENSIVE TARS DIAGNOSTICS WITH ENHANCED STYLING GENERATED!")
                    logger.LogInformation("📂 File: {TempPath}", tempPath)
                    logger.LogInformation("🌐 Opening enhanced TARS diagnostics in browser...")
                    
                    // Open in browser
                    try
                        let psi = System.Diagnostics.ProcessStartInfo(tempPath, UseShellExecute = true)
                        System.Diagnostics.Process.Start(psi) |> ignore
                        logger.LogInformation("🌐 Browser opened successfully!")
                    with
                    | ex -> logger.LogWarning("⚠️ Could not open browser: {Error}", ex.Message)
                    
                    // Show ALL comprehensive TARS subsystems
                    logger.LogInformation("")
                    logger.LogInformation("🧠 COMPREHENSIVE TARS SUBSYSTEMS LOADED:")
                    logger.LogInformation("   🔢 Total Subsystems: {Count}", tarsModel.AllSubsystems.Length)
                    logger.LogInformation("   ⚡ Overall TARS Health: {Health.ToString("F1")}%", tarsModel.OverallTarsHealth)
                    logger.LogInformation("   🤖 Active Agents: {Agents}", tarsModel.ActiveAgents)
                    logger.LogInformation("   ⚙️ Processing Tasks: {Tasks}", tarsModel.ProcessingTasks)

                    logger.LogInformation("")
                    logger.LogInformation("🔍 ALL TARS SUBSYSTEMS:")
                    for subsystem in tarsModel.AllSubsystems do
                        let statusEmoji =
                            match subsystem.Status with
                            | TarsEngine.FSharp.Cli.UI.TarsElmishDiagnostics.Operational -> "✅"
                            | TarsEngine.FSharp.Cli.UI.TarsElmishDiagnostics.Evolving -> "🔄"
                            | _ -> "⚠️"
                        logger.LogInformation("   {StatusEmoji} {Name}: {Health.ToString("F1")}% ({Components} components)",
                            statusEmoji, subsystem.Name, subsystem.HealthPercentage, subsystem.ActiveComponents)

                    logger.LogInformation("")
                    logger.LogInformation("🎨 ENHANCED UI FEATURES:")
                    logger.LogInformation("   🧭 Navigation Sidebar with all subsystems")
                    logger.LogInformation("   📊 Real-time health monitoring")
                    logger.LogInformation("   ⚡ Performance metrics for each subsystem")
                    logger.LogInformation("   🏗️ System architecture visualization")
                    logger.LogInformation("   🎛️ Interactive controls and settings")
                    logger.LogInformation("   🎨 Enhanced styling with backdrop blur and animations")
                    logger.LogInformation("   🌈 Professional dark theme with TARS branding")
                    logger.LogInformation("   ⚡ Performance metrics for each subsystem")
                    logger.LogInformation("   🏗️ System architecture visualization")
                    logger.LogInformation("   🎛️ Interactive controls and settings")
                    
                    logger.LogInformation("")
                    logger.LogInformation("💡 Press any key to continue...")
                    Console.ReadKey() |> ignore
                    
                    return { Success = true; Message = "Elmish diagnostics completed successfully"; ExitCode = 0 }
                    
                with
                | ex ->
                    logger.LogError(ex, "❌ Elmish diagnostics failed")
                    return { Success = false; Message = sprintf "Elmish diagnostics failed: %s" ex.Message; ExitCode = 1 }
            }

