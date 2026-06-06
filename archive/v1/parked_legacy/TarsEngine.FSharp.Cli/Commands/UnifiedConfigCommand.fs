namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Configuration.UnifiedConfigurationManager
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Core.UnifiedTypes

/// Unified Configuration Command - Demonstrates centralized configuration management
module UnifiedConfigCommand =
    
    /// Demonstrate the unified configuration system
    let demonstrateConfigurationSystem (logger: ITarsLogger) =
        task {
            try
                AnsiConsole.MarkupLine("[bold cyan]⚙️ TARS Unified Configuration Management Demo[/]")
                AnsiConsole.MarkupLine("[dim]Centralized configuration for all TARS systems[/]")
                AnsiConsole.WriteLine()
                
                // Create configuration manager
                use configManager = createConfigurationManager logger
                
                AnsiConsole.MarkupLine("[yellow]🔧 Initializing Configuration Manager...[/]")
                let! initResult = configManager.InitializeAsync(CancellationToken.None)
                
                match initResult with
                | Success (_, metadata) ->
                    let environment = metadata.["environment"] :?> string
                    let configCount = metadata.["configCount"] :?> int
                    
                    AnsiConsole.MarkupLine($"[green]✅ Configuration manager initialized[/]")
                    AnsiConsole.MarkupLine($"[dim]Environment: {environment}[/]")
                    AnsiConsole.MarkupLine($"[dim]Loaded configurations: {configCount}[/]")
                
                | Failure (error, corrId) ->
                    AnsiConsole.MarkupLine($"[red]❌ Configuration initialization failed: {TarsError.toString error}[/]")
                    return 1
                
                AnsiConsole.WriteLine()
                
                // Show current configuration values
                AnsiConsole.MarkupLine("[yellow]📋 Current Configuration Values:[/]")
                let allValues = configManager.GetAllValues()
                
                if allValues.IsEmpty then
                    AnsiConsole.MarkupLine("[dim]No configuration values found[/]")
                else
                    for kvp in allValues do
                        let valueStr = 
                            match kvp.Value with
                            | StringValue s -> $"\"{s}\""
                            | IntValue i -> i.ToString()
                            | FloatValue f -> f.ToString("F2")
                            | BoolValue b -> b.ToString().ToLower()
                            | ArrayValue arr -> $"[{arr.Length} items]"
                            | ObjectValue obj -> $"{{{obj.Count} properties}}"
                            | NullValue -> "null"
                        
                        AnsiConsole.MarkupLine($"  [cyan]{kvp.Key}[/]: [yellow]{valueStr}[/]")
                
                AnsiConsole.WriteLine()
                
                // Demonstrate configuration by category
                AnsiConsole.MarkupLine("[yellow]📂 Configuration by Category:[/]")
                let categories = ["Core"; "Agents"; "CUDA"; "Proof"]
                
                for category in categories do
                    let categoryValues = configManager.GetValuesByCategory(category)
                    AnsiConsole.MarkupLine($"  [bold cyan]{category}[/] ({categoryValues.Count} values):")
                    
                    for kvp in categoryValues do
                        let valueStr = 
                            match kvp.Value with
                            | StringValue s -> $"\"{s}\""
                            | IntValue i -> i.ToString()
                            | FloatValue f -> f.ToString("F2")
                            | BoolValue b -> b.ToString().ToLower()
                            | _ -> "complex"
                        
                        AnsiConsole.MarkupLine($"    [dim]{kvp.Key}[/]: [yellow]{valueStr}[/]")
                
                AnsiConsole.WriteLine()
                
                // Demonstrate setting configuration values
                AnsiConsole.MarkupLine("[yellow]✏️ Setting Configuration Values...[/]")
                
                let configUpdates = [
                    ("tars.demo.stringValue", "Hello TARS Configuration!")
                    ("tars.demo.intValue", 42)
                    ("tars.demo.boolValue", true)
                    ("tars.demo.floatValue", 3.14159)
                ]
                
                for (key, value) in configUpdates do
                    let! setResult = 
                        match value with
                        | :? string as s -> configManager.SetValueAsync(key, s, None)
                        | :? int as i -> configManager.SetValueAsync(key, i, None)
                        | :? bool as b -> configManager.SetValueAsync(key, b, None)
                        | :? float as f -> configManager.SetValueAsync(key, f, None)
                        | _ -> configManager.SetValueAsync(key, value.ToString(), None)
                    
                    match setResult with
                    | Success _ ->
                        AnsiConsole.MarkupLine($"  ✅ [green]Set {key}[/] = [yellow]{value}[/]")
                    | Failure (error, _) ->
                        AnsiConsole.MarkupLine($"  ❌ [red]Failed to set {key}: {TarsError.toString error}[/]")
                
                AnsiConsole.WriteLine()
                
                // Demonstrate configuration retrieval with extensions
                AnsiConsole.MarkupLine("[yellow]🔍 Retrieving Configuration Values...[/]")
                
                let stringVal = ConfigurationExtensions.getString configManager "tars.demo.stringValue" "default"
                let intVal = ConfigurationExtensions.getInt configManager "tars.demo.intValue" 0
                let boolVal = ConfigurationExtensions.getBool configManager "tars.demo.boolValue" false
                let floatVal = ConfigurationExtensions.getFloat configManager "tars.demo.floatValue" 0.0
                
                AnsiConsole.MarkupLine(sprintf "  [cyan]String Value[/]: [yellow]\"%s\"[/]" stringVal)
                AnsiConsole.MarkupLine(sprintf "  [cyan]Integer Value[/]: [yellow]%d[/]" intVal)
                AnsiConsole.MarkupLine(sprintf "  [cyan]Boolean Value[/]: [yellow]%s[/]" (boolVal.ToString().ToLower()))
                AnsiConsole.MarkupLine(sprintf "  [cyan]Float Value[/]: [yellow]%s[/]" (floatVal.ToString("F5")))
                
                AnsiConsole.WriteLine()
                
                // Demonstrate configuration change subscription
                AnsiConsole.MarkupLine("[yellow]📡 Testing Configuration Change Notifications...[/]")
                
                let mutable changeCount = 0
                let changeCallback (changeEvent: ConfigChangeEvent) =
                    changeCount <- changeCount + 1
                    AnsiConsole.MarkupLine(sprintf "  🔔 [dim]Change detected: %s at %s[/]" changeEvent.Key (changeEvent.Timestamp.ToString("HH:mm:ss")))
                
                configManager.SubscribeToChanges(changeCallback)
                
                // Make some changes to trigger notifications
                let! _ = configManager.SetValueAsync("tars.demo.notificationTest", "change1", None)
                let! _ = configManager.SetValueAsync("tars.demo.notificationTest", "change2", None)
                let! _ = configManager.SetValueAsync("tars.demo.notificationTest", "change3", None)
                
                AnsiConsole.MarkupLine(sprintf "  [green]✅ Received %d change notifications[/]" changeCount)
                
                AnsiConsole.WriteLine()
                
                // Create configuration snapshot
                AnsiConsole.MarkupLine("[yellow]📸 Creating Configuration Snapshot...[/]")
                let! snapshotResult = configManager.CreateSnapshotAsync("demo_snapshot", CancellationToken.None)
                
                match snapshotResult with
                | Success (snapshot, metadata) ->
                    let snapshotId = metadata.["snapshotId"] :?> string
                    AnsiConsole.MarkupLine(sprintf "  ✅ [green]Snapshot created: %s[/]" snapshotId)
                    AnsiConsole.MarkupLine(sprintf "  [dim]Timestamp: %s[/]" (snapshot.Timestamp.ToString("yyyy-MM-dd HH:mm:ss")))
                    AnsiConsole.MarkupLine(sprintf "  [dim]Configuration count: %d[/]" snapshot.Configuration.Count)
                | Failure (error, _) ->
                    AnsiConsole.MarkupLine(sprintf "  ❌ [red]Snapshot creation failed: %s[/]" (TarsError.toString error))
                
                AnsiConsole.WriteLine()
                
                // Save configuration to file
                AnsiConsole.MarkupLine("[yellow]💾 Saving Configuration to File...[/]")
                let! saveResult = configManager.SaveConfigurationAsync(CancellationToken.None)
                
                match saveResult with
                | Success (_, metadata) ->
                    let configCount = metadata.["configCount"] :?> int
                    AnsiConsole.MarkupLine(sprintf "  ✅ [green]Saved %d configuration values to file[/]" configCount)
                | Failure (error, _) ->
                    AnsiConsole.MarkupLine(sprintf "  ❌ [red]Save failed: %s[/]" (TarsError.toString error))
                
                AnsiConsole.WriteLine()
                
                // Show configuration statistics
                let statistics = configManager.GetStatistics()
                AnsiConsole.MarkupLine("[bold cyan]📊 Configuration Statistics:[/]")
                
                let totalConfigs = statistics.["totalConfigurations"] :?> int
                let totalSchemas = statistics.["totalSchemas"] :?> int
                let totalEnvironments = statistics.["totalEnvironments"] :?> int
                let totalSnapshots = statistics.["totalSnapshots"] :?> int
                let currentEnv = statistics.["currentEnvironment"] :?> string
                let isInitialized = statistics.["isInitialized"] :?> bool
                
                AnsiConsole.MarkupLine(sprintf "  Total Configurations: [yellow]%d[/]" totalConfigs)
                AnsiConsole.MarkupLine(sprintf "  Total Schemas: [cyan]%d[/]" totalSchemas)
                AnsiConsole.MarkupLine(sprintf "  Total Environments: [blue]%d[/]" totalEnvironments)
                AnsiConsole.MarkupLine(sprintf "  Total Snapshots: [magenta]%d[/]" totalSnapshots)
                AnsiConsole.MarkupLine(sprintf "  Current Environment: [green]%s[/]" currentEnv)
                AnsiConsole.MarkupLine(sprintf "  Initialized: [yellow]%b[/]" isInitialized)
                
                AnsiConsole.WriteLine()
                
                // Performance analysis
                AnsiConsole.MarkupLine("[yellow]📈 Configuration Performance Analysis:[/]")
                
                let startTime = DateTime.UtcNow
                let iterations = 1000
                
                // Test configuration retrieval performance
                for i in 1..iterations do
                    let _ = ConfigurationExtensions.getString configManager "tars.demo.stringValue" "default"
                    let _ = ConfigurationExtensions.getInt configManager "tars.demo.intValue" 0
                    let _ = ConfigurationExtensions.getBool configManager "tars.demo.boolValue" false
                    ()
                
                let retrievalTime = DateTime.UtcNow - startTime
                let retrievalsPerSecond = float (iterations * 3) / retrievalTime.TotalSeconds
                
                AnsiConsole.MarkupLine(sprintf "  Retrieval Performance: [cyan]%s ops/sec[/]" (retrievalsPerSecond.ToString("F0")))
                AnsiConsole.MarkupLine(sprintf "  Average Retrieval Time: [yellow]%sms[/]" ((retrievalTime.TotalMilliseconds / float (iterations * 3)).ToString("F3")))
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[bold green]🎉 Unified Configuration System Demo Completed Successfully![/]")
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[bold cyan]🚀 CONFIGURATION SYSTEM ACHIEVEMENTS:[/]")
                AnsiConsole.MarkupLine("  ✅ [green]Centralized configuration[/] - Single source of truth for all settings")
                AnsiConsole.MarkupLine("  ✅ [green]Schema validation[/] - Type-safe configuration with validation rules")
                AnsiConsole.MarkupLine("  ✅ [green]Environment support[/] - Multiple configuration environments")
                AnsiConsole.MarkupLine("  ✅ [green]Change notifications[/] - Real-time configuration change events")
                AnsiConsole.MarkupLine("  ✅ [green]Configuration snapshots[/] - Versioning and backup capabilities")
                AnsiConsole.MarkupLine("  ✅ [green]Hot-reloading[/] - Dynamic configuration updates without restart")
                AnsiConsole.MarkupLine("  ✅ [green]High performance[/] - Optimized for frequent configuration access")
                
                return 0
            
            with
            | ex ->
                AnsiConsole.MarkupLine(sprintf "[red]❌ Configuration system demo failed: %s[/]" ex.Message)
                return 1
        }
    
    /// Unified Configuration Command implementation
    type UnifiedConfigCommand() =
        interface ICommand with
            member _.Name = "config"
            member _.Description = "Demonstrate TARS unified configuration management system"
            member _.Usage = "tars config [--demo] [--show] [--set key=value]"
            member _.Examples = [
                "tars config --demo           # Run configuration system demonstration"
                "tars config --show           # Show current configuration"
                "tars config --set key=value  # Set configuration value"
                "tars config                  # Show configuration overview"
            ]
            
            member _.ValidateOptions(options: CommandOptions) = true
            
            member _.ExecuteAsync(options: CommandOptions) =
                task {
                    try
                        let logger = createLogger "UnifiedConfigCommand"
                        
                        let isDemoMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--demo")
                        
                        let isShowMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--show")
                        
                        let setArgs = 
                            options.Arguments 
                            |> List.filter (fun arg -> arg.StartsWith("--set"))
                        
                        if isDemoMode then
                            let! result = demonstrateConfigurationSystem logger
                            return { Message = ""; ExitCode = result; Success = result = 0 }
                        elif isShowMode then
                            use configManager = createConfigurationManager logger
                            let! _ = configManager.InitializeAsync(CancellationToken.None)
                            
                            AnsiConsole.MarkupLine("[bold cyan]⚙️ TARS Configuration Values[/]")
                            AnsiConsole.WriteLine()
                            
                            let allValues = configManager.GetAllValues()
                            if allValues.IsEmpty then
                                AnsiConsole.MarkupLine("[dim]No configuration values found[/]")
                            else
                                for kvp in allValues do
                                    let valueStr = 
                                        match kvp.Value with
                                        | StringValue s -> sprintf "\"%s\"" s
                                        | IntValue i -> i.ToString()
                                        | FloatValue f -> f.ToString("F2")
                                        | BoolValue b -> b.ToString().ToLower()
                                        | _ -> "complex"

                                    AnsiConsole.MarkupLine(sprintf "  [cyan]%s[/]: [yellow]%s[/]" kvp.Key valueStr)
                            
                            return { Message = ""; ExitCode = 0; Success = true }
                        elif setArgs.Length > 0 then
                            use configManager = createConfigurationManager logger
                            let! _ = configManager.InitializeAsync(CancellationToken.None)
                            
                            AnsiConsole.MarkupLine("[bold cyan]⚙️ Setting Configuration Values[/]")
                            AnsiConsole.WriteLine()
                            
                            for setArg in setArgs do
                                if setArg.Contains("=") then
                                    let parts = setArg.Substring(5).Split('=', 2) // Remove "--set" prefix
                                    if parts.Length = 2 then
                                        let key = parts.[0].Trim()
                                        let value = parts.[1].Trim()
                                        
                                        let! result = configManager.SetValueAsync(key, value, None)
                                        match result with
                                        | Success _ ->
                                            AnsiConsole.MarkupLine(sprintf "  ✅ [green]Set %s[/] = [yellow]%s[/]" key value)
                                        | Failure (error, _) ->
                                            AnsiConsole.MarkupLine(sprintf "  ❌ [red]Failed to set %s: %s[/]" key (TarsError.toString error))
                            
                            let! _ = configManager.SaveConfigurationAsync(CancellationToken.None)
                            return { Message = ""; ExitCode = 0; Success = true }
                        else
                            AnsiConsole.MarkupLine("[bold cyan]⚙️ TARS Unified Configuration Management[/]")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("This system provides centralized configuration management")
                            AnsiConsole.MarkupLine("for all TARS modules with schema validation and hot-reloading.")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("[bold yellow]Features:[/]")
                            AnsiConsole.MarkupLine("  ⚙️ [cyan]Centralized Configuration[/] - Single source of truth")
                            AnsiConsole.MarkupLine("  🔍 [blue]Schema Validation[/] - Type-safe configuration")
                            AnsiConsole.MarkupLine("  🌍 [green]Environment Support[/] - Multiple environments")
                            AnsiConsole.MarkupLine("  📡 [yellow]Change Notifications[/] - Real-time updates")
                            AnsiConsole.MarkupLine("  📸 [magenta]Snapshots[/] - Configuration versioning")
                            AnsiConsole.MarkupLine("  🔥 [red]Hot-reloading[/] - Dynamic updates")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Available options:")
                            AnsiConsole.MarkupLine("  [yellow]--demo[/]           Run configuration demonstration")
                            AnsiConsole.MarkupLine("  [yellow]--show[/]           Show current configuration")
                            AnsiConsole.MarkupLine("  [yellow]--set key=value[/]  Set configuration value")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Examples:")
                            AnsiConsole.MarkupLine("  [dim]tars config --demo[/]")
                            AnsiConsole.MarkupLine("  [dim]tars config --show[/]")
                            AnsiConsole.MarkupLine("  [dim]tars config --set tars.core.logLevel=Debug[/]")
                            return { Message = ""; ExitCode = 0; Success = true }
                    
                    with
                    | ex ->
                        AnsiConsole.MarkupLine(sprintf "[red]❌ Command failed: %s[/]" ex.Message)
                        return { Message = ""; ExitCode = 1; Success = false }
                }

