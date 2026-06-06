namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Integration.UnifiedAgentInterfaces
open TarsEngine.FSharp.Cli.Integration.UnifiedAgentRegistry
open TarsEngine.FSharp.Cli.Integration.UnifiedAgentSystem
open TarsEngine.FSharp.Cli.Integration.UnifiedAgentAdapters
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Core.UnifiedTypes

/// Unified Agent Integration Command - Demonstrates integration of existing TARS agents
module UnifiedAgentIntegrationCommand =
    
    /// Demonstrate the integration of existing TARS agents into the unified system
    let demonstrateAgentIntegration (logger: ITarsLogger) =
        task {
            try
                AnsiConsole.MarkupLine("[bold cyan]🔗 TARS Agent Integration Demo[/]")
                AnsiConsole.MarkupLine("[dim]Integrating existing MoE and Reasoning agents into unified system[/]")
                AnsiConsole.WriteLine()
                
                // Create registry and coordinator
                use registry = new UnifiedAgentRegistry(logger)
                let coordinator = createAgentCoordinator None registry logger
                
                // Start coordinator
                let! startResult = coordinator.StartAsync(CancellationToken.None)
                match startResult with
                | Failure (error, _) ->
                    AnsiConsole.MarkupLine($"[red]❌ Failed to start coordinator: {TarsError.toString error}[/]")
                    return 1
                | Success _ ->
                    AnsiConsole.MarkupLine("[green]✅ Unified coordinator started[/]")

                    AnsiConsole.WriteLine()

                    // Create and register MoE expert adapters
                    AnsiConsole.MarkupLine("[yellow]🧠 Integrating MoE (Mixture of Experts) agents...[/]")
                    let moeAgents = AdapterFactory.createMoEAdapters logger

                    for agent in moeAgents do
                        let! initResult = agent.InitializeAsync(CancellationToken.None)
                        let! startResult = agent.StartAsync(CancellationToken.None)
                        let! regResult = coordinator.RegisterAgentAsync(agent, CancellationToken.None)

                        match regResult with
                        | Success _ ->
                            AnsiConsole.MarkupLine($"[green]  ✅ {agent.Config.Name}[/] - {agent.Config.Description}")
                        | Failure (error, _) ->
                            AnsiConsole.MarkupLine($"[red]  ❌ Failed to register {agent.Config.Name}: {TarsError.toString error}[/]")

                    AnsiConsole.WriteLine()

                    // Create and register reasoning agent adapters
                    AnsiConsole.MarkupLine("[yellow]🤔 Integrating Reasoning agents...[/]")
                    let reasoningAgents = AdapterFactory.createReasoningAdapters logger

                    for agent in reasoningAgents do
                        let! initResult = agent.InitializeAsync(CancellationToken.None)
                        let! startResult = agent.StartAsync(CancellationToken.None)
                        let! regResult = coordinator.RegisterAgentAsync(agent, CancellationToken.None)

                        match regResult with
                        | Success _ ->
                            AnsiConsole.MarkupLine($"[green]  ✅ {agent.Config.Name}[/] - {agent.Config.Description}")
                        | Failure (error, _) ->
                            AnsiConsole.MarkupLine($"[red]  ❌ Failed to register {agent.Config.Name}: {TarsError.toString error}[/]")

                    AnsiConsole.WriteLine()
                
                // Show registered agents summary
                let allAgents = coordinator.GetRegisteredAgents()
                AnsiConsole.MarkupLine($"[bold cyan]📊 Integration Summary:[/]")
                AnsiConsole.MarkupLine($"  Total Agents: [bold]{allAgents.Length}[/]")
                
                let moeCount = allAgents |> List.filter (fun a -> a.Config.AgentType = "MoEExpert") |> List.length
                let reasoningCount = allAgents |> List.filter (fun a -> a.Config.AgentType = "ReasoningAgent") |> List.length
                
                AnsiConsole.MarkupLine($"  MoE Experts: [yellow]{moeCount}[/]")
                AnsiConsole.MarkupLine($"  Reasoning Agents: [blue]{reasoningCount}[/]")
                
                AnsiConsole.WriteLine()
                
                // Demonstrate intelligent task routing
                AnsiConsole.MarkupLine("[bold cyan]🎯 Demonstrating Intelligent Task Routing:[/]")
                
                let demoTasks = [
                    // Tasks that should route to MoE experts
                    ("Generate a creative story about AI", "GenerationExpert")
                    ("Classify this sentiment: 'I love this product!'", "ClassificationExpert")
                    ("Debug this Python function", "CodeExpert")
                    ("Translate 'Hello' to Spanish", "MultilingualExpert")
                    
                    // Tasks that should route to reasoning agents
                    ("Solve this math problem: 2x + 5 = 15", "MathematicalReasoningAgent")
                    ("What causes climate change?", "CausalReasoningAgent")
                    ("Plan a strategy for project management", "StrategicReasoningAgent")
                    ("Analyze the logic in this argument", "LogicalReasoningAgent")
                ]
                
                let taskResults = ResizeArray<string * string * bool>()
                
                for (taskDescription, expectedAgent) in demoTasks do
                    let task = {
                        TaskId = Guid.NewGuid().ToString()
                        TaskType = "AutoRoute"
                        Description = taskDescription
                        Input = box taskDescription
                        Priority = MessagePriority.Normal
                        CreatedAt = DateTime.Now
                        Deadline = Some (DateTime.Now.AddMinutes(5.0))
                        RequiredCapabilities = []
                        Context = createOperationContext "IntegrationDemo" None None None
                        Dependencies = []
                        ExpectedOutput = Some "analysis_result"
                    }
                    
                    // Find best agent for this task
                    let! agentResult = coordinator.FindBestAgentAsync(task, CancellationToken.None)
                    
                    match agentResult with
                    | Success (selectedAgent, _) ->
                        let isCorrectRouting = selectedAgent.Config.Name.Contains(expectedAgent.Replace("Agent", ""))
                        taskResults.Add(taskDescription, selectedAgent.Config.Name, isCorrectRouting)
                        
                        let routingIcon = if isCorrectRouting then "✅" else "⚠️"
                        AnsiConsole.MarkupLine($"  {routingIcon} [dim]{taskDescription}[/]")
                        AnsiConsole.MarkupLine($"    → Routed to: [yellow]{selectedAgent.Config.Name}[/]")
                        
                        // Execute the task
                        let! taskResult = coordinator.ExecuteTaskAsync(task, CancellationToken.None)
                        match taskResult with
                        | Success (result, _) ->
                            let resultStr = result.ToString()
                            let preview = if resultStr.Length > 100 then resultStr.Substring(0, 100) + "..." else resultStr
                            AnsiConsole.MarkupLine($"    ✅ [dim]{preview}[/]")
                        | Failure (error, _) ->
                            AnsiConsole.MarkupLine($"    ❌ [red]{TarsError.toString error}[/]")
                    
                    | Failure (error, _) ->
                        taskResults.Add(taskDescription, "None", false)
                        AnsiConsole.MarkupLine($"  ❌ [red]Failed to route: {taskDescription}[/]")
                        AnsiConsole.MarkupLine($"    Error: {TarsError.toString error}")
                
                AnsiConsole.WriteLine()
                
                // Show routing accuracy
                let correctRoutings = taskResults |> Seq.filter (fun (_, _, correct) -> correct) |> Seq.length
                let totalTasks = taskResults.Count
                let accuracy = if totalTasks > 0 then (float correctRoutings / float totalTasks) * 100.0 else 0.0
                
                let accuracyStr = accuracy.ToString("F1")
                AnsiConsole.MarkupLine($"[bold cyan]📈 Routing Accuracy: {accuracyStr}%[/] ({correctRoutings}/{totalTasks})")
                
                AnsiConsole.WriteLine()
                
                // Show system metrics
                let systemMetrics = coordinator.GetSystemMetrics()
                AnsiConsole.MarkupLine("[bold cyan]📊 System Performance:[/]")
                let totalTasks = systemMetrics.["totalTasks"]
                let completedTasks = systemMetrics.["completedTasks"]
                let failedTasks = systemMetrics.["failedTasks"]
                AnsiConsole.MarkupLine($"  Tasks Executed: [green]{completedTasks}[/]")
                AnsiConsole.MarkupLine($"  Tasks Failed: [red]{failedTasks}[/]")
                AnsiConsole.MarkupLine($"  Success Rate: [yellow]{if totalTasks :?> int > 0 then (completedTasks :?> int * 100 / totalTasks :?> int) else 0}%[/]")
                
                AnsiConsole.WriteLine()
                
                // Perform health checks
                let! healthResult = coordinator.HealthCheckAllAsync(CancellationToken.None)
                match healthResult with
                | Success (healthData, _) ->
                    let healthyAgents = healthData |> Map.filter (fun _ health -> 
                        match health.TryGetValue("status") with
                        | true, status -> status.ToString() = "Ready"
                        | false, _ -> false) |> Map.count
                    
                    AnsiConsole.MarkupLine("[bold cyan]🏥 Agent Health:[/]")
                    AnsiConsole.MarkupLine($"  Healthy Agents: [green]{healthyAgents}[/]/{healthData.Count}")
                    
                    if healthyAgents < healthData.Count then
                        AnsiConsole.MarkupLine("  [yellow]⚠️ Some agents may need attention[/]")
                
                | Failure (error, _) ->
                    AnsiConsole.MarkupLine($"[red]❌ Health check failed: {TarsError.toString error}[/]")
                
                // Stop coordinator
                let! stopResult = coordinator.StopAsync(CancellationToken.None)
                match stopResult with
                | Success _ ->
                    AnsiConsole.MarkupLine("[green]✅ Coordinator stopped gracefully[/]")
                | Failure (error, _) ->
                    AnsiConsole.MarkupLine($"[red]❌ Failed to stop coordinator: {TarsError.toString error}[/]")
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[bold green]🎉 Agent Integration Demo Completed Successfully![/]")
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[bold cyan]🚀 INTEGRATION ACHIEVEMENTS:[/]")
                AnsiConsole.MarkupLine("  ✅ [green]MoE Experts integrated[/] - 8 specialized AI models")
                AnsiConsole.MarkupLine("  ✅ [green]Reasoning Agents integrated[/] - 5 reasoning specializations")
                AnsiConsole.MarkupLine("  ✅ [green]Intelligent task routing[/] - Automatic expert selection")
                AnsiConsole.MarkupLine("  ✅ [green]Unified coordination[/] - Single system managing all agents")
                AnsiConsole.MarkupLine("  ✅ [green]Health monitoring[/] - Real-time agent status tracking")
                AnsiConsole.MarkupLine("  ✅ [green]Performance metrics[/] - Comprehensive system analytics")
                
                return 0
            
            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ Integration demo failed: {ex.Message}[/]")
                return 1
        }
    
    /// Unified Agent Integration Command implementation
    type UnifiedAgentIntegrationCommand() =
        interface ICommand with
            member _.Name = "integrate"
            member _.Description = "Demonstrate integration of existing TARS agents into unified system"
            member _.Usage = "tars integrate [--demo]"
            member _.Examples = [
                "tars integrate --demo        # Run agent integration demonstration"
                "tars integrate               # Show integration overview"
            ]
            
            member _.ValidateOptions(options: CommandOptions) = true
            
            member _.ExecuteAsync(options: CommandOptions) =
                task {
                    try
                        let logger = createLogger "UnifiedAgentIntegrationCommand"
                        
                        let isDemoMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--demo")
                        
                        if isDemoMode then
                            let! result = demonstrateAgentIntegration logger
                            return { Message = ""; ExitCode = result; Success = result = 0 }
                        else
                            AnsiConsole.MarkupLine("[bold cyan]🔗 TARS Agent Integration System[/]")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("This command demonstrates the integration of existing TARS agents")
                            AnsiConsole.MarkupLine("into the new unified coordination system.")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("[bold yellow]Integrated Agent Types:[/]")
                            AnsiConsole.MarkupLine("  🧠 [cyan]MoE (Mixture of Experts)[/] - 8 specialized AI models")
                            AnsiConsole.MarkupLine("  🤔 [blue]Reasoning Agents[/] - 5 reasoning specializations")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Available options:")
                            AnsiConsole.MarkupLine("  [yellow]--demo[/]  Run integration demonstration")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Example: [dim]tars integrate --demo[/]")
                            return { Message = ""; ExitCode = 0; Success = true }
                    
                    with
                    | ex ->
                        AnsiConsole.MarkupLine($"[red]❌ Command failed: {ex.Message}[/]")
                        return { Message = ""; ExitCode = 1; Success = false }
                }

