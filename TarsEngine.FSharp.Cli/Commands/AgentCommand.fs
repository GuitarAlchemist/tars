namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Agents.AgentTypes
open TarsEngine.FSharp.Agents.AgentPersonas
open TarsEngine.FSharp.Agents.AgentOrchestrator

/// CLI command for managing TARS multi-agent system
type AgentCommand(logger: ILogger<AgentCommand>) =
    
    let mutable orchestrator: TarsAgentOrchestrator option = None
    
    interface ICommand with
        member _.Name = "agent"
        member _.Description = "Manage TARS multi-agent system with personas, teams, and communication"
        member _.Usage = "tars agent <subcommand> [options]"
        member _.Examples = [
            "tars agent start"
            "tars agent create-team development"
            "tars agent status"
            "tars agent assign-task \"code analysis\" --capabilities CodeAnalysis,Testing"
            "tars agent list-personas"
            "tars agent stop"
        ]
        member _.ValidateOptions(_) = true
        
        member _.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | "start" :: _ ->
                        return! this.StartAgentSystemAsync()
                    
                    | "stop" :: _ ->
                        return! this.StopAgentSystemAsync()
                    
                    | "status" :: _ ->
                        return! this.ShowStatusAsync()
                    
                    | "create-team" :: teamType :: _ ->
                        return! this.CreateTeamAsync(teamType)
                    
                    | "assign-task" :: taskName :: rest ->
                        let capabilities = this.ParseCapabilities(rest)
                        return! this.AssignTaskAsync(taskName, capabilities)
                    
                    | "list-personas" :: _ ->
                        return this.ListPersonas()
                    
                    | "demo" :: _ ->
                        return! this.RunDemoAsync()
                    
                    | [] | ["help"] ->
                        return this.ShowHelp()
                    
                    | unknown :: _ ->
                        printfn "❌ Unknown agent command: %s" unknown
                        return this.ShowHelp()
                        
                with
                | ex ->
                    logger.LogError(ex, "Error executing agent command")
                    return CommandResult.error($"Agent command failed: {ex.Message}")
            }
    
    /// Start the multi-agent system
    member private this.StartAgentSystemAsync() =
        task {
            try
                printfn "🤖 STARTING TARS MULTI-AGENT SYSTEM"
                printfn "==================================="
                printfn ""
                
                if orchestrator.IsSome then
                    printfn "⚠️ Agent system is already running"
                    return CommandResult.error("Agent system already running")
                else
                    // Create orchestrator configuration
                    let config = {
                        MaxAgents = 20
                        DefaultTeamSize = 5
                        MetascriptDirectory = ".tars/metascripts"
                        AutoStartAgents = true
                        HealthCheckInterval = TimeSpan.FromMinutes(5)
                        MessageRetentionHours = 24
                    }
                    
                    printfn "📊 Configuration:"
                    printfn "   Max Agents: %d" config.MaxAgents
                    printfn "   Metascript Directory: %s" config.MetascriptDirectory
                    printfn "   Auto Start: %b" config.AutoStartAgents
                    printfn ""
                    
                    // Initialize orchestrator
                    let newOrchestrator = TarsAgentOrchestrator(config, logger)
                    orchestrator <- Some newOrchestrator
                    
                    printfn "✅ Multi-agent orchestrator initialized"
                    printfn "🎯 Ready to create agents and teams"
                    printfn ""
                    printfn "💡 Next steps:"
                    printfn "   tars agent create-team development"
                    printfn "   tars agent create-team research"
                    printfn "   tars agent status"
                    
                    return CommandResult.success("Multi-agent system started successfully")
                    
            with
            | ex ->
                logger.LogError(ex, "Failed to start agent system")
                return CommandResult.error($"Failed to start agent system: {ex.Message}")
        }
    
    /// Stop the multi-agent system
    member private this.StopAgentSystemAsync() =
        task {
            try
                printfn "🛑 STOPPING TARS MULTI-AGENT SYSTEM"
                printfn "==================================="
                printfn ""
                
                match orchestrator with
                | Some orch ->
                    printfn "⏳ Stopping all agents..."
                    do! orch.StopAllAgentsAsync()
                    
                    orchestrator <- None
                    
                    printfn "✅ Multi-agent system stopped"
                    return CommandResult.success("Multi-agent system stopped successfully")
                | None ->
                    printfn "⚠️ Agent system is not running"
                    return CommandResult.error("Agent system is not running")
                    
            with
            | ex ->
                logger.LogError(ex, "Failed to stop agent system")
                return CommandResult.error($"Failed to stop agent system: {ex.Message}")
        }
    
    /// Show system status
    member private this.ShowStatusAsync() =
        task {
            try
                printfn "📊 TARS MULTI-AGENT SYSTEM STATUS"
                printfn "================================="
                printfn ""
                
                match orchestrator with
                | Some orch ->
                    let status = orch.GetStatus()
                    
                    printfn "🤖 SYSTEM OVERVIEW:"
                    printfn "   Status: OPERATIONAL"
                    printfn "   Active Agents: %d" status.ActiveAgents
                    printfn "   Active Teams: %d" status.ActiveTeams
                    printfn "   Uptime: %A" status.Uptime
                    printfn "   Message Bus: %s" (if status.MessageBusActive then "ACTIVE" else "INACTIVE")
                    printfn ""
                    
                    if status.AgentDetails.Length > 0 then
                        printfn "🤖 ACTIVE AGENTS:"
                        status.AgentDetails |> List.iteri (fun i agent ->
                            printfn "   %d. %s (ID: %A)" (i + 1) agent.Persona agent.Id
                            printfn "      Status: %A" agent.Status
                            printfn "      Tasks Completed: %d" agent.TasksCompleted
                            printfn "      Success Rate: %.1f%%" (agent.SuccessRate * 100.0)
                            printfn "      Last Activity: %A" agent.LastActivity
                            printfn "")
                    else
                        printfn "🤖 ACTIVE AGENTS: None"
                        printfn ""
                    
                    if status.TeamDetails.Length > 0 then
                        printfn "👥 ACTIVE TEAMS:"
                        status.TeamDetails |> List.iteri (fun i team ->
                            printfn "   %d. %s" (i + 1) team.Name
                            printfn "      Members: %d agents" team.MemberCount
                            printfn "      Tasks Completed: %d" team.TasksCompleted
                            printfn "      Success Rate: %.1f%%" (team.SuccessRate * 100.0)
                            printfn "      Collaboration Score: %.1f/10" (team.CollaborationScore * 10.0)
                            printfn "")
                    else
                        printfn "👥 ACTIVE TEAMS: None"
                        printfn ""
                    
                    return CommandResult.successWithData("System status retrieved" ) status
                | None ->
                    printfn "❌ SYSTEM STATUS: NOT RUNNING"
                    printfn ""
                    printfn "💡 Start the system with: tars agent start"
                    
                    return CommandResult.error("Agent system is not running")
                    
            with
            | ex ->
                logger.LogError(ex, "Failed to get system status")
                return CommandResult.error($"Failed to get system status: {ex.Message}")
        }
    
    /// Create a team
    member private this.CreateTeamAsync(teamType: string) =
        task {
            try
                printfn "👥 CREATING %s TEAM" (teamType.ToUpper())
                printfn "=========================="
                printfn ""
                
                match orchestrator with
                | Some orch ->
                    match teamType.ToLower() with
                    | "development" | "dev" ->
                        printfn "🏗️ Creating development team with specialized agents..."
                        let! team = orch.CreateDevelopmentTeamAsync()
                        
                        match team with
                        | Some t ->
                            let config = t.GetConfiguration()
                            printfn "✅ Development team created successfully!"
                            printfn "👑 Leader: %A" config.LeaderAgent
                            printfn "👥 Members: %d agents" config.Members.Length
                            printfn "🎯 Objectives: %s" (String.concat ", " config.SharedObjectives)
                            
                            return CommandResult.success("Development team created successfully")
                        | None ->
                            return CommandResult.error("Failed to create development team")
                    
                    | "research" ->
                        printfn "🔬 Creating research team with exploration focus..."
                        let! team = orch.CreateResearchTeamAsync()
                        
                        match team with
                        | Some t ->
                            let config = t.GetConfiguration()
                            printfn "✅ Research team created successfully!"
                            printfn "👥 Members: %d agents" config.Members.Length
                            printfn "🎯 Focus: %s" config.Description
                            
                            return CommandResult.success("Research team created successfully")
                        | None ->
                            return CommandResult.error("Failed to create research team")
                    
                    | unknown ->
                        printfn "❌ Unknown team type: %s" unknown
                        printfn "Available types: development, research"
                        return CommandResult.error($"Unknown team type: {unknown}")
                        
                | None ->
                    printfn "❌ Agent system is not running"
                    printfn "💡 Start with: tars agent start"
                    return CommandResult.error("Agent system is not running")
                    
            with
            | ex ->
                logger.LogError(ex, "Failed to create team")
                return CommandResult.error($"Failed to create team: {ex.Message}")
        }
    
    /// Assign a task
    member private this.AssignTaskAsync(taskName: string, capabilities: AgentCapability list) =
        task {
            try
                printfn "📋 ASSIGNING TASK: %s" taskName
                printfn "=========================="
                printfn ""
                
                match orchestrator with
                | Some orch ->
                    printfn "🎯 Task: %s" taskName
                    printfn "📊 Required Capabilities: %s" (capabilities |> List.map (fun c -> c.ToString()) |> String.concat ", ")
                    printfn ""
                    
                    let! result = orch.AssignTaskAsync(taskName, $"Task: {taskName}", capabilities)
                    
                    match result with
                    | Some (Choice1Of2 agentId) ->
                        printfn "✅ Task assigned to agent: %A" agentId
                        return CommandResult.success($"Task assigned to agent {agentId}")
                    | Some (Choice2Of2 teamName) ->
                        printfn "✅ Task assigned to team: %s" teamName
                        return CommandResult.success($"Task assigned to team {teamName}")
                    | None ->
                        printfn "❌ No suitable agents or teams found for this task"
                        return CommandResult.error("No suitable agents found for task")
                        
                | None ->
                    printfn "❌ Agent system is not running"
                    return CommandResult.error("Agent system is not running")
                    
            with
            | ex ->
                logger.LogError(ex, "Failed to assign task")
                return CommandResult.error($"Failed to assign task: {ex.Message}")
        }
    
    /// List available personas
    member private this.ListPersonas() =
        printfn "🎭 AVAILABLE AGENT PERSONAS"
        printfn "=========================="
        printfn ""
        
        let personas = getAllPersonas()
        
        personas |> List.iteri (fun i persona ->
            printfn "%d. %s" (i + 1) persona.Name
            printfn "   Description: %s" persona.Description
            printfn "   Specialization: %s" persona.Specialization
            printfn "   Capabilities: %s" (persona.Capabilities |> List.map (fun c -> c.ToString()) |> String.concat ", ")
            printfn "   Personality: %s" (persona.Personality |> List.map (fun p -> p.ToString()) |> String.concat ", ")
            printfn "   Collaboration: %.1f/10" (persona.CollaborationPreference * 10.0)
            printfn "   Learning Rate: %.1f/10" (persona.LearningRate * 10.0)
            printfn "")
        
        CommandResult.successWithData "Listed all personas" personas
    
    /// Run demonstration
    member private this.RunDemoAsync() =
        task {
            try
                printfn "🎬 RUNNING MULTI-AGENT SYSTEM DEMO"
                printfn "=================================="
                printfn ""
                
                // Start system if not running
                if orchestrator.IsNone then
                    let! startResult = this.StartAgentSystemAsync()
                    if not startResult.Success then
                        return startResult
                
                // Create teams
                let! devTeamResult = this.CreateTeamAsync("development")
                let! researchTeamResult = this.CreateTeamAsync("research")
                
                // Show status
                let! statusResult = this.ShowStatusAsync()
                
                printfn "🎉 Demo completed successfully!"
                printfn "💡 Use 'tars agent status' to monitor the system"
                
                return CommandResult.success("Multi-agent demo completed successfully")
                
            with
            | ex ->
                logger.LogError(ex, "Demo failed")
                return CommandResult.error($"Demo failed: {ex.Message}")
        }
    
    /// Parse capabilities from command line
    member private this.ParseCapabilities(args: string list) =
        args
        |> List.tryFind (fun arg -> arg.StartsWith("--capabilities"))
        |> Option.map (fun arg -> 
            arg.Substring(arg.IndexOf('=') + 1).Split(',')
            |> Array.choose (fun cap ->
                match cap.Trim() with
                | "CodeAnalysis" -> Some CodeAnalysis
                | "ProjectGeneration" -> Some ProjectGeneration
                | "Documentation" -> Some Documentation
                | "Testing" -> Some Testing
                | "Research" -> Some Research
                | "Planning" -> Some Planning
                | _ -> None)
            |> Array.toList)
        |> Option.defaultValue [CodeAnalysis; Planning]
    
    /// Show help
    member private this.ShowHelp() =
        printfn """🤖 TARS MULTI-AGENT SYSTEM COMMANDS
===================================

USAGE:
  tars agent <subcommand> [options]

SUBCOMMANDS:
  start                    Start the multi-agent system
  stop                     Stop the multi-agent system
  status                   Show system status and agent details
  create-team <type>       Create a team (development, research)
  assign-task <name>       Assign task to suitable agents
  list-personas            List all available agent personas
  demo                     Run complete demonstration
  help                     Show this help

EXAMPLES:
  tars agent start
  tars agent create-team development
  tars agent assign-task "code analysis" --capabilities=CodeAnalysis,Testing
  tars agent status
  tars agent demo

FEATURES:
  • 7 distinct agent personas with unique behaviors
  • Team formation and coordination
  • In-process communication via .NET Channels
  • Long-running agents with FSharp.Control.TaskSeq
  • Autonomous task assignment and execution
  • Real-time multi-agent collaboration"""
        
        CommandResult.success("Help displayed")
