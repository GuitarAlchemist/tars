namespace TarsEngine.FSharp.Agents

open System
open System.IO
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open FSharp.Control
open AgentTypes
open AgentPersonas
open AgentCommunication
open AgentTeams

/// TARS multi-agent orchestrator - manages agent lifecycle and coordination
module AgentOrchestrator =
    
    /// Agent orchestrator configuration
    type OrchestratorConfig = {
        MaxAgents: int
        DefaultTeamSize: int
        MetascriptDirectory: string
        AutoStartAgents: bool
        HealthCheckInterval: TimeSpan
        MessageRetentionHours: int
    }
    
    /// Orchestrator state
    type OrchestratorState = {
        ActiveAgents: Map<AgentId, Agent>
        ActiveTeams: Map<string, AgentTeam>
        MessageBus: MessageBus
        Config: OrchestratorConfig
        StartTime: DateTime
        LastHealthCheck: DateTime
    }
    
    /// Simple logger wrapper
    type SimpleLogger<'T>(baseLogger: ILogger) =
        interface ILogger<'T> with
            member _.BeginScope(state) = baseLogger.BeginScope(state)
            member _.IsEnabled(logLevel) = baseLogger.IsEnabled(logLevel)
            member _.Log(logLevel, eventId, state, ex, formatter) = baseLogger.Log(logLevel, eventId, state, ex, formatter)

    /// TARS Agent Orchestrator
    type TarsAgentOrchestrator(config: OrchestratorConfig, logger: ILogger<TarsAgentOrchestrator>) =

        let messageBusLogger = SimpleLogger<MessageBus>(logger :> ILogger)
        let messageBus = MessageBus(messageBusLogger)
        
        let mutable state = {
            ActiveAgents = Map.empty
            ActiveTeams = Map.empty
            MessageBus = messageBus
            Config = config
            StartTime = DateTime.UtcNow
            LastHealthCheck = DateTime.UtcNow
        }
        
        /// Create agent from persona and metascript
        member this.CreateAgentAsync(persona: AgentPersona, metascriptPath: string option) =
            task {
                try
                    let scriptPath = 
                        match metascriptPath with
                        | Some path -> path
                        | None -> 
                            // Find suitable metascript for persona
                            this.FindMetascriptForPersona(persona)
                    
                    if File.Exists(scriptPath) then
                        let agentId = AgentId(Guid.NewGuid())
                        let agent = {
                            Id = agentId
                            Persona = persona
                            Status = Initializing
                            Context = {
                                AgentId = agentId
                                WorkingDirectory = config.MetascriptDirectory
                                Variables = Map.empty
                                SharedMemory = Map.empty
                                CancellationToken = CancellationToken.None
                                Logger = logger :> ILogger
                            }
                            CurrentTasks = []
                            MessageQueue = System.Threading.Channels.Channel.CreateUnbounded<AgentMessage>()
                            MetascriptPath = Some scriptPath
                            StartTime = DateTime.UtcNow
                            LastActivity = DateTime.UtcNow
                            Statistics = Map.empty
                        }

                        state <- { state with ActiveAgents = state.ActiveAgents |> Map.add agentId agent }

                        logger.LogInformation("Created agent {AgentId} with persona {PersonaName} and metascript {ScriptPath}",
                                             agentId, persona.Name, scriptPath)

                        return Some agent
                    else
                        logger.LogWarning("Metascript not found: {ScriptPath}", scriptPath)
                        return None
                        
                with
                | ex ->
                    logger.LogError(ex, "Failed to create agent with persona {PersonaName}", persona.Name)
                    return None
            }
        
        /// Create a complete development team
        member this.CreateDevelopmentTeamAsync() =
            task {
                logger.LogInformation("Creating development team with multiple agent personas")
                
                let teamAgents = ResizeArray<Agent>()
                
                // Create architect agent
                let! architectAgent = this.CreateAgentAsync(architect, None)
                match architectAgent with
                | Some agent -> teamAgents.Add(agent)
                | None -> logger.LogWarning("Failed to create architect agent")
                
                // Create developer agent
                let! developerAgent = this.CreateAgentAsync(developer, None)
                match developerAgent with
                | Some agent -> teamAgents.Add(agent)
                | None -> logger.LogWarning("Failed to create developer agent")
                
                // Create researcher agent
                let! researcherAgent = this.CreateAgentAsync(researcher, None)
                match researcherAgent with
                | Some agent -> teamAgents.Add(agent)
                | None -> logger.LogWarning("Failed to create researcher agent")
                
                // Create optimizer agent
                let! optimizerAgent = this.CreateAgentAsync(optimizer, None)
                match optimizerAgent with
                | Some agent -> teamAgents.Add(agent)
                | None -> logger.LogWarning("Failed to create optimizer agent")
                
                // Create guardian agent
                let! guardianAgent = this.CreateAgentAsync(guardian, None)
                match guardianAgent with
                | Some agent -> teamAgents.Add(agent)
                | None -> logger.LogWarning("Failed to create guardian agent")
                
                if teamAgents.Count > 0 then
                    // Create team configuration
                    let teamConfig = {
                        CoordinationPatterns.developmentTeam with
                            Members = teamAgents |> Seq.map (fun a -> a.Id) |> Seq.toList
                            LeaderAgent = teamAgents |> Seq.tryFind (fun a -> a.Persona.Name = "Architect") |> Option.map (fun a -> a.Id)
                    }
                    
                    let teamLogger = SimpleLogger<AgentTeam>(logger :> ILogger)
                    let team = AgentTeam(teamConfig, messageBus, teamLogger)
                    state <- { state with ActiveTeams = state.ActiveTeams |> Map.add teamConfig.Name team }
                    
                    logger.LogInformation("Development team created with {AgentCount} agents", teamAgents.Count)
                    return Some team
                else
                    logger.LogError("Failed to create development team - no agents created")
                    return None
            }
        
        /// Create research team
        member this.CreateResearchTeamAsync() =
            task {
                logger.LogInformation("Creating research team")
                
                let teamAgents = ResizeArray<Agent>()
                
                // Create multiple researcher agents
                for i in 1..3 do
                    let! researcherAgent = this.CreateAgentAsync(researcher, None)
                    match researcherAgent with
                    | Some agent -> teamAgents.Add(agent)
                    | None -> logger.LogWarning("Failed to create researcher agent {Index}", i)
                
                // Add innovator agent
                let! innovatorAgent = this.CreateAgentAsync(innovator, None)
                match innovatorAgent with
                | Some agent -> teamAgents.Add(agent)
                | None -> logger.LogWarning("Failed to create innovator agent")
                
                if teamAgents.Count > 0 then
                    let teamConfig = {
                        CoordinationPatterns.researchTeam with
                            Members = teamAgents |> Seq.map (fun a -> a.Id) |> Seq.toList
                    }
                    
                    let teamLogger = SimpleLogger<AgentTeam>(logger :> ILogger)
                    let team = AgentTeam(teamConfig, messageBus, teamLogger)
                    state <- { state with ActiveTeams = state.ActiveTeams |> Map.add teamConfig.Name team }
                    
                    logger.LogInformation("Research team created with {AgentCount} agents", teamAgents.Count)
                    return Some team
                else
                    return None
            }
        
        /// Find suitable metascript for persona
        member private this.FindMetascriptForPersona(persona: AgentPersona) =
            let metascriptDir = config.MetascriptDirectory
            
            // Try to find persona-specific metascript
            let personaScript = Path.Combine(metascriptDir, $"{persona.Name.ToLower()}_agent.trsx")
            if File.Exists(personaScript) then
                personaScript
            else
                // Try preferred metascripts
                let preferredScript = 
                    persona.PreferredMetascripts
                    |> List.map (fun script -> Path.Combine(metascriptDir, script))
                    |> List.tryFind File.Exists
                
                match preferredScript with
                | Some script -> script
                | None ->
                    // Use default autonomous metascript
                    let defaultScript = Path.Combine(metascriptDir, "autonomous_improvement.trsx")
                    if File.Exists(defaultScript) then
                        defaultScript
                    else
                        // Create a basic metascript
                        let basicScript = Path.Combine(metascriptDir, $"basic_{persona.Name.ToLower()}.trsx")
                        this.CreateBasicMetascript(basicScript, persona)
                        basicScript
        
        /// Create basic metascript for persona
        member private this.CreateBasicMetascript(scriptPath: string, persona: AgentPersona) =
            let capabilitiesStr = persona.Capabilities |> List.map (fun c -> c.ToString()) |> String.concat ", "
            let metascriptContent =
                "DESCRIBE {\n" +
                "    name: \"" + persona.Name + " Agent Metascript\"\n" +
                "    version: \"1.0\"\n" +
                "    author: \"TARS Agent Orchestrator\"\n" +
                "    description: \"Basic metascript for " + persona.Name + " persona\"\n" +
                "    autonomous: true\n" +
                "}\n\n" +
                "CONFIG {\n" +
                "    model: \"llama3\"\n" +
                "    temperature: 0.3\n" +
                "    persona: \"" + persona.Name + "\"\n" +
                "}\n\n" +
                "VARIABLE agent_capabilities {\n" +
                "    value: \"" + capabilitiesStr + "\"\n" +
                "}\n\n" +
                "FSHARP {\n" +
                "    open System\n" +
                "    \n" +
                "    let executeAgentTask() =\n" +
                "        async {\n" +
                "            printfn \"🤖 " + persona.Name + " Agent: Starting autonomous task execution\"\n" +
                "            printfn \"📊 Capabilities: %s\" agent_capabilities\n" +
                "            printfn \"🎯 Specialization: " + persona.Specialization + "\"\n" +
                "            printfn \"🤝 Collaboration Preference: " + persona.CollaborationPreference.ToString("F1") + "\"\n" +
                "            \n" +
                "            // Simulate agent work based on persona\n" +
                "            for i in 1..5 do\n" +
                "                printfn \"📋 " + persona.Name + ": Executing step %d/5\" i\n" +
                "                do! Async.Sleep(2000)\n" +
                "                \n" +
                "                // Agent-specific behavior\n" +
                "                match \"" + persona.Name + "\" with\n" +
                "                | \"Architect\" -> printfn \"🏗️ Designing system architecture...\"\n" +
                "                | \"Developer\" -> printfn \"💻 Writing and testing code...\"\n" +
                "                | \"Researcher\" -> printfn \"🔬 Gathering knowledge and insights...\"\n" +
                "                | \"Optimizer\" -> printfn \"⚡ Analyzing performance metrics...\"\n" +
                "                | \"Communicator\" -> printfn \"🤝 Facilitating team coordination...\"\n" +
                "                | \"Guardian\" -> printfn \"🛡️ Ensuring quality and security...\"\n" +
                "                | \"Innovator\" -> printfn \"💡 Exploring creative solutions...\"\n" +
                "                | _ -> printfn \"🤖 Performing general agent tasks...\"\n" +
                "            \n" +
                "            printfn \"✅ " + persona.Name + " Agent: Task execution completed\"\n" +
                "            return true\n" +
                "        }\n" +
                "    \n" +
                "    let! result = executeAgentTask()\n" +
                "    result\n" +
                "    \n" +
                "    output_variable: \"agent_execution_result\"\n" +
                "}\n\n" +
                "ACTION {\n" +
                "    type: \"agent_completion\"\n" +
                "    description: \"" + persona.Name + " agent autonomous execution completed\"\n" +
                "    \n" +
                "    FSHARP {\n" +
                "        printfn \"\"\n" +
                "        printfn \"🎉 " + persona.Name + " AGENT EXECUTION SUMMARY:\"\n" +
                "        printfn \"✅ Autonomous execution: COMPLETED\"\n" +
                "        printfn \"🎯 Persona specialization: " + persona.Specialization + "\"\n" +
                "        printfn \"📊 Task result: %A\" agent_execution_result\n" +
                "        printfn \"🤖 Agent ready for team collaboration\"\n" +
                "        \n" +
                "        true\n" +
                "    }\n" +
                "}"
            
            Directory.CreateDirectory(Path.GetDirectoryName(scriptPath)) |> ignore
            File.WriteAllText(scriptPath, metascriptContent)
            logger.LogInformation("Created basic metascript for {PersonaName}: {ScriptPath}", persona.Name, scriptPath)
        
        /// Start all agents
        member this.StartAllAgentsAsync() =
            task {
                logger.LogInformation("Starting all agents")

                // Update agent status to Running
                let updatedAgents =
                    state.ActiveAgents
                    |> Map.map (fun _ agent -> { agent with Status = Running; LastActivity = DateTime.UtcNow })

                state <- { state with ActiveAgents = updatedAgents }

                logger.LogInformation("All {AgentCount} agents started", state.ActiveAgents.Count)
            }
        
        /// Stop all agents
        member this.StopAllAgentsAsync() =
            task {
                logger.LogInformation("Stopping all agents")

                // Update agent status to Stopped
                let updatedAgents =
                    state.ActiveAgents
                    |> Map.map (fun _ agent -> { agent with Status = Stopped; LastActivity = DateTime.UtcNow })

                state <- { state with ActiveAgents = updatedAgents }

                logger.LogInformation("All agents stopped")
            }
        
        /// Get orchestrator status
        member this.GetStatus() =
            {|
                ActiveAgents = state.ActiveAgents.Count
                ActiveTeams = state.ActiveTeams.Count
                Uptime = DateTime.UtcNow - state.StartTime
                LastHealthCheck = state.LastHealthCheck
                MessageBusActive = true
                AgentDetails =
                    state.ActiveAgents.Values
                    |> Seq.map (fun agent ->
                        {|
                            Id = agent.Id
                            Persona = agent.Persona.Name
                            Status = agent.Status
                            TasksCompleted = agent.CurrentTasks.Length
                            SuccessRate = 1.0 // Placeholder
                            LastActivity = agent.LastActivity
                        |})
                    |> Seq.toList
                TeamDetails =
                    state.ActiveTeams.Values
                    |> Seq.map (fun team ->
                        let config = team.GetConfiguration()
                        let metrics = team.GetMetrics()
                        {|
                            Name = config.Name
                            MemberCount = config.Members.Length
                            TasksCompleted = metrics.TasksCompleted
                            SuccessRate = metrics.SuccessRate
                            CollaborationScore = metrics.CollaborationScore
                        |})
                    |> Seq.toList
            |}
        
        /// Assign task to best suited agent or team
        member this.AssignTaskAsync(taskName: string, description: string, requiredCapabilities: AgentCapability list) =
            task {
                logger.LogInformation("Assigning task {TaskName} with capabilities: {Capabilities}", 
                                     taskName, requiredCapabilities)
                
                // Find suitable agents
                let suitableAgents =
                    state.ActiveAgents.Values
                    |> Seq.filter (fun agent ->
                        requiredCapabilities |> List.forall (fun cap -> agent.Persona.Capabilities |> List.contains cap))
                    |> Seq.toList
                
                match suitableAgents with
                | [] ->
                    logger.LogWarning("No suitable agents found for task {TaskName}", taskName)
                    return None
                | [singleAgent] ->
                    // Assign to single agent
                    logger.LogInformation("Task {TaskName} assigned to agent {AgentId}",
                                         taskName, singleAgent.Id)
                    return Some (Choice1Of2 singleAgent.Id)
                | multipleAgents ->
                    // Assign to team if multiple agents needed
                    let teamName = $"Task Team for {taskName}"
                    let teamConfig = {
                        Name = teamName
                        Description = $"Temporary team for task: {taskName}"
                        LeaderAgent = Some ((multipleAgents |> List.head).Id)
                        Members = multipleAgents |> List.map (fun a -> a.Id)
                        SharedObjectives = [taskName]
                        CommunicationProtocol = "Task-focused collaboration"
                        DecisionMakingProcess = "Leader-guided consensus"
                        ConflictResolution = "Capability-based resolution"
                    }
                    
                    let teamLogger = SimpleLogger<AgentTeam>(logger :> ILogger)
                    let team = AgentTeam(teamConfig, messageBus, teamLogger)
                    state <- { state with ActiveTeams = state.ActiveTeams |> Map.add teamName team }
                    
                    let! assignedAgent = team.AssignTaskAsync(taskName, description, requiredCapabilities)
                    
                    logger.LogInformation("Task {TaskName} assigned to team {TeamName}", taskName, teamName)
                    return Some (Choice2Of2 teamName)
            }
        
        /// Get all active agents
        member this.GetActiveAgents() = state.ActiveAgents.Values |> Seq.toList
        
        /// Get all active teams
        member this.GetActiveTeams() = state.ActiveTeams.Values |> Seq.toList
        
        /// Get message bus
        member this.GetMessageBus() = messageBus
