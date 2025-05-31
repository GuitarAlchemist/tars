namespace TarsEngine.FSharp.Agents

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open FSharp.Control
open AgentTypes
open AgentPersonas
open AgentCommunication
open AgentTeams
open MetascriptAgent

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
        ActiveAgents: Map<AgentId, MetascriptAgent>
        ActiveTeams: Map<string, AgentTeam>
        MessageBus: MessageBus
        Config: OrchestratorConfig
        StartTime: DateTime
        LastHealthCheck: DateTime
    }
    
    /// TARS Agent Orchestrator
    type TarsAgentOrchestrator(config: OrchestratorConfig, logger: ILogger<TarsAgentOrchestrator>) =
        
        let messageBus = MessageBus(logger)
        
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
                        let agent = MetascriptAgent(persona, scriptPath, messageBus, logger)
                        let agentId = agent.GetId()
                        
                        state <- { state with ActiveAgents = state.ActiveAgents |> Map.add agentId agent }
                        
                        logger.LogInformation("Created agent {AgentId} with persona {PersonaName} and metascript {ScriptPath}", 
                                             agentId, persona.Name, scriptPath)
                        
                        if config.AutoStartAgents then
                            do! agent.StartAsync()
                        
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
                
                let teamAgents = ResizeArray<MetascriptAgent>()
                
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
                            Members = teamAgents |> Seq.map (fun a -> a.GetId()) |> Seq.toList
                            LeaderAgent = teamAgents |> Seq.tryFind (fun a -> a.GetPersona().Name = "Architect") |> Option.map (fun a -> a.GetId())
                    }
                    
                    let team = AgentTeam(teamConfig, messageBus, logger)
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
                
                let teamAgents = ResizeArray<MetascriptAgent>()
                
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
                            Members = teamAgents |> Seq.map (fun a -> a.GetId()) |> Seq.toList
                    }
                    
                    let team = AgentTeam(teamConfig, messageBus, logger)
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
            let metascriptContent = $"""DESCRIBE {{
    name: "{persona.Name} Agent Metascript"
    version: "1.0"
    author: "TARS Agent Orchestrator"
    description: "Basic metascript for {persona.Name} persona"
    autonomous: true
}}

CONFIG {{
    model: "llama3"
    temperature: 0.3
    persona: "{persona.Name}"
}}

VARIABLE agent_capabilities {{
    value: {persona.Capabilities |> List.map (fun c -> c.ToString()) |> String.concat ", "}
}}

FSHARP {{
    open System
    
    let executeAgentTask() =
        async {{
            printfn "🤖 {persona.Name} Agent: Starting autonomous task execution"
            printfn "📊 Capabilities: %s" agent_capabilities
            printfn "🎯 Specialization: {persona.Specialization}"
            printfn "🤝 Collaboration Preference: {persona.CollaborationPreference:F1}"
            
            // Simulate agent work based on persona
            for i in 1..5 do
                printfn "📋 {persona.Name}: Executing step %d/5" i
                do! Async.Sleep(2000)
                
                // Agent-specific behavior
                match "{persona.Name}" with
                | "Architect" -> printfn "🏗️ Designing system architecture..."
                | "Developer" -> printfn "💻 Writing and testing code..."
                | "Researcher" -> printfn "🔬 Gathering knowledge and insights..."
                | "Optimizer" -> printfn "⚡ Analyzing performance metrics..."
                | "Communicator" -> printfn "🤝 Facilitating team coordination..."
                | "Guardian" -> printfn "🛡️ Ensuring quality and security..."
                | "Innovator" -> printfn "💡 Exploring creative solutions..."
                | _ -> printfn "🤖 Performing general agent tasks..."
            
            printfn "✅ {persona.Name} Agent: Task execution completed"
            return true
        }}
    
    let! result = executeAgentTask()
    result
    
    output_variable: "agent_execution_result"
}}

ACTION {{
    type: "agent_completion"
    description: "{persona.Name} agent autonomous execution completed"
    
    FSHARP {{
        printfn ""
        printfn "🎉 {persona.Name} AGENT EXECUTION SUMMARY:"
        printfn "✅ Autonomous execution: COMPLETED"
        printfn "🎯 Persona specialization: {persona.Specialization}"
        printfn "📊 Task result: %A" agent_execution_result
        printfn "🤖 Agent ready for team collaboration"
        
        true
    }}
}}"""
            
            Directory.CreateDirectory(Path.GetDirectoryName(scriptPath)) |> ignore
            File.WriteAllText(scriptPath, metascriptContent)
            logger.LogInformation("Created basic metascript for {PersonaName}: {ScriptPath}", persona.Name, scriptPath)
        
        /// Start all agents
        member this.StartAllAgentsAsync() =
            task {
                logger.LogInformation("Starting all agents")
                
                let startTasks = 
                    state.ActiveAgents.Values
                    |> Seq.map (fun agent -> agent.StartAsync())
                    |> Seq.toArray
                
                do! Task.WhenAll(startTasks)
                
                logger.LogInformation("All {AgentCount} agents started", state.ActiveAgents.Count)
            }
        
        /// Stop all agents
        member this.StopAllAgentsAsync() =
            task {
                logger.LogInformation("Stopping all agents")
                
                let stopTasks = 
                    state.ActiveAgents.Values
                    |> Seq.map (fun agent -> agent.StopAsync())
                    |> Seq.toArray
                
                do! Task.WhenAll(stopTasks)
                
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
                        let agentState = agent.GetState()
                        {|
                            Id = agent.GetId()
                            Persona = agent.GetPersona().Name
                            Status = agentState.Agent.Status
                            TasksCompleted = agentState.PerformanceMetrics.TasksCompleted
                            SuccessRate = agentState.PerformanceMetrics.EfficiencyRating
                            LastActivity = agentState.Agent.LastActivity
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
                        let persona = agent.GetPersona()
                        requiredCapabilities |> List.forall (fun cap -> persona.Capabilities |> List.contains cap))
                    |> Seq.toList
                
                match suitableAgents with
                | [] ->
                    logger.LogWarning("No suitable agents found for task {TaskName}", taskName)
                    return None
                | [singleAgent] ->
                    // Assign to single agent
                    do! singleAgent.SendMessageAsync(
                        singleAgent.GetId(),
                        "TaskAssignment",
                        {| TaskName = taskName; Description = description; Requirements = requiredCapabilities |})
                    
                    logger.LogInformation("Task {TaskName} assigned to agent {AgentId}", 
                                         taskName, singleAgent.GetId())
                    return Some (Choice1Of2 singleAgent.GetId())
                | multipleAgents ->
                    // Assign to team if multiple agents needed
                    let teamName = $"Task Team for {taskName}"
                    let teamConfig = {
                        Name = teamName
                        Description = $"Temporary team for task: {taskName}"
                        LeaderAgent = Some (multipleAgents |> List.head).GetId()
                        Members = multipleAgents |> List.map (fun a -> a.GetId())
                        SharedObjectives = [taskName]
                        CommunicationProtocol = "Task-focused collaboration"
                        DecisionMakingProcess = "Leader-guided consensus"
                        ConflictResolution = "Capability-based resolution"
                    }
                    
                    let team = AgentTeam(teamConfig, messageBus, logger)
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
