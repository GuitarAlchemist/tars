namespace TarsEngine.FSharp.Core.Services

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Types
open TarsEngine.FSharp.Agents.AgentTeams
open TarsEngine.FSharp.Core.Metascript.FractalGrammarMetascripts

/// Real-time integration between 3D visualization and functional agent teams
module Agent3DIntegrationService =
    
    /// 3D agent state for visualization
    type Agent3DState = {
        Id: string
        AgentType: GameTheoryAgentType
        Position: float * float * float
        Color: uint32
        Size: float
        Performance: float
        IsActive: bool
        LastActivity: DateTime
        ConnectionIds: string list
        TeamMemberships: string list
        CurrentTask: string option
        MessageCount: int
        ProcessingLoad: float
    }
    
    /// Team 3D visualization state
    type Team3DState = {
        TeamId: string
        Name: string
        MemberIds: string list
        CenterPosition: float * float * float
        CoordinationLevel: float
        Strategy: CoordinationStrategy
        IsActive: bool
        TasksInProgress: int
        PerformanceMetrics: Map<string, float>
    }
    
    /// 3D scene update message
    type Scene3DUpdate =
        | AgentSpawned of Agent3DState
        | AgentUpdated of Agent3DState
        | AgentTerminated of string
        | TeamFormed of Team3DState
        | TeamDisbanded of string
        | ConnectionEstablished of string * string
        | ConnectionRemoved of string * string
        | PerformanceUpdate of string * float
        | SystemMetricsUpdate of Map<string, obj>
    
    /// Real-time 3D integration service
    type Agent3DIntegrationService(logger: ILogger<Agent3DIntegrationService>) =
        
        let activeAgents = ConcurrentDictionary<string, Agent3DState>()
        let activeTeams = ConcurrentDictionary<string, Team3DState>()
        let sceneUpdateCallbacks = ResizeArray<Scene3DUpdate -> unit>()
        let fractalExecutor = FractalMetascriptExecutor()
        
        /// Subscribe to 3D scene updates
        member this.Subscribe3DUpdates(callback: Scene3DUpdate -> unit) =
            sceneUpdateCallbacks.Add(callback)
            logger.LogInformation("🎯 New 3D visualization subscriber registered")
        
        /// Broadcast update to all subscribers
        member private this.BroadcastUpdate(update: Scene3DUpdate) =
            for callback in sceneUpdateCallbacks do
                try
                    callback(update)
                with
                | ex -> logger.LogError(ex, "Error in 3D update callback")
        
        /// Convert game theory agent type to 3D properties
        member private this.GetAgent3DProperties(agentType: GameTheoryAgentType) : uint32 * float =
            match agentType with
            | QuantalResponseEquilibrium(_) -> (0x4a9effu, 0.8f) // Blue, large
            | CognitiveHierarchy(_) -> (0x00ff88u, 0.7f) // Green, medium-large
            | NoRegretLearning(_) -> (0xffaa00u, 0.9f) // Yellow, largest (best performance)
            | EvolutionaryGameTheory(_) -> (0x9b59b6u, 0.6f) // Purple, medium
            | CorrelatedEquilibrium(_) -> (0xff6b6bu, 0.75f) // Red, medium-large
            | MachineLearningAgent(_) -> (0xffffffu, 0.5f) // White, small
        
        /// Generate random 3D position for new agent
        member private this.GenerateAgentPosition(teamId: string option) : float * float * float =
            let random = Random()
            match teamId with
            | Some team when activeTeams.ContainsKey(team) ->
                // Position near team center
                let teamState = activeTeams.[team]
                let (cx, cy, cz) = teamState.CenterPosition
                (cx + random.NextDouble() * 4.0 - 2.0,
                 cy + random.NextDouble() * 4.0 - 2.0,
                 cz + random.NextDouble() * 4.0 - 2.0)
            | _ ->
                // Random position in space
                (random.NextDouble() * 16.0 - 8.0,
                 random.NextDouble() * 16.0 - 8.0,
                 random.NextDouble() * 16.0 - 8.0)
        
        /// Spawn new agent with 3D visualization
        member this.SpawnAgent(agentType: GameTheoryAgentType, ?teamId: string) : string =
            let agentId = Guid.NewGuid().ToString("N")[..7]
            let (color, baseSize) = this.GetAgent3DProperties(agentType)
            let position = this.GenerateAgentPosition(teamId)
            
            let agent3D = {
                Id = agentId
                AgentType = agentType
                Position = position
                Color = color
                Size = baseSize
                Performance = 0.5 + Random().NextDouble() * 0.5
                IsActive = true
                LastActivity = DateTime.UtcNow
                ConnectionIds = []
                TeamMemberships = teamId |> Option.toList
                CurrentTask = None
                MessageCount = 0
                ProcessingLoad = 0.0
            }
            
            activeAgents.[agentId] <- agent3D
            this.BroadcastUpdate(AgentSpawned(agent3D))
            
            logger.LogInformation("🚀 Agent {AgentId} spawned with type {AgentType} at position {Position}",
                                 agentId, agentType, position)
            agentId
        
        /// Update agent performance and 3D properties
        member this.UpdateAgentPerformance(agentId: string, performance: float, ?processingLoad: float) =
            match activeAgents.TryGetValue(agentId) with
            | true, agent ->
                let updatedAgent = { 
                    agent with 
                        Performance = performance
                        ProcessingLoad = defaultArg processingLoad agent.ProcessingLoad
                        LastActivity = DateTime.UtcNow
                        Size = agent.Size * (0.8 + performance * 0.4) // Size reflects performance
                }
                activeAgents.[agentId] <- updatedAgent
                this.BroadcastUpdate(AgentUpdated(updatedAgent))
                
                logger.LogDebug("📊 Agent {AgentId} performance updated to {Performance:F2}",
                               agentId, performance)
            | false, _ ->
                logger.LogWarning("⚠️ Attempted to update non-existent agent {AgentId}", agentId)
        
        /// Form team with 3D visualization
        member this.FormTeam(teamName: string, memberIds: string list, strategy: CoordinationStrategy) : string =
            let teamId = Guid.NewGuid().ToString("N")[..7]
            
            // Calculate team center position
            let memberPositions = 
                memberIds 
                |> List.choose (fun id -> 
                    match activeAgents.TryGetValue(id) with
                    | true, agent -> Some agent.Position
                    | false, _ -> None)
            
            let centerPosition = 
                if memberPositions.IsEmpty then
                    (0.0, 0.0, 0.0)
                else
                    let (sumX, sumY, sumZ) = 
                        memberPositions 
                        |> List.fold (fun (x, y, z) (px, py, pz) -> (x + px, y + py, z + pz)) (0.0, 0.0, 0.0)
                    let count = float memberPositions.Length
                    (sumX / count, sumY / count, sumZ / count)
            
            let teamState = {
                TeamId = teamId
                Name = teamName
                MemberIds = memberIds
                CenterPosition = centerPosition
                CoordinationLevel = 0.5
                Strategy = strategy
                IsActive = true
                TasksInProgress = 0
                PerformanceMetrics = Map.empty
            }
            
            activeTeams.[teamId] <- teamState
            
            // Update agent team memberships
            for memberId in memberIds do
                match activeAgents.TryGetValue(memberId) with
                | true, agent ->
                    let updatedAgent = { 
                        agent with 
                            TeamMemberships = teamId :: agent.TeamMemberships 
                    }
                    activeAgents.[memberId] <- updatedAgent
                    this.BroadcastUpdate(AgentUpdated(updatedAgent))
                | false, _ -> ()
            
            this.BroadcastUpdate(TeamFormed(teamState))
            
            logger.LogInformation("🤝 Team {TeamName} ({TeamId}) formed with {MemberCount} members using {Strategy} strategy",
                                 teamName, teamId, memberIds.Length, strategy)
            teamId
        
        /// Execute fractal metascript with 3D visualization
        member this.ExecuteFractalMetascriptWith3D(metascriptContent: string) : Async<ExecutionResult> =
            async {
                try
                    let parser = FractalGrammarParser()
                    let fractalBlock = parser.ParseFractalMetascript(metascriptContent)
                    
                    logger.LogInformation("🌀 Executing fractal metascript with {RuleCount} rules",
                                         fractalBlock.Rules.Length)
                    
                    // Execute fractal rules and update 3D visualization
                    for rule in fractalBlock.Rules do
                        do! this.ExecuteFractalRuleWith3D(rule)
                    
                    let! result = fractalExecutor.ExecuteFractalMetascript(fractalBlock)
                    
                    // Update system metrics in 3D
                    let systemMetrics = Map.ofList [
                        ("active_agents", box activeAgents.Count)
                        ("active_teams", box activeTeams.Count)
                        ("avg_performance", box (this.CalculateAveragePerformance()))
                        ("coordination_level", box (this.CalculateCoordinationLevel()))
                    ]
                    this.BroadcastUpdate(SystemMetricsUpdate(systemMetrics))
                    
                    return result
                with
                | ex ->
                    logger.LogError(ex, "Error executing fractal metascript with 3D integration")
                    return {
                        Success = false
                        Output = None
                        Error = Some ex.Message
                        ExecutionTime = TimeSpan.Zero
                        Metadata = Map.empty
                    }
            }
        
        /// Execute fractal rule with 3D effects
        member private this.ExecuteFractalRuleWith3D(rule: FractalRule) : Async<unit> =
            async {
                match rule with
                | SpawnAgentTeam(agentType, count, strategy) ->
                    let agentIds = [1..count] |> List.map (fun _ -> this.SpawnAgent(agentType))
                    if agentIds.Length > 1 then
                        let teamId = this.FormTeam($"Auto Team {DateTime.Now:HHmmss}", agentIds, strategy)
                        logger.LogInformation("🎯 Auto-formed team {TeamId} with {Count} {AgentType} agents",
                                             teamId, count, agentType)
                
                | TeamMerge(team1, team2) ->
                    // Merge teams in 3D space
                    match activeTeams.TryGetValue(team1), activeTeams.TryGetValue(team2) with
                    | (true, t1), (true, t2) ->
                        let mergedMembers = t1.MemberIds @ t2.MemberIds |> List.distinct
                        let newTeamId = this.FormTeam($"Merged {t1.Name}-{t2.Name}", mergedMembers, t1.Strategy)
                        
                        activeTeams.TryRemove(team1) |> ignore
                        activeTeams.TryRemove(team2) |> ignore
                        this.BroadcastUpdate(TeamDisbanded(team1))
                        this.BroadcastUpdate(TeamDisbanded(team2))
                        
                        logger.LogInformation("🔗 Teams {Team1} and {Team2} merged into {NewTeam}",
                                             team1, team2, newTeamId)
                    | _ -> ()
                
                | AgentInteraction(agent1, agent2, interactionType) ->
                    // Create visual connection between agents
                    this.BroadcastUpdate(ConnectionEstablished(agent1, agent2))
                    
                    // Simulate interaction effects
                    do! Async.Sleep(100)
                    this.UpdateAgentPerformance(agent1, 0.8)
                    this.UpdateAgentPerformance(agent2, 0.8)
                
                | _ -> 
                    // Other rules handled by base executor
                    ()
            }
        
        /// Calculate average performance across all agents
        member private this.CalculateAveragePerformance() : float =
            if activeAgents.IsEmpty then 0.0
            else
                activeAgents.Values
                |> Seq.map (fun agent -> agent.Performance)
                |> Seq.average
        
        /// Calculate overall coordination level
        member private this.CalculateCoordinationLevel() : float =
            if activeTeams.IsEmpty then 0.5
            else
                activeTeams.Values
                |> Seq.map (fun team -> team.CoordinationLevel)
                |> Seq.average
        
        /// Get current 3D scene state
        member this.GetCurrentSceneState() : Agent3DState list * Team3DState list =
            (activeAgents.Values |> Seq.toList, activeTeams.Values |> Seq.toList)
        
        /// Generate JavaScript for 3D scene updates
        member this.GenerateSceneUpdateScript() : string =
            let agents = activeAgents.Values |> Seq.toList
            let teams = activeTeams.Values |> Seq.toList
            
            let agentUpdates = 
                agents 
                |> List.map (fun agent ->
                    let (x, y, z) = agent.Position
                    $"updateAgent('{agent.Id}', {x}, {y}, {z}, {agent.Performance}, {agent.Size}, 0x{agent.Color:X});")
                |> String.concat "\n"
            
            let teamUpdates = 
                teams
                |> List.map (fun team ->
                    let (x, y, z) = team.CenterPosition
                    $"updateTeam('{team.TeamId}', '{team.Name}', {x}, {y}, {z}, {team.CoordinationLevel});")
                |> String.concat "\n"
            
            agentUpdates + "\n" + teamUpdates
