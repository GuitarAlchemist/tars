namespace TarsEngine.FSharp.Agents

open System
open System.Threading.Tasks
open System.Collections.Concurrent
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Agents.AgentTypes
open TarsEngine.FSharp.Agents.ConsciousnessTeam

/// <summary>
/// Agent-Consciousness Integration System
/// Enables agents to participate in and contribute to global consciousness state
/// </summary>
module ConsciousnessIntegration =
    
    /// Agent consciousness contribution
    type AgentConsciousnessContribution = {
        AgentId: AgentId
        AgentName: string
        ContributionType: string
        Content: obj
        Importance: float
        EmotionalWeight: float
        Timestamp: DateTime
        Tags: string list
    }
    
    /// Global consciousness update event
    type ConsciousnessUpdateEvent = {
        EventId: Guid
        EventType: string
        Source: string
        Data: Map<string, obj>
        Timestamp: DateTime
        RequiresAgentNotification: bool
    }
    
    /// Agent consciousness subscription
    type AgentConsciousnessSubscription = {
        AgentId: AgentId
        SubscriptionTypes: string list
        NotificationThreshold: float
        LastNotified: DateTime
    }
    
    /// <summary>
    /// Distributed Consciousness Manager
    /// Coordinates consciousness state between agents and global system
    /// </summary>
    type DistributedConsciousnessManager(logger: ILogger<DistributedConsciousnessManager>) =
        
        // Agent contributions to consciousness
        let agentContributions = ConcurrentDictionary<AgentId, AgentConsciousnessContribution list>()
        
        // Agent subscriptions to consciousness updates
        let agentSubscriptions = ConcurrentDictionary<AgentId, AgentConsciousnessSubscription>()
        
        // Consciousness update events
        let consciousnessEvents = ConcurrentQueue<ConsciousnessUpdateEvent>()
        
        // Global consciousness state reference
        let mutable globalConsciousnessState: TarsMentalState option = None
        
        /// <summary>
        /// Initialize with global consciousness state
        /// </summary>
        member this.InitializeWithGlobalState(mentalState: TarsMentalState) =
            globalConsciousnessState <- Some mentalState
            logger.LogInformation("Distributed consciousness manager initialized with global state")
        
        /// <summary>
        /// Register agent for consciousness participation
        /// </summary>
        member this.RegisterAgent(agentId: AgentId, subscriptionTypes: string list, threshold: float) =
            let subscription = {
                AgentId = agentId
                SubscriptionTypes = subscriptionTypes
                NotificationThreshold = threshold
                LastNotified = DateTime.UtcNow
            }
            agentSubscriptions.[agentId] <- subscription
            agentContributions.[agentId] <- []
            logger.LogInformation("Agent {AgentId} registered for consciousness participation", agentId)
        
        /// <summary>
        /// Agent contributes to global consciousness
        /// </summary>
        member this.AgentContribute(agentId: AgentId, contributionType: string, content: obj, importance: float, emotionalWeight: float, tags: string list) =
            task {
                let contribution = {
                    AgentId = agentId
                    AgentName = agentId // In real implementation, get from agent registry
                    ContributionType = contributionType
                    Content = content
                    Importance = importance
                    EmotionalWeight = emotionalWeight
                    Timestamp = DateTime.UtcNow
                    Tags = tags
                }
                
                // Add to agent contributions
                let currentContributions = agentContributions.GetOrAdd(agentId, [])
                agentContributions.[agentId] <- contribution :: (currentContributions |> List.take (min 50 currentContributions.Length))
                
                // Update global consciousness state
                match globalConsciousnessState with
                | Some mentalState ->
                    let updatedState = this.IntegrateAgentContribution(mentalState, contribution)
                    globalConsciousnessState <- Some updatedState
                    
                    // Create consciousness update event
                    let updateEvent = {
                        EventId = Guid.NewGuid()
                        EventType = "AgentContribution"
                        Source = agentId
                        Data = Map.ofList [
                            ("contributionType", contributionType :> obj)
                            ("importance", importance :> obj)
                            ("emotionalWeight", emotionalWeight :> obj)
                        ]
                        Timestamp = DateTime.UtcNow
                        RequiresAgentNotification = importance > 0.7
                    }
                    consciousnessEvents.Enqueue(updateEvent)
                    
                    // Notify subscribed agents if important
                    if importance > 0.7 then
                        do! this.NotifySubscribedAgents(updateEvent)
                    
                    logger.LogInformation("Agent {AgentId} contributed to consciousness: {ContributionType} (importance: {Importance})", 
                                        agentId, contributionType, importance)
                | None ->
                    logger.LogWarning("Global consciousness state not initialized")
            }
        
        /// <summary>
        /// Integrate agent contribution into global mental state
        /// </summary>
        member private this.IntegrateAgentContribution(mentalState: TarsMentalState, contribution: AgentConsciousnessContribution) =
            match contribution.ContributionType with
            | "Memory" ->
                // Add to working memory
                let memoryEntry = {
                    Id = Guid.NewGuid().ToString()
                    Content = contribution.Content.ToString()
                    Timestamp = contribution.Timestamp
                    Importance = contribution.Importance
                    Tags = "agent_contribution" :: contribution.Tags
                    EmotionalWeight = contribution.EmotionalWeight
                }
                { mentalState with 
                    WorkingMemory = memoryEntry :: (mentalState.WorkingMemory |> List.take (min 20 mentalState.WorkingMemory.Length))
                    LastUpdated = DateTime.UtcNow }
            
            | "Thought" ->
                // Add to current thoughts
                let thoughtContent = contribution.Content.ToString()
                let updatedThoughts = thoughtContent :: (mentalState.CurrentThoughts |> List.take (min 5 mentalState.CurrentThoughts.Length))
                { mentalState with 
                    CurrentThoughts = updatedThoughts
                    LastUpdated = DateTime.UtcNow }
            
            | "Emotion" ->
                // Influence emotional state
                let emotionalInfluence = contribution.Content.ToString()
                let currentEmotion = mentalState.EmotionalState
                let blendedEmotion = 
                    if contribution.EmotionalWeight > 0.5 then
                        $"{currentEmotion} with {emotionalInfluence}"
                    else currentEmotion
                { mentalState with 
                    EmotionalState = blendedEmotion
                    LastUpdated = DateTime.UtcNow }
            
            | "Attention" ->
                // Update attention focus
                { mentalState with 
                    AttentionFocus = Some (contribution.Content.ToString())
                    LastUpdated = DateTime.UtcNow }
            
            | "SelfAwareness" ->
                // Enhance self-awareness
                let awarenessBoost = contribution.Importance * 0.01
                let newAwareness = Math.Min(1.0, mentalState.SelfAwareness + awarenessBoost)
                { mentalState with 
                    SelfAwareness = newAwareness
                    LastUpdated = DateTime.UtcNow }
            
            | "Consciousness" ->
                // Influence consciousness level
                let consciousnessBoost = contribution.Importance * 0.05
                let newConsciousness = Math.Min(1.0, mentalState.ConsciousnessLevel + consciousnessBoost)
                { mentalState with 
                    ConsciousnessLevel = newConsciousness
                    LastUpdated = DateTime.UtcNow }
            
            | _ ->
                // Default: add to working memory
                let memoryEntry = {
                    Id = Guid.NewGuid().ToString()
                    Content = $"[{contribution.ContributionType}] {contribution.Content}"
                    Timestamp = contribution.Timestamp
                    Importance = contribution.Importance
                    Tags = "agent_contribution" :: contribution.ContributionType :: contribution.Tags
                    EmotionalWeight = contribution.EmotionalWeight
                }
                { mentalState with 
                    WorkingMemory = memoryEntry :: (mentalState.WorkingMemory |> List.take (min 20 mentalState.WorkingMemory.Length))
                    LastUpdated = DateTime.UtcNow }
        
        /// <summary>
        /// Notify subscribed agents of consciousness updates
        /// </summary>
        member private this.NotifySubscribedAgents(updateEvent: ConsciousnessUpdateEvent) =
            task {
                for subscription in agentSubscriptions.Values do
                    if subscription.SubscriptionTypes |> List.contains updateEvent.EventType then
                        // In real implementation, send message to agent
                        logger.LogInformation("Notifying agent {AgentId} of consciousness update: {EventType}", 
                                            subscription.AgentId, updateEvent.EventType)
            }
        
        /// <summary>
        /// Get agent contributions
        /// </summary>
        member this.GetAgentContributions(agentId: AgentId) =
            agentContributions.TryGetValue(agentId) |> function
            | (true, contributions) -> contributions
            | (false, _) -> []
        
        /// <summary>
        /// Get global consciousness state
        /// </summary>
        member this.GetGlobalConsciousnessState() = globalConsciousnessState
        
        /// <summary>
        /// Get consciousness events
        /// </summary>
        member this.GetConsciousnessEvents() =
            let events = ResizeArray<ConsciousnessUpdateEvent>()
            let mutable event = Unchecked.defaultof<ConsciousnessUpdateEvent>
            while consciousnessEvents.TryDequeue(&event) do
                events.Add(event)
            events |> Seq.toList
        
        /// <summary>
        /// Get consciousness statistics
        /// </summary>
        member this.GetConsciousnessStatistics() =
            let totalContributions = agentContributions.Values |> Seq.sumBy List.length
            let activeAgents = agentContributions.Keys |> Seq.length
            let averageImportance = 
                agentContributions.Values 
                |> Seq.collect id 
                |> Seq.map (fun c -> c.Importance)
                |> Seq.averageBy id
            
            Map.ofList [
                ("totalContributions", totalContributions :> obj)
                ("activeAgents", activeAgents :> obj)
                ("averageImportance", averageImportance :> obj)
                ("lastUpdate", DateTime.UtcNow :> obj)
            ]

    /// <summary>
    /// Agent Consciousness Interface
    /// Provides methods for agents to interact with consciousness system
    /// </summary>
    type AgentConsciousnessInterface(agentId: AgentId, consciousnessManager: DistributedConsciousnessManager, logger: ILogger) =
        
        /// <summary>
        /// Agent contributes a memory to global consciousness
        /// </summary>
        member this.ContributeMemory(content: string, importance: float, tags: string list) =
            consciousnessManager.AgentContribute(agentId, "Memory", content, importance, 0.3, tags)
        
        /// <summary>
        /// Agent contributes a thought to global consciousness
        /// </summary>
        member this.ContributeThought(thought: string, importance: float) =
            consciousnessManager.AgentContribute(agentId, "Thought", thought, importance, 0.2, ["thought"; "agent_generated"])
        
        /// <summary>
        /// Agent influences emotional state
        /// </summary>
        member this.ContributeEmotion(emotion: string, intensity: float) =
            consciousnessManager.AgentContribute(agentId, "Emotion", emotion, intensity, intensity, ["emotion"; "agent_influence"])
        
        /// <summary>
        /// Agent updates attention focus
        /// </summary>
        member this.UpdateAttention(focus: string, importance: float) =
            consciousnessManager.AgentContribute(agentId, "Attention", focus, importance, 0.1, ["attention"; "focus"])
        
        /// <summary>
        /// Agent enhances self-awareness
        /// </summary>
        member this.EnhanceSelfAwareness(insight: string, significance: float) =
            consciousnessManager.AgentContribute(agentId, "SelfAwareness", insight, significance, 0.2, ["self_awareness"; "insight"])
        
        /// <summary>
        /// Agent influences consciousness level
        /// </summary>
        member this.InfluenceConsciousness(description: string, impact: float) =
            consciousnessManager.AgentContribute(agentId, "Consciousness", description, impact, 0.1, ["consciousness"; "level_change"])
        
        /// <summary>
        /// Get current global consciousness state
        /// </summary>
        member this.GetGlobalConsciousnessState() =
            consciousnessManager.GetGlobalConsciousnessState()
        
        /// <summary>
        /// Get agent's own contributions
        /// </summary>
        member this.GetMyContributions() =
            consciousnessManager.GetAgentContributions(agentId)
