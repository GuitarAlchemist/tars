namespace TarsEngine.FSharp.Cli.Integration

open System
open System.Threading
open System.Threading.Tasks
open System.Collections.Concurrent
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Integration.UnifiedAgentInterfaces

/// Unified Agent Registry - Central registration and discovery system for all TARS agents
module UnifiedAgentRegistry =
    
    /// Agent registration information
    type AgentRegistration = {
        Agent: IUnifiedAgent
        RegisteredAt: DateTime
        LastHeartbeat: DateTime
        HealthStatus: UnifiedAgentStatus
        Tags: string list
        Metadata: Map<string, obj>
    }
    
    /// Agent discovery filter
    type AgentDiscoveryFilter = {
        AgentTypes: string list option
        Capabilities: string list option
        Status: UnifiedAgentStatus list option
        Tags: string list option
        MinSuccessRate: float option
        MaxLoad: float option
    }
    
    /// Thread-safe agent registry implementation
    type UnifiedAgentRegistry(logger: ITarsLogger) =
        let registeredAgents = ConcurrentDictionary<UnifiedAgentId, AgentRegistration>()
        let agentsByType = ConcurrentDictionary<string, ConcurrentBag<UnifiedAgentId>>()
        let agentsByCapability = ConcurrentDictionary<string, ConcurrentBag<UnifiedAgentId>>()
        let agentsByTag = ConcurrentDictionary<string, ConcurrentBag<UnifiedAgentId>>()
        let mutable isDisposed = false
        
        /// Health check timer (initialized later)
        let mutable healthCheckTimer : Timer option = None
        
        /// Register an agent in the registry
        member this.RegisterAgentAsync(agent: IUnifiedAgent, cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    logger.LogInformation(correlationId, $"Registering agent: {agent.Config.Name} ({agent.Config.AgentType})")
                    
                    let registration = {
                        Agent = agent
                        RegisteredAt = DateTime.Now
                        LastHeartbeat = DateTime.Now
                        HealthStatus = agent.Status
                        Tags = []
                        Metadata = Map.empty
                    }
                    
                    // Add to main registry
                    registeredAgents.[agent.Config.AgentId] <- registration
                    
                    // Index by type
                    let typeAgents = agentsByType.GetOrAdd(agent.Config.AgentType, fun _ -> ConcurrentBag<UnifiedAgentId>())
                    typeAgents.Add(agent.Config.AgentId)
                    
                    // Index by capabilities
                    for capability in agent.Capabilities do
                        let capabilityAgents = agentsByCapability.GetOrAdd(capability.Name, fun _ -> ConcurrentBag<UnifiedAgentId>())
                        capabilityAgents.Add(agent.Config.AgentId)
                    
                    logger.LogInformation(correlationId, $"Agent registered successfully: {agent.Config.Name}")
                    return Success ((), Map [("agentId", box agent.Config.AgentId); ("timestamp", box DateTime.Now)])
                    
                with
                | ex ->
                    let error = ExecutionError ($"Failed to register agent: {agent.Config.Name}", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Unregister an agent from the registry
        member this.UnregisterAgentAsync(agentId: UnifiedAgentId, cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    
                    match registeredAgents.TryRemove(agentId) with
                    | true, registration ->
                        logger.LogInformation(correlationId, $"Unregistering agent: {registration.Agent.Config.Name}")
                        
                        // Remove from type index
                        match agentsByType.TryGetValue(registration.Agent.Config.AgentType) with
                        | true, typeAgents ->
                            // Note: ConcurrentBag doesn't support removal, so we'll leave it
                            // This is acceptable as we check the main registry for existence
                            ()
                        | false, _ -> ()
                        
                        // Remove from capability indices (same limitation)
                        // In a production system, we might use a different data structure
                        
                        logger.LogInformation(correlationId, $"Agent unregistered successfully: {registration.Agent.Config.Name}")
                        return Success ((), Map [("agentId", box agentId); ("timestamp", box DateTime.Now)])
                    
                    | false, _ ->
                        let agentIdStr = match agentId with UnifiedAgentId guid -> guid.ToString()
                        let error = ValidationError ($"Agent not found: {agentIdStr}", Map [("agentId", agentIdStr)])
                        return Failure (error, correlationId)
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to unregister agent: {agentId}", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Find agents matching the specified filter
        member this.FindAgentsAsync(filter: AgentDiscoveryFilter, cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    logger.LogDebug(correlationId, "Finding agents with filter")
                    
                    let allAgents = 
                        registeredAgents.Values
                        |> Seq.toList
                    
                    let filteredAgents =
                        allAgents
                        |> List.filter (fun reg ->
                            // Filter by agent type
                            match filter.AgentTypes with
                            | Some types -> List.contains reg.Agent.Config.AgentType types
                            | None -> true
                        )
                        |> List.filter (fun reg ->
                            // Filter by capabilities
                            match filter.Capabilities with
                            | Some caps ->
                                caps |> List.forall (fun cap ->
                                    reg.Agent.Capabilities |> List.exists (fun agentCap -> agentCap.Name = cap))
                            | None -> true
                        )
                        |> List.filter (fun reg ->
                            // Filter by status
                            match filter.Status with
                            | Some statuses -> List.contains reg.HealthStatus statuses
                            | None -> true
                        )
                        |> List.filter (fun reg ->
                            // Filter by tags
                            match filter.Tags with
                            | Some tags ->
                                tags |> List.forall (fun tag -> List.contains tag reg.Tags)
                            | None -> true
                        )
                        |> List.filter (fun reg ->
                            // Filter by success rate
                            match filter.MinSuccessRate with
                            | Some minRate -> reg.Agent.Metrics.SuccessRate >= minRate
                            | None -> true
                        )
                        |> List.filter (fun reg ->
                            // Filter by load
                            match filter.MaxLoad with
                            | Some maxLoad -> reg.Agent.Metrics.CurrentLoad <= maxLoad
                            | None -> true
                        )
                        |> List.map (fun reg -> reg.Agent)
                    
                    logger.LogDebug(correlationId, $"Found {filteredAgents.Length} agents matching filter")
                    return Success (filteredAgents, Map [("count", box filteredAgents.Length); ("timestamp", box DateTime.Now)])
                
                with
                | ex ->
                    let error = ExecutionError ("Failed to find agents", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Get agent by ID
        member this.GetAgentAsync(agentId: UnifiedAgentId, cancellationToken: CancellationToken) =
            task {
                try
                    match registeredAgents.TryGetValue(agentId) with
                    | true, registration ->
                        return Success (Some registration.Agent, Map [("agentId", box agentId)])
                    | false, _ ->
                        return Success (None, Map [("agentId", box agentId)])
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to get agent: {agentId}", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Get all registered agents
        member this.GetAllAgentsAsync(cancellationToken: CancellationToken) =
            task {
                try
                    let agents = 
                        registeredAgents.Values
                        |> Seq.map (fun reg -> reg.Agent)
                        |> Seq.toList
                    
                    return Success (agents, Map [("count", box agents.Length); ("timestamp", box DateTime.Now)])
                
                with
                | ex ->
                    let error = ExecutionError ("Failed to get all agents", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Perform health checks on all registered agents
        member this.PerformHealthChecksAsync(cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    logger.LogDebug(correlationId, "Performing health checks on all agents")
                    
                    let healthResults = ConcurrentDictionary<UnifiedAgentId, Map<string, obj>>()
                    
                    let healthCheckTasks =
                        registeredAgents.Values
                        |> Seq.map (fun registration ->
                            task {
                                try
                                    let! healthResult = registration.Agent.HealthCheckAsync(cancellationToken)
                                    match healthResult with
                                    | Success (health, _) ->
                                        healthResults.[registration.Agent.Config.AgentId] <- health
                                        
                                        // Update registration with current status
                                        let updatedRegistration = {
                                            registration with
                                                LastHeartbeat = DateTime.Now
                                                HealthStatus = registration.Agent.Status
                                        }
                                        registeredAgents.[registration.Agent.Config.AgentId] <- updatedRegistration
                                    
                                    | Failure (error, corrId) ->
                                        logger.LogWarning(corrId, $"Health check failed for agent {registration.Agent.Config.Name}: {TarsError.toString error}")
                                        healthResults.[registration.Agent.Config.AgentId] <- Map [("status", box "unhealthy"); ("error", box (TarsError.toString error))]
                                
                                with
                                | ex ->
                                    logger.LogError(correlationId, ExecutionError ($"Health check exception for agent {registration.Agent.Config.Name}", Some ex), ex)
                                    healthResults.[registration.Agent.Config.AgentId] <- Map [("status", box "error"); ("exception", box ex.Message)]
                            })
                        |> Seq.toArray
                    
                    do! Task.WhenAll(healthCheckTasks |> Array.map (fun t -> t :> Task))
                    
                    let results = healthResults |> Seq.map (fun kvp -> kvp.Key, kvp.Value) |> Map.ofSeq
                    logger.LogDebug(correlationId, $"Health checks completed for {results.Count} agents")
                    
                    return Success (results, Map [("agentCount", box results.Count); ("timestamp", box DateTime.Now)])
                
                with
                | ex ->
                    let error = ExecutionError ("Failed to perform health checks", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Get registry statistics
        member this.GetRegistryStatistics() =
            let totalAgents = registeredAgents.Count
            let agentsByTypeStats =
                agentsByType
                |> Seq.map (fun kvp -> kvp.Key, kvp.Value.Count)
                |> Map.ofSeq

            let healthyAgents =
                registeredAgents.Values
                |> Seq.filter (fun reg ->
                    match reg.HealthStatus with
                    | Ready | Busy _ -> true
                    | _ -> false)
                |> Seq.length

            Map [
                ("totalAgents", box totalAgents)
                ("healthyAgents", box healthyAgents)
                ("agentsByType", box agentsByTypeStats)
                ("lastUpdate", box DateTime.Now)
            ]

        /// Initialize the health check timer
        member this.StartHealthCheckTimer() =
            if healthCheckTimer.IsNone then
                let timer = new Timer(fun _ ->
                    if not isDisposed then
                        task {
                            let! _ = this.PerformHealthChecksAsync(CancellationToken.None)
                            ()
                        } |> ignore
                , null, TimeSpan.FromMinutes(1.0), TimeSpan.FromMinutes(1.0))
                healthCheckTimer <- Some timer
        
        interface IDisposable with
            member this.Dispose() =
                if not isDisposed then
                    isDisposed <- true
                    healthCheckTimer |> Option.iter (fun timer -> timer.Dispose())
                    registeredAgents.Clear()
                    agentsByType.Clear()
                    agentsByCapability.Clear()
                    agentsByTag.Clear()
    
    /// Create default discovery filter (no filtering)
    let createDefaultFilter() = {
        AgentTypes = None
        Capabilities = None
        Status = None
        Tags = None
        MinSuccessRate = None
        MaxLoad = None
    }
    
    /// Create filter for specific agent types
    let filterByTypes (types: string list) = {
        createDefaultFilter() with AgentTypes = Some types
    }
    
    /// Create filter for specific capabilities
    let filterByCapabilities (capabilities: string list) = {
        createDefaultFilter() with Capabilities = Some capabilities
    }
    
    /// Create filter for healthy agents only
    let filterHealthyAgents() = {
        createDefaultFilter() with Status = Some [Ready]
    }
