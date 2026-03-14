namespace TarsEngine.FSharp.WindowsService.Agents

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Agent manager for orchestrating multiple agents
/// </summary>
type AgentManager(
    logger: ILogger<AgentManager>,
    registry: AgentRegistry,
    communication: AgentCommunication,
    host: AgentHost) =
    
    let managedAgents = ConcurrentDictionary<string, IAgent>()
    let agentConfigurations = ConcurrentDictionary<string, AgentConfig>()
    let healthCheckTasks = ConcurrentDictionary<string, Task>()
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    
    /// Configure the agent manager with agent configurations
    member this.ConfigureAsync(agentConfigs: AgentConfig list, cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation($"Configuring agent manager with {agentConfigs.Length} agent configurations")
            
            // Store configurations
            for config in agentConfigs do
                agentConfigurations.[config.Name] <- config
                logger.LogDebug($"Stored configuration for agent: {config.Name}")
            
            logger.LogInformation("Agent manager configuration completed")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to configure agent manager")
            raise
    }
    
    /// Start the agent manager
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting agent manager...")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Start enabled agents
            let enabledConfigs = agentConfigurations.Values |> Seq.filter (fun c -> c.Enabled) |> List.ofSeq
            
            for config in enabledConfigs do
                do! this.StartAgentAsync(config, cancellationToken)
                
                // Add startup delay if specified
                if config.StartupDelay > TimeSpan.Zero then
                    logger.LogDebug($"Applying startup delay for agent {config.Name}: {config.StartupDelay}")
                    do! Task.Delay(config.StartupDelay, cancellationToken)
            
            // Start health monitoring
            do! this.StartHealthMonitoringAsync(cancellationToken)
            
            logger.LogInformation($"Agent manager started successfully with {enabledConfigs.Length} agents")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to start agent manager")
            isRunning <- false
            raise
    }
    
    /// Stop the agent manager
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping agent manager...")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Stop all managed agents
            let stopTasks = 
                managedAgents.Values
                |> Seq.map (fun agent -> this.StopAgentAsync(agent.Id, cancellationToken))
                |> Array.ofSeq
            
            do! Task.WhenAll(stopTasks)
            
            // Wait for health check tasks to complete
            let healthTasks = healthCheckTasks.Values |> Array.ofSeq
            if healthTasks.Length > 0 then
                try
                    do! Task.WhenAll(healthTasks).WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning("Health check tasks did not complete within timeout")
                | ex ->
                    logger.LogWarning(ex, "Error waiting for health check tasks to complete")
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            managedAgents.Clear()
            healthCheckTasks.Clear()
            
            logger.LogInformation("Agent manager stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping agent manager")
    }
    
    /// Start a specific agent
    member private this.StartAgentAsync(config: AgentConfig, cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation($"Starting agent: {config.Name} of type {config.Type}")
            
            // Create agent instances based on MaxInstances
            for i in 1 .. config.MaxInstances do
                let instanceName = if config.MaxInstances = 1 then config.Name else $"{config.Name}-{i}"
                
                // Create agent instance
                let! createResult = registry.CreateAgentInstance(config.Type, { config with Name = instanceName })
                match createResult with
                | Ok instanceId ->
                    match registry.GetAgent(instanceId) with
                    | Some agent ->
                        // Register for communication
                        let! commResult = communication.RegisterAgent(agent.Id)
                        match commResult with
                        | Ok () ->
                            // Start hosting the agent
                            let! hostResult = host.StartAgentAsync(agent, config)
                            match hostResult with
                            | Ok context ->
                                managedAgents.[agent.Id] <- agent
                                
                                // Start health monitoring for this agent
                                do! this.StartAgentHealthMonitoringAsync(agent, config, cancellationToken)
                                
                                logger.LogInformation($"Agent {instanceName} ({agent.Id}) started successfully")
                            
                            | Error error ->
                                logger.LogError($"Failed to start hosting for agent {instanceName}: {error}")
                        
                        | Error error ->
                            logger.LogError($"Failed to register agent {instanceName} for communication: {error}")
                    
                    | None ->
                        logger.LogError($"Failed to get agent instance after creation: {instanceName}")
                
                | Error error ->
                    logger.LogError($"Failed to create agent instance {instanceName}: {error}")
                    
        with
        | ex ->
            logger.LogError(ex, $"Error starting agent: {config.Name}")
    }
    
    /// Stop a specific agent
    member private this.StopAgentAsync(agentId: string, cancellationToken: CancellationToken) = task {
        try
            match managedAgents.TryGetValue(agentId) with
            | true, agent ->
                logger.LogInformation($"Stopping agent: {agent.Name} ({agentId})")
                
                // Stop health monitoring
                match healthCheckTasks.TryRemove(agentId) with
                | true, healthTask ->
                    try
                        do! healthTask.WaitAsync(TimeSpan.FromSeconds(5.0), cancellationToken)
                    with
                    | :? TimeoutException ->
                        logger.LogWarning($"Health monitoring task for agent {agentId} did not complete within timeout")
                    | ex ->
                        logger.LogWarning(ex, $"Error stopping health monitoring for agent {agentId}")
                | false, _ -> ()
                
                // Stop hosting
                let! hostResult = host.StopAgentAsync(agentId)
                match hostResult with
                | Ok () ->
                    logger.LogDebug($"Agent hosting stopped for: {agentId}")
                | Error error ->
                    logger.LogWarning($"Error stopping agent hosting for {agentId}: {error}")
                
                // Unregister from communication
                let! commResult = communication.UnregisterAgent(agentId)
                match commResult with
                | Ok () ->
                    logger.LogDebug($"Agent communication unregistered for: {agentId}")
                | Error error ->
                    logger.LogWarning($"Error unregistering agent communication for {agentId}: {error}")
                
                // Remove from registry
                let! registryResult = registry.RemoveAgentInstance(agentId)
                match registryResult with
                | Ok () ->
                    logger.LogDebug($"Agent removed from registry: {agentId}")
                | Error error ->
                    logger.LogWarning($"Error removing agent from registry {agentId}: {error}")
                
                managedAgents.TryRemove(agentId) |> ignore
                logger.LogInformation($"Agent {agent.Name} ({agentId}) stopped successfully")
            
            | false, _ ->
                logger.LogWarning($"Agent {agentId} is not managed by this manager")
                
        with
        | ex ->
            logger.LogError(ex, $"Error stopping agent: {agentId}")
    }
    
    /// Start health monitoring for all agents
    member private this.StartHealthMonitoringAsync(cancellationToken: CancellationToken) = task {
        logger.LogInformation("Starting health monitoring for all agents")
        
        for kvp in managedAgents do
            let agent = kvp.Value
            match agentConfigurations.TryGetValue(agent.Name) with
            | true, config ->
                do! this.StartAgentHealthMonitoringAsync(agent, config, cancellationToken)
            | false, _ ->
                logger.LogWarning($"No configuration found for agent: {agent.Name}")
    }
    
    /// Start health monitoring for a specific agent
    member private this.StartAgentHealthMonitoringAsync(agent: IAgent, config: AgentConfig, cancellationToken: CancellationToken) = task {
        let healthTask = task {
            try
                logger.LogDebug($"Starting health monitoring for agent: {agent.Name} ({agent.Id})")
                
                while not cancellationToken.IsCancellationRequested && isRunning do
                    try
                        // Check agent health
                        let! health = agent.GetHealthAsync(cancellationToken)
                        
                        if not health.IsHealthy then
                            logger.LogWarning($"Agent {agent.Name} ({agent.Id}) is unhealthy: {health.Status}")
                            
                            // Increment error count
                            let! errorResult = registry.IncrementAgentErrorCount(agent.Id)
                            match errorResult with
                            | Ok errorCount ->
                                logger.LogDebug($"Agent {agent.Id} error count: {errorCount}")
                                
                                // Check if restart is needed
                                if config.RestartOnFailure && errorCount >= 3 then
                                    logger.LogWarning($"Agent {agent.Name} ({agent.Id}) will be restarted due to repeated failures")
                                    do! this.RestartAgentAsync(agent.Id, cancellationToken)
                            
                            | Error error ->
                                logger.LogWarning($"Failed to increment error count for agent {agent.Id}: {error}")
                        else
                            // Update status to healthy
                            registry.UpdateAgentInstanceStatus(agent.Id, AgentStatus.Running) |> ignore
                        
                        // Wait for next health check
                        do! Task.Delay(config.HealthCheckInterval, cancellationToken)
                        
                    with
                    | :? OperationCanceledException ->
                        break
                    | ex ->
                        logger.LogWarning(ex, $"Error during health check for agent {agent.Name} ({agent.Id})")
                        do! Task.Delay(config.HealthCheckInterval, cancellationToken)
                        
            with
            | :? OperationCanceledException ->
                logger.LogDebug($"Health monitoring cancelled for agent: {agent.Name} ({agent.Id})")
            | ex ->
                logger.LogError(ex, $"Health monitoring failed for agent: {agent.Name} ({agent.Id})")
        }
        
        healthCheckTasks.[agent.Id] <- healthTask
    }
    
    /// Restart a specific agent
    member private this.RestartAgentAsync(agentId: string, cancellationToken: CancellationToken) = task {
        try
            match managedAgents.TryGetValue(agentId) with
            | true, agent ->
                logger.LogInformation($"Restarting agent: {agent.Name} ({agentId})")
                
                match agentConfigurations.TryGetValue(agent.Name) with
                | true, config ->
                    // Stop the agent
                    do! this.StopAgentAsync(agentId, cancellationToken)
                    
                    // Wait a moment before restarting
                    do! Task.Delay(TimeSpan.FromSeconds(5.0), cancellationToken)
                    
                    // Start the agent again
                    do! this.StartAgentAsync(config, cancellationToken)
                    
                    logger.LogInformation($"Agent {agent.Name} restarted successfully")
                
                | false, _ ->
                    logger.LogError($"No configuration found for agent restart: {agent.Name}")
            
            | false, _ ->
                logger.LogWarning($"Agent {agentId} not found for restart")
                
        with
        | ex ->
            logger.LogError(ex, $"Error restarting agent: {agentId}")
    }
    
    /// Reconfigure the agent manager
    member this.ReconfigureAsync(agentConfigs: AgentConfig list, cancellationToken: CancellationToken) = task {
        logger.LogInformation("Reconfiguring agent manager...")
        
        // Stop current agents
        do! this.StopAsync(cancellationToken)
        
        // Apply new configuration
        do! this.ConfigureAsync(agentConfigs, cancellationToken)
        
        // Start with new configuration
        do! this.StartAsync(cancellationToken)
        
        logger.LogInformation("Agent manager reconfiguration completed")
    }
    
    /// Get agent manager status
    member this.GetStatus() =
        if isRunning then
            $"Running with {managedAgents.Count} agents"
        else
            "Stopped"
    
    /// Get agent manager metrics
    member this.GetMetrics() =
        let totalAgents = managedAgents.Count
        let healthyAgents = 
            managedAgents.Values
            |> Seq.filter (fun agent -> 
                try
                    let health = agent.GetHealthAsync(CancellationToken.None).Result
                    health.IsHealthy
                with
                | _ -> false)
            |> Seq.length
        
        Map.ofList [
            ("TotalAgents", totalAgents :> obj)
            ("HealthyAgents", healthyAgents :> obj)
            ("UnhealthyAgents", (totalAgents - healthyAgents) :> obj)
            ("IsRunning", isRunning :> obj)
        ]
