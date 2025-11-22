namespace TarsEngine.FSharp.WindowsService.Agents

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Agent status enumeration
/// </summary>
type AgentStatus =
    | Stopped
    | Starting
    | Running
    | Stopping
    | Failed
    | Restarting

/// <summary>
/// Agent instance information
/// </summary>
type AgentInstance = {
    Id: string
    Name: string
    Type: string
    Status: AgentStatus
    StartTime: DateTime option
    LastHeartbeat: DateTime option
    Configuration: Map<string, obj>
    Metrics: Map<string, obj>
    ErrorCount: int
    RestartCount: int
}

/// <summary>
/// Agent registration information
/// </summary>
type AgentRegistration = {
    Name: string
    Type: string
    Factory: AgentConfig -> IAgent
    MaxInstances: int
    HealthCheckInterval: TimeSpan
    RestartOnFailure: bool
}

/// <summary>
/// Agent interface for all TARS agents
/// </summary>
and IAgent =
    abstract member Id: string
    abstract member Name: string
    abstract member Type: string
    abstract member Status: AgentStatus
    abstract member StartAsync: CancellationToken -> Task
    abstract member StopAsync: CancellationToken -> Task
    abstract member GetHealthAsync: CancellationToken -> Task<AgentHealthStatus>
    abstract member GetMetricsAsync: CancellationToken -> Task<Map<string, obj>>
    abstract member ConfigureAsync: Map<string, obj> -> CancellationToken -> Task

/// <summary>
/// Agent health status
/// </summary>
and AgentHealthStatus = {
    IsHealthy: bool
    Status: string
    LastCheck: DateTime
    Issues: string list
    Metrics: Map<string, obj>
}

/// <summary>
/// Agent registry for managing agent types and instances
/// </summary>
type AgentRegistry(logger: ILogger<AgentRegistry>) =
    
    let registrations = ConcurrentDictionary<string, AgentRegistration>()
    let instances = ConcurrentDictionary<string, AgentInstance>()
    let agents = ConcurrentDictionary<string, IAgent>()
    
    /// Register an agent type
    member this.RegisterAgentType(registration: AgentRegistration) =
        try
            logger.LogInformation($"Registering agent type: {registration.Type}")
            
            if registrations.ContainsKey(registration.Type) then
                logger.LogWarning($"Agent type {registration.Type} is already registered, overwriting")
            
            registrations.[registration.Type] <- registration
            logger.LogInformation($"Agent type {registration.Type} registered successfully")
            Ok ()
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to register agent type: {registration.Type}")
            Error ex.Message
    
    /// Unregister an agent type
    member this.UnregisterAgentType(agentType: string) =
        try
            logger.LogInformation($"Unregistering agent type: {agentType}")
            
            // Stop all instances of this type first
            let instancesToStop = 
                instances.Values 
                |> Seq.filter (fun i -> i.Type = agentType)
                |> List.ofSeq
            
            for instance in instancesToStop do
                this.RemoveAgentInstance(instance.Id) |> ignore
            
            registrations.TryRemove(agentType) |> ignore
            logger.LogInformation($"Agent type {agentType} unregistered successfully")
            Ok ()
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to unregister agent type: {agentType}")
            Error ex.Message
    
    /// Create an agent instance
    member this.CreateAgentInstance(agentType: string, config: AgentConfig) =
        try
            logger.LogInformation($"Creating agent instance: {config.Name} of type {agentType}")
            
            match registrations.TryGetValue(agentType) with
            | true, registration ->
                // Check instance limits
                let currentInstances = 
                    instances.Values 
                    |> Seq.filter (fun i -> i.Type = agentType) 
                    |> Seq.length
                
                if currentInstances >= registration.MaxInstances then
                    let error = $"Maximum instances ({registration.MaxInstances}) reached for agent type {agentType}"
                    logger.LogWarning(error)
                    Error error
                else
                    // Create the agent
                    let agent = registration.Factory(config)
                    let instanceId = Guid.NewGuid().ToString()
                    
                    let instance = {
                        Id = instanceId
                        Name = config.Name
                        Type = agentType
                        Status = Stopped
                        StartTime = None
                        LastHeartbeat = None
                        Configuration = config.Configuration
                        Metrics = Map.empty
                        ErrorCount = 0
                        RestartCount = 0
                    }
                    
                    instances.[instanceId] <- instance
                    agents.[instanceId] <- agent
                    
                    logger.LogInformation($"Agent instance {config.Name} ({instanceId}) created successfully")
                    Ok instanceId
            
            | false, _ ->
                let error = $"Agent type {agentType} is not registered"
                logger.LogError(error)
                Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to create agent instance: {config.Name}")
            Error ex.Message
    
    /// Remove an agent instance
    member this.RemoveAgentInstance(instanceId: string) =
        try
            logger.LogInformation($"Removing agent instance: {instanceId}")
            
            match instances.TryGetValue(instanceId) with
            | true, instance ->
                // Stop the agent if running
                match agents.TryGetValue(instanceId) with
                | true, agent ->
                    if agent.Status = Running then
                        // Note: In a real implementation, we'd await this
                        agent.StopAsync(CancellationToken.None) |> ignore
                    agents.TryRemove(instanceId) |> ignore
                | false, _ -> ()
                
                instances.TryRemove(instanceId) |> ignore
                logger.LogInformation($"Agent instance {instance.Name} ({instanceId}) removed successfully")
                Ok ()
            
            | false, _ ->
                let error = $"Agent instance {instanceId} not found"
                logger.LogWarning(error)
                Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to remove agent instance: {instanceId}")
            Error ex.Message
    
    /// Get agent instance
    member this.GetAgentInstance(instanceId: string) =
        match instances.TryGetValue(instanceId) with
        | true, instance -> Some instance
        | false, _ -> None
    
    /// Get all agent instances
    member this.GetAllAgentInstances() =
        instances.Values |> List.ofSeq
    
    /// Get agent instances by type
    member this.GetAgentInstancesByType(agentType: string) =
        instances.Values 
        |> Seq.filter (fun i -> i.Type = agentType)
        |> List.ofSeq
    
    /// Get agent instances by status
    member this.GetAgentInstancesByStatus(status: AgentStatus) =
        instances.Values 
        |> Seq.filter (fun i -> i.Status = status)
        |> List.ofSeq
    
    /// Get registered agent types
    member this.GetRegisteredAgentTypes() =
        registrations.Keys |> List.ofSeq
    
    /// Get agent registration
    member this.GetAgentRegistration(agentType: string) =
        match registrations.TryGetValue(agentType) with
        | true, registration -> Some registration
        | false, _ -> None
    
    /// Update agent instance status
    member this.UpdateAgentInstanceStatus(instanceId: string, status: AgentStatus) =
        match instances.TryGetValue(instanceId) with
        | true, instance ->
            let updatedInstance = { instance with Status = status; LastHeartbeat = Some DateTime.UtcNow }
            instances.[instanceId] <- updatedInstance
            logger.LogDebug($"Agent instance {instance.Name} ({instanceId}) status updated to {status}")
            Ok ()
        | false, _ ->
            let error = $"Agent instance {instanceId} not found"
            logger.LogWarning(error)
            Error error
    
    /// Update agent instance metrics
    member this.UpdateAgentInstanceMetrics(instanceId: string, metrics: Map<string, obj>) =
        match instances.TryGetValue(instanceId) with
        | true, instance ->
            let updatedInstance = { instance with Metrics = metrics; LastHeartbeat = Some DateTime.UtcNow }
            instances.[instanceId] <- updatedInstance
            logger.LogDebug($"Agent instance {instance.Name} ({instanceId}) metrics updated")
            Ok ()
        | false, _ ->
            let error = $"Agent instance {instanceId} not found"
            logger.LogWarning(error)
            Error error
    
    /// Increment agent error count
    member this.IncrementAgentErrorCount(instanceId: string) =
        match instances.TryGetValue(instanceId) with
        | true, instance ->
            let updatedInstance = { instance with ErrorCount = instance.ErrorCount + 1 }
            instances.[instanceId] <- updatedInstance
            logger.LogWarning($"Agent instance {instance.Name} ({instanceId}) error count incremented to {updatedInstance.ErrorCount}")
            Ok updatedInstance.ErrorCount
        | false, _ ->
            let error = $"Agent instance {instanceId} not found"
            logger.LogWarning(error)
            Error error
    
    /// Increment agent restart count
    member this.IncrementAgentRestartCount(instanceId: string) =
        match instances.TryGetValue(instanceId) with
        | true, instance ->
            let updatedInstance = { instance with RestartCount = instance.RestartCount + 1 }
            instances.[instanceId] <- updatedInstance
            logger.LogInformation($"Agent instance {instance.Name} ({instanceId}) restart count incremented to {updatedInstance.RestartCount}")
            Ok updatedInstance.RestartCount
        | false, _ ->
            let error = $"Agent instance {instanceId} not found"
            logger.LogWarning(error)
            Error error
    
    /// Get agent by instance ID
    member this.GetAgent(instanceId: string) =
        match agents.TryGetValue(instanceId) with
        | true, agent -> Some agent
        | false, _ -> None
    
    /// Get registry statistics
    member this.GetStatistics() =
        let totalInstances = instances.Count
        let runningInstances = instances.Values |> Seq.filter (fun i -> i.Status = Running) |> Seq.length
        let failedInstances = instances.Values |> Seq.filter (fun i -> i.Status = Failed) |> Seq.length
        let registeredTypes = registrations.Count
        
        {|
            TotalInstances = totalInstances
            RunningInstances = runningInstances
            FailedInstances = failedInstances
            RegisteredTypes = registeredTypes
            InstancesByType = 
                instances.Values 
                |> Seq.groupBy (fun i -> i.Type)
                |> Seq.map (fun (t, instances) -> (t, instances |> Seq.length))
                |> Map.ofSeq
            InstancesByStatus = 
                instances.Values 
                |> Seq.groupBy (fun i -> i.Status)
                |> Seq.map (fun (s, instances) -> (s.ToString(), instances |> Seq.length))
                |> Map.ofSeq
        |}
