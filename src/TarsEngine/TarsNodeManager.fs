namespace TarsEngine

open System
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Configuration
open TarsEngine.TarsNode
open TarsEngine.TarsNodeAdapters

/// TARS Node Manager - Orchestrates TARS Nodes across all platforms
module TarsNodeManager =
    
    /// TARS Node Manager Implementation
    type TarsNodeManager(logger: ILogger<TarsNodeManager>, config: IConfiguration) =
        
        let mutable platformAdapters: ITarsNodePlatformAdapter list = []
        let mutable managedNodes: Dictionary<string, TarsNodeConfig> = new Dictionary<string, TarsNodeConfig>()
        let mutable nodeHealth: Dictionary<string, TarsNodeHealth> = new Dictionary<string, TarsNodeHealth>()
        
        /// Initialize platform adapters
        do
            let kubernetesAdapter = new KubernetesTarsNodeAdapter(logger) :> ITarsNodePlatformAdapter
            let windowsServiceAdapter = new WindowsServiceTarsNodeAdapter(logger) :> ITarsNodePlatformAdapter
            let edgeDeviceAdapter = new EdgeDeviceTarsNodeAdapter(logger) :> ITarsNodePlatformAdapter
            let cloudInstanceAdapter = new CloudInstanceTarsNodeAdapter(logger) :> ITarsNodePlatformAdapter
            
            platformAdapters <- [kubernetesAdapter; windowsServiceAdapter; edgeDeviceAdapter; cloudInstanceAdapter]
            logger.LogInformation($"🤖 TARS Node Manager initialized with {platformAdapters.Length} platform adapters")
        
        /// Register a new platform adapter
        member _.RegisterPlatformAdapter(adapter: ITarsNodePlatformAdapter) =
            platformAdapters <- adapter :: platformAdapters
            logger.LogInformation("📦 New platform adapter registered")
        
        /// Create a new TARS Node configuration
        member _.CreateNode(config: TarsNodeConfig) = async {
            logger.LogInformation($"🆕 Creating TARS Node: {config.NodeName}")
            
            // Validate configuration
            if String.IsNullOrEmpty(config.NodeId) then
                failwith "Node ID cannot be empty"
            
            if managedNodes.ContainsKey(config.NodeId) then
                failwith $"Node with ID {config.NodeId} already exists"
            
            // Add to managed nodes
            managedNodes.[config.NodeId] <- config
            
            // Initialize health tracking
            nodeHealth.[config.NodeId] <- {
                State = Initializing
                Uptime = TimeSpan.Zero
                CpuUsage = 0.0
                MemoryUsage = 0.0
                StorageUsage = 0.0
                NetworkLatency = 0.0
                RequestsPerSecond = 0.0
                ErrorRate = 0.0
                LastHealthCheck = DateTime.UtcNow
                HealthScore = 0.0
            }
            
            logger.LogInformation($"✅ TARS Node {config.NodeName} created successfully")
            return config.NodeId
        }
        
        /// Deploy a TARS Node to a specific platform
        member this.DeployNode(nodeId: string) (platform: TarsNodePlatform) = async {
            logger.LogInformation($"🚀 Deploying TARS Node {nodeId} to {platform}")
            
            if not (managedNodes.ContainsKey(nodeId)) then
                failwith $"Node {nodeId} not found"
            
            let nodeConfig = managedNodes.[nodeId]
            
            // Find appropriate platform adapter
            let adapter = platformAdapters |> List.tryFind (fun a -> a.CanDeploy(platform))
            
            match adapter with
            | Some(adapter) ->
                // Update node configuration with target platform
                let updatedConfig = { nodeConfig with Platform = platform }
                managedNodes.[nodeId] <- updatedConfig
                
                // Create deployment specification
                let deployment = {
                    Config = updatedConfig
                    DeploymentStrategy = "rolling_update"
                    RolloutPolicy = "gradual"
                    HealthChecks = ["liveness"; "readiness"; "startup"]
                    MonitoringConfig = Map [
                        ("metrics_enabled", "true")
                        ("logging_level", "info")
                        ("tracing_enabled", "true")
                    ]
                    BackupStrategy = "automated"
                    UpdatePolicy = "zero_downtime"
                }
                
                // Deploy using platform adapter
                let! deploymentId = adapter.Deploy(deployment) |> Async.AwaitTask
                
                // Update node state
                if nodeHealth.ContainsKey(nodeId) then
                    let currentHealth = nodeHealth.[nodeId]
                    nodeHealth.[nodeId] <- { currentHealth with State = Starting }
                
                logger.LogInformation($"✅ TARS Node {nodeId} deployed successfully to {platform}")
                return true
            
            | None ->
                logger.LogError($"❌ No platform adapter found for {platform}")
                return false
        }
        
        /// Get health status of a TARS Node
        member _.GetNodeHealth(nodeId: string) = async {
            logger.LogDebug($"📊 Getting health for TARS Node {nodeId}")
            
            if not (managedNodes.ContainsKey(nodeId)) then
                failwith $"Node {nodeId} not found"
            
            let nodeConfig = managedNodes.[nodeId]
            
            // Find appropriate platform adapter
            let adapter = platformAdapters |> List.tryFind (fun a -> a.CanDeploy(nodeConfig.Platform))
            
            match adapter with
            | Some(adapter) ->
                let! health = adapter.GetHealth(nodeId) |> Async.AwaitTask
                nodeHealth.[nodeId] <- health
                return health
            
            | None ->
                logger.LogWarning($"⚠️ No platform adapter found for {nodeConfig.Platform}")
                return nodeHealth.[nodeId]
        }
        
        /// Scale TARS Nodes of a specific role
        member this.ScaleNodes(role: TarsNodeRole) (targetCount: int) = async {
            logger.LogInformation($"📈 Scaling TARS Nodes with role {role} to {targetCount} instances")
            
            // Find existing nodes with the specified role
            let existingNodes = 
                managedNodes.Values 
                |> Seq.filter (fun node -> node.Role = role)
                |> Seq.toList
            
            let currentCount = existingNodes.Length
            
            if targetCount > currentCount then
                // Scale up - create new nodes
                let nodesToCreate = targetCount - currentCount
                let! newNodeIds = 
                    [1..nodesToCreate]
                    |> List.map (fun i -> async {
                        // Create new node configuration based on existing nodes
                        let templateNode = existingNodes |> List.head
                        let newNodeConfig = {
                            templateNode with
                                NodeId = Guid.NewGuid().ToString("N")[..7]
                                NodeName = sprintf "%s-scale-%d" templateNode.NodeName i
                        }
                        
                        let! nodeId = this.CreateNode(newNodeConfig)
                        let! deployed = this.DeployNode(nodeId) (newNodeConfig.Platform)
                        
                        if deployed then
                            return Some nodeId
                        else
                            return None
                    })
                    |> Async.Parallel
                
                let successfulNodes = newNodeIds |> Array.choose id |> Array.toList
                logger.LogInformation($"✅ Scaled up: Created {successfulNodes.Length} new nodes")
                return successfulNodes
            
            elif targetCount < currentCount then
                // Scale down - remove excess nodes
                let nodesToRemove = currentCount - targetCount
                let nodesToRemoveList = existingNodes |> List.take nodesToRemove
                
                let! removedNodes =
                    nodesToRemoveList
                    |> List.map (fun node -> async {
                        let adapter = platformAdapters |> List.tryFind (fun a -> a.CanDeploy(node.Platform))
                        match adapter with
                        | Some(adapter) ->
                            let! removed = adapter.Remove(node.NodeId) |> Async.AwaitTask
                            if removed then
                                managedNodes.Remove(node.NodeId) |> ignore
                                nodeHealth.Remove(node.NodeId) |> ignore
                                return Some node.NodeId
                            else
                                return None
                        | None ->
                            return None
                    })
                    |> Async.Parallel
                
                let successfulRemovals = removedNodes |> Array.choose id |> Array.toList
                logger.LogInformation($"✅ Scaled down: Removed {successfulRemovals.Length} nodes")
                return successfulRemovals
            
            else
                logger.LogInformation($"✅ No scaling needed: Already at target count {targetCount}")
                return []
        }
        
        /// Migrate a TARS Node to a different platform
        member this.MigrateNode(nodeId: string) (targetPlatform: TarsNodePlatform) = async {
            logger.LogInformation($"🔄 Migrating TARS Node {nodeId} to {targetPlatform}")
            
            if not (managedNodes.ContainsKey(nodeId)) then
                failwith $"Node {nodeId} not found"
            
            let nodeConfig = managedNodes.[nodeId]
            let currentPlatform = nodeConfig.Platform
            
            if currentPlatform = targetPlatform then
                logger.LogInformation($"✅ Node {nodeId} already on target platform")
                return true
            
            // Find adapters for both platforms
            let sourceAdapter = platformAdapters |> List.tryFind (fun a -> a.CanDeploy(currentPlatform))
            let targetAdapter = platformAdapters |> List.tryFind (fun a -> a.CanDeploy(targetPlatform))
            
            match sourceAdapter, targetAdapter with
            | Some(source), Some(target) ->
                try
                    // Update node state to migrating
                    if nodeHealth.ContainsKey(nodeId) then
                        let currentHealth = nodeHealth.[nodeId]
                        nodeHealth.[nodeId] <- { currentHealth with State = Migrating(targetPlatform) }
                    
                    // Deploy to target platform
                    let updatedConfig = { nodeConfig with Platform = targetPlatform }
                    let deployment = {
                        Config = updatedConfig
                        DeploymentStrategy = "blue_green"
                        RolloutPolicy = "zero_downtime"
                        HealthChecks = ["liveness"; "readiness"; "startup"]
                        MonitoringConfig = Map [
                            ("migration_mode", "true")
                            ("source_platform", currentPlatform.ToString())
                            ("target_platform", targetPlatform.ToString())
                        ]
                        BackupStrategy = "pre_migration_backup"
                        UpdatePolicy = "zero_downtime"
                    }
                    
                    let! deploymentId = target.Deploy(deployment) |> Async.AwaitTask
                    
                    // Wait for target to be healthy
                    let! targetHealth = target.GetHealth(nodeId) |> Async.AwaitTask
                    
                    if targetHealth.HealthScore > 0.8 then
                        // Remove from source platform
                        let! removed = source.Remove(nodeId) |> Async.AwaitTask
                        
                        if removed then
                            // Update managed configuration
                            managedNodes.[nodeId] <- updatedConfig
                            nodeHealth.[nodeId] <- { targetHealth with State = Running }
                            
                            logger.LogInformation($"✅ Successfully migrated node {nodeId} from {currentPlatform} to {targetPlatform}")
                            return true
                        else
                            logger.LogError($"❌ Failed to remove node {nodeId} from source platform")
                            return false
                    else
                        logger.LogError($"❌ Target platform health check failed for node {nodeId}")
                        return false
                
                with ex ->
                    logger.LogError($"❌ Migration failed for node {nodeId}: {ex.Message}")
                    
                    // Attempt rollback
                    if nodeHealth.ContainsKey(nodeId) then
                        let currentHealth = nodeHealth.[nodeId]
                        nodeHealth.[nodeId] <- { currentHealth with State = Running }
                    
                    return false
            
            | _ ->
                logger.LogError($"❌ Platform adapters not found for migration")
                return false
        }
        
        /// Discover existing TARS Nodes on platforms
        member _.DiscoverNodes(platform: TarsNodePlatform option) = async {
            logger.LogInformation("🔍 Discovering existing TARS Nodes...")
            
            let adaptersToCheck = 
                match platform with
                | Some(p) -> platformAdapters |> List.filter (fun a -> a.CanDeploy(p))
                | None -> platformAdapters
            
            // In a real implementation, this would query each platform for existing TARS nodes
            // For now, return the currently managed nodes
            let discoveredNodes = managedNodes.Values |> Seq.toList
            
            logger.LogInformation($"🔍 Discovered {discoveredNodes.Length} TARS Nodes")
            return discoveredNodes
        }
        
        /// Optimize TARS Node deployment based on constraints
        member _.OptimizeDeployment(constraints: string list) = async {
            logger.LogInformation($"🎯 Optimizing TARS Node deployment with constraints: {String.Join(", ", constraints)}")
            
            // Analyze current deployment
            let currentNodes = managedNodes.Values |> Seq.toList
            
            // Generate optimization recommendations
            let optimizations = [
                // Example optimization: Consolidate underutilized nodes
                if currentNodes.Length > 3 then
                    yield {
                        Config = TarsNodeFactory.CreateKubernetesNode("tars", "optimized", HybridNode([
                            CoreNode(["executive"; "operations"])
                            SpecializedNode("ui")
                            SpecializedNode("knowledge")
                        ]))
                        DeploymentStrategy = "consolidation"
                        RolloutPolicy = "gradual"
                        HealthChecks = ["liveness"; "readiness"]
                        MonitoringConfig = Map [("optimization", "consolidation")]
                        BackupStrategy = "automated"
                        UpdatePolicy = "zero_downtime"
                    }
                
                // Example optimization: Add edge nodes for better performance
                if constraints |> List.contains "low_latency" then
                    yield {
                        Config = TarsNodeFactory.CreateEdgeDeviceNode("edge-opt-001", "user-location", ["local_inference"; "caching"])
                        DeploymentStrategy = "edge_optimization"
                        RolloutPolicy = "immediate"
                        HealthChecks = ["liveness"]
                        MonitoringConfig = Map [("optimization", "latency")]
                        BackupStrategy = "local_with_sync"
                        UpdatePolicy = "rolling"
                    }
            ]
            
            logger.LogInformation($"🎯 Generated {optimizations.Length} optimization recommendations")
            return optimizations
        }
        
        interface ITarsNodeManager with
            member this.RegisterPlatformAdapter(adapter) = this.RegisterPlatformAdapter(adapter)
            member this.CreateNode(config) = this.CreateNode(config) |> Async.StartAsTask
            member this.DeployNode(nodeId) (platform) = this.DeployNode nodeId platform |> Async.StartAsTask
            member this.GetNodeHealth(nodeId) = this.GetNodeHealth(nodeId) |> Async.StartAsTask
            member this.ScaleNodes(role) (targetCount) = this.ScaleNodes role targetCount |> Async.StartAsTask
            member this.MigrateNode(nodeId) (targetPlatform) = this.MigrateNode nodeId targetPlatform |> Async.StartAsTask
            member this.DiscoverNodes(platform) = this.DiscoverNodes(platform) |> Async.StartAsTask
            member this.OptimizeDeployment(constraints) = this.OptimizeDeployment(constraints) |> Async.StartAsTask
