namespace TarsEngine.FSharp.WindowsService.Semantic

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Agent capability profile with performance history
/// </summary>
type AgentCapabilityProfile = {
    AgentId: string
    AgentName: string
    AgentType: string
    Capabilities: Map<string, AgentCapability>
    Specializations: string list
    PreferredTaskTypes: string list
    PerformanceHistory: PerformanceHistory
    Availability: AgentAvailability
    LoadFactor: float
    LastUpdated: DateTime
    CapabilityScore: float
    ReliabilityScore: float
    EfficiencyScore: float
}

/// <summary>
/// Performance history for an agent
/// </summary>
and PerformanceHistory = {
    TasksCompleted: int64
    TasksSuccessful: int64
    TasksFailed: int64
    AverageExecutionTime: TimeSpan
    AverageQualityScore: float
    RecentPerformance: PerformanceMetric list
    CapabilityPerformance: Map<string, CapabilityPerformance>
    TrendDirection: TrendDirection
}

/// <summary>
/// Performance metric data point
/// </summary>
and PerformanceMetric = {
    TaskId: string
    Capability: string
    ExecutionTime: TimeSpan
    QualityScore: float
    Success: bool
    Timestamp: DateTime
    Feedback: string option
}

/// <summary>
/// Capability-specific performance
/// </summary>
and CapabilityPerformance = {
    CapabilityName: string
    TasksCompleted: int64
    SuccessRate: float
    AverageExecutionTime: TimeSpan
    AverageQualityScore: float
    ImprovementRate: float
    LastUsed: DateTime
    Confidence: float
}

/// <summary>
/// Capability learning progress
/// </summary>
type CapabilityLearning = {
    CapabilityName: string
    InitialLevel: CapabilityLevel
    CurrentLevel: CapabilityLevel
    TargetLevel: CapabilityLevel
    LearningRate: float
    ExperiencePoints: int
    Milestones: LearningMilestone list
    NextMilestone: LearningMilestone option
}

/// <summary>
/// Learning milestone
/// </summary>
and LearningMilestone = {
    Level: CapabilityLevel
    RequiredExperience: int
    AchievedAt: DateTime option
    Requirements: string list
    Rewards: string list
}

/// <summary>
/// Capability recommendation
/// </summary>
type CapabilityRecommendation = {
    AgentId: string
    RecommendationType: RecommendationType
    CapabilityName: string
    CurrentLevel: CapabilityLevel option
    RecommendedLevel: CapabilityLevel
    Priority: RecommendationPriority
    Reasoning: string list
    EstimatedLearningTime: TimeSpan
    Prerequisites: string list
    Benefits: string list
}

/// <summary>
/// Recommendation types
/// </summary>
and RecommendationType =
    | LearnNew
    | Improve
    | Specialize
    | Diversify
    | Refresh

/// <summary>
/// Recommendation priority
/// </summary>
and RecommendationPriority =
    | Critical
    | High
    | Medium
    | Low
    | Optional

/// <summary>
/// Agent capability profiler for skill management and optimization
/// </summary>
type AgentCapabilityProfiler(logger: ILogger<AgentCapabilityProfiler>, semanticAnalyzer: SemanticAnalyzer) =
    
    let agentProfiles = ConcurrentDictionary<string, AgentCapabilityProfile>()
    let capabilityLearning = ConcurrentDictionary<string, Map<string, CapabilityLearning>>()
    let performanceMetrics = ConcurrentQueue<PerformanceMetric>()
    let capabilityRegistry = ConcurrentDictionary<string, CapabilityDefinition>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable analysisTask: Task option = None
    
    let maxPerformanceHistory = 1000
    let learningAnalysisInterval = TimeSpan.FromHours(1.0)
    
    /// Start the capability profiler
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting agent capability profiler...")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Initialize built-in capabilities
            this.InitializeBuiltInCapabilities()
            
            // Start analysis loop
            let analysisLoop = this.AnalysisLoopAsync(cancellationTokenSource.Value.Token)
            analysisTask <- Some analysisLoop
            
            logger.LogInformation("Agent capability profiler started successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to start agent capability profiler")
            isRunning <- false
            raise
    }
    
    /// Stop the capability profiler
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping agent capability profiler...")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Wait for analysis task to complete
            match analysisTask with
            | Some task ->
                try
                    do! task.WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning("Capability profiler analysis task did not complete within timeout")
                | ex ->
                    logger.LogWarning(ex, "Error waiting for capability profiler analysis task to complete")
            | None -> ()
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            analysisTask <- None
            
            logger.LogInformation("Agent capability profiler stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping agent capability profiler")
    }
    
    /// Register an agent with initial capabilities
    member this.RegisterAgentAsync(agentId: string, agentName: string, agentType: string, initialCapabilities: AgentCapability list) = task {
        try
            logger.LogInformation($"Registering agent: {agentName} ({agentId})")
            
            let capabilityMap = 
                initialCapabilities
                |> List.map (fun cap -> (cap.Name, cap))
                |> Map.ofList
            
            let profile = {
                AgentId = agentId
                AgentName = agentName
                AgentType = agentType
                Capabilities = capabilityMap
                Specializations = []
                PreferredTaskTypes = []
                PerformanceHistory = {
                    TasksCompleted = 0L
                    TasksSuccessful = 0L
                    TasksFailed = 0L
                    AverageExecutionTime = TimeSpan.Zero
                    AverageQualityScore = 0.0
                    RecentPerformance = []
                    CapabilityPerformance = Map.empty
                    TrendDirection = TrendDirection.Stable
                }
                Availability = AgentAvailability.Available
                LoadFactor = 0.0
                LastUpdated = DateTime.UtcNow
                CapabilityScore = this.CalculateCapabilityScore(capabilityMap)
                ReliabilityScore = 1.0
                EfficiencyScore = 1.0
            }
            
            agentProfiles.[agentId] <- profile
            
            // Initialize learning for each capability
            let learning = 
                initialCapabilities
                |> List.map (fun cap -> 
                    let learningData = {
                        CapabilityName = cap.Name
                        InitialLevel = cap.Level
                        CurrentLevel = cap.Level
                        TargetLevel = CapabilityLevel.Expert
                        LearningRate = 1.0
                        ExperiencePoints = 0
                        Milestones = this.CreateLearningMilestones(cap.Name)
                        NextMilestone = None
                    }
                    (cap.Name, learningData))
                |> Map.ofList
            
            capabilityLearning.[agentId] <- learning
            
            logger.LogInformation($"Agent registered successfully: {agentName} with {initialCapabilities.Length} capabilities")
            return Ok ()
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to register agent: {agentName}")
            return Error ex.Message
    }
    
    /// Update agent capabilities based on task performance
    member this.UpdateCapabilitiesAsync(agentId: string, taskId: string, capability: string, executionTime: TimeSpan, qualityScore: float, success: bool, feedback: string option) = task {
        try
            logger.LogDebug($"Updating capabilities for agent: {agentId}")
            
            // Record performance metric
            let metric = {
                TaskId = taskId
                Capability = capability
                ExecutionTime = executionTime
                QualityScore = qualityScore
                Success = success
                Timestamp = DateTime.UtcNow
                Feedback = feedback
            }
            
            performanceMetrics.Enqueue(metric)
            
            // Keep performance history manageable
            while performanceMetrics.Count > maxPerformanceHistory do
                performanceMetrics.TryDequeue() |> ignore
            
            // Update agent profile
            match agentProfiles.TryGetValue(agentId) with
            | true, profile ->
                // Update performance history
                let updatedHistory = this.UpdatePerformanceHistory(profile.PerformanceHistory, metric)
                
                // Update capability performance
                let updatedCapabilities = this.UpdateCapabilityPerformance(profile.Capabilities, capability, metric)
                
                // Calculate new scores
                let capabilityScore = this.CalculateCapabilityScore(updatedCapabilities)
                let reliabilityScore = this.CalculateReliabilityScore(updatedHistory)
                let efficiencyScore = this.CalculateEfficiencyScore(updatedHistory)
                
                let updatedProfile = {
                    profile with
                        Capabilities = updatedCapabilities
                        PerformanceHistory = updatedHistory
                        LastUpdated = DateTime.UtcNow
                        CapabilityScore = capabilityScore
                        ReliabilityScore = reliabilityScore
                        EfficiencyScore = efficiencyScore
                }
                
                agentProfiles.[agentId] <- updatedProfile
                
                // Update learning progress
                do! this.UpdateLearningProgressAsync(agentId, capability, success, qualityScore)
                
                logger.LogDebug($"Capabilities updated for agent: {agentId}")
                return Ok ()
            
            | false, _ ->
                let error = $"Agent not found: {agentId}"
                logger.LogWarning(error)
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Failed to update capabilities for agent: {agentId}")
            return Error ex.Message
    }
    
    /// Get agent capability profile
    member this.GetAgentProfileAsync(agentId: string) = task {
        match agentProfiles.TryGetValue(agentId) with
        | true, profile -> return Some profile
        | false, _ -> return None
    }
    
    /// Find agents by capability requirements
    member this.FindAgentsByCapabilityAsync(requirements: CapabilityRequirement list) = task {
        try
            logger.LogDebug($"Finding agents by capability requirements: {requirements.Length} requirements")
            
            let matchingAgents = 
                agentProfiles.Values
                |> Seq.filter (fun profile -> profile.Availability = AgentAvailability.Available)
                |> Seq.map (fun profile ->
                    let matchScore = this.CalculateCapabilityMatch(profile, requirements)
                    (profile, matchScore))
                |> Seq.filter (fun (_, score) -> score > 0.5) // Minimum 50% match
                |> Seq.sortByDescending snd
                |> Seq.map fst
                |> List.ofSeq
            
            logger.LogDebug($"Found {matchingAgents.Length} matching agents")
            return matchingAgents
            
        with
        | ex ->
            logger.LogError(ex, "Error finding agents by capability")
            return []
    }
    
    /// Generate capability recommendations for an agent
    member this.GenerateRecommendationsAsync(agentId: string) = task {
        try
            logger.LogDebug($"Generating capability recommendations for agent: {agentId}")
            
            match agentProfiles.TryGetValue(agentId) with
            | true, profile ->
                let recommendations = ResizeArray<CapabilityRecommendation>()
                
                // Analyze current capabilities and performance
                for kvp in profile.Capabilities do
                    let capability = kvp.Value
                    
                    // Recommend improvement if performance is below average
                    if capability.SuccessRate < 0.8 then
                        let recommendation = {
                            AgentId = agentId
                            RecommendationType = Improve
                            CapabilityName = capability.Name
                            CurrentLevel = Some capability.Level
                            RecommendedLevel = capability.Level
                            Priority = High
                            Reasoning = [$"Success rate is {capability.SuccessRate:P0}, below 80% threshold"]
                            EstimatedLearningTime = TimeSpan.FromHours(8.0)
                            Prerequisites = []
                            Benefits = ["Improved task success rate"; "Better performance scores"]
                        }
                        recommendations.Add(recommendation)
                
                // Recommend new capabilities based on market demand
                let marketDemandCapabilities = ["WebAPI"; "Infrastructure"; "DataProcessor"; "TestGenerator"]
                for capabilityName in marketDemandCapabilities do
                    if not (profile.Capabilities.ContainsKey(capabilityName)) then
                        let recommendation = {
                            AgentId = agentId
                            RecommendationType = LearnNew
                            CapabilityName = capabilityName
                            CurrentLevel = None
                            RecommendedLevel = CapabilityLevel.Beginner
                            Priority = Medium
                            Reasoning = ["High market demand"; "Expands agent versatility"]
                            EstimatedLearningTime = TimeSpan.FromHours(16.0)
                            Prerequisites = []
                            Benefits = ["Access to more tasks"; "Increased earning potential"]
                        }
                        recommendations.Add(recommendation)
                
                logger.LogDebug($"Generated {recommendations.Count} recommendations for agent: {agentId}")
                return recommendations |> List.ofSeq
            
            | false, _ ->
                logger.LogWarning($"Agent not found for recommendations: {agentId}")
                return []
                
        with
        | ex ->
            logger.LogError(ex, $"Error generating recommendations for agent: {agentId}")
            return []
    }
    
    /// Initialize built-in capabilities
    member private this.InitializeBuiltInCapabilities() =
        let builtInCapabilities = [
            ("WebAPI", "Web API development and REST services")
            ("Infrastructure", "Infrastructure automation and DevOps")
            ("DataProcessor", "Data processing and transformation")
            ("TestGenerator", "Test generation and quality assurance")
            ("DocumentationGenerator", "Documentation creation and maintenance")
            ("CodeAnalyzer", "Code analysis and quality assessment")
            ("DatabaseMigration", "Database schema management")
            ("DeploymentScript", "Deployment automation")
            ("MonitoringDashboard", "Monitoring and observability")
            ("ProjectManagement", "Project planning and coordination")
        ]
        
        for (name, description) in builtInCapabilities do
            let definition = {
                Name = name
                Description = description
                Category = "Technical"
                Prerequisites = []
                LearningPath = []
                AssessmentCriteria = []
            }
            capabilityRegistry.[name] <- definition
        
        logger.LogInformation($"Initialized {builtInCapabilities.Length} built-in capabilities")
    
    /// Analysis loop for continuous learning and optimization
    member private this.AnalysisLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting capability analysis loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Analyze agent performance trends
                    this.AnalyzePerformanceTrends()
                    
                    // Update capability levels based on learning
                    this.UpdateCapabilityLevels()
                    
                    // Generate insights and recommendations
                    this.GenerateInsights()
                    
                    // Wait for next analysis cycle
                    do! Task.Delay(learningAnalysisInterval, cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, "Error in capability analysis loop")
                    do! Task.Delay(learningAnalysisInterval, cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Capability analysis loop cancelled")
        | ex ->
            logger.LogError(ex, "Capability analysis loop failed")
    }
    
    /// Helper methods for calculations and updates
    member private this.CalculateCapabilityScore(capabilities: Map<string, AgentCapability>) =
        if capabilities.IsEmpty then 0.0
        else
            capabilities.Values
            |> Seq.averageBy (fun cap -> float cap.Level * cap.Confidence)
    
    member private this.CalculateReliabilityScore(history: PerformanceHistory) =
        if history.TasksCompleted = 0L then 1.0
        else float history.TasksSuccessful / float history.TasksCompleted
    
    member private this.CalculateEfficiencyScore(history: PerformanceHistory) =
        // Simplified efficiency calculation
        if history.AverageExecutionTime = TimeSpan.Zero then 1.0
        else min 1.0 (TimeSpan.FromHours(1.0).TotalSeconds / history.AverageExecutionTime.TotalSeconds)
    
    member private this.CalculateCapabilityMatch(profile: AgentCapabilityProfile, requirements: CapabilityRequirement list) =
        if requirements.IsEmpty then 1.0
        else
            let totalWeight = requirements |> List.sumBy (fun req -> req.Weight)
            let matchedWeight = 
                requirements
                |> List.sumBy (fun req ->
                    match profile.Capabilities.TryGetValue(req.Name) with
                    | true, capability when capability.Level >= req.Level -> req.Weight
                    | true, capability -> req.Weight * 0.5 // Partial match
                    | false, _ -> 0.0)
            
            if totalWeight = 0.0 then 0.0 else matchedWeight / totalWeight
    
    // Additional helper methods would be implemented here...
    member private this.UpdatePerformanceHistory(history: PerformanceHistory, metric: PerformanceMetric) = history
    member private this.UpdateCapabilityPerformance(capabilities: Map<string, AgentCapability>, capability: string, metric: PerformanceMetric) = capabilities
    member private this.UpdateLearningProgressAsync(agentId: string, capability: string, success: bool, qualityScore: float) = task { () }
    member private this.CreateLearningMilestones(capabilityName: string) = []
    member private this.AnalyzePerformanceTrends() = ()
    member private this.UpdateCapabilityLevels() = ()
    member private this.GenerateInsights() = ()

/// <summary>
/// Capability definition for the registry
/// </summary>
and CapabilityDefinition = {
    Name: string
    Description: string
    Category: string
    Prerequisites: string list
    LearningPath: string list
    AssessmentCriteria: string list
}
