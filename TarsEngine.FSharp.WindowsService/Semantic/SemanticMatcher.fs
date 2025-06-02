namespace TarsEngine.FSharp.WindowsService.Semantic

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Matching criteria for task-agent pairing
/// </summary>
type MatchingCriteria = {
    CapabilityWeight: float
    AvailabilityWeight: float
    PerformanceWeight: float
    LoadBalanceWeight: float
    PreferenceWeight: float
    ProximityWeight: float
    CostWeight: float
}

/// <summary>
/// Agent match result
/// </summary>
type AgentMatchResult = {
    Agent: AgentCapabilityProfile
    MatchScore: float
    CapabilityMatch: float
    AvailabilityScore: float
    PerformanceScore: float
    LoadBalanceScore: float
    PreferenceScore: float
    ProximityScore: float
    CostScore: float
    Confidence: float
    Reasoning: string list
    EstimatedExecutionTime: TimeSpan
    EstimatedCost: float
}

/// <summary>
/// Matching strategy
/// </summary>
type MatchingStrategy =
    | BestMatch          // Single best agent
    | TopN of int        // Top N agents
    | Threshold of float // All agents above threshold
    | LoadBalanced       // Distribute load evenly
    | Specialized        // Prefer specialists
    | Diversified        // Prefer diverse capabilities

/// <summary>
/// Matching context
/// </summary>
type MatchingContext = {
    TaskRequest: TaskRequest
    AvailableAgents: AgentCapabilityProfile list
    Strategy: MatchingStrategy
    Criteria: MatchingCriteria
    Constraints: MatchingConstraint list
    Preferences: MatchingPreference list
}

/// <summary>
/// Matching constraints
/// </summary>
and MatchingConstraint =
    | RequiredCapability of string * CapabilityLevel
    | MaxExecutionTime of TimeSpan
    | MaxCost of float
    | ExcludeAgent of string
    | RequireAgent of string
    | MinSuccessRate of float
    | MaxLoadFactor of float

/// <summary>
/// Matching preferences
/// </summary>
and MatchingPreference =
    | PreferAgent of string * float
    | PreferCapability of string * float
    | PreferPerformance of float
    | PreferAvailability of float
    | PreferCost of float

/// <summary>
/// Matching result
/// </summary>
type MatchingResult = {
    TaskRequestId: string
    MatchedAgents: AgentMatchResult list
    Strategy: MatchingStrategy
    TotalCandidates: int
    MatchingTime: TimeSpan
    Confidence: float
    Reasoning: string list
    Alternatives: AgentMatchResult list
    Recommendations: string list
}

/// <summary>
/// Intelligent semantic matcher for task-agent pairing
/// </summary>
type SemanticMatcher(logger: ILogger<SemanticMatcher>, semanticAnalyzer: SemanticAnalyzer, capabilityProfiler: AgentCapabilityProfiler) =
    
    let matchingHistory = ConcurrentQueue<MatchingResult>()
    let matchingStatistics = ConcurrentDictionary<string, int64>()
    let learningData = ConcurrentDictionary<string, float>()
    
    let defaultCriteria = {
        CapabilityWeight = 0.4
        AvailabilityWeight = 0.2
        PerformanceWeight = 0.2
        LoadBalanceWeight = 0.1
        PreferenceWeight = 0.05
        ProximityWeight = 0.03
        CostWeight = 0.02
    }
    
    let maxMatchingHistory = 10000
    
    /// Find best matching agents for a task request
    member this.FindBestMatchAsync(taskRequest: TaskRequest, strategy: MatchingStrategy, criteria: MatchingCriteria option) = task {
        try
            let startTime = DateTime.UtcNow
            logger.LogInformation($"Finding best match for task: {taskRequest.Title}")
            
            // Get available agents
            let! availableAgents = capabilityProfiler.GetAllAgentProfiles() |> Task.FromResult
            let activeAgents = availableAgents |> List.filter (fun a -> a.Availability = AgentAvailability.Available)
            
            if activeAgents.IsEmpty then
                logger.LogWarning($"No available agents for task: {taskRequest.Id}")
                return Error "No available agents"
            
            // Create matching context
            let matchingContext = {
                TaskRequest = taskRequest
                AvailableAgents = activeAgents
                Strategy = strategy
                Criteria = criteria |> Option.defaultValue defaultCriteria
                Constraints = []
                Preferences = []
            }
            
            // Perform matching
            let! matchResults = this.PerformMatchingAsync(matchingContext)
            
            // Apply strategy
            let selectedAgents = this.ApplyMatchingStrategy(matchResults, strategy)
            
            let matchingTime = DateTime.UtcNow - startTime
            
            let result = {
                TaskRequestId = taskRequest.Id
                MatchedAgents = selectedAgents
                Strategy = strategy
                TotalCandidates = activeAgents.Length
                MatchingTime = matchingTime
                Confidence = this.CalculateOverallConfidence(selectedAgents)
                Reasoning = this.GenerateMatchingReasoning(selectedAgents, matchingContext)
                Alternatives = matchResults |> List.except selectedAgents |> List.take (min 3 (matchResults.Length - selectedAgents.Length))
                Recommendations = this.GenerateRecommendations(matchResults, matchingContext)
            }
            
            // Store result for learning
            matchingHistory.Enqueue(result)
            while matchingHistory.Count > maxMatchingHistory do
                matchingHistory.TryDequeue() |> ignore
            
            // Update statistics
            this.UpdateMatchingStatistics(result)
            
            logger.LogInformation($"Matching completed: {selectedAgents.Length} agents selected in {matchingTime.TotalMilliseconds:F0}ms")
            return Ok result
            
        with
        | ex ->
            logger.LogError(ex, $"Error finding best match for task: {taskRequest.Title}")
            return Error ex.Message
    }
    
    /// Perform detailed matching analysis
    member private this.PerformMatchingAsync(context: MatchingContext) = task {
        let matchResults = ResizeArray<AgentMatchResult>()
        
        for agent in context.AvailableAgents do
            try
                // Calculate capability match
                let capabilityMatch = this.CalculateCapabilityMatch(agent, context.TaskRequest.RequiredCapabilities)
                
                // Calculate availability score
                let availabilityScore = this.CalculateAvailabilityScore(agent)
                
                // Calculate performance score
                let performanceScore = this.CalculatePerformanceScore(agent, context.TaskRequest)
                
                // Calculate load balance score
                let loadBalanceScore = this.CalculateLoadBalanceScore(agent)
                
                // Calculate preference score
                let preferenceScore = this.CalculatePreferenceScore(agent, context.TaskRequest)
                
                // Calculate proximity score (simplified)
                let proximityScore = 1.0 // Would be based on geographic or network proximity
                
                // Calculate cost score
                let costScore = this.CalculateCostScore(agent, context.TaskRequest)
                
                // Calculate overall match score
                let overallScore = 
                    capabilityMatch * context.Criteria.CapabilityWeight +
                    availabilityScore * context.Criteria.AvailabilityWeight +
                    performanceScore * context.Criteria.PerformanceWeight +
                    loadBalanceScore * context.Criteria.LoadBalanceWeight +
                    preferenceScore * context.Criteria.PreferenceWeight +
                    proximityScore * context.Criteria.ProximityWeight +
                    costScore * context.Criteria.CostWeight
                
                // Calculate confidence
                let confidence = this.CalculateMatchConfidence(agent, context.TaskRequest, overallScore)
                
                // Generate reasoning
                let reasoning = this.GenerateAgentReasoning(agent, context.TaskRequest, capabilityMatch, performanceScore)
                
                // Estimate execution time and cost
                let estimatedTime = this.EstimateExecutionTime(agent, context.TaskRequest)
                let estimatedCost = this.EstimateCost(agent, context.TaskRequest, estimatedTime)
                
                let matchResult = {
                    Agent = agent
                    MatchScore = overallScore
                    CapabilityMatch = capabilityMatch
                    AvailabilityScore = availabilityScore
                    PerformanceScore = performanceScore
                    LoadBalanceScore = loadBalanceScore
                    PreferenceScore = preferenceScore
                    ProximityScore = proximityScore
                    CostScore = costScore
                    Confidence = confidence
                    Reasoning = reasoning
                    EstimatedExecutionTime = estimatedTime
                    EstimatedCost = estimatedCost
                }
                
                matchResults.Add(matchResult)
                
            with
            | ex ->
                logger.LogWarning(ex, $"Error calculating match for agent: {agent.AgentName}")
        
        // Sort by match score
        let sortedResults = 
            matchResults
            |> Seq.sortByDescending (fun r -> r.MatchScore)
            |> List.ofSeq
        
        return sortedResults
    }
    
    /// Calculate capability match score
    member private this.CalculateCapabilityMatch(agent: AgentCapabilityProfile, requirements: CapabilityRequirement list) =
        if requirements.IsEmpty then 1.0
        else
            let totalWeight = requirements |> List.sumBy (fun req -> req.Weight)
            let matchedWeight = 
                requirements
                |> List.sumBy (fun req ->
                    match agent.Capabilities.TryGetValue(req.Name) with
                    | true, capability ->
                        let levelMatch = 
                            if capability.Level >= req.Level then 1.0
                            else float capability.Level / float req.Level
                        let confidenceMatch = capability.Confidence
                        req.Weight * levelMatch * confidenceMatch
                    | false, _ -> 0.0)
            
            if totalWeight = 0.0 then 0.0 else matchedWeight / totalWeight
    
    /// Calculate availability score
    member private this.CalculateAvailabilityScore(agent: AgentCapabilityProfile) =
        match agent.Availability with
        | AgentAvailability.Available -> 1.0 - agent.LoadFactor
        | AgentAvailability.Busy -> 0.3
        | AgentAvailability.Overloaded -> 0.1
        | AgentAvailability.Maintenance -> 0.0
        | AgentAvailability.Offline -> 0.0
        | AgentAvailability.Restricted -> 0.2
    
    /// Calculate performance score
    member private this.CalculatePerformanceScore(agent: AgentCapabilityProfile, taskRequest: TaskRequest) =
        let reliabilityWeight = 0.4
        let efficiencyWeight = 0.3
        let qualityWeight = 0.3
        
        agent.ReliabilityScore * reliabilityWeight +
        agent.EfficiencyScore * efficiencyWeight +
        agent.PerformanceHistory.AverageQualityScore * qualityWeight
    
    /// Calculate load balance score
    member private this.CalculateLoadBalanceScore(agent: AgentCapabilityProfile) =
        1.0 - agent.LoadFactor
    
    /// Calculate preference score
    member private this.CalculatePreferenceScore(agent: AgentCapabilityProfile, taskRequest: TaskRequest) =
        // Check if agent specializes in the task domain
        let domainMatch = 
            if agent.Specializations |> List.contains taskRequest.SemanticMetadata.Domain then 0.8
            else 0.5
        
        // Check if agent prefers this type of task
        let typeMatch = 
            if agent.PreferredTaskTypes |> List.exists (fun t -> taskRequest.SemanticMetadata.Intent.Contains(t)) then 0.7
            else 0.5
        
        (domainMatch + typeMatch) / 2.0
    
    /// Calculate cost score
    member private this.CalculateCostScore(agent: AgentCapabilityProfile, taskRequest: TaskRequest) =
        // Simplified cost calculation (in production, would consider actual pricing)
        let baseCost = 1.0
        let complexityMultiplier = 
            match taskRequest.Complexity with
            | TaskComplexity.Simple -> 1.0
            | TaskComplexity.Moderate -> 1.5
            | TaskComplexity.Complex -> 2.0
            | TaskComplexity.Expert -> 3.0
            | TaskComplexity.Collaborative -> 2.5
        
        let agentCostMultiplier = agent.CapabilityScore // Higher capability = higher cost
        let totalCost = baseCost * complexityMultiplier * agentCostMultiplier
        
        // Return inverse score (lower cost = higher score)
        1.0 / (1.0 + totalCost)
    
    /// Apply matching strategy to select final agents
    member private this.ApplyMatchingStrategy(matchResults: AgentMatchResult list, strategy: MatchingStrategy) =
        match strategy with
        | BestMatch -> 
            matchResults |> List.take (min 1 matchResults.Length)
        
        | TopN n -> 
            matchResults |> List.take (min n matchResults.Length)
        
        | Threshold threshold -> 
            matchResults |> List.filter (fun r -> r.MatchScore >= threshold)
        
        | LoadBalanced -> 
            // Select agents with lowest load factors
            matchResults 
            |> List.sortBy (fun r -> r.Agent.LoadFactor)
            |> List.take (min 3 matchResults.Length)
        
        | Specialized -> 
            // Prefer agents with high capability scores in required areas
            matchResults 
            |> List.sortByDescending (fun r -> r.CapabilityMatch * r.Agent.CapabilityScore)
            |> List.take (min 2 matchResults.Length)
        
        | Diversified -> 
            // Select agents with diverse capabilities
            let selectedAgents = ResizeArray<AgentMatchResult>()
            let usedCapabilities = ResizeArray<string>()
            
            for result in matchResults do
                let agentCapabilities = result.Agent.Capabilities.Keys |> Set.ofSeq
                let hasNewCapability = agentCapabilities |> Set.exists (fun cap -> not (usedCapabilities.Contains(cap)))
                
                if hasNewCapability && selectedAgents.Count < 3 then
                    selectedAgents.Add(result)
                    usedCapabilities.AddRange(agentCapabilities)
            
            selectedAgents |> List.ofSeq
    
    /// Calculate overall confidence for the matching result
    member private this.CalculateOverallConfidence(selectedAgents: AgentMatchResult list) =
        if selectedAgents.IsEmpty then 0.0
        else selectedAgents |> List.averageBy (fun a -> a.Confidence)
    
    /// Generate reasoning for the matching decision
    member private this.GenerateMatchingReasoning(selectedAgents: AgentMatchResult list, context: MatchingContext) =
        let reasoning = ResizeArray<string>()
        
        reasoning.Add($"Selected {selectedAgents.Length} agents from {context.AvailableAgents.Length} candidates")
        reasoning.Add($"Strategy: {context.Strategy}")
        
        for agent in selectedAgents do
            reasoning.Add($"Agent {agent.Agent.AgentName}: {agent.MatchScore:F2} match score ({agent.Confidence:F2} confidence)")
        
        reasoning |> List.ofSeq
    
    /// Generate agent-specific reasoning
    member private this.GenerateAgentReasoning(agent: AgentCapabilityProfile, taskRequest: TaskRequest, capabilityMatch: float, performanceScore: float) =
        let reasoning = ResizeArray<string>()
        
        reasoning.Add($"Capability match: {capabilityMatch:P0}")
        reasoning.Add($"Performance score: {performanceScore:F2}")
        reasoning.Add($"Availability: {agent.Availability}")
        reasoning.Add($"Load factor: {agent.LoadFactor:P0}")
        
        if agent.Specializations.Length > 0 then
            reasoning.Add($"Specializations: {String.Join(", ", agent.Specializations)}")
        
        reasoning |> List.ofSeq
    
    /// Calculate match confidence
    member private this.CalculateMatchConfidence(agent: AgentCapabilityProfile, taskRequest: TaskRequest, matchScore: float) =
        let factorCount = 4.0
        let capabilityConfidence = if agent.Capabilities.Count > 0 then agent.Capabilities.Values |> Seq.averageBy (fun c -> c.Confidence) else 0.5
        let performanceConfidence = agent.ReliabilityScore
        let availabilityConfidence = if agent.Availability = AgentAvailability.Available then 1.0 else 0.5
        let scoreConfidence = matchScore
        
        (capabilityConfidence + performanceConfidence + availabilityConfidence + scoreConfidence) / factorCount
    
    /// Estimate execution time
    member private this.EstimateExecutionTime(agent: AgentCapabilityProfile, taskRequest: TaskRequest) =
        let baseTime = taskRequest.ResourceRequirements.EstimatedDuration
        let complexityMultiplier = 
            match taskRequest.Complexity with
            | TaskComplexity.Simple -> 0.8
            | TaskComplexity.Moderate -> 1.0
            | TaskComplexity.Complex -> 1.5
            | TaskComplexity.Expert -> 2.0
            | TaskComplexity.Collaborative -> 1.8
        
        let agentEfficiencyMultiplier = 2.0 - agent.EfficiencyScore // Higher efficiency = lower time
        
        TimeSpan.FromMilliseconds(baseTime.TotalMilliseconds * complexityMultiplier * agentEfficiencyMultiplier)
    
    /// Estimate cost
    member private this.EstimateCost(agent: AgentCapabilityProfile, taskRequest: TaskRequest, estimatedTime: TimeSpan) =
        let hourlyRate = 50.0 + (agent.CapabilityScore * 50.0) // $50-100 per hour based on capability
        let hours = estimatedTime.TotalHours
        hourlyRate * hours
    
    /// Generate recommendations
    member private this.GenerateRecommendations(matchResults: AgentMatchResult list, context: MatchingContext) =
        let recommendations = ResizeArray<string>()
        
        if matchResults.IsEmpty then
            recommendations.Add("No suitable agents found. Consider adjusting requirements or training new agents.")
        elif matchResults.Length = 1 then
            recommendations.Add("Consider having backup agents available for critical tasks.")
        else
            let topAgent = matchResults |> List.head
            if topAgent.MatchScore < 0.7 then
                recommendations.Add("Match score is below optimal. Consider agent training or requirement adjustment.")
        
        recommendations |> List.ofSeq
    
    /// Update matching statistics
    member private this.UpdateMatchingStatistics(result: MatchingResult) =
        matchingStatistics.AddOrUpdate("TotalMatches", 1L, fun _ current -> current + 1L) |> ignore
        matchingStatistics.AddOrUpdate($"Strategy_{result.Strategy}", 1L, fun _ current -> current + 1L) |> ignore
        
        let avgMatchingTime = matchingStatistics.GetOrAdd("TotalMatchingTimeMs", 0L)
        let newAvgTime = (avgMatchingTime + int64 result.MatchingTime.TotalMilliseconds) / 2L
        matchingStatistics.["TotalMatchingTimeMs"] <- newAvgTime
    
    /// Get matching statistics
    member this.GetMatchingStatistics() =
        let totalMatches = matchingStatistics.GetOrAdd("TotalMatches", 0L)
        let avgMatchingTime = matchingStatistics.GetOrAdd("TotalMatchingTimeMs", 0L)
        
        Map.ofSeq [
            ("TotalMatches", totalMatches :> obj)
            ("AverageMatchingTimeMs", avgMatchingTime :> obj)
            ("MatchingHistorySize", matchingHistory.Count :> obj)
        ]
    
    /// Get recent matching results
    member this.GetRecentMatchingResults(count: int) =
        matchingHistory 
        |> Seq.take (min count matchingHistory.Count)
        |> List.ofSeq
