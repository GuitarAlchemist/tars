namespace TarsEngine.FSharp.WindowsService.Roadmap

open System
open System.IO
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core
open TarsEngine.FSharp.WindowsService.Agents

/// <summary>
/// Roadmap analysis result
/// </summary>
type RoadmapAnalysisResult = {
    RoadmapId: string
    AnalysisType: AnalysisType
    Findings: AnalysisFinding list
    Recommendations: AchievementRecommendation list
    RiskAssessment: RiskAssessment
    PerformanceMetrics: AchievementMetrics
    AnalyzedAt: DateTime
    AnalysisTime: TimeSpan
    Confidence: float
}

/// <summary>
/// Types of roadmap analysis
/// </summary>
and AnalysisType =
    | ProgressAnalysis
    | RiskAnalysis
    | PerformanceAnalysis
    | DependencyAnalysis
    | ResourceAnalysis
    | QualityAnalysis
    | ComprehensiveAnalysis

/// <summary>
/// Analysis finding
/// </summary>
and AnalysisFinding = {
    Type: FindingType
    Severity: FindingSeverity
    Title: string
    Description: string
    AffectedAchievements: string list
    Evidence: string list
    Recommendations: string list
    EstimatedImpact: float
}

/// <summary>
/// Finding types
/// </summary>
and FindingType =
    | ProgressDelay
    | ResourceBottleneck
    | DependencyIssue
    | QualityRisk
    | ScopeCreep
    | EstimationError
    | BlockerDetected
    | OpportunityIdentified

/// <summary>
/// Finding severity levels
/// </summary>
and FindingSeverity =
    | Critical
    | High
    | Medium
    | Low
    | Info

/// <summary>
/// Risk assessment
/// </summary>
and RiskAssessment = {
    OverallRisk: RiskLevel
    RiskFactors: RiskFactor list
    MitigationStrategies: string list
    ContingencyPlans: string list
    MonitoringRecommendations: string list
}

/// <summary>
/// Risk levels
/// </summary>
and RiskLevel =
    | VeryHigh
    | High
    | Medium
    | Low
    | VeryLow

/// <summary>
/// Risk factor
/// </summary>
and RiskFactor = {
    Type: RiskType
    Description: string
    Probability: float
    Impact: float
    RiskScore: float
    AffectedAchievements: string list
}

/// <summary>
/// Risk types
/// </summary>
and RiskType =
    | ScheduleRisk
    | ResourceRisk
    | TechnicalRisk
    | DependencyRisk
    | QualityRisk
    | ExternalRisk

/// <summary>
/// Roadmap analysis agent for autonomous roadmap management
/// </summary>
type RoadmapAnalysisAgent(agentId: string, logger: ILogger<RoadmapAnalysisAgent>, roadmapStorage: RoadmapStorage) =
    inherit BaseAgent(agentId, "RoadmapAnalysisAgent", logger)
    
    let analysisResults = ConcurrentQueue<RoadmapAnalysisResult>()
    let analysisHistory = ConcurrentDictionary<string, RoadmapAnalysisResult list>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable analysisTask: Task option = None
    
    let analysisInterval = TimeSpan.FromHours(1.0) // Analyze every hour
    let maxAnalysisHistory = 100
    
    /// Start the roadmap analysis agent
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation($"Starting roadmap analysis agent: {agentId}")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Start analysis loop
            let analysisLoop = this.AnalysisLoopAsync(cancellationTokenSource.Value.Token)
            analysisTask <- Some analysisLoop
            
            logger.LogInformation($"Roadmap analysis agent started: {agentId}")
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to start roadmap analysis agent: {agentId}")
            isRunning <- false
            raise
    }
    
    /// Stop the roadmap analysis agent
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation($"Stopping roadmap analysis agent: {agentId}")
            
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
                    logger.LogWarning($"Analysis task did not complete within timeout for agent: {agentId}")
                | ex ->
                    logger.LogWarning(ex, $"Error waiting for analysis task to complete for agent: {agentId}")
            | None -> ()
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            analysisTask <- None
            
            logger.LogInformation($"Roadmap analysis agent stopped: {agentId}")
            
        with
        | ex ->
            logger.LogError(ex, $"Error stopping roadmap analysis agent: {agentId}")
    }
    
    /// Analyze specific roadmap
    member this.AnalyzeRoadmapAsync(roadmapId: string, analysisType: AnalysisType) = task {
        try
            let startTime = DateTime.UtcNow
            logger.LogDebug($"Analyzing roadmap: {roadmapId} with type: {analysisType}")
            
            match! roadmapStorage.LoadRoadmapAsync(roadmapId) with
            | Some roadmap ->
                let! analysisResult = this.PerformAnalysisAsync(roadmap, analysisType)
                
                // Store analysis result
                analysisResults.Enqueue(analysisResult)
                
                // Update analysis history
                let history = analysisHistory.GetOrAdd(roadmapId, [])
                let updatedHistory = analysisResult :: history |> List.take maxAnalysisHistory
                analysisHistory.[roadmapId] <- updatedHistory
                
                // Apply recommendations if they are low-risk
                do! this.ApplyLowRiskRecommendationsAsync(roadmap, analysisResult.Recommendations)
                
                let analysisTime = DateTime.UtcNow - startTime
                logger.LogInformation($"Roadmap analysis completed: {roadmapId} in {analysisTime.TotalSeconds:F1}s")
                
                return Ok analysisResult
            
            | None ->
                let error = $"Roadmap not found: {roadmapId}"
                logger.LogWarning(error)
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Error analyzing roadmap: {roadmapId}")
            return Error ex.Message
    }
    
    /// Analyze all roadmaps
    member this.AnalyzeAllRoadmapsAsync() = task {
        try
            logger.LogDebug("Analyzing all roadmaps")
            
            let! roadmaps = roadmapStorage.GetAllRoadmapsAsync()
            let results = ResizeArray<RoadmapAnalysisResult>()
            
            for roadmap in roadmaps do
                let! analysisResult = this.AnalyzeRoadmapAsync(roadmap.Id, ComprehensiveAnalysis)
                match analysisResult with
                | Ok result -> results.Add(result)
                | Error error -> logger.LogWarning($"Failed to analyze roadmap {roadmap.Id}: {error}")
            
            logger.LogInformation($"Analyzed {results.Count} roadmaps")
            return results |> List.ofSeq
            
        with
        | ex ->
            logger.LogError(ex, "Error analyzing all roadmaps")
            return []
    }
    
    /// Get analysis results for roadmap
    member this.GetAnalysisResultsAsync(roadmapId: string, limit: int option) = task {
        match analysisHistory.TryGetValue(roadmapId) with
        | true, history ->
            let results = 
                history
                |> List.sortByDescending (fun r -> r.AnalyzedAt)
                |> (fun list -> 
                    match limit with
                    | Some l -> list |> List.take (min l list.Length)
                    | None -> list)
            return results
        | false, _ ->
            return []
    }
    
    /// Get recent analysis results
    member this.GetRecentAnalysisResults(limit: int) =
        analysisResults
        |> Seq.take limit
        |> List.ofSeq
    
    /// Perform detailed analysis
    member private this.PerformAnalysisAsync(roadmap: TarsRoadmap, analysisType: AnalysisType) = task {
        let startTime = DateTime.UtcNow
        
        let findings = ResizeArray<AnalysisFinding>()
        let recommendations = ResizeArray<AchievementRecommendation>()
        
        // Extract all achievements for analysis
        let allAchievements = 
            roadmap.Phases
            |> List.collect (fun phase -> phase.Milestones)
            |> List.collect (fun milestone -> milestone.Achievements)
        
        // Perform different types of analysis
        match analysisType with
        | ProgressAnalysis -> 
            do! this.AnalyzeProgressAsync(roadmap, allAchievements, findings, recommendations)
        | RiskAnalysis -> 
            do! this.AnalyzeRisksAsync(roadmap, allAchievements, findings, recommendations)
        | PerformanceAnalysis -> 
            do! this.AnalyzePerformanceAsync(roadmap, allAchievements, findings, recommendations)
        | DependencyAnalysis -> 
            do! this.AnalyzeDependenciesAsync(roadmap, allAchievements, findings, recommendations)
        | ResourceAnalysis -> 
            do! this.AnalyzeResourcesAsync(roadmap, allAchievements, findings, recommendations)
        | QualityAnalysis -> 
            do! this.AnalyzeQualityAsync(roadmap, allAchievements, findings, recommendations)
        | ComprehensiveAnalysis -> 
            do! this.AnalyzeProgressAsync(roadmap, allAchievements, findings, recommendations)
            do! this.AnalyzeRisksAsync(roadmap, allAchievements, findings, recommendations)
            do! this.AnalyzePerformanceAsync(roadmap, allAchievements, findings, recommendations)
            do! this.AnalyzeDependenciesAsync(roadmap, allAchievements, findings, recommendations)
        
        // Calculate performance metrics
        let performanceMetrics = RoadmapHelpers.calculateAchievementMetrics allAchievements
        
        // Assess overall risk
        let riskAssessment = this.AssessRisks(findings |> List.ofSeq)
        
        // Calculate confidence based on data quality
        let confidence = this.CalculateAnalysisConfidence(roadmap, allAchievements)
        
        let analysisTime = DateTime.UtcNow - startTime
        
        return {
            RoadmapId = roadmap.Id
            AnalysisType = analysisType
            Findings = findings |> List.ofSeq
            Recommendations = recommendations |> List.ofSeq
            RiskAssessment = riskAssessment
            PerformanceMetrics = performanceMetrics
            AnalyzedAt = DateTime.UtcNow
            AnalysisTime = analysisTime
            Confidence = confidence
        }
    }
    
    /// Analyze progress and identify delays
    member private this.AnalyzeProgressAsync(roadmap: TarsRoadmap, achievements: Achievement list, findings: ResizeArray<AnalysisFinding>, recommendations: ResizeArray<AchievementRecommendation>) = task {
        // Identify overdue achievements
        let overdueAchievements = 
            achievements
            |> List.filter (fun a -> 
                a.DueDate.IsSome && 
                a.DueDate.Value < DateTime.UtcNow && 
                a.Status <> AchievementStatus.Completed)
        
        if not overdueAchievements.IsEmpty then
            let finding = {
                Type = ProgressDelay
                Severity = if overdueAchievements.Length > 5 then High else Medium
                Title = $"{overdueAchievements.Length} overdue achievements detected"
                Description = "Several achievements have passed their due dates without completion"
                AffectedAchievements = overdueAchievements |> List.map (fun a -> a.Id)
                Evidence = overdueAchievements |> List.map (fun a -> $"{a.Title}: due {a.DueDate.Value.ToString(\"yyyy-MM-dd\")}")
                Recommendations = ["Review and update schedules"; "Reassign resources if needed"; "Break down large achievements"]
                EstimatedImpact = float overdueAchievements.Length * 0.1
            }
            findings.Add(finding)
        
        // Identify achievements with low progress
        let stalledAchievements = 
            achievements
            |> List.filter (fun a -> 
                a.Status = AchievementStatus.InProgress && 
                a.CompletionPercentage < 10.0 &&
                a.StartedAt.IsSome &&
                (DateTime.UtcNow - a.StartedAt.Value).TotalDays > 7.0)
        
        if not stalledAchievements.IsEmpty then
            let finding = {
                Type = ProgressDelay
                Severity = Medium
                Title = $"{stalledAchievements.Length} stalled achievements detected"
                Description = "Achievements started but showing minimal progress"
                AffectedAchievements = stalledAchievements |> List.map (fun a -> a.Id)
                Evidence = stalledAchievements |> List.map (fun a -> $"{a.Title}: {a.CompletionPercentage:F1}% in {(DateTime.UtcNow - a.StartedAt.Value).TotalDays:F0} days")
                Recommendations = ["Investigate blockers"; "Provide additional support"; "Consider reassignment"]
                EstimatedImpact = float stalledAchievements.Length * 0.15
            }
            findings.Add(finding)
    }
    
    /// Analyze risks and potential issues
    member private this.AnalyzeRisksAsync(roadmap: TarsRoadmap, achievements: Achievement list, findings: ResizeArray<AnalysisFinding>, recommendations: ResizeArray<AchievementRecommendation>) = task {
        // Identify blocked achievements
        let blockedAchievements = achievements |> List.filter (fun a -> a.Status = AchievementStatus.Blocked)
        
        if not blockedAchievements.IsEmpty then
            let finding = {
                Type = BlockerDetected
                Severity = High
                Title = $"{blockedAchievements.Length} blocked achievements"
                Description = "Achievements are blocked and cannot proceed"
                AffectedAchievements = blockedAchievements |> List.map (fun a -> a.Id)
                Evidence = blockedAchievements |> List.collect (fun a -> a.Blockers)
                Recommendations = ["Address blockers immediately"; "Find alternative approaches"; "Escalate if needed"]
                EstimatedImpact = float blockedAchievements.Length * 0.2
            }
            findings.Add(finding)
        
        // Identify dependency risks
        let achievementsWithManyDeps = 
            achievements |> List.filter (fun a -> a.Dependencies.Length > 3)
        
        if not achievementsWithManyDeps.IsEmpty then
            let finding = {
                Type = DependencyIssue
                Severity = Medium
                Title = $"{achievementsWithManyDeps.Length} achievements with high dependency risk"
                Description = "Achievements with many dependencies are at risk of delays"
                AffectedAchievements = achievementsWithManyDeps |> List.map (fun a -> a.Id)
                Evidence = achievementsWithManyDeps |> List.map (fun a -> $"{a.Title}: {a.Dependencies.Length} dependencies")
                Recommendations = ["Review dependency necessity"; "Consider parallel execution"; "Create fallback plans"]
                EstimatedImpact = float achievementsWithManyDeps.Length * 0.1
            }
            findings.Add(finding)
    }
    
    /// Analyze performance trends
    member private this.AnalyzePerformanceAsync(roadmap: TarsRoadmap, achievements: Achievement list, findings: ResizeArray<AnalysisFinding>, recommendations: ResizeArray<AchievementRecommendation>) = task {
        // Analyze estimation accuracy
        let achievementsWithEstimates = 
            achievements 
            |> List.filter (fun a -> a.ActualHours.IsSome && a.EstimatedHours > 0.0)
        
        if not achievementsWithEstimates.IsEmpty then
            let estimationErrors = 
                achievementsWithEstimates
                |> List.map (fun a -> 
                    let actual = a.ActualHours.Value
                    let estimated = a.EstimatedHours
                    abs(actual - estimated) / max actual estimated)
            
            let averageError = estimationErrors |> List.average
            
            if averageError > 0.3 then // More than 30% error
                let finding = {
                    Type = EstimationError
                    Severity = Medium
                    Title = "Poor estimation accuracy detected"
                    Description = $"Average estimation error is {averageError:P0}"
                    AffectedAchievements = achievementsWithEstimates |> List.map (fun a -> a.Id)
                    Evidence = [$"Average error: {averageError:P0}"; $"Analyzed {achievementsWithEstimates.Length} achievements"]
                    Recommendations = ["Improve estimation techniques"; "Use historical data"; "Break down large tasks"]
                    EstimatedImpact = averageError
                }
                findings.Add(finding)
    }
    
    /// Analyze dependencies and identify issues
    member private this.AnalyzeDependenciesAsync(roadmap: TarsRoadmap, achievements: Achievement list, findings: ResizeArray<AnalysisFinding>, recommendations: ResizeArray<AchievementRecommendation>) = task {
        // Create dependency map
        let achievementMap = achievements |> List.map (fun a -> (a.Id, a)) |> Map.ofList
        
        // Check for circular dependencies (simplified check)
        for achievement in achievements do
            for depId in achievement.Dependencies do
                match achievementMap.TryGetValue(depId) with
                | true, dependency ->
                    if dependency.Dependencies |> List.contains achievement.Id then
                        let finding = {
                            Type = DependencyIssue
                            Severity = High
                            Title = "Circular dependency detected"
                            Description = $"Circular dependency between {achievement.Title} and {dependency.Title}"
                            AffectedAchievements = [achievement.Id; dependency.Id]
                            Evidence = [$"{achievement.Title} depends on {dependency.Title}"; $"{dependency.Title} depends on {achievement.Title}"]
                            Recommendations = ["Remove one dependency"; "Restructure achievements"; "Create intermediate milestone"]
                            EstimatedImpact = 0.5
                        }
                        findings.Add(finding)
                | false, _ ->
                    let finding = {
                        Type = DependencyIssue
                        Severity = Medium
                        Title = "Missing dependency detected"
                        Description = $"Achievement {achievement.Title} depends on non-existent achievement {depId}"
                        AffectedAchievements = [achievement.Id]
                        Evidence = [$"Dependency {depId} not found in roadmap"]
                        Recommendations = ["Remove invalid dependency"; "Create missing achievement"; "Update dependency reference"]
                        EstimatedImpact = 0.2
                    }
                    findings.Add(finding)
    }
    
    /// Analyze resource allocation
    member private this.AnalyzeResourcesAsync(roadmap: TarsRoadmap, achievements: Achievement list, findings: ResizeArray<AnalysisFinding>, recommendations: ResizeArray<AchievementRecommendation>) = task {
        // Analyze agent workload
        let agentWorkloads = 
            achievements
            |> List.filter (fun a -> a.AssignedAgent.IsSome && a.Status = AchievementStatus.InProgress)
            |> List.groupBy (fun a -> a.AssignedAgent.Value)
            |> List.map (fun (agent, tasks) -> (agent, tasks |> List.sumBy (fun t -> t.EstimatedHours)))
        
        let overloadedAgents = agentWorkloads |> List.filter (fun (_, hours) -> hours > 40.0)
        
        if not overloadedAgents.IsEmpty then
            let finding = {
                Type = ResourceBottleneck
                Severity = Medium
                Title = $"{overloadedAgents.Length} overloaded agents detected"
                Description = "Some agents have excessive workloads"
                AffectedAchievements = []
                Evidence = overloadedAgents |> List.map (fun (agent, hours) -> $"{agent}: {hours:F1} hours")
                Recommendations = ["Redistribute workload"; "Add more agents"; "Prioritize critical tasks"]
                EstimatedImpact = float overloadedAgents.Length * 0.15
            }
            findings.Add(finding)
    }
    
    /// Analyze quality metrics
    member private this.AnalyzeQualityAsync(roadmap: TarsRoadmap, achievements: Achievement list, findings: ResizeArray<AnalysisFinding>, recommendations: ResizeArray<AchievementRecommendation>) = task {
        // Check for achievements without proper descriptions
        let poorlyDefinedAchievements = 
            achievements |> List.filter (fun a -> a.Description.Length < 20)
        
        if not poorlyDefinedAchievements.IsEmpty then
            let finding = {
                Type = QualityRisk
                Severity = Low
                Title = $"{poorlyDefinedAchievements.Length} poorly defined achievements"
                Description = "Some achievements lack detailed descriptions"
                AffectedAchievements = poorlyDefinedAchievements |> List.map (fun a -> a.Id)
                Evidence = poorlyDefinedAchievements |> List.map (fun a -> $"{a.Title}: {a.Description.Length} characters")
                Recommendations = ["Add detailed descriptions"; "Define acceptance criteria"; "Include examples"]
                EstimatedImpact = float poorlyDefinedAchievements.Length * 0.05
            }
            findings.Add(finding)
    }
    
    /// Assess overall risks
    member private this.AssessRisks(findings: AnalysisFinding list) =
        let criticalFindings = findings |> List.filter (fun f -> f.Severity = Critical)
        let highFindings = findings |> List.filter (fun f -> f.Severity = High)
        let mediumFindings = findings |> List.filter (fun f -> f.Severity = Medium)
        
        let overallRisk = 
            if criticalFindings.Length > 0 then VeryHigh
            elif highFindings.Length > 2 then High
            elif highFindings.Length > 0 || mediumFindings.Length > 3 then Medium
            elif mediumFindings.Length > 0 then Low
            else VeryLow
        
        let riskFactors = 
            findings
            |> List.map (fun finding -> {
                Type = match finding.Type with
                       | ProgressDelay -> ScheduleRisk
                       | ResourceBottleneck -> ResourceRisk
                       | DependencyIssue -> DependencyRisk
                       | QualityRisk -> QualityRisk
                       | BlockerDetected -> TechnicalRisk
                       | _ -> TechnicalRisk
                Description = finding.Description
                Probability = 0.7 // Would be calculated based on historical data
                Impact = finding.EstimatedImpact
                RiskScore = 0.7 * finding.EstimatedImpact
                AffectedAchievements = finding.AffectedAchievements
            })
        
        {
            OverallRisk = overallRisk
            RiskFactors = riskFactors
            MitigationStrategies = [
                "Regular progress monitoring"
                "Proactive blocker resolution"
                "Resource reallocation as needed"
                "Dependency management"
            ]
            ContingencyPlans = [
                "Scope reduction if needed"
                "Timeline extension options"
                "Alternative implementation approaches"
                "External resource acquisition"
            ]
            MonitoringRecommendations = [
                "Daily progress checks for critical items"
                "Weekly risk assessment reviews"
                "Monthly roadmap health reports"
                "Quarterly strategic reviews"
            ]
        }
    
    /// Calculate analysis confidence
    member private this.CalculateAnalysisConfidence(roadmap: TarsRoadmap, achievements: Achievement list) =
        let factors = [
            // Data completeness
            if achievements |> List.forall (fun a -> not (String.IsNullOrWhiteSpace(a.Description))) then 0.2 else 0.1
            
            // Time estimates available
            if achievements |> List.forall (fun a -> a.EstimatedHours > 0.0) then 0.2 else 0.1
            
            // Historical data available
            if achievements |> List.exists (fun a -> a.ActualHours.IsSome) then 0.2 else 0.1
            
            // Recent updates
            if achievements |> List.exists (fun a -> (DateTime.UtcNow - a.UpdatedAt).TotalDays < 7.0) then 0.2 else 0.1
            
            // Assignment clarity
            if achievements |> List.filter (fun a -> a.Status = AchievementStatus.InProgress) |> List.forall (fun a -> a.AssignedAgent.IsSome) then 0.2 else 0.1
        ]
        
        factors |> List.sum
    
    /// Apply low-risk recommendations automatically
    member private this.ApplyLowRiskRecommendationsAsync(roadmap: TarsRoadmap, recommendations: AchievementRecommendation list) = task {
        let lowRiskRecommendations = 
            recommendations |> List.filter (fun r -> r.EstimatedEffort < 1.0 && r.EstimatedImpact > 0.1)
        
        for recommendation in lowRiskRecommendations do
            try
                match recommendation.Type with
                | UpdateExisting when recommendation.Achievement.IsSome ->
                    // Auto-update low-risk changes
                    logger.LogInformation($"Auto-applying recommendation: {recommendation.Type}")
                    // Implementation would update the achievement
                | _ ->
                    // Log other recommendations for manual review
                    logger.LogInformation($"Recommendation requires manual review: {recommendation.Type}")
            with
            | ex ->
                logger.LogWarning(ex, $"Failed to apply recommendation: {recommendation.Type}")
    }
    
    /// Analysis loop for continuous monitoring
    member private this.AnalysisLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug($"Starting analysis loop for agent: {agentId}")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Analyze all roadmaps
                    let! _ = this.AnalyzeAllRoadmapsAsync()
                    
                    // Wait for next analysis cycle
                    do! Task.Delay(analysisInterval, cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    () // Exit the loop
                | ex ->
                    logger.LogWarning(ex, $"Error in analysis loop for agent: {agentId}")
                    do! Task.Delay(TimeSpan.FromMinutes(5.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug($"Analysis loop cancelled for agent: {agentId}")
        | ex ->
            logger.LogError(ex, $"Analysis loop failed for agent: {agentId}")
    }
