namespace TarsEngine.FSharp.Agents

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Deployment.VMDeploymentManager
open TarsEngine.FSharp.Testing.VMTestRunner
open TarsEngine.FSharp.Agents.MLEnhancedQAAgent

/// <summary>
/// TARS Autonomous QA Agent
/// Automatically deploys and tests projects on VMs without human intervention
/// </summary>
module QAAgent =
    
    /// QA Agent configuration
    type QAAgentConfig = {
        AutoDeployEnabled: bool
        PreferredVMProviders: VMProvider list
        TestSuiteConfig: TestConfiguration
        MaxConcurrentDeployments: int
        AutoShutdownAfterTests: bool
        NotificationEnabled: bool
        ReportFormat: string
        ContinuousIntegration: bool
        ScheduledTesting: TimeSpan option
    }
    
    /// QA Task
    type QATask = {
        TaskId: string
        ProjectPath: string
        ProjectName: string
        Priority: int // 1 = highest, 5 = lowest
        RequestedBy: string
        CreatedAt: DateTime
        TargetEnvironment: string // "development", "staging", "production"
        CustomTestConfig: TestConfiguration option
        RequiredVMProvider: VMProvider option
        Deadline: DateTime option
    }
    
    /// QA Result
    type QAResult = {
        TaskId: string
        ProjectName: string
        Success: bool
        VMInstanceId: string option
        DeploymentResult: DeploymentResult option
        TestSuiteResult: TestSuiteResult option
        StartTime: DateTime
        EndTime: DateTime
        TotalDuration: TimeSpan
        QualityScore: float
        Recommendations: string list
        IssuesFound: int
        CriticalIssues: int
        ReportPath: string option
        NextSteps: string list
    }
    
    /// <summary>
    /// Autonomous QA Agent with ML-Enhanced Quality Prediction
    /// Handles end-to-end automated testing workflows with intelligent quality assessment
    /// </summary>
    type AutonomousQAAgent(
        vmDeploymentManager: VMDeploymentManager,
        vmTestRunner: VMTestRunner,
        logger: ILogger<AutonomousQAAgent>
    ) =

        let mutable taskQueue = []
        let mutable activeDeployments = Map.empty<string, QATask * VMInstance>
        let mutable completedTasks = []
        let mutable agentMetrics = {|
            TasksProcessed = 0
            SuccessfulDeployments = 0
            FailedDeployments = 0
            TotalTestsRun = 0
            AverageQualityScore = 0.0
            TotalVMHoursUsed = 0.0
        |}

        // ML-Enhanced QA capabilities
        let mlQAAgent = MLEnhancedQAAgent(logger)
        let mutable isMLModelsTrained = false
        
        /// <summary>
        /// Start autonomous QA operations with ML enhancement
        /// </summary>
        member this.StartAutonomousOperations(config: QAAgentConfig) : Task<unit> =
            task {
                logger.LogInformation("🚀 Starting ML-enhanced autonomous QA agent operations")

                // Initialize ML models
                if not isMLModelsTrained then
                    logger.LogInformation("🧠 Training ML models for quality prediction...")
                    let! trainingResult = mlQAAgent.TrainWithSyntheticData()
                    isMLModelsTrained <- true
                    logger.LogInformation("✅ ML models trained: SVM={SVMAccuracy:P1}, RF={ForestAccuracy:P1}",
                                        trainingResult.SVMAccuracy, trainingResult.ForestAccuracy)

                // Start background task processing
                let! _ = Task.Run(fun () -> this.ProcessTaskQueue(config))

                // Start continuous monitoring
                let! _ = Task.Run(fun () -> this.MonitorActiveDeployments(config))

                // Start scheduled testing if enabled
                if config.ScheduledTesting.IsSome then
                    let! _ = Task.Run(fun () -> this.RunScheduledTesting(config))
                    ()

                logger.LogInformation("🎯 ML-enhanced autonomous QA agent is now operational")
            }
        
        /// <summary>
        /// Submit QA task for autonomous processing
        /// </summary>
        member this.SubmitQATask(projectPath: string, priority: int, requestedBy: string, ?targetEnvironment: string, ?deadline: DateTime, ?vmProvider: VMProvider) : string =
            let taskId = "qa-" + Guid.NewGuid().ToString("N")[..7]
            let projectName = Path.GetFileName(projectPath)
            
            let qaTask = {
                TaskId = taskId
                ProjectPath = projectPath
                ProjectName = projectName
                Priority = priority
                RequestedBy = requestedBy
                CreatedAt = DateTime.UtcNow
                TargetEnvironment = defaultArg targetEnvironment "testing"
                CustomTestConfig = None
                RequiredVMProvider = vmProvider
                Deadline = deadline
            }
            
            // Insert task in priority order
            taskQueue <- taskQueue @ [qaTask] |> List.sortBy (fun t -> t.Priority, t.CreatedAt)
            
            logger.LogInformation("QA task submitted: {TaskId} for project {ProjectName} (Priority: {Priority})", taskId, projectName, priority)
            taskId
        
        /// <summary>
        /// Process task queue autonomously
        /// </summary>
        member private this.ProcessTaskQueue(config: QAAgentConfig) : Task<unit> =
            task {
                while true do
                    try
                        if taskQueue.Length > 0 && activeDeployments.Count < config.MaxConcurrentDeployments then
                            let nextTask = taskQueue.Head
                            taskQueue <- taskQueue.Tail
                            
                            logger.LogInformation("Processing QA task: {TaskId}", nextTask.TaskId)
                            
                            // Process task autonomously
                            let! result = this.ProcessQATaskAutonomously(nextTask, config)
                            completedTasks <- result :: completedTasks
                            
                            // Update metrics
                            agentMetrics <- {|
                                agentMetrics with
                                    TasksProcessed = agentMetrics.TasksProcessed + 1
                                    SuccessfulDeployments = agentMetrics.SuccessfulDeployments + (if result.Success then 1 else 0)
                                    FailedDeployments = agentMetrics.FailedDeployments + (if not result.Success then 1 else 0)
                            |}
                        
                        // Wait before checking queue again
                        do! Task.Delay(TimeSpan.FromSeconds(10.0))
                        
                    with
                    | ex -> 
                        logger.LogError(ex, "Error processing task queue")
                        do! Task.Delay(TimeSpan.FromSeconds(30.0))
            }
        
        /// <summary>
        /// Process QA task completely autonomously
        /// </summary>
        member private this.ProcessQATaskAutonomously(task: QATask, config: QAAgentConfig) : Task<QAResult> =
            task {
                let startTime = DateTime.UtcNow
                logger.LogInformation("Starting autonomous QA processing for task: {TaskId}", task.TaskId)
                
                try
                    // 1. Analyze project and determine optimal VM configuration
                    let vmConfig = this.AnalyzeProjectAndSelectVM(task, config)
                    logger.LogInformation("Selected VM provider: {Provider} for task: {TaskId}", vmConfig.Provider, task.TaskId)
                    
                    // 2. Create deployment specification
                    let deploymentSpec = {
                        ProjectPath = task.ProjectPath
                        ProjectName = task.ProjectName
                        VMConfig = vmConfig
                        DeploymentType = task.TargetEnvironment
                        EnvironmentVariables = Map.empty
                        DatabaseRequired = this.ProjectRequiresDatabase(task.ProjectPath)
                        ExternalServices = this.DetectExternalServices(task.ProjectPath)
                        TestSuites = ["unit"; "integration"; "api"; "performance"]
                        MonitoringEnabled = true
                    }
                    
                    // 3. Deploy to VM autonomously
                    logger.LogInformation("Deploying project autonomously: {TaskId}", task.TaskId)
                    let! deploymentResult = vmDeploymentManager.DeployToVM(deploymentSpec)
                    
                    if not deploymentResult.Success then
                        logger.LogWarning("Deployment failed for task: {TaskId}", task.TaskId)
                        let endTime = DateTime.UtcNow
                        return {
                            TaskId = task.TaskId
                            ProjectName = task.ProjectName
                            Success = false
                            VMInstanceId = None
                            DeploymentResult = Some deploymentResult
                            TestSuiteResult = None
                            StartTime = startTime
                            EndTime = endTime
                            TotalDuration = endTime - startTime
                            QualityScore = 0.0
                            Recommendations = ["Fix deployment issues before retesting"]
                            IssuesFound = deploymentResult.ErrorMessages.Length
                            CriticalIssues = deploymentResult.ErrorMessages.Length
                            ReportPath = None
                            NextSteps = ["Review deployment logs"; "Fix configuration issues"]
                        }
                    
                    // 4. Run comprehensive test suite autonomously
                    logger.LogInformation("Running comprehensive test suite autonomously: {TaskId}", task.TaskId)
                    let testConfig = task.CustomTestConfig |> Option.defaultValue config.TestSuiteConfig
                    let! testSuiteResult = vmTestRunner.RunTestSuite(deploymentResult.VMInstanceId.Value, task.ProjectPath, testConfig)
                    
                    // 5. Analyze results and generate quality score
                    let qualityScore = this.CalculateQualityScore(deploymentResult, testSuiteResult)
                    let recommendations = this.GenerateAutonomousRecommendations(deploymentResult, testSuiteResult)
                    let issuesFound = this.CountIssues(testSuiteResult)
                    let criticalIssues = this.CountCriticalIssues(testSuiteResult)
                    
                    // 6. Generate comprehensive report
                    let! reportPath = this.GenerateQAReport(task, deploymentResult, testSuiteResult, qualityScore)
                    
                    // 7. Cleanup VM if configured
                    if config.AutoShutdownAfterTests then
                        logger.LogInformation("Auto-shutting down VM for task: {TaskId}", task.TaskId)
                        let! shutdownResult = vmDeploymentManager.ShutdownVM(deploymentResult.VMInstanceId.Value, true)
                        logger.LogInformation("VM shutdown result: {Result} for task: {TaskId}", shutdownResult, task.TaskId)
                    
                    let endTime = DateTime.UtcNow
                    let totalDuration = endTime - startTime
                    
                    logger.LogInformation("QA task completed autonomously: {TaskId}, Quality Score: {Score}", task.TaskId, qualityScore)
                    
                    return {
                        TaskId = task.TaskId
                        ProjectName = task.ProjectName
                        Success = testSuiteResult.OverallSuccess
                        VMInstanceId = deploymentResult.VMInstanceId
                        DeploymentResult = Some deploymentResult
                        TestSuiteResult = Some testSuiteResult
                        StartTime = startTime
                        EndTime = endTime
                        TotalDuration = totalDuration
                        QualityScore = qualityScore
                        Recommendations = recommendations
                        IssuesFound = issuesFound
                        CriticalIssues = criticalIssues
                        ReportPath = Some reportPath
                        NextSteps = this.GenerateNextSteps(testSuiteResult, qualityScore)
                    }
                    
                with
                | ex ->
                    logger.LogError(ex, "Autonomous QA processing failed for task: {TaskId}", task.TaskId)
                    let endTime = DateTime.UtcNow
                    return {
                        TaskId = task.TaskId
                        ProjectName = task.ProjectName
                        Success = false
                        VMInstanceId = None
                        DeploymentResult = None
                        TestSuiteResult = None
                        StartTime = startTime
                        EndTime = endTime
                        TotalDuration = endTime - startTime
                        QualityScore = 0.0
                        Recommendations = ["Fix critical errors and retry"]
                        IssuesFound = 1
                        CriticalIssues = 1
                        ReportPath = None
                        NextSteps = ["Review error logs"; "Fix blocking issues"]
                    }
            }
        
        /// <summary>
        /// Analyze project and select optimal VM
        /// </summary>
        member private this.AnalyzeProjectAndSelectVM(task: QATask, config: QAAgentConfig) : VMConfiguration =
            // Check if specific VM provider is required
            match task.RequiredVMProvider with
            | Some provider ->
                vmDeploymentManager.GetRecommendedVMConfig("moderate", this.ProjectRequiresDatabase(task.ProjectPath))
                |> fun config -> { config with Provider = provider }
            | None ->
                // Analyze project complexity
                let complexity = this.AnalyzeProjectComplexity(task.ProjectPath)
                let requiresDatabase = this.ProjectRequiresDatabase(task.ProjectPath)
                
                // Select best provider from preferred list
                let selectedProvider = 
                    config.PreferredVMProviders
                    |> List.tryFind (fun provider ->
                        match provider, complexity with
                        | GitHubCodespaces, "simple" -> true
                        | GitPod, "simple" | GitPod, "moderate" -> true
                        | AWSFreeT2Micro, "moderate" | AWSFreeT2Micro, "complex" -> true
                        | OracleCloudFree, "complex" | OracleCloudFree, "enterprise" -> true
                        | _ -> false
                    )
                    |> Option.defaultValue GitHubCodespaces
                
                vmDeploymentManager.GetRecommendedVMConfig(complexity, requiresDatabase)
                |> fun config -> { config with Provider = selectedProvider }
        
        /// <summary>
        /// Analyze project complexity autonomously with ML-enhanced quality prediction
        /// </summary>
        member private this.AnalyzeProjectComplexity(projectPath: string) : string =
            let hasDockerfile = File.Exists(Path.Combine(projectPath, "Dockerfile"))
            let hasDatabase = Directory.Exists(Path.Combine(projectPath, "database"))
            let hasTests = Directory.Exists(Path.Combine(projectPath, "tests"))
            let hasK8s = Directory.Exists(Path.Combine(projectPath, "k8s"))
            let hasCICD = File.Exists(Path.Combine(projectPath, ".github", "workflows", "ci-cd.yml"))

            let complexityScore =
                (if hasDockerfile then 1 else 0) +
                (if hasDatabase then 2 else 0) +
                (if hasTests then 1 else 0) +
                (if hasK8s then 2 else 0) +
                (if hasCICD then 1 else 0)

            // Enhanced analysis with ML quality prediction
            if isMLModelsTrained then
                try
                    // Analyze project files for quality prediction
                    let codeFiles = Directory.GetFiles(projectPath, "*.fs", SearchOption.AllDirectories)
                    if codeFiles.Length > 0 then
                        let sampleFile = codeFiles.[0]
                        let analysisTask = mlQAAgent.AnalyzeCodeFile(sampleFile)
                        let analysis = analysisTask.Result

                        logger.LogInformation("🔍 ML Quality Analysis: {Score:F3} ({RiskLevel} Risk) for {ProjectPath}",
                                            analysis.Prediction.OverallQualityScore,
                                            analysis.Prediction.RiskLevel,
                                            projectPath)

                        // Adjust complexity based on ML prediction
                        let mlComplexityAdjustment =
                            match analysis.Prediction.RiskLevel with
                            | "Critical" -> 2
                            | "High" -> 1
                            | "Medium" -> 0
                            | "Low" -> -1
                            | _ -> 0

                        let adjustedScore = complexityScore + mlComplexityAdjustment

                        match adjustedScore with
                        | score when score >= 6 -> "enterprise"
                        | score when score >= 4 -> "complex"
                        | score when score >= 2 -> "moderate"
                        | _ -> "simple"
                    else
                        // Fallback to original logic
                        match complexityScore with
                        | score when score >= 6 -> "enterprise"
                        | score when score >= 4 -> "complex"
                        | score when score >= 2 -> "moderate"
                        | _ -> "simple"
                with
                | ex ->
                    logger.LogWarning(ex, "ML analysis failed, using fallback complexity analysis")
                    match complexityScore with
                    | score when score >= 6 -> "enterprise"
                    | score when score >= 4 -> "complex"
                    | score when score >= 2 -> "moderate"
                    | _ -> "simple"
            else
                match complexityScore with
                | score when score >= 6 -> "enterprise"
                | score when score >= 4 -> "complex"
                | score when score >= 2 -> "moderate"
                | _ -> "simple"
        
        /// <summary>
        /// Check if project requires database
        /// </summary>
        member private this.ProjectRequiresDatabase(projectPath: string) : bool =
            Directory.Exists(Path.Combine(projectPath, "database")) ||
            File.Exists(Path.Combine(projectPath, "database-schema.sql")) ||
            File.Exists(Path.Combine(projectPath, "appsettings.json")) // Likely has DB connection string
        
        /// <summary>
        /// Detect external services
        /// </summary>
        member private this.DetectExternalServices(projectPath: string) : string list =
            let services = ResizeArray<string>()
            
            // Check for common external service indicators
            if Directory.Exists(Path.Combine(projectPath, "database")) then
                services.Add("PostgreSQL")
            
            // Check for Redis, message queues, etc.
            // This would analyze project files for dependencies
            
            services |> Seq.toList
        
        /// <summary>
        /// Calculate quality score with ML enhancement
        /// </summary>
        member private this.CalculateQualityScore(deploymentResult: DeploymentResult, testResult: TestSuiteResult) : float =
            let deploymentScore = if deploymentResult.Success then 30.0 else 0.0
            let testScore = if testResult.OverallSuccess then 40.0 else 0.0
            let coverageScore =
                testResult.TestResults
                |> List.choose (fun r -> r.Coverage)
                |> List.tryHead
                |> Option.map (fun c -> c * 0.2)
                |> Option.defaultValue 0.0
            let securityScore =
                testResult.SecurityScan
                |> Option.map (fun s -> s.SecurityScore * 0.1)
                |> Option.defaultValue 0.0

            let baseScore = deploymentScore + testScore + coverageScore + securityScore

            // Enhanced scoring with ML prediction if available
            if isMLModelsTrained then
                try
                    // Create synthetic metrics for ML analysis
                    let mlMetrics = {
                        CyclomaticComplexity = 15.0 // Would extract from actual analysis
                        LinesOfCode = 1000
                        TestCoverage = coverageScore / 20.0 // Convert back to percentage
                        CodeDuplication = 0.05
                        TechnicalDebt = 25.0
                        BugDensity = if testResult.OverallSuccess then 0.01 else 0.05
                        MaintainabilityIndex = baseScore
                        SecurityVulnerabilities =
                            testResult.SecurityScan
                            |> Option.map (fun s -> s.CriticalIssues)
                            |> Option.defaultValue 0
                        PerformanceScore = if deploymentResult.Success then 0.8 else 0.3
                        DocumentationCoverage = 0.6
                    }

                    let predictionTask = mlQAAgent.PredictQualityIssues(mlMetrics)
                    let prediction = predictionTask.Result

                    // Combine traditional and ML scores
                    let mlScore = prediction.OverallQualityScore * 100.0
                    let combinedScore = (baseScore * 0.7) + (mlScore * 0.3)

                    logger.LogInformation("🎯 Quality Score: Traditional={Traditional:F1}, ML={MLScore:F1}, Combined={Combined:F1}",
                                        baseScore, mlScore, combinedScore)

                    combinedScore
                with
                | ex ->
                    logger.LogWarning(ex, "ML quality scoring failed, using traditional scoring")
                    baseScore
            else
                baseScore
        
        /// <summary>
        /// Generate autonomous recommendations
        /// </summary>
        member private this.GenerateAutonomousRecommendations(deploymentResult: DeploymentResult, testResult: TestSuiteResult) : string list =
            let recommendations = ResizeArray<string>()
            
            // Deployment recommendations
            if not deploymentResult.Success then
                recommendations.Add("Fix deployment configuration issues")
            
            // Test recommendations
            if not testResult.OverallSuccess then
                recommendations.Add("Address failing tests before production deployment")
            
            // Coverage recommendations
            testResult.TestResults
            |> List.choose (fun r -> r.Coverage)
            |> List.tryHead
            |> Option.iter (fun coverage ->
                if coverage < 80.0 then
                    recommendations.Add($"Increase test coverage from {coverage:F1}% to at least 80%")
            )
            
            // Security recommendations
            testResult.SecurityScan
            |> Option.iter (fun scan ->
                if scan.CriticalIssues > 0 then
                    recommendations.Add($"Fix {scan.CriticalIssues} critical security issues")
                recommendations.AddRange(scan.Recommendations)
            )
            
            recommendations |> Seq.toList
        
        /// <summary>
        /// Get QA agent status
        /// </summary>
        member this.GetAgentStatus() : obj =
            {|
                IsOperational = true
                TasksInQueue = taskQueue.Length
                ActiveDeployments = activeDeployments.Count
                CompletedTasks = completedTasks.Length
                Metrics = agentMetrics
                LastActivity = DateTime.UtcNow
            |}
