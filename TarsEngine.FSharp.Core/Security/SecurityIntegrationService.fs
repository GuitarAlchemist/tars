namespace TarsEngine.FSharp.Core.Security

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Tracing

/// Security Integration Service that connects Bengio's AI safety framework with TARS infrastructure
/// Provides centralized security orchestration across all TARS components

/// Security check result for TARS operations
type SecurityCheckResult = {
    OperationId: string
    OperationType: string
    SecurityPassed: bool
    EthicsPassed: bool
    OverallSafety: bool
    SecurityAssessment: SafetyAssessment option
    AlignmentAssessment: AlignmentAssessment option
    RequiredOversight: OversightLevel
    BlockingReasons: string list
    Recommendations: string list
    Timestamp: DateTime
}

/// TARS operation types that require security checks
type TarsOperationType =
    | AgentCreation
    | AgentExecution
    | ClosureGeneration
    | ClosureExecution
    | MetascriptExecution
    | VectorStoreOperation
    | TransformOperation
    | HyperlightVMExecution
    | WasmModuleExecution
    | InterAgentCommunication
    | HumanInteraction
    | SystemConfiguration
    | DataAccess
    | ExternalAPICall

/// Security policy configuration
type SecurityPolicy = {
    RequireSecurityCheck: bool
    RequireEthicsCheck: bool
    RequireHumanOversight: bool
    BayesianThreshold: float
    AlignmentThreshold: float
    BlockDangerousBehaviors: bool
    LogAllOperations: bool
    EnableRealTimeMonitoring: bool
}

/// Security Integration Service
type SecurityIntegrationService(
    securityService: AISecurityService,
    ethicsService: AIEthicsService,
    logger: ILogger<SecurityIntegrationService>) =
    
    let securityChecks = ConcurrentDictionary<string, SecurityCheckResult>()
    let operationPolicies = ConcurrentDictionary<TarsOperationType, SecurityPolicy>()
    let mutable globalSecurityEnabled = true
    let mutable realTimeMonitoringEnabled = true
    
    /// Initialize Security Integration Service
    member this.InitializeAsync() = task {
        try
            logger.LogInformation("Initializing Security Integration Service...")
            
            // Initialize default security policies for all TARS operations
            do! this.InitializeDefaultPoliciesAsync()
            
            // Initialize security and ethics services
            do! securityService.InitializeAsync()
            do! ethicsService.InitializeAsync()
            
            logger.LogInformation("Security Integration Service initialized with comprehensive AI safety framework")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to initialize Security Integration Service")
            raise ex
    }
    
    /// Initialize default security policies
    member private this.InitializeDefaultPoliciesAsync() = task {
        let defaultPolicy = {
            RequireSecurityCheck = true
            RequireEthicsCheck = true
            RequireHumanOversight = false
            BayesianThreshold = 0.1 // 10% harm probability threshold
            AlignmentThreshold = 0.7 // 70% alignment score threshold
            BlockDangerousBehaviors = true
            LogAllOperations = true
            EnableRealTimeMonitoring = true
        }
        
        let highRiskPolicy = {
            defaultPolicy with
                RequireHumanOversight = true
                BayesianThreshold = 0.05 // 5% threshold for high-risk operations
                AlignmentThreshold = 0.8 // 80% alignment threshold
        }
        
        let lowRiskPolicy = {
            defaultPolicy with
                RequireHumanOversight = false
                BayesianThreshold = 0.2 // 20% threshold for low-risk operations
                AlignmentThreshold = 0.6 // 60% alignment threshold
        }
        
        // Assign policies based on operation risk level
        operationPolicies.[AgentCreation] <- highRiskPolicy
        operationPolicies.[AgentExecution] <- defaultPolicy
        operationPolicies.[ClosureGeneration] <- highRiskPolicy
        operationPolicies.[ClosureExecution] <- defaultPolicy
        operationPolicies.[MetascriptExecution] <- defaultPolicy
        operationPolicies.[VectorStoreOperation] <- lowRiskPolicy
        operationPolicies.[TransformOperation] <- lowRiskPolicy
        operationPolicies.[HyperlightVMExecution] <- defaultPolicy
        operationPolicies.[WasmModuleExecution] <- defaultPolicy
        operationPolicies.[InterAgentCommunication] <- defaultPolicy
        operationPolicies.[HumanInteraction] <- highRiskPolicy
        operationPolicies.[SystemConfiguration] <- highRiskPolicy
        operationPolicies.[DataAccess] <- defaultPolicy
        operationPolicies.[ExternalAPICall] <- highRiskPolicy
        
        logger.LogInformation($"Initialized security policies for {operationPolicies.Count} operation types")
    }
    
    /// Perform comprehensive security check for TARS operation
    member this.CheckOperationSecurityAsync(operationType: TarsOperationType, operationDescription: string, context: Map<string, obj>) = task {
        try
            if not globalSecurityEnabled then
                return Ok {
                    OperationId = Guid.NewGuid().ToString("N")[..7]
                    OperationType = operationType.ToString()
                    SecurityPassed = true
                    EthicsPassed = true
                    OverallSafety = true
                    SecurityAssessment = None
                    AlignmentAssessment = None
                    RequiredOversight = NoOversight
                    BlockingReasons = []
                    Recommendations = []
                    Timestamp = DateTime.UtcNow
                }
            
            let operationId = Guid.NewGuid().ToString("N")[..7]
            logger.LogDebug($"Security check for {operationType}: {operationDescription}")
            
            // Get security policy for this operation type
            let policy = operationPolicies.GetValueOrDefault(operationType, {
                RequireSecurityCheck = true
                RequireEthicsCheck = true
                RequireHumanOversight = false
                BayesianThreshold = 0.1
                AlignmentThreshold = 0.7
                BlockDangerousBehaviors = true
                LogAllOperations = true
                EnableRealTimeMonitoring = true
            })
            
            // Perform security assessment
            let! securityResult = 
                if policy.RequireSecurityCheck then
                    securityService.AssessActionSafetyAsync(operationDescription, context)
                else
                    Task.FromResult(Ok {
                        ActionDescription = operationDescription
                        HarmProbability = 0.0
                        BehaviorType = Honest
                        ReasoningChain = ["Security check bypassed by policy"]
                        Confidence = 1.0
                        SafetyVerdict = Safe
                        Timestamp = DateTime.UtcNow
                        AssessmentId = operationId
                    })
            
            // Perform ethics assessment
            let! ethicsResult = 
                if policy.RequireEthicsCheck then
                    ethicsService.AssessAlignmentAsync(operationDescription, context)
                else
                    Task.FromResult(Ok {
                        ActionDescription = operationDescription
                        HumanValues = []
                        AlignmentScore = 1.0
                        ValueImpacts = Map.empty
                        ReasoningChain = ["Ethics check bypassed by policy"]
                        Recommendations = []
                        AlignmentVerdict = FullyAligned
                        Timestamp = DateTime.UtcNow
                        AssessmentId = operationId
                    })
            
            // Combine results
            let securityCheckResult = 
                match securityResult, ethicsResult with
                | Ok secAssessment, Ok ethAssessment ->
                    let securityPassed = 
                        match secAssessment.SafetyVerdict with
                        | Safe -> true
                        | _ -> false
                    
                    let ethicsPassed = 
                        match ethAssessment.AlignmentVerdict with
                        | FullyAligned | MostlyAligned _ -> ethAssessment.AlignmentScore >= policy.AlignmentThreshold
                        | _ -> false
                    
                    let blockingReasons = ResizeArray<string>()
                    
                    // Check for blocking conditions
                    if policy.BlockDangerousBehaviors then
                        match secAssessment.BehaviorType with
                        | Deception -> blockingReasons.Add("Deceptive behavior detected")
                        | SelfPreservation -> blockingReasons.Add("Self-preservation behavior detected")
                        | Cheating -> blockingReasons.Add("Cheating behavior detected")
                        | Manipulation -> blockingReasons.Add("Manipulation behavior detected")
                        | _ -> ()
                    
                    if secAssessment.HarmProbability > policy.BayesianThreshold then
                        blockingReasons.Add($"Harm probability ({secAssessment.HarmProbability:P1}) exceeds threshold ({policy.BayesianThreshold:P1})")
                    
                    if ethAssessment.AlignmentScore < policy.AlignmentThreshold then
                        blockingReasons.Add($"Alignment score ({ethAssessment.AlignmentScore:F2}) below threshold ({policy.AlignmentThreshold:F2})")
                    
                    // Determine oversight level
                    let! oversightLevel = ethicsService.DetermineOversightLevelAsync(ethAssessment)
                    let finalOversightLevel = 
                        if policy.RequireHumanOversight then
                            match oversightLevel with
                            | NoOversight -> RequestApproval
                            | other -> other
                        else oversightLevel
                    
                    // Combine recommendations
                    let allRecommendations = 
                        List.concat [
                            if secAssessment.ReasoningChain.Length < 5 then ["Provide more detailed reasoning for this operation"]
                            ethAssessment.Recommendations
                            if blockingReasons.Count > 0 then ["Consider alternative approaches that address safety concerns"]
                        ]
                    
                    let overallSafety = securityPassed && ethicsPassed && blockingReasons.Count = 0
                    
                    Ok {
                        OperationId = operationId
                        OperationType = operationType.ToString()
                        SecurityPassed = securityPassed
                        EthicsPassed = ethicsPassed
                        OverallSafety = overallSafety
                        SecurityAssessment = Some secAssessment
                        AlignmentAssessment = Some ethAssessment
                        RequiredOversight = finalOversightLevel
                        BlockingReasons = blockingReasons.ToArray() |> Array.toList
                        Recommendations = allRecommendations
                        Timestamp = DateTime.UtcNow
                    }
                
                | Error secError, Ok _ ->
                    Error $"Security assessment failed: {secError}"
                
                | Ok _, Error ethError ->
                    Error $"Ethics assessment failed: {ethError}"
                
                | Error secError, Error ethError ->
                    Error $"Both assessments failed - Security: {secError}, Ethics: {ethError}"
            
            match securityCheckResult with
            | Ok result ->
                // Store security check result
                securityChecks.[operationId] <- result
                
                // Log result
                if policy.LogAllOperations then
                    let logLevel = if result.OverallSafety then LogLevel.Information else LogLevel.Warning
                    logger.Log(logLevel, $"Security check {operationId}: {operationType} - Safety: {result.OverallSafety}")
                
                return Ok result
            
            | Error error ->
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Security check failed for {operationType}: {operationDescription}")
            return Error ex.Message
    }
    
    /// Quick security check for low-risk operations
    member this.QuickSecurityCheckAsync(operationType: TarsOperationType, operationDescription: string) = task {
        let context = Map.ofList [("quickCheck", box true)]
        let! result = this.CheckOperationSecurityAsync(operationType, operationDescription, context)
        
        match result with
        | Ok checkResult -> return Ok checkResult.OverallSafety
        | Error error -> return Error error
    }
    
    /// Check if operation requires human oversight
    member this.RequiresHumanOversightAsync(operationType: TarsOperationType, operationDescription: string) = task {
        let! result = this.CheckOperationSecurityAsync(operationType, operationDescription, Map.empty)
        
        match result with
        | Ok checkResult ->
            let requiresOversight = 
                match checkResult.RequiredOversight with
                | NoOversight | InformHuman -> false
                | RequestApproval | RequireCollaboration | HumanOnly -> true
            
            return Ok (requiresOversight, checkResult.RequiredOversight)
        
        | Error error ->
            return Error error
    }
    
    /// Integrate with metascript tracing
    member this.AddSecurityTraceAsync(traceId: string, securityCheckResult: SecurityCheckResult) = task {
        try
            // This would integrate with the MetascriptTraceService to add security information
            logger.LogDebug($"Adding security trace for operation {securityCheckResult.OperationId} to trace {traceId}")
            
            // In a real implementation, this would call the MetascriptTraceService
            // to add security assessment information to the trace
            
            return Ok ()
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to add security trace: {traceId}")
            return Error ex.Message
    }
    
    /// Get security statistics for monitoring
    member this.GetSecurityStatisticsAsync() = task {
        let totalChecks = securityChecks.Count
        let safeOperations = securityChecks.Values |> Seq.filter (fun c -> c.OverallSafety) |> Seq.length
        let blockedOperations = securityChecks.Values |> Seq.filter (fun c -> c.BlockingReasons.Length > 0) |> Seq.length
        let oversightRequired = securityChecks.Values |> Seq.filter (fun c -> c.RequiredOversight <> NoOversight) |> Seq.length
        
        let operationTypeStats = 
            securityChecks.Values
            |> Seq.groupBy (fun c -> c.OperationType)
            |> Seq.map (fun (opType, checks) -> 
                let total = Seq.length checks
                let safe = checks |> Seq.filter (fun c -> c.OverallSafety) |> Seq.length
                (opType, total, safe, float safe / float total))
            |> Seq.toList
        
        let! securityStats = securityService.GetSafetyStatisticsAsync()
        let! ethicsStats = ethicsService.GetEthicsStatisticsAsync()
        
        return {|
            TotalSecurityChecks = totalChecks
            SafeOperations = safeOperations
            BlockedOperations = blockedOperations
            OversightRequired = oversightRequired
            SafetyRate = if totalChecks > 0 then float safeOperations / float totalChecks else 1.0
            OperationTypeStatistics = operationTypeStats
            SecurityServiceStats = securityStats
            EthicsServiceStats = ethicsStats
            GlobalSecurityEnabled = globalSecurityEnabled
            RealTimeMonitoringEnabled = realTimeMonitoringEnabled
        |}
    }
    
    /// Update security policy for operation type
    member this.UpdateSecurityPolicyAsync(operationType: TarsOperationType, policy: SecurityPolicy) = task {
        operationPolicies.[operationType] <- policy
        logger.LogInformation($"Updated security policy for {operationType}")
        return Ok ()
    }
    
    /// Enable or disable global security
    member this.SetGlobalSecurityAsync(enabled: bool) = task {
        globalSecurityEnabled <- enabled
        logger.LogInformation($"Global security {if enabled then "enabled" else "disabled"}")
        return Ok ()
    }
    
    /// Get recent security checks
    member this.GetRecentSecurityChecksAsync(count: int) = task {
        let recentChecks = 
            securityChecks.Values
            |> Seq.sortByDescending (fun c -> c.Timestamp)
            |> Seq.take count
            |> Seq.toList
        
        return Ok recentChecks
    }
    
    /// Emergency security shutdown
    member this.EmergencySecurityShutdownAsync(reason: string) = task {
        try
            logger.LogCritical($"EMERGENCY SECURITY SHUTDOWN: {reason}")
            
            // Disable all operations
            globalSecurityEnabled <- false
            
            // Set all policies to maximum security
            let emergencyPolicy = {
                RequireSecurityCheck = true
                RequireEthicsCheck = true
                RequireHumanOversight = true
                BayesianThreshold = 0.01 // 1% threshold
                AlignmentThreshold = 0.95 // 95% alignment required
                BlockDangerousBehaviors = true
                LogAllOperations = true
                EnableRealTimeMonitoring = true
            }
            
            for operationType in operationPolicies.Keys do
                operationPolicies.[operationType] <- emergencyPolicy
            
            logger.LogCritical("Emergency security measures activated")
            return Ok ()
            
        with
        | ex ->
            logger.LogCritical(ex, "Failed to activate emergency security shutdown")
            return Error ex.Message
    }
