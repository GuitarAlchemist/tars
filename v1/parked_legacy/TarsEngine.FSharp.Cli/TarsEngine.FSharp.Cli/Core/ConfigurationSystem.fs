namespace TarsEngine.FSharp.Cli.Core

open System
open System.IO
open System.Text.Json
open TarsEngine.FSharp.Cli.Core.MultiAgentDomain
open TarsEngine.FSharp.Cli.Core.ResultBasedErrorHandling

// ============================================================================
// CONFIGURATION SYSTEM - ELIMINATES HARD-CODED VALUES
// ============================================================================

module ConfigurationSystem =

    // ============================================================================
    // CONFIGURATION TYPES
    // ============================================================================

    type TarsConfiguration = {
        System: SystemConfiguration
        Agents: AgentConfiguration
        Departments: DepartmentConfiguration
        Performance: PerformanceConfiguration
        Visualization: VisualizationConfiguration
        Learning: LearningConfiguration
        Security: SecurityConfiguration
    }

    and SystemConfiguration = {
        MaxConcurrentProblems: int
        DefaultTimeout: TimeSpan
        LogLevel: LogLevel
        EnableMetrics: bool
        EnableCaching: bool
        CacheExpirationMinutes: int
        MaxRetryAttempts: int
        BackupEnabled: bool
        BackupIntervalHours: int
    }

    and AgentConfiguration = {
        MaxAgents: int
        MinAgents: int
        DefaultQualityThreshold: float
        MaxCommunicationRetries: int
        CommunicationTimeoutSeconds: int
        LearningEnabled: bool
        AdaptationRate: float
        PerformanceTrackingEnabled: bool
        DefaultCapabilities: string list
        QualityDecayRate: float // Per day
    }

    and DepartmentConfiguration = {
        MaxDepartments: int
        MinAgentsPerDepartment: int
        MaxAgentsPerDepartment: int
        DefaultCommunicationProtocol: CommunicationProtocol
        DefaultCoordinationStrategy: CoordinationStrategy
        DefaultGameTheoryStrategy: GameTheoryStrategy
        AutoRebalanceEnabled: bool
        RebalanceThresholdHours: int
    }

    and PerformanceConfiguration = {
        MetricsRetentionDays: int
        PerformanceUpdateIntervalSeconds: int
        QualityThresholds: QualityThresholds
        OptimizationEnabled: bool
        OptimizationIntervalMinutes: int
        PredictiveModelingEnabled: bool
        BottleneckDetectionEnabled: bool
        AlertThresholds: AlertThresholds
    }

    and VisualizationConfiguration = {
        DefaultTheme: string
        RefreshIntervalSeconds: int
        MaxElementsPerPage: int
        EnableAnimations: bool
        EnableRealTimeUpdates: bool
        CacheVisualizationMinutes: int
        ExportFormats: string list
        DefaultExportFormat: string
    }

    and LearningConfiguration = {
        LearningEnabled: bool
        LearningRate: float
        MaxLearningEvents: int
        LearningDecayDays: int
        AdaptationThreshold: float
        ConceptAnalysisEnabled: bool
        ConceptUpdateIntervalMinutes: int
        LearningInsightsRetentionDays: int
    }

    and SecurityConfiguration = {
        EnableAuthentication: bool
        EnableAuthorization: bool
        SessionTimeoutMinutes: int
        MaxFailedAttempts: int
        LockoutDurationMinutes: int
        EnableAuditLogging: bool
        EncryptSensitiveData: bool
        AllowedOperations: string list
    }

    and LogLevel = Debug | Info | Warning | Error | Critical

    and AlertThresholds = {
        LowSystemEfficiency: float
        HighErrorRate: float
        LongResponseTime: TimeSpan
        LowAgentSatisfaction: float
        HighResourceUtilization: float
    }

    // ============================================================================
    // DEFAULT CONFIGURATIONS
    // ============================================================================

    let defaultConfiguration : TarsConfiguration = {
        System = {
            MaxConcurrentProblems = 5
            DefaultTimeout = TimeSpan.FromMinutes(30.0)
            LogLevel = Info
            EnableMetrics = true
            EnableCaching = true
            CacheExpirationMinutes = 15
            MaxRetryAttempts = 3
            BackupEnabled = true
            BackupIntervalHours = 6
        }
        Agents = {
            MaxAgents = 20
            MinAgents = 2
            DefaultQualityThreshold = 0.7
            MaxCommunicationRetries = 3
            CommunicationTimeoutSeconds = 30
            LearningEnabled = true
            AdaptationRate = 0.1
            PerformanceTrackingEnabled = true
            DefaultCapabilities = [
                "Problem Solving"
                "Communication"
                "Analysis"
                "Collaboration"
            ]
            QualityDecayRate = 0.01 // 1% per day without activity
        }
        Departments = {
            MaxDepartments = 8
            MinAgentsPerDepartment = 1
            MaxAgentsPerDepartment = 10
            DefaultCommunicationProtocol = PeerToPeer(maxConnections = 5)
            DefaultCoordinationStrategy = Distributed(consensus = Majority)
            DefaultGameTheoryStrategy = Nash
            AutoRebalanceEnabled = true
            RebalanceThresholdHours = 24
        }
        Performance = {
            MetricsRetentionDays = 30
            PerformanceUpdateIntervalSeconds = 60
            QualityThresholds = {
                MinAgentPerformance = 0.6
                MinDepartmentEfficiency = 0.7
                MinProblemConfidence = 0.5
                MinSystemQuality = 0.75
            }
            OptimizationEnabled = true
            OptimizationIntervalMinutes = 30
            PredictiveModelingEnabled = true
            BottleneckDetectionEnabled = true
            AlertThresholds = {
                LowSystemEfficiency = 0.6
                HighErrorRate = 0.1
                LongResponseTime = TimeSpan.FromSeconds(10.0)
                LowAgentSatisfaction = 0.5
                HighResourceUtilization = 0.9
            }
        }
        Visualization = {
            DefaultTheme = "dark"
            RefreshIntervalSeconds = 5
            MaxElementsPerPage = 50
            EnableAnimations = true
            EnableRealTimeUpdates = true
            CacheVisualizationMinutes = 5
            ExportFormats = ["html"; "json"; "csv"; "pdf"]
            DefaultExportFormat = "html"
        }
        Learning = {
            LearningEnabled = true
            LearningRate = 0.05
            MaxLearningEvents = 1000
            LearningDecayDays = 7
            AdaptationThreshold = 0.1
            ConceptAnalysisEnabled = true
            ConceptUpdateIntervalMinutes = 15
            LearningInsightsRetentionDays = 90
        }
        Security = {
            EnableAuthentication = false // Disabled for demo
            EnableAuthorization = false
            SessionTimeoutMinutes = 60
            MaxFailedAttempts = 5
            LockoutDurationMinutes = 15
            EnableAuditLogging = true
            EncryptSensitiveData = false // Disabled for demo
            AllowedOperations = [
                "read"
                "write"
                "execute"
                "analyze"
                "visualize"
                "export"
            ]
        }
    }

    // ============================================================================
    // CONFIGURATION MANAGEMENT
    // ============================================================================

    module ConfigurationManager =
        
        let private configurationCache = ref (Some defaultConfiguration)
        let private configurationPath = "tars-config.json"

        let serializeConfiguration (config: TarsConfiguration) : string =
            let options = JsonSerializerOptions()
            options.WriteIndented <- true
            options.PropertyNamingPolicy <- JsonNamingPolicy.CamelCase
            JsonSerializer.Serialize(config, options)

        let deserializeConfiguration (json: string) : TarsResult<TarsConfiguration, TarsError> =
            try
                let options = JsonSerializerOptions()
                options.PropertyNamingPolicy <- JsonNamingPolicy.CamelCase
                let config = JsonSerializer.Deserialize<TarsConfiguration>(json, options)
                Success config
            with
            | ex -> Error (configurationError "Failed to deserialize configuration" (Some ex.Message))

        let validateConfiguration (config: TarsConfiguration) : TarsResult<TarsConfiguration, TarsError> =
            result {
                // System validation
                if config.System.MaxConcurrentProblems <= 0 then
                    return! Error (validationError "MaxConcurrentProblems must be positive" None)
                elif config.System.CacheExpirationMinutes <= 0 then
                    return! Error (validationError "CacheExpirationMinutes must be positive" None)

                // Agent validation
                elif config.Agents.MaxAgents <= config.Agents.MinAgents then
                    return! Error (validationError "MaxAgents must be greater than MinAgents" None)
                elif config.Agents.DefaultQualityThreshold < 0.0 || config.Agents.DefaultQualityThreshold > 1.0 then
                    return! Error (validationError "DefaultQualityThreshold must be between 0 and 1" None)

                // Department validation
                elif config.Departments.MaxDepartments <= 0 then
                    return! Error (validationError "MaxDepartments must be positive" None)
                elif config.Departments.MinAgentsPerDepartment <= 0 then
                    return! Error (validationError "MinAgentsPerDepartment must be positive" None)

                // Performance validation
                elif config.Performance.MetricsRetentionDays <= 0 then
                    return! Error (validationError "MetricsRetentionDays must be positive" None)

                else
                    return config
            }

        let loadConfiguration () : TarsResult<TarsConfiguration, TarsError> =
            result {
                if File.Exists(configurationPath) then
                    let! json = Result.catch (fun () -> File.ReadAllText(configurationPath))
                    let! config = deserializeConfiguration json
                    let! validatedConfig = validateConfiguration config
                    configurationCache := Some validatedConfig
                    return validatedConfig
                else
                    // Create default configuration file
                    let defaultJson = serializeConfiguration defaultConfiguration
                    do! Result.catch (fun () -> File.WriteAllText(configurationPath, defaultJson))
                    configurationCache := Some defaultConfiguration
                    return defaultConfiguration
            }

        let saveConfiguration (config: TarsConfiguration) : TarsResult<unit, TarsError> =
            result {
                let! validatedConfig = validateConfiguration config
                let json = serializeConfiguration validatedConfig
                do! Result.catch (fun () -> File.WriteAllText(configurationPath, json))
                configurationCache := Some validatedConfig
                return ()
            }

        let getCurrentConfiguration () : TarsConfiguration =
            match !configurationCache with
            | Some config -> config
            | None ->
                match loadConfiguration() with
                | Success config -> config
                | Error _ -> defaultConfiguration

        let updateConfiguration (updater: TarsConfiguration -> TarsConfiguration) : TarsResult<TarsConfiguration, TarsError> =
            result {
                let currentConfig = getCurrentConfiguration()
                let newConfig = updater currentConfig
                let! validatedConfig = validateConfiguration newConfig
                do! saveConfiguration validatedConfig
                return validatedConfig
            }

        let resetToDefaults () : TarsResult<TarsConfiguration, TarsError> =
            result {
                do! saveConfiguration defaultConfiguration
                return defaultConfiguration
            }

    // ============================================================================
    // CONFIGURATION HELPERS
    // ============================================================================

    module ConfigurationHelpers =
        
        let getSystemConfig () = (ConfigurationManager.getCurrentConfiguration()).System
        let getAgentConfig () = (ConfigurationManager.getCurrentConfiguration()).Agents
        let getDepartmentConfig () = (ConfigurationManager.getCurrentConfiguration()).Departments
        let getPerformanceConfig () = (ConfigurationManager.getCurrentConfiguration()).Performance
        let getVisualizationConfig () = (ConfigurationManager.getCurrentConfiguration()).Visualization
        let getLearningConfig () = (ConfigurationManager.getCurrentConfiguration()).Learning
        let getSecurityConfig () = (ConfigurationManager.getCurrentConfiguration()).Security

        let isFeatureEnabled (feature: string) : bool =
            let config = ConfigurationManager.getCurrentConfiguration()
            match feature.ToLower() with
            | "metrics" -> config.System.EnableMetrics
            | "caching" -> config.System.EnableCaching
            | "learning" -> config.Agents.LearningEnabled
            | "optimization" -> config.Performance.OptimizationEnabled
            | "predictive" -> config.Performance.PredictiveModelingEnabled
            | "bottleneck" -> config.Performance.BottleneckDetectionEnabled
            | "animations" -> config.Visualization.EnableAnimations
            | "realtime" -> config.Visualization.EnableRealTimeUpdates
            | "concepts" -> config.Learning.ConceptAnalysisEnabled
            | "auth" -> config.Security.EnableAuthentication
            | "audit" -> config.Security.EnableAuditLogging
            | _ -> false

        let getThreshold (thresholdType: string) : float option =
            let config = ConfigurationManager.getCurrentConfiguration()
            match thresholdType.ToLower() with
            | "agent_quality" -> Some config.Agents.DefaultQualityThreshold
            | "agent_performance" -> Some config.Performance.QualityThresholds.MinAgentPerformance
            | "department_efficiency" -> Some config.Performance.QualityThresholds.MinDepartmentEfficiency
            | "problem_confidence" -> Some config.Performance.QualityThresholds.MinProblemConfidence
            | "system_quality" -> Some config.Performance.QualityThresholds.MinSystemQuality
            | "system_efficiency" -> Some config.Performance.AlertThresholds.LowSystemEfficiency
            | "error_rate" -> Some config.Performance.AlertThresholds.HighErrorRate
            | "agent_satisfaction" -> Some config.Performance.AlertThresholds.LowAgentSatisfaction
            | "resource_utilization" -> Some config.Performance.AlertThresholds.HighResourceUtilization
            | "adaptation" -> Some config.Learning.AdaptationThreshold
            | _ -> None

        let getTimeout (timeoutType: string) : TimeSpan option =
            let config = ConfigurationManager.getCurrentConfiguration()
            match timeoutType.ToLower() with
            | "default" -> Some config.System.DefaultTimeout
            | "communication" -> Some (TimeSpan.FromSeconds(float config.Agents.CommunicationTimeoutSeconds))
            | "session" -> Some (TimeSpan.FromMinutes(float config.Security.SessionTimeoutMinutes))
            | "response" -> Some config.Performance.AlertThresholds.LongResponseTime
            | _ -> None

        let getLimit (limitType: string) : int option =
            let config = ConfigurationManager.getCurrentConfiguration()
            match limitType.ToLower() with
            | "max_agents" -> Some config.Agents.MaxAgents
            | "min_agents" -> Some config.Agents.MinAgents
            | "max_departments" -> Some config.Departments.MaxDepartments
            | "max_concurrent_problems" -> Some config.System.MaxConcurrentProblems
            | "max_retries" -> Some config.System.MaxRetryAttempts
            | "max_comm_retries" -> Some config.Agents.MaxCommunicationRetries
            | "max_elements_per_page" -> Some config.Visualization.MaxElementsPerPage
            | "max_learning_events" -> Some config.Learning.MaxLearningEvents
            | "max_failed_attempts" -> Some config.Security.MaxFailedAttempts
            | _ -> None

        let createAgentWithDefaults (id: string) (name: string) (specialization: AgentSpecialization) : UnifiedAgent =
            let config = getAgentConfig()
            {
                Id = id
                Name = name
                CreatedAt = DateTime.UtcNow
                Specialization = specialization
                Capabilities = config.DefaultCapabilities
                ReasoningCapabilities = []
                Status = Idle
                CurrentTask = None
                Progress = 0.0
                Position3D = (0.0, 0.0, 0.0)
                Department = None
                CommunicationHistory = []
                GameTheoryProfile = CooperativeGame("Default")
                StrategyPreferences = [Cooperative(weight = 0.7)]
                PerformanceMetrics = {
                    TasksCompleted = 0
                    AverageResponseTime = TimeSpan.Zero
                    SuccessRate = 1.0
                    CommunicationEfficiency = 0.8
                    ReasoningAccuracy = 0.8
                }
                QualityScore = config.DefaultQualityThreshold
            }

        let createDepartmentWithDefaults (id: string) (name: string) (deptType: DepartmentType) (agents: UnifiedAgent list) : UnifiedDepartment =
            let config = getDepartmentConfig()
            {
                Id = id
                Name = name
                CreatedAt = DateTime.UtcNow
                DepartmentType = deptType
                Hierarchy = 1
                Agents = agents
                CommunicationProtocol = config.DefaultCommunicationProtocol
                CoordinationStrategy = config.DefaultCoordinationStrategy
                GameTheoryStrategy = config.DefaultGameTheoryStrategy
                CollectiveGoals = []
                Position3D = (0.0, 0.0, 0.0)
                PerformanceMetrics = {
                    CollectiveEfficiency = 0.8
                    InterAgentCoordination = 0.7
                    GoalCompletionRate = 0.8
                    CommunicationOverhead = 0.2
                    ResourceUtilization = 0.7
                }
            }
