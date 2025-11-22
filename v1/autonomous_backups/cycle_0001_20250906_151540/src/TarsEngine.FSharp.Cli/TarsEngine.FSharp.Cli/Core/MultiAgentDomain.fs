namespace TarsEngine.FSharp.Cli.Core

open System
open TarsEngine.FSharp.Core.Agents.AgentSystem

// ============================================================================
// UNIFIED MULTI-AGENT REASONING DOMAIN MODEL
// ============================================================================

module MultiAgentDomain =

    // ============================================================================
    // CORE DOMAIN TYPES
    // ============================================================================

    /// Unified agent representation combining all aspects
    type UnifiedAgent = {
        // Core Identity
        Id: string
        Name: string
        CreatedAt: DateTime
        
        // Capabilities
        Specialization: AgentSpecialization
        Capabilities: string list
        ReasoningCapabilities: ReasoningCapability list
        
        // State
        Status: AgentStatus
        CurrentTask: string option
        Progress: float
        
        // Spatial & Communication
        Position3D: float * float * float
        Department: string option
        CommunicationHistory: AgentMessage list
        
        // Game Theory & Strategy
        GameTheoryProfile: GameTheoryModel
        StrategyPreferences: StrategyPreference list
        
        // Performance Metrics
        PerformanceMetrics: PerformanceMetrics
        QualityScore: float
    }

    and ReasoningCapability =
        | ProblemDecomposition of complexity: int
        | ConceptAnalysis of domains: string list
        | PatternRecognition of accuracy: float
        | StrategicPlanning of horizon: TimeSpan
        | KnowledgeIntegration of sources: string list

    and AgentStatus =
        | Idle
        | Working of task: string
        | Communicating of target: string
        | Reasoning of problem: string
        | Completed of result: string

    and StrategyPreference =
        | Cooperative of weight: float
        | Competitive of weight: float
        | Adaptive of threshold: float
        | Specialized of domain: string

    and PerformanceMetrics = {
        TasksCompleted: int
        AverageResponseTime: TimeSpan
        SuccessRate: float
        CommunicationEfficiency: float
        ReasoningAccuracy: float
    }

    /// Unified department with clear responsibilities
    type UnifiedDepartment = {
        // Core Identity
        Id: string
        Name: string
        CreatedAt: DateTime
        
        // Structure
        DepartmentType: DepartmentType
        Hierarchy: int
        Agents: UnifiedAgent list
        
        // Communication & Coordination
        CommunicationProtocol: CommunicationProtocol
        CoordinationStrategy: CoordinationStrategy
        
        // Game Theory & Strategy
        GameTheoryStrategy: GameTheoryStrategy
        CollectiveGoals: Goal list
        
        // Spatial & Performance
        Position3D: float * float * float
        PerformanceMetrics: DepartmentPerformanceMetrics
    }

    and CommunicationProtocol =
        | Hierarchical of levels: int
        | PeerToPeer of maxConnections: int
        | Broadcast of scope: BroadcastScope
        | GameTheoretic of mechanism: string
        | Mesh of redundancy: int

    and BroadcastScope =
        | Department
        | AllAgents
        | Selective of criteria: string

    and CoordinationStrategy =
        | Centralized of coordinator: string
        | Distributed of consensus: ConsensusType
        | Hybrid of primary: string * backup: string list

    and ConsensusType =
        | Majority
        | Unanimous
        | Weighted of weights: Map<string, float>

    and Goal = {
        Id: string
        Description: string
        Priority: Priority
        Deadline: DateTime option
        AssignedAgents: string list
        Progress: float
    }

    and Priority = Low | Medium | High | Critical

    and DepartmentPerformanceMetrics = {
        CollectiveEfficiency: float
        InterAgentCoordination: float
        GoalCompletionRate: float
        CommunicationOverhead: float
        ResourceUtilization: float
    }

    /// Unified problem representation
    type UnifiedProblem = {
        // Core Problem
        Id: string
        OriginalStatement: string
        Domain: string
        CreatedAt: DateTime
        
        // Decomposition
        Complexity: ProblemComplexity
        SubProblems: SubProblem list
        Dependencies: Dependency list
        
        // Analysis
        ConceptAnalysis: ConceptAnalysisResult option
        ReasoningSteps: ReasoningStep list
        
        // Solution Strategy
        SolutionStrategy: SolutionStrategy
        RequiredExpertise: ExpertiseRequirement list
        EstimatedEffort: EffortEstimate
        
        // Quality & Confidence
        ConfidenceScore: float
        QualityMetrics: ProblemQualityMetrics
    }

    and ProblemComplexity =
        | Simple of difficulty: int
        | Moderate of subProblems: int * difficulty: int
        | Complex of subProblems: int * depth: int * difficulty: int
        | Adaptive of baseComplexity: ProblemComplexity * adaptationFactors: AdaptationFactor list

    and AdaptationFactor =
        | TimeConstraint of urgency: float
        | ResourceConstraint of availability: float
        | KnowledgeGap of severity: float
        | StakeholderComplexity of count: int

    and SubProblem = {
        Id: string
        Title: string
        Description: string
        ParentId: string option
        RequiredExpertise: string list
        EstimatedComplexity: int
        Dependencies: string list
        ExpectedOutput: string
        AssignedAgents: string list
        Status: SubProblemStatus
    }

    and SubProblemStatus =
        | NotStarted
        | InProgress of progress: float
        | Blocked of reason: string
        | Completed of result: string
        | Failed of error: string

    and Dependency = {
        FromSubProblem: string
        ToSubProblem: string
        DependencyType: DependencyType
        Strength: float
    }

    and DependencyType =
        | Sequential
        | Parallel
        | Conditional of condition: string
        | Resource of resource: string

    and ConceptAnalysisResult = {
        DominantConcepts: (string * float) list
        SemanticSummary: string
        ConceptWeights: Map<string, float>
        AnalysisConfidence: float
    }

    and ReasoningStep = {
        Id: string
        Description: string
        InputConcepts: string list
        OutputConcepts: string list
        ReasoningType: ReasoningType
        Confidence: float
        ExecutionTime: TimeSpan
    }

    and ReasoningType =
        | Deductive
        | Inductive
        | Abductive
        | Analogical
        | Causal

    and SolutionStrategy =
        | Divide of approach: DivideStrategy
        | Collaborate of coordination: CollaborationStrategy
        | Iterate of cycles: IterationStrategy
        | Hybrid of strategies: SolutionStrategy list

    and DivideStrategy = {
        PartitioningMethod: PartitioningMethod
        MergeStrategy: MergeStrategy
        QualityControl: QualityControlMethod
    }

    and PartitioningMethod =
        | ByExpertise
        | ByComplexity
        | ByDependency
        | ByResource

    and MergeStrategy =
        | Sequential
        | Parallel
        | Hierarchical
        | Consensus

    and QualityControlMethod =
        | PeerReview
        | ExpertValidation
        | AutomatedTesting
        | CrossValidation

    and CollaborationStrategy = {
        CommunicationPattern: CommunicationPattern
        DecisionMaking: DecisionMakingMethod
        ConflictResolution: ConflictResolutionMethod
    }

    and CommunicationPattern =
        | AllToAll
        | StarTopology of center: string
        | RingTopology
        | TreeTopology of root: string

    and DecisionMakingMethod =
        | Democratic
        | Authoritative of authority: string
        | Expertise of weightByExpertise: bool
        | GameTheoretic of mechanism: string

    and ConflictResolutionMethod =
        | Voting
        | Mediation of mediator: string
        | Escalation of escalationPath: string list
        | Negotiation

    and IterationStrategy = {
        MaxIterations: int
        ConvergenceCriteria: ConvergenceCriteria
        FeedbackMechanism: FeedbackMechanism
    }

    and ConvergenceCriteria =
        | QualityThreshold of threshold: float
        | ChangeThreshold of threshold: float
        | TimeLimit of limit: TimeSpan
        | ResourceLimit of limit: float

    and FeedbackMechanism =
        | Continuous
        | Periodic of interval: TimeSpan
        | Milestone of milestones: string list
        | Adaptive of sensitivity: float

    and ExpertiseRequirement = {
        Domain: string
        Level: ExpertiseLevel
        IsCritical: bool
        Alternatives: string list
    }

    and ExpertiseLevel = Novice | Intermediate | Advanced | Expert

    and EffortEstimate = {
        TimeEstimate: TimeSpan
        ResourceEstimate: ResourceEstimate
        RiskFactors: RiskFactor list
        ConfidenceInterval: float * float
    }

    and ResourceEstimate = {
        ComputationalResources: float
        HumanResources: float
        DataResources: string list
        ExternalDependencies: string list
    }

    and RiskFactor = {
        Description: string
        Probability: float
        Impact: float
        MitigationStrategy: string option
    }

    and ProblemQualityMetrics = {
        Completeness: float
        Consistency: float
        Feasibility: float
        Clarity: float
        Testability: float
    }

    // ============================================================================
    // UNIFIED SYSTEM STATE
    // ============================================================================

    type UnifiedReasoningSystem = {
        // Core State
        Id: string
        Name: string
        CreatedAt: DateTime
        Status: SystemStatus
        
        // Components
        Problems: UnifiedProblem list
        Departments: UnifiedDepartment list
        Agents: UnifiedAgent list
        
        // Configuration
        Configuration: SystemConfiguration
        
        // Performance & Metrics
        SystemMetrics: SystemMetrics
        QualityScore: float
    }

    and SystemStatus =
        | Initializing
        | Ready
        | Processing of problems: string list
        | Optimizing
        | Error of error: string

    and SystemConfiguration = {
        MaxAgents: int
        MaxDepartments: int
        DefaultGameTheoryStrategy: GameTheoryStrategy
        CommunicationTimeout: TimeSpan
        QualityThresholds: QualityThresholds
    }

    and QualityThresholds = {
        MinAgentPerformance: float
        MinDepartmentEfficiency: float
        MinProblemConfidence: float
        MinSystemQuality: float
    }

    and SystemMetrics = {
        TotalProblemsProcessed: int
        AverageProcessingTime: TimeSpan
        SystemEfficiency: float
        ResourceUtilization: float
        AgentSatisfaction: float
    }
