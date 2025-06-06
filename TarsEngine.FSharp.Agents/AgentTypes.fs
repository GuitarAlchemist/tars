namespace TarsEngine.FSharp.Agents

open System
open System.Threading
open System.Threading.Channels
open Microsoft.Extensions.Logging

/// Core types for TARS multi-agent system
module AgentTypes =
    
    /// Unique identifier for agents
    type AgentId = AgentId of Guid
    
    /// Agent status
    type AgentStatus =
        | Initializing
        | Running
        | Paused
        | Stopping
        | Stopped
        | Failed of string
    
    /// Agent capability
    type AgentCapability =
        | CodeAnalysis
        | ProjectGeneration
        | Documentation
        | Testing
        | Deployment
        | Monitoring
        | Research
        | Communication
        | Planning
        | Execution
        | Learning
        | SelfImprovement
        | Analysis
        | Automation
        | SystemManagement
    
    /// Agent personality traits
    type PersonalityTrait =
        | Analytical
        | Creative
        | Methodical
        | Innovative
        | Collaborative
        | Independent
        | Optimistic
        | Cautious
        | Aggressive
        | Patient
    
    /// Agent communication priority
    type MessagePriority =
        | Low = 1
        | Normal = 2
        | High = 3
        | Critical = 4
        | Emergency = 5
    
    /// Inter-agent message
    type AgentMessage = {
        Id: Guid
        FromAgent: AgentId
        ToAgent: AgentId option // None for broadcast
        MessageType: string
        Content: obj
        Priority: MessagePriority
        Timestamp: DateTime
        CorrelationId: Guid option
        ReplyTo: AgentId option
    }
    
    /// Agent execution context
    type AgentContext = {
        AgentId: AgentId
        WorkingDirectory: string
        Variables: Map<string, obj>
        SharedMemory: Map<string, obj>
        CancellationToken: CancellationToken
        Logger: ILogger
    }
    
    /// Agent task result
    type AgentTaskResult = {
        Success: bool
        Output: obj option
        Error: string option
        ExecutionTime: TimeSpan
        Metadata: Map<string, obj>
    }
    
    /// Long-running agent task with streaming results
    type AgentTask = {
        Id: Guid
        Name: string
        Description: string
        StartTime: DateTime
        EstimatedDuration: TimeSpan option
        Progress: float // 0.0 to 1.0
        Status: AgentStatus
        Results: seq<AgentTaskResult>
    }
    
    /// Agent persona definition
    type AgentPersona = {
        Name: string
        Description: string
        Capabilities: AgentCapability list
        Personality: PersonalityTrait list
        Specialization: string
        PreferredMetascripts: string list
        CommunicationStyle: string
        DecisionMakingStyle: string
        LearningRate: float
        CollaborationPreference: float // 0.0 (independent) to 1.0 (highly collaborative)
    }
    
    /// Agent instance
    type Agent = {
        Id: AgentId
        Persona: AgentPersona
        Status: AgentStatus
        Context: AgentContext
        CurrentTasks: AgentTask list
        MessageQueue: Channel<AgentMessage>
        MetascriptPath: string option
        StartTime: DateTime
        LastActivity: DateTime
        Statistics: Map<string, obj>
    }
    
    /// Agent team configuration
    type TeamConfiguration = {
        Name: string
        Description: string
        LeaderAgent: AgentId option
        Members: AgentId list
        SharedObjectives: string list
        CommunicationProtocol: string
        DecisionMakingProcess: string
        ConflictResolution: string
    }
    
    /// Team coordination message
    type TeamMessage = {
        TeamName: string
        MessageType: string
        Content: obj
        Priority: MessagePriority
        Timestamp: DateTime
        RequiresConsensus: bool
        VotingDeadline: DateTime option
    }
    
    /// Agent performance metrics
    type AgentMetrics = {
        TasksCompleted: int
        TasksSuccessful: int
        AverageExecutionTime: TimeSpan
        MessagesProcessed: int
        CollaborationScore: float
        LearningProgress: float
        EfficiencyRating: float
        LastUpdated: DateTime
    }
    
    /// Agent learning data
    type LearningData = {
        ExperienceType: string
        Context: Map<string, obj>
        Action: string
        Result: AgentTaskResult
        Feedback: float // -1.0 (negative) to 1.0 (positive)
        Timestamp: DateTime
    }
