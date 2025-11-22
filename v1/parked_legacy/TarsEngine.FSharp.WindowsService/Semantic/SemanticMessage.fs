namespace TarsEngine.FSharp.WindowsService.Semantic

open System
open System.Collections.Generic

/// <summary>
/// Semantic message priority levels
/// </summary>
type SemanticPriority =
    | Critical = 0
    | High = 1
    | Normal = 2
    | Low = 3
    | Background = 4

/// <summary>
/// Message urgency levels
/// </summary>
type MessageUrgency =
    | Immediate     // Requires immediate response
    | Urgent        // Response needed within minutes
    | Standard      // Response needed within hours
    | Deferred      // Response can be delayed
    | Batch         // Can be processed in batch

/// <summary>
/// Task complexity assessment
/// </summary>
type TaskComplexity =
    | Simple        // Basic operations, single step
    | Moderate      // Multiple steps, some logic
    | Complex       // Advanced logic, multiple components
    | Expert        // Requires specialized knowledge
    | Collaborative // Requires multiple agents

/// <summary>
/// Agent capability requirement
/// </summary>
type CapabilityRequirement = {
    Name: string
    Level: CapabilityLevel
    Required: bool
    Weight: float
    Description: string
}

/// <summary>
/// Capability proficiency levels
/// </summary>
and CapabilityLevel =
    | Beginner = 1
    | Intermediate = 2
    | Advanced = 3
    | Expert = 4
    | Master = 5

/// <summary>
/// Resource requirements for task execution
/// </summary>
type ResourceRequirement = {
    EstimatedDuration: TimeSpan
    MemoryRequiredMB: int
    CpuIntensive: bool
    NetworkAccess: bool
    FileSystemAccess: bool
    ExternalDependencies: string list
    ConcurrencyLevel: int
}

/// <summary>
/// Semantic metadata for NLP analysis
/// </summary>
type SemanticMetadata = {
    Keywords: string list
    Entities: Map<string, string>
    Intent: string
    Domain: string
    Language: string
    Confidence: float
    Embeddings: float array option
    Sentiment: float
    Topics: string list
}

/// <summary>
/// Task request with semantic information
/// </summary>
type TaskRequest = {
    Id: string
    Title: string
    Description: string
    RequiredCapabilities: CapabilityRequirement list
    ResourceRequirements: ResourceRequirement
    Priority: SemanticPriority
    Urgency: MessageUrgency
    Complexity: TaskComplexity
    Deadline: DateTime option
    Context: Map<string, obj>
    SemanticMetadata: SemanticMetadata
    RequesterId: string
    CreatedAt: DateTime
    ExpiresAt: DateTime option
    Tags: string list
}

/// <summary>
/// Agent capability declaration
/// </summary>
type AgentCapability = {
    Name: string
    Level: CapabilityLevel
    Confidence: float
    Experience: int
    LastUsed: DateTime
    SuccessRate: float
    AverageExecutionTime: TimeSpan
    Description: string
    Examples: string list
    Dependencies: string list
}

/// <summary>
/// Agent response to task request
/// </summary>
type CapabilityResponse = {
    Id: string
    TaskRequestId: string
    AgentId: string
    AgentName: string
    CanHandle: bool
    Confidence: float
    EstimatedDuration: TimeSpan
    EstimatedCost: float
    MatchingCapabilities: AgentCapability list
    ProposedApproach: string
    ResourceRequirements: ResourceRequirement
    Availability: DateTime
    Conditions: string list
    AlternativeAgents: string list
    ResponseTime: TimeSpan
    CreatedAt: DateTime
    ExpiresAt: DateTime
}

/// <summary>
/// Semantic message envelope
/// </summary>
type SemanticMessage = {
    Id: string
    CorrelationId: string option
    ConversationId: string option
    MessageType: SemanticMessageType
    Priority: SemanticPriority
    Urgency: MessageUrgency
    SenderId: string
    Recipients: string list
    BroadcastScope: BroadcastScope option
    Content: SemanticMessageContent
    Metadata: Map<string, obj>
    CreatedAt: DateTime
    ExpiresAt: DateTime option
    DeliveryAttempts: int
    MaxDeliveryAttempts: int
    RequiresAcknowledgment: bool
    IsEncrypted: bool
}

/// <summary>
/// Types of semantic messages
/// </summary>
and SemanticMessageType =
    | TaskRequest
    | CapabilityResponse
    | TaskAssignment
    | TaskUpdate
    | TaskCompletion
    | CapabilityAnnouncement
    | AgentStatusUpdate
    | CollaborationRequest
    | ResourceRequest
    | SystemNotification

/// <summary>
/// Broadcast scope for messages
/// </summary>
and BroadcastScope =
    | AllAgents
    | AgentsByCapability of string list
    | AgentsByType of string list
    | AgentsByTag of string list
    | SpecificAgents of string list
    | NearbyAgents of float // radius

/// <summary>
/// Content of semantic messages
/// </summary>
and SemanticMessageContent =
    | TaskRequestContent of TaskRequest
    | CapabilityResponseContent of CapabilityResponse
    | TaskAssignmentContent of TaskAssignment
    | TaskUpdateContent of TaskUpdate
    | TaskCompletionContent of TaskCompletion
    | CapabilityAnnouncementContent of CapabilityAnnouncement
    | AgentStatusUpdateContent of AgentStatusUpdate
    | CollaborationRequestContent of CollaborationRequest
    | ResourceRequestContent of ResourceRequest
    | SystemNotificationContent of SystemNotification
    | CustomContent of Map<string, obj>

/// <summary>
/// Task assignment information
/// </summary>
and TaskAssignment = {
    TaskId: string
    AssignedAgentId: string
    AssignedAt: DateTime
    ExpectedCompletion: DateTime
    Instructions: string
    Resources: Map<string, obj>
    Constraints: string list
    SuccessCriteria: string list
}

/// <summary>
/// Task progress update
/// </summary>
and TaskUpdate = {
    TaskId: string
    AgentId: string
    Status: TaskStatus
    Progress: float
    Message: string
    EstimatedCompletion: DateTime option
    Issues: string list
    ResourcesUsed: Map<string, obj>
    UpdatedAt: DateTime
}

/// <summary>
/// Task completion information
/// </summary>
and TaskCompletion = {
    TaskId: string
    AgentId: string
    Status: TaskStatus
    Result: obj option
    ExecutionTime: TimeSpan
    ResourcesUsed: Map<string, obj>
    QualityScore: float
    Feedback: string
    Artifacts: string list
    CompletedAt: DateTime
}

/// <summary>
/// Agent capability announcement
/// </summary>
and CapabilityAnnouncement = {
    AgentId: string
    Capabilities: AgentCapability list
    Availability: AgentAvailability
    LoadFactor: float
    Specializations: string list
    PreferredTaskTypes: string list
    AnnouncedAt: DateTime
}

/// <summary>
/// Agent status update
/// </summary>
and AgentStatusUpdate = {
    AgentId: string
    Status: AgentStatus
    Availability: AgentAvailability
    CurrentTasks: string list
    LoadFactor: float
    Performance: AgentPerformance
    UpdatedAt: DateTime
}

/// <summary>
/// Collaboration request between agents
/// </summary>
and CollaborationRequest = {
    RequestId: string
    InitiatorAgentId: string
    TaskId: string
    CollaborationType: CollaborationType
    RequiredCapabilities: CapabilityRequirement list
    ProposedAgents: string list
    Timeline: TimeSpan
    Description: string
    RequestedAt: DateTime
}

/// <summary>
/// Resource request from agent
/// </summary>
and ResourceRequest = {
    RequestId: string
    AgentId: string
    ResourceType: ResourceType
    Quantity: int
    Duration: TimeSpan
    Priority: SemanticPriority
    Justification: string
    RequestedAt: DateTime
}

/// <summary>
/// System notification
/// </summary>
and SystemNotification = {
    NotificationId: string
    Type: NotificationType
    Severity: NotificationSeverity
    Title: string
    Message: string
    AffectedAgents: string list
    ActionRequired: bool
    ExpiresAt: DateTime option
    CreatedAt: DateTime
}

/// <summary>
/// Task execution status
/// </summary>
and TaskStatus =
    | Pending
    | Assigned
    | InProgress
    | Paused
    | Completed
    | Failed
    | Cancelled
    | Expired

/// <summary>
/// Agent availability status
/// </summary>
and AgentAvailability =
    | Available
    | Busy
    | Overloaded
    | Maintenance
    | Offline
    | Restricted

/// <summary>
/// Agent operational status
/// </summary>
and AgentStatus =
    | Online
    | Offline
    | Starting
    | Stopping
    | Error
    | Maintenance

/// <summary>
/// Agent performance metrics
/// </summary>
and AgentPerformance = {
    TasksCompleted: int64
    SuccessRate: float
    AverageExecutionTime: TimeSpan
    QualityScore: float
    ResponseTime: TimeSpan
    Reliability: float
    Efficiency: float
}

/// <summary>
/// Types of collaboration
/// </summary>
and CollaborationType =
    | Sequential      // Tasks executed in sequence
    | Parallel        // Tasks executed in parallel
    | Pipeline        // Output of one feeds into another
    | Consensus       // Multiple agents reach consensus
    | Competition     // Best result wins
    | Assistance      // One agent helps another

/// <summary>
/// Types of resources
/// </summary>
and ResourceType =
    | ComputeTime
    | Memory
    | Storage
    | NetworkBandwidth
    | ExternalAPI
    | Database
    | FileSystem
    | Specialized of string

/// <summary>
/// Notification types
/// </summary>
and NotificationType =
    | Information
    | Warning
    | Error
    | Maintenance
    | Update
    | Security
    | Performance

/// <summary>
/// Notification severity levels
/// </summary>
and NotificationSeverity =
    | Low
    | Medium
    | High
    | Critical

/// <summary>
/// Message delivery status
/// </summary>
type DeliveryStatus =
    | Pending
    | Delivered
    | Failed
    | Expired
    | Acknowledged
    | Rejected

/// <summary>
/// Message delivery receipt
/// </summary>
type DeliveryReceipt = {
    MessageId: string
    RecipientId: string
    Status: DeliveryStatus
    DeliveredAt: DateTime option
    AcknowledgedAt: DateTime option
    Error: string option
    Attempts: int
}

/// <summary>
/// Semantic message validation result
/// </summary>
type MessageValidationResult = {
    IsValid: bool
    Errors: string list
    Warnings: string list
    Suggestions: string list
    Score: float
}

/// <summary>
/// Message routing decision
/// </summary>
type RoutingDecision = {
    MessageId: string
    SelectedAgents: string list
    RoutingScore: float
    RoutingReason: string
    AlternativeAgents: string list
    RoutingTime: TimeSpan
    DecisionAt: DateTime
}

/// <summary>
/// Semantic message utilities and helpers
/// </summary>
module SemanticMessageHelpers =
    
    /// Create a new task request
    let createTaskRequest title description capabilities priority urgency requesterId =
        {
            Id = Guid.NewGuid().ToString()
            Title = title
            Description = description
            RequiredCapabilities = capabilities
            ResourceRequirements = {
                EstimatedDuration = TimeSpan.FromMinutes(30.0)
                MemoryRequiredMB = 256
                CpuIntensive = false
                NetworkAccess = false
                FileSystemAccess = false
                ExternalDependencies = []
                ConcurrencyLevel = 1
            }
            Priority = priority
            Urgency = urgency
            Complexity = TaskComplexity.Moderate
            Deadline = None
            Context = Map.empty
            SemanticMetadata = {
                Keywords = []
                Entities = Map.empty
                Intent = ""
                Domain = ""
                Language = "en"
                Confidence = 0.0
                Embeddings = None
                Sentiment = 0.0
                Topics = []
            }
            RequesterId = requesterId
            CreatedAt = DateTime.UtcNow
            ExpiresAt = Some (DateTime.UtcNow.AddHours(24.0))
            Tags = []
        }
    
    /// Create a capability response
    let createCapabilityResponse taskRequestId agentId agentName canHandle confidence =
        {
            Id = Guid.NewGuid().ToString()
            TaskRequestId = taskRequestId
            AgentId = agentId
            AgentName = agentName
            CanHandle = canHandle
            Confidence = confidence
            EstimatedDuration = TimeSpan.FromMinutes(15.0)
            EstimatedCost = 1.0
            MatchingCapabilities = []
            ProposedApproach = ""
            ResourceRequirements = {
                EstimatedDuration = TimeSpan.FromMinutes(15.0)
                MemoryRequiredMB = 128
                CpuIntensive = false
                NetworkAccess = false
                FileSystemAccess = false
                ExternalDependencies = []
                ConcurrencyLevel = 1
            }
            Availability = DateTime.UtcNow
            Conditions = []
            AlternativeAgents = []
            ResponseTime = TimeSpan.Zero
            CreatedAt = DateTime.UtcNow
            ExpiresAt = DateTime.UtcNow.AddHours(1.0)
        }
    
    /// Create a semantic message
    let createSemanticMessage messageType priority senderId recipients content =
        {
            Id = Guid.NewGuid().ToString()
            CorrelationId = None
            ConversationId = None
            MessageType = messageType
            Priority = priority
            Urgency = MessageUrgency.Standard
            SenderId = senderId
            Recipients = recipients
            BroadcastScope = None
            Content = content
            Metadata = Map.empty
            CreatedAt = DateTime.UtcNow
            ExpiresAt = Some (DateTime.UtcNow.AddHours(24.0))
            DeliveryAttempts = 0
            MaxDeliveryAttempts = 3
            RequiresAcknowledgment = false
            IsEncrypted = false
        }
    
    /// Validate a semantic message
    let validateMessage (message: SemanticMessage) =
        let errors = ResizeArray<string>()
        let warnings = ResizeArray<string>()
        let suggestions = ResizeArray<string>()
        
        // Basic validation
        if String.IsNullOrWhiteSpace(message.Id) then
            errors.Add("Message ID is required")
        
        if String.IsNullOrWhiteSpace(message.SenderId) then
            errors.Add("Sender ID is required")
        
        if message.Recipients.IsEmpty && message.BroadcastScope.IsNone then
            errors.Add("Message must have recipients or broadcast scope")
        
        // Content validation
        match message.Content with
        | TaskRequestContent taskRequest ->
            if String.IsNullOrWhiteSpace(taskRequest.Title) then
                errors.Add("Task request title is required")
            if String.IsNullOrWhiteSpace(taskRequest.Description) then
                errors.Add("Task request description is required")
        | CapabilityResponseContent response ->
            if String.IsNullOrWhiteSpace(response.TaskRequestId) then
                errors.Add("Task request ID is required for capability response")
        | _ -> ()
        
        // Expiration validation
        match message.ExpiresAt with
        | Some expiry when expiry <= DateTime.UtcNow ->
            warnings.Add("Message has already expired")
        | Some expiry when expiry <= DateTime.UtcNow.AddMinutes(5.0) ->
            warnings.Add("Message expires very soon")
        | _ -> ()
        
        // Suggestions
        if message.Priority = SemanticPriority.Critical && message.Urgency = MessageUrgency.Deferred then
            suggestions.Add("Critical priority messages should have urgent delivery")
        
        {
            IsValid = errors.Count = 0
            Errors = errors |> List.ofSeq
            Warnings = warnings |> List.ofSeq
            Suggestions = suggestions |> List.ofSeq
            Score = if errors.Count = 0 then 1.0 - (float warnings.Count * 0.1) else 0.0
        }
    
    /// Calculate message priority score
    let calculatePriorityScore (message: SemanticMessage) =
        let priorityScore = 
            match message.Priority with
            | SemanticPriority.Critical -> 100.0
            | SemanticPriority.High -> 80.0
            | SemanticPriority.Normal -> 60.0
            | SemanticPriority.Low -> 40.0
            | SemanticPriority.Background -> 20.0
        
        let urgencyScore = 
            match message.Urgency with
            | MessageUrgency.Immediate -> 50.0
            | MessageUrgency.Urgent -> 40.0
            | MessageUrgency.Standard -> 30.0
            | MessageUrgency.Deferred -> 20.0
            | MessageUrgency.Batch -> 10.0
        
        let ageScore = 
            let age = DateTime.UtcNow - message.CreatedAt
            max 0.0 (10.0 - age.TotalMinutes)
        
        priorityScore + urgencyScore + ageScore
    
    /// Extract keywords from text
    let extractKeywords (text: string) =
        text.Split([|' '; '.'; ','; ';'; '!'; '?'; '\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        |> Array.filter (fun word -> word.Length > 3)
        |> Array.map (fun word -> word.ToLower().Trim())
        |> Array.distinct
        |> List.ofArray
    
    /// Calculate semantic similarity between two texts
    let calculateSimilarity (text1: string) (text2: string) =
        let keywords1 = extractKeywords text1 |> Set.ofList
        let keywords2 = extractKeywords text2 |> Set.ofList
        
        let intersection = Set.intersect keywords1 keywords2
        let union = Set.union keywords1 keywords2
        
        if union.Count = 0 then 0.0
        else float intersection.Count / float union.Count
