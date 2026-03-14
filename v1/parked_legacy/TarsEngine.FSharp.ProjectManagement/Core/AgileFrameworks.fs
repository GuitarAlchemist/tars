namespace TarsEngine.FSharp.ProjectManagement.Core

open System
open System.Collections.Generic

/// Core types for agile methodologies and project management
module AgileFrameworks =
    
    /// Work item priority levels
    type Priority = 
        | Critical = 1
        | High = 2
        | Medium = 3
        | Low = 4
        | Backlog = 5
    
    /// Work item status for different frameworks
    type WorkItemStatus =
        // Kanban statuses
        | ToDo
        | InProgress
        | InReview
        | Done
        | Blocked
        // Scrum-specific statuses
        | ProductBacklog
        | SprintBacklog
        | InSprint
        | SprintReview
        | SprintDone
        // SAFe-specific statuses
        | ProgramBacklog
        | PIPlanning
        | InDevelopment
        | SystemDemo
        | PIComplete
    
    /// Story point estimation
    type StoryPoints = 
        | XS = 1
        | S = 2
        | M = 3
        | L = 5
        | XL = 8
        | XXL = 13
        | Epic = 21
    
    /// Work item types
    type WorkItemType =
        | UserStory
        | Task
        | Bug
        | Epic
        | Feature
        | Spike
        | TechnicalDebt
        | Improvement
    
    /// Team member with skills and capacity
    type TeamMember = {
        Id: Guid
        Name: string
        Email: string
        Role: string
        Skills: string list
        Capacity: float // Hours per sprint/week
        Availability: float // 0.0 to 1.0
        HourlyRate: decimal option
        TeamIds: Guid list
    }
    
    /// Work item definition
    type WorkItem = {
        Id: Guid
        Title: string
        Description: string
        Type: WorkItemType
        Status: WorkItemStatus
        Priority: Priority
        StoryPoints: StoryPoints option
        Assignee: Guid option
        Reporter: Guid
        CreatedDate: DateTime
        UpdatedDate: DateTime
        DueDate: DateTime option
        CompletedDate: DateTime option
        Tags: string list
        Dependencies: Guid list
        Attachments: string list
        Comments: Comment list
        AcceptanceCriteria: string list
        EstimatedHours: float option
        ActualHours: float option
        SprintId: Guid option
        EpicId: Guid option
    }
    
    and Comment = {
        Id: Guid
        AuthorId: Guid
        Content: string
        CreatedDate: DateTime
        IsInternal: bool
    }
    
    /// Sprint definition for Scrum
    type Sprint = {
        Id: Guid
        Name: string
        Goal: string
        StartDate: DateTime
        EndDate: DateTime
        TeamId: Guid
        Capacity: float // Total team capacity in hours
        CommittedStoryPoints: int
        CompletedStoryPoints: int
        WorkItems: Guid list
        Status: SprintStatus
        Retrospective: Retrospective option
    }
    
    and SprintStatus =
        | Planning
        | Active
        | Review
        | Completed
        | Cancelled
    
    and Retrospective = {
        WentWell: string list
        NeedsImprovement: string list
        ActionItems: ActionItem list
        TeamMood: int // 1-10 scale
        Date: DateTime
    }
    
    and ActionItem = {
        Id: Guid
        Description: string
        AssigneeId: Guid option
        DueDate: DateTime option
        Status: ActionItemStatus
    }
    
    and ActionItemStatus =
        | Open
        | InProgress
        | Completed
        | Cancelled
    
    /// Kanban board configuration
    type KanbanBoard = {
        Id: Guid
        Name: string
        Description: string
        TeamId: Guid
        Columns: KanbanColumn list
        WipLimits: Map<string, int>
        WorkItems: Guid list
        CreatedDate: DateTime
        UpdatedDate: DateTime
    }
    
    and KanbanColumn = {
        Id: Guid
        Name: string
        Position: int
        WipLimit: int option
        Definition: string
        Color: string
    }
    
    /// SAFe Program Increment
    type ProgramIncrement = {
        Id: Guid
        Name: string
        StartDate: DateTime
        EndDate: DateTime
        Objectives: PIObjective list
        Teams: Guid list
        Features: Guid list
        Status: PIStatus
        Confidence: int // 1-5 scale
        BusinessValue: int
    }
    
    and PIObjective = {
        Id: Guid
        Description: string
        BusinessValue: int
        Confidence: int
        AssignedTeam: Guid
        Features: Guid list
        Status: ObjectiveStatus
    }
    
    and PIStatus =
        | Planning
        | Execution
        | Innovation
        | SystemDemo
        | Inspect
        | Adapt
    
    and ObjectiveStatus =
        | NotStarted
        | InProgress
        | AtRisk
        | Completed
        | Deferred
    
    /// Team definition
    type AgileTeam = {
        Id: Guid
        Name: string
        Description: string
        Type: TeamType
        Members: Guid list
        ScrumMaster: Guid option
        ProductOwner: Guid option
        Capacity: float
        Velocity: float list // Historical velocity
        CurrentSprint: Guid option
        KanbanBoard: Guid option
        CreatedDate: DateTime
        Settings: TeamSettings
    }
    
    and TeamType =
        | ScrumTeam
        | KanbanTeam
        | SAFeTeam
        | HybridTeam
    
    and TeamSettings = {
        SprintLength: int // Days
        WorkingDays: DayOfWeek list
        WorkingHours: float
        TimeZone: string
        Methodology: AgileMethodology
        AutoAssignment: bool
        NotificationSettings: NotificationSettings
    }
    
    and AgileMethodology =
        | Scrum
        | Kanban
        | Scrumban
        | SAFe
        | Custom of string
    
    and NotificationSettings = {
        DailyStandupReminder: bool
        SprintEndingAlert: bool
        BlockedItemAlert: bool
        WipLimitExceeded: bool
        EmailNotifications: bool
        SlackIntegration: bool
    }
    
    /// Project definition
    type Project = {
        Id: Guid
        Name: string
        Description: string
        StartDate: DateTime
        EndDate: DateTime option
        Budget: decimal option
        Status: ProjectStatus
        Teams: Guid list
        Epics: Guid list
        Milestones: Milestone list
        Stakeholders: Stakeholder list
        RiskRegister: Risk list
        CreatedDate: DateTime
        UpdatedDate: DateTime
    }
    
    and ProjectStatus =
        | Initiation
        | Planning
        | Execution
        | Monitoring
        | Closure
        | OnHold
        | Cancelled
    
    and Milestone = {
        Id: Guid
        Name: string
        Description: string
        DueDate: DateTime
        Status: MilestoneStatus
        Dependencies: Guid list
        Deliverables: string list
    }
    
    and MilestoneStatus =
        | NotStarted
        | InProgress
        | Completed
        | Delayed
        | AtRisk
    
    and Stakeholder = {
        Id: Guid
        Name: string
        Role: string
        Influence: InfluenceLevel
        Interest: InterestLevel
        ContactInfo: string
        PreferredCommunication: CommunicationMethod
    }
    
    and InfluenceLevel = High | Medium | Low
    and InterestLevel = High | Medium | Low
    and CommunicationMethod = Email | Slack | Teams | Phone | InPerson
    
    and Risk = {
        Id: Guid
        Description: string
        Category: RiskCategory
        Probability: float // 0.0 to 1.0
        Impact: ImpactLevel
        MitigationPlan: string
        Owner: Guid
        Status: RiskStatus
        IdentifiedDate: DateTime
        ReviewDate: DateTime
    }
    
    and RiskCategory =
        | Technical
        | Schedule
        | Budget
        | Resource
        | External
        | Quality
    
    and ImpactLevel = Critical | High | Medium | Low
    and RiskStatus = Open | Mitigating | Closed | Accepted
    
    /// Metrics and reporting
    type TeamMetrics = {
        TeamId: Guid
        Period: DateRange
        Velocity: float
        Throughput: int
        LeadTime: TimeSpan
        CycleTime: TimeSpan
        DefectRate: float
        TeamSatisfaction: float
        BurndownData: BurndownPoint list
        CumulativeFlowData: FlowData list
    }
    
    and DateRange = {
        StartDate: DateTime
        EndDate: DateTime
    }
    
    and BurndownPoint = {
        Date: DateTime
        RemainingWork: float
        IdealRemaining: float
    }
    
    and FlowData = {
        Date: DateTime
        ColumnCounts: Map<string, int>
    }
    
    /// Gantt chart data
    type GanttTask = {
        Id: Guid
        Name: string
        StartDate: DateTime
        EndDate: DateTime
        Duration: TimeSpan
        Progress: float // 0.0 to 1.0
        Dependencies: Guid list
        AssignedResources: Guid list
        IsMilestone: bool
        IsCriticalPath: bool
        ParentTask: Guid option
        SubTasks: Guid list
    }
    
    type GanttChart = {
        Id: Guid
        ProjectId: Guid
        Name: string
        Tasks: GanttTask list
        CriticalPath: Guid list
        Baseline: GanttBaseline option
        CreatedDate: DateTime
        UpdatedDate: DateTime
    }
    
    and GanttBaseline = {
        Name: string
        CreatedDate: DateTime
        Tasks: Map<Guid, GanttTask>
    }
