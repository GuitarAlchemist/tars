namespace TarsEngine.FSharp.WindowsService.Roadmap

open System
open System.Collections.Generic

/// <summary>
/// Individual achievement or milestone representing a specific deliverable or goal.
/// Contains comprehensive tracking information for autonomous progress management.
/// </summary>
[<CLIMutable>]
type Achievement = {
    /// <summary>
    /// Unique identifier for the achievement.
    /// Used for referencing in dependencies, updates, and tracking.
    /// </summary>
    Id: string

    /// <summary>
    /// Human-readable title describing the achievement.
    /// Should be concise but descriptive enough to understand the goal.
    /// </summary>
    Title: string

    /// <summary>
    /// Detailed description of what needs to be accomplished.
    /// Includes acceptance criteria, scope, and any special requirements.
    /// </summary>
    Description: string

    /// <summary>
    /// Category classification for organization and reporting.
    /// Helps group related achievements and analyze progress by type.
    /// </summary>
    Category: AchievementCategory

    /// <summary>
    /// Priority level for resource allocation and scheduling.
    /// Higher priority achievements should be completed first.
    /// </summary>
    Priority: AchievementPriority

    /// <summary>
    /// Current status indicating progress through the development lifecycle.
    /// Automatically updated based on progress and external events.
    /// </summary>
    Status: AchievementStatus

    /// <summary>
    /// Complexity assessment for effort estimation and skill matching.
    /// Used by autonomous agents for task assignment and planning.
    /// </summary>
    Complexity: AchievementComplexity

    /// <summary>
    /// Estimated hours required to complete the achievement.
    /// Used for planning, scheduling, and resource allocation.
    /// </summary>
    EstimatedHours: float

    /// <summary>
    /// Actual hours spent on the achievement (when available).
    /// Used for improving estimation accuracy and performance analysis.
    /// </summary>
    ActualHours: float option

    /// <summary>
    /// Completion percentage (0.0 to 100.0).
    /// Provides granular progress tracking within the current status.
    /// </summary>
    CompletionPercentage: float

    /// <summary>
    /// List of achievement IDs that must be completed before this one.
    /// Used for scheduling and ensuring proper development sequence.
    /// </summary>
    Dependencies: string list

    /// <summary>
    /// List of current blockers preventing progress.
    /// Automatically monitored by analysis agents for resolution.
    /// </summary>
    Blockers: string list

    /// <summary>
    /// Flexible tags for categorization and filtering.
    /// Supports custom organization and search capabilities.
    /// </summary>
    Tags: string list

    /// <summary>
    /// ID of the agent currently assigned to work on this achievement.
    /// Used for workload balancing and progress tracking.
    /// </summary>
    AssignedAgent: string option

    /// <summary>
    /// Timestamp when the achievement was created.
    /// Immutable record of initial planning and scope definition.
    /// </summary>
    CreatedAt: DateTime

    /// <summary>
    /// Timestamp of the most recent update to any field.
    /// Automatically updated on any change for audit and analysis.
    /// </summary>
    UpdatedAt: DateTime

    /// <summary>
    /// Timestamp when work actually began on the achievement.
    /// Set automatically when status changes to InProgress.
    /// </summary>
    StartedAt: DateTime option

    /// <summary>
    /// Timestamp when the achievement was completed.
    /// Set automatically when status changes to Completed.
    /// </summary>
    CompletedAt: DateTime option

    /// <summary>
    /// Target completion date for planning and commitment tracking.
    /// May be set by planning agents or external requirements.
    /// </summary>
    DueDate: DateTime option

    /// <summary>
    /// Flexible metadata storage for additional information.
    /// Supports custom fields, quality scores, and integration data.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Roadmap milestone containing multiple achievements.
/// Represents a significant deliverable or project phase with grouped achievements.
/// </summary>
[<CLIMutable>]
type Milestone = {
    /// <summary>
    /// Unique identifier for the milestone.
    /// Used for referencing in dependencies and progress tracking.
    /// </summary>
    Id: string

    /// <summary>
    /// Human-readable title describing the milestone.
    /// Should clearly indicate the major deliverable or goal.
    /// </summary>
    Title: string

    /// <summary>
    /// Detailed description of the milestone scope and objectives.
    /// Includes success criteria and key deliverables.
    /// </summary>
    Description: string

    /// <summary>
    /// Version identifier for the milestone deliverable.
    /// Follows semantic versioning or project-specific versioning scheme.
    /// </summary>
    Version: string

    /// <summary>
    /// List of achievements that must be completed for this milestone.
    /// Automatically aggregated for progress and metrics calculation.
    /// </summary>
    Achievements: Achievement list

    /// <summary>
    /// Current status derived from constituent achievement progress.
    /// Automatically calculated based on achievement completion.
    /// </summary>
    Status: AchievementStatus

    /// <summary>
    /// Priority level for milestone scheduling and resource allocation.
    /// Influences the priority of constituent achievements.
    /// </summary>
    Priority: AchievementPriority

    /// <summary>
    /// Total estimated hours calculated from all achievements.
    /// Automatically aggregated from constituent achievements.
    /// </summary>
    EstimatedHours: float

    /// <summary>
    /// Total actual hours spent on all achievements.
    /// Automatically calculated from completed achievements.
    /// </summary>
    ActualHours: float option

    /// <summary>
    /// Overall completion percentage (0.0 to 100.0).
    /// Calculated from achievement progress and completion status.
    /// </summary>
    CompletionPercentage: float

    /// <summary>
    /// List of milestone IDs that must be completed before this one.
    /// Used for milestone sequencing and dependency management.
    /// </summary>
    Dependencies: string list

    /// <summary>
    /// Timestamp when the milestone was created.
    /// Immutable record of initial milestone definition.
    /// </summary>
    CreatedAt: DateTime

    /// <summary>
    /// Timestamp of the most recent update to the milestone.
    /// Updated when achievements change or milestone is modified.
    /// </summary>
    UpdatedAt: DateTime

    /// <summary>
    /// Timestamp when work began on the first achievement.
    /// Set automatically when first achievement starts.
    /// </summary>
    StartedAt: DateTime option

    /// <summary>
    /// Timestamp when all achievements were completed.
    /// Set automatically when milestone reaches 100% completion.
    /// </summary>
    CompletedAt: DateTime option

    /// <summary>
    /// Target completion date for milestone delivery.
    /// Used for planning and commitment tracking.
    /// </summary>
    TargetDate: DateTime option

    /// <summary>
    /// Flexible metadata storage for milestone-specific information.
    /// Supports custom fields, quality metrics, and integration data.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Roadmap phase containing multiple milestones.
/// Represents a major development phase with related milestones and strategic objectives.
/// </summary>
[<CLIMutable>]
type RoadmapPhase = {
    /// <summary>
    /// Unique identifier for the roadmap phase.
    /// Used for referencing in dependencies and progress tracking.
    /// </summary>
    Id: string

    /// <summary>
    /// Human-readable title describing the phase.
    /// Should clearly indicate the strategic objective or development focus.
    /// </summary>
    Title: string

    /// <summary>
    /// Detailed description of the phase scope and strategic goals.
    /// Includes key objectives, success criteria, and expected outcomes.
    /// </summary>
    Description: string

    /// <summary>
    /// List of milestones that comprise this development phase.
    /// Automatically aggregated for phase-level progress and metrics.
    /// </summary>
    Milestones: Milestone list

    /// <summary>
    /// Current status derived from constituent milestone progress.
    /// Automatically calculated based on milestone completion.
    /// </summary>
    Status: AchievementStatus

    /// <summary>
    /// Priority level for phase scheduling and strategic importance.
    /// Influences resource allocation and milestone prioritization.
    /// </summary>
    Priority: AchievementPriority

    /// <summary>
    /// Total estimated hours calculated from all milestones.
    /// Automatically aggregated from constituent milestones and achievements.
    /// </summary>
    EstimatedHours: float

    /// <summary>
    /// Total actual hours spent on all milestones.
    /// Automatically calculated from completed milestones and achievements.
    /// </summary>
    ActualHours: float option

    /// <summary>
    /// Overall completion percentage (0.0 to 100.0).
    /// Calculated from milestone progress and completion status.
    /// </summary>
    CompletionPercentage: float

    /// <summary>
    /// Timestamp when the phase was created.
    /// Immutable record of initial phase definition and planning.
    /// </summary>
    CreatedAt: DateTime

    /// <summary>
    /// Timestamp of the most recent update to the phase.
    /// Updated when milestones change or phase is modified.
    /// </summary>
    UpdatedAt: DateTime

    /// <summary>
    /// Timestamp when work began on the first milestone.
    /// Set automatically when first milestone starts.
    /// </summary>
    StartedAt: DateTime option

    /// <summary>
    /// Timestamp when all milestones were completed.
    /// Set automatically when phase reaches 100% completion.
    /// </summary>
    CompletedAt: DateTime option

    /// <summary>
    /// Target completion date for phase delivery.
    /// Used for strategic planning and roadmap scheduling.
    /// </summary>
    TargetDate: DateTime option

    /// <summary>
    /// Flexible metadata storage for phase-specific information.
    /// Supports custom fields, strategic metrics, and planning data.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Complete TARS roadmap representing the entire development plan.
/// Root container for all phases, milestones, and achievements with comprehensive tracking.
/// </summary>
[<CLIMutable>]
type TarsRoadmap = {
    /// <summary>
    /// Unique identifier for the roadmap.
    /// Used for referencing, storage, and cross-roadmap dependencies.
    /// </summary>
    Id: string

    /// <summary>
    /// Human-readable title describing the roadmap.
    /// Should clearly indicate the project or initiative scope.
    /// </summary>
    Title: string

    /// <summary>
    /// Detailed description of the roadmap scope and strategic vision.
    /// Includes project goals, success criteria, and strategic context.
    /// </summary>
    Description: string

    /// <summary>
    /// Version identifier for the roadmap.
    /// Follows semantic versioning for roadmap evolution tracking.
    /// </summary>
    Version: string

    /// <summary>
    /// List of development phases that comprise the complete roadmap.
    /// Automatically aggregated for roadmap-level progress and metrics.
    /// </summary>
    Phases: RoadmapPhase list

    /// <summary>
    /// Current status derived from constituent phase progress.
    /// Automatically calculated based on phase and milestone completion.
    /// </summary>
    Status: AchievementStatus

    /// <summary>
    /// Total estimated hours calculated from all phases.
    /// Automatically aggregated from all constituent phases, milestones, and achievements.
    /// </summary>
    EstimatedHours: float

    /// <summary>
    /// Total actual hours spent on the entire roadmap.
    /// Automatically calculated from all completed work across phases.
    /// </summary>
    ActualHours: float option

    /// <summary>
    /// Overall completion percentage (0.0 to 100.0).
    /// Calculated from phase progress and strategic milestone completion.
    /// </summary>
    CompletionPercentage: float

    /// <summary>
    /// Timestamp when the roadmap was created.
    /// Immutable record of initial roadmap conception and planning.
    /// </summary>
    CreatedAt: DateTime

    /// <summary>
    /// Timestamp of the most recent update to the roadmap.
    /// Updated when phases change or roadmap structure is modified.
    /// </summary>
    UpdatedAt: DateTime

    /// <summary>
    /// Timestamp when work began on the first phase.
    /// Set automatically when first phase and achievement start.
    /// </summary>
    StartedAt: DateTime option

    /// <summary>
    /// Timestamp when all phases were completed.
    /// Set automatically when roadmap reaches 100% completion.
    /// </summary>
    CompletedAt: DateTime option

    /// <summary>
    /// Target completion date for the entire roadmap.
    /// Used for strategic planning, commitments, and stakeholder communication.
    /// </summary>
    TargetDate: DateTime option

    /// <summary>
    /// Flexible metadata storage for roadmap-specific information.
    /// Supports custom fields, strategic metrics, stakeholder data, and integration information.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Achievement update event
/// </summary>
type AchievementUpdate = {
    AchievementId: string
    UpdateType: AchievementUpdateType
    OldValue: obj option
    NewValue: obj option
    UpdatedBy: string
    UpdatedAt: DateTime
    Notes: string option
}

/// <summary>
/// Types of achievement updates
/// </summary>
and AchievementUpdateType =
    | StatusChange
    | ProgressUpdate
    | PriorityChange
    | AssignmentChange
    | DependencyAdded
    | DependencyRemoved
    | BlockerAdded
    | BlockerRemoved
    | EstimateRevised
    | CompletionDateChanged
    | NotesAdded

/// <summary>
/// Achievement analytics and metrics
/// </summary>
type AchievementMetrics = {
    TotalAchievements: int
    CompletedAchievements: int
    InProgressAchievements: int
    BlockedAchievements: int
    OverdueAchievements: int
    CompletionRate: float
    AverageCompletionTime: TimeSpan
    EstimationAccuracy: float
    VelocityTrend: float
    BurndownRate: float
    PredictedCompletionDate: DateTime option
    RiskFactors: string list
}

/// <summary>
/// Roadmap analytics
/// </summary>
type RoadmapAnalytics = {
    RoadmapId: string
    GeneratedAt: DateTime
    OverallMetrics: AchievementMetrics
    PhaseMetrics: Map<string, AchievementMetrics>
    MilestoneMetrics: Map<string, AchievementMetrics>
    CategoryMetrics: Map<AchievementCategory, AchievementMetrics>
    PriorityMetrics: Map<AchievementPriority, AchievementMetrics>
    TrendAnalysis: TrendAnalysis
    Recommendations: string list
}

/// <summary>
/// Trend analysis data
/// </summary>
and TrendAnalysis = {
    VelocityTrend: TrendDirection
    QualityTrend: TrendDirection
    ComplexityTrend: TrendDirection
    EstimationTrend: TrendDirection
    ProductivityTrend: TrendDirection
    HistoricalData: Map<DateTime, AchievementMetrics>
    Predictions: Map<DateTime, AchievementMetrics>
}

/// <summary>
/// Trend direction
/// </summary>
and TrendDirection =
    | Improving
    | Stable
    | Declining
    | Volatile
    | Unknown

/// <summary>
/// Achievement search criteria
/// </summary>
type AchievementSearchCriteria = {
    Keywords: string list
    Categories: AchievementCategory list
    Statuses: AchievementStatus list
    Priorities: AchievementPriority list
    AssignedAgents: string list
    Tags: string list
    DateRange: (DateTime * DateTime) option
    CompletionRange: (float * float) option
    EstimatedHoursRange: (float * float) option
}

/// <summary>
/// Achievement search result
/// </summary>
type AchievementSearchResult = {
    Achievement: Achievement
    RelevanceScore: float
    MatchingCriteria: string list
    Context: string option
}

/// <summary>
/// Roadmap validation result
/// </summary>
type RoadmapValidationResult = {
    IsValid: bool
    Errors: string list
    Warnings: string list
    Suggestions: string list
    ValidationTime: TimeSpan
}

/// <summary>
/// Achievement recommendation
/// </summary>
type AchievementRecommendation = {
    Type: RecommendationType
    Achievement: Achievement option
    Priority: AchievementPriority
    Reasoning: string list
    EstimatedImpact: float
    EstimatedEffort: float
    Dependencies: string list
    Risks: string list
    Benefits: string list
}

/// <summary>
/// Recommendation types
/// </summary>
and RecommendationType =
    | CreateNew
    | UpdateExisting
    | ReprioritizeExisting
    | SplitAchievement
    | MergeAchievements
    | AddDependency
    | RemoveDependency
    | ReassignAgent
    | AdjustEstimate
    | SetDeadline

/// <summary>
/// Roadmap helper functions
/// </summary>
module RoadmapHelpers =
    
    /// Create a new achievement
    let createAchievement title description category priority complexity estimatedHours =
        {
            Id = Guid.NewGuid().ToString()
            Title = title
            Description = description
            Category = category
            Priority = priority
            Status = AchievementStatus.NotStarted
            Complexity = complexity
            EstimatedHours = estimatedHours
            ActualHours = None
            CompletionPercentage = 0.0
            Dependencies = []
            Blockers = []
            Tags = []
            AssignedAgent = None
            CreatedAt = DateTime.UtcNow
            UpdatedAt = DateTime.UtcNow
            StartedAt = None
            CompletedAt = None
            DueDate = None
            Metadata = Map.empty
        }
    
    /// Create a new milestone
    let createMilestone title description version achievements =
        let totalHours = achievements |> List.sumBy (fun a -> a.EstimatedHours)
        let completedHours = achievements |> List.sumBy (fun a -> a.ActualHours |> Option.defaultValue 0.0)
        let completionPercentage = 
            if totalHours > 0.0 then (completedHours / totalHours) * 100.0 else 0.0
        
        {
            Id = Guid.NewGuid().ToString()
            Title = title
            Description = description
            Version = version
            Achievements = achievements
            Status = if completionPercentage >= 100.0 then AchievementStatus.Completed 
                    elif completionPercentage > 0.0 then AchievementStatus.InProgress 
                    else AchievementStatus.NotStarted
            Priority = AchievementPriority.Medium
            EstimatedHours = totalHours
            ActualHours = Some completedHours
            CompletionPercentage = completionPercentage
            Dependencies = []
            CreatedAt = DateTime.UtcNow
            UpdatedAt = DateTime.UtcNow
            StartedAt = None
            CompletedAt = None
            TargetDate = None
            Metadata = Map.empty
        }
    
    /// Update achievement status
    let updateAchievementStatus achievement newStatus updatedBy =
        let now = DateTime.UtcNow
        let updatedAchievement = 
            { achievement with 
                Status = newStatus
                UpdatedAt = now
                StartedAt = if newStatus = AchievementStatus.InProgress && achievement.StartedAt.IsNone then Some now else achievement.StartedAt
                CompletedAt = if newStatus = AchievementStatus.Completed then Some now else None
                CompletionPercentage = if newStatus = AchievementStatus.Completed then 100.0 else achievement.CompletionPercentage
            }
        
        let update = {
            AchievementId = achievement.Id
            UpdateType = StatusChange
            OldValue = Some (achievement.Status :> obj)
            NewValue = Some (newStatus :> obj)
            UpdatedBy = updatedBy
            UpdatedAt = now
            Notes = None
        }
        
        (updatedAchievement, update)
    
    /// Update achievement progress
    let updateAchievementProgress achievement newProgress updatedBy =
        let now = DateTime.UtcNow
        let clampedProgress = max 0.0 (min 100.0 newProgress)
        let newStatus = 
            if clampedProgress >= 100.0 then AchievementStatus.Completed
            elif clampedProgress > 0.0 then AchievementStatus.InProgress
            else achievement.Status
        
        let updatedAchievement = 
            { achievement with 
                CompletionPercentage = clampedProgress
                Status = newStatus
                UpdatedAt = now
                StartedAt = if newStatus = AchievementStatus.InProgress && achievement.StartedAt.IsNone then Some now else achievement.StartedAt
                CompletedAt = if newStatus = AchievementStatus.Completed then Some now else None
            }
        
        let update = {
            AchievementId = achievement.Id
            UpdateType = ProgressUpdate
            OldValue = Some (achievement.CompletionPercentage :> obj)
            NewValue = Some (clampedProgress :> obj)
            UpdatedBy = updatedBy
            UpdatedAt = now
            Notes = None
        }
        
        (updatedAchievement, update)
    
    /// Calculate achievement metrics
    let calculateAchievementMetrics achievements =
        let total = achievements |> List.length
        let completed = achievements |> List.filter (fun a -> a.Status = AchievementStatus.Completed) |> List.length
        let inProgress = achievements |> List.filter (fun a -> a.Status = AchievementStatus.InProgress) |> List.length
        let blocked = achievements |> List.filter (fun a -> a.Status = AchievementStatus.Blocked) |> List.length
        let overdue = achievements |> List.filter (fun a -> 
            a.DueDate.IsSome && a.DueDate.Value < DateTime.UtcNow && a.Status <> AchievementStatus.Completed) |> List.length
        
        let completionRate = if total > 0 then float completed / float total * 100.0 else 0.0
        
        let completedWithTimes = 
            achievements 
            |> List.filter (fun a -> a.Status = AchievementStatus.Completed && a.StartedAt.IsSome && a.CompletedAt.IsSome)
        
        let averageCompletionTime = 
            if completedWithTimes.IsEmpty then TimeSpan.Zero
            else
                let totalTime = 
                    completedWithTimes 
                    |> List.sumBy (fun a -> (a.CompletedAt.Value - a.StartedAt.Value).TotalMilliseconds)
                TimeSpan.FromMilliseconds(totalTime / float completedWithTimes.Length)
        
        let estimationAccuracy = 
            let achievementsWithEstimates = 
                achievements 
                |> List.filter (fun a -> a.ActualHours.IsSome && a.EstimatedHours > 0.0)
            
            if achievementsWithEstimates.IsEmpty then 0.0
            else
                let accuracySum = 
                    achievementsWithEstimates
                    |> List.sumBy (fun a -> 
                        let actual = a.ActualHours.Value
                        let estimated = a.EstimatedHours
                        1.0 - abs(actual - estimated) / max actual estimated)
                accuracySum / float achievementsWithEstimates.Length * 100.0
        
        {
            TotalAchievements = total
            CompletedAchievements = completed
            InProgressAchievements = inProgress
            BlockedAchievements = blocked
            OverdueAchievements = overdue
            CompletionRate = completionRate
            AverageCompletionTime = averageCompletionTime
            EstimationAccuracy = estimationAccuracy
            VelocityTrend = 0.0 // Would be calculated from historical data
            BurndownRate = 0.0 // Would be calculated from progress over time
            PredictedCompletionDate = None // Would be calculated based on current velocity
            RiskFactors = []
        }
    
    /// Validate roadmap structure
    let validateRoadmap roadmap =
        let errors = ResizeArray<string>()
        let warnings = ResizeArray<string>()
        let suggestions = ResizeArray<string>()
        
        // Basic validation
        if String.IsNullOrWhiteSpace(roadmap.Title) then
            errors.Add("Roadmap title is required")
        
        if roadmap.Phases.IsEmpty then
            warnings.Add("Roadmap has no phases defined")
        
        // Phase validation
        for phase in roadmap.Phases do
            if String.IsNullOrWhiteSpace(phase.Title) then
                errors.Add($"Phase {phase.Id} is missing a title")
            
            if phase.Milestones.IsEmpty then
                warnings.Add($"Phase '{phase.Title}' has no milestones")
            
            // Milestone validation
            for milestone in phase.Milestones do
                if String.IsNullOrWhiteSpace(milestone.Title) then
                    errors.Add($"Milestone {milestone.Id} is missing a title")
                
                if milestone.Achievements.IsEmpty then
                    warnings.Add($"Milestone '{milestone.Title}' has no achievements")
                
                // Achievement validation
                for achievement in milestone.Achievements do
                    if String.IsNullOrWhiteSpace(achievement.Title) then
                        errors.Add($"Achievement {achievement.Id} is missing a title")
                    
                    if achievement.EstimatedHours <= 0.0 then
                        warnings.Add($"Achievement '{achievement.Title}' has no time estimate")
                    
                    // Dependency validation
                    for dependency in achievement.Dependencies do
                        let dependencyExists = 
                            roadmap.Phases
                            |> List.collect (fun p -> p.Milestones)
                            |> List.collect (fun m -> m.Achievements)
                            |> List.exists (fun a -> a.Id = dependency)
                        
                        if not dependencyExists then
                            errors.Add($"Achievement '{achievement.Title}' has invalid dependency: {dependency}")
        
        // Suggestions
        if roadmap.TargetDate.IsNone then
            suggestions.Add("Consider setting a target completion date for the roadmap")
        
        let totalEstimatedHours = 
            roadmap.Phases
            |> List.collect (fun p -> p.Milestones)
            |> List.collect (fun m -> m.Achievements)
            |> List.sumBy (fun a -> a.EstimatedHours)
        
        if totalEstimatedHours > 2000.0 then
            suggestions.Add("Roadmap is very large (>2000 hours). Consider breaking into smaller roadmaps")
        
        {
            IsValid = errors.Count = 0
            Errors = errors |> List.ofSeq
            Warnings = warnings |> List.ofSeq
            Suggestions = suggestions |> List.ofSeq
            ValidationTime = TimeSpan.FromMilliseconds(10.0)
        }
