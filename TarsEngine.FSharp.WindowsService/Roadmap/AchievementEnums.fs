namespace TarsEngine.FSharp.WindowsService.Roadmap

/// <summary>
/// Achievement status levels indicating the current state of an achievement.
/// Used to track progress through the development lifecycle.
/// </summary>
type AchievementStatus =
    /// <summary>
    /// Achievement has been defined but work has not yet started.
    /// Initial state for all new achievements.
    /// </summary>
    | NotStarted = 0
    
    /// <summary>
    /// Achievement is currently being worked on.
    /// Progress tracking and monitoring are active.
    /// </summary>
    | InProgress = 1
    
    /// <summary>
    /// Achievement has been successfully completed.
    /// All acceptance criteria have been met and deliverables are ready.
    /// </summary>
    | Completed = 2
    
    /// <summary>
    /// Achievement is temporarily blocked by external dependencies or issues.
    /// Requires intervention to resolve blockers before work can continue.
    /// </summary>
    | Blocked = 3
    
    /// <summary>
    /// Achievement has been cancelled and will not be completed.
    /// May be due to changing requirements or strategic decisions.
    /// </summary>
    | Cancelled = 4
    
    /// <summary>
    /// Achievement is temporarily paused but may resume in the future.
    /// Different from blocked - this is an intentional pause.
    /// </summary>
    | OnHold = 5

/// <summary>
/// Achievement priority levels for resource allocation and scheduling.
/// Higher priority achievements should be completed first.
/// </summary>
type AchievementPriority =
    /// <summary>
    /// Critical priority - must be completed immediately.
    /// System-blocking or security-critical achievements.
    /// </summary>
    | Critical = 0
    
    /// <summary>
    /// High priority - should be completed soon.
    /// Important features or significant improvements.
    /// </summary>
    | High = 1
    
    /// <summary>
    /// Medium priority - normal development work.
    /// Standard features and enhancements.
    /// </summary>
    | Medium = 2
    
    /// <summary>
    /// Low priority - can be deferred if needed.
    /// Nice-to-have features or minor improvements.
    /// </summary>
    | Low = 3
    
    /// <summary>
    /// Future priority - planned for later releases.
    /// Long-term goals and research items.
    /// </summary>
    | Future = 4

/// <summary>
/// Achievement categories for organization and reporting.
/// Helps classify achievements by their primary purpose.
/// </summary>
type AchievementCategory =
    /// <summary>
    /// Infrastructure and foundational work.
    /// Core systems, frameworks, and architectural components.
    /// </summary>
    | Infrastructure
    
    /// <summary>
    /// New features and functionality.
    /// User-facing capabilities and enhancements.
    /// </summary>
    | Features
    
    /// <summary>
    /// Quality assurance and testing work.
    /// Bug fixes, test coverage, and quality improvements.
    /// </summary>
    | Quality
    
    /// <summary>
    /// Performance optimization and efficiency.
    /// Speed improvements, resource optimization, and scalability.
    /// </summary>
    | Performance
    
    /// <summary>
    /// Documentation and knowledge management.
    /// User guides, API docs, and technical documentation.
    /// </summary>
    | Documentation
    
    /// <summary>
    /// Research and experimental work.
    /// Proof of concepts, technology evaluation, and innovation.
    /// </summary>
    | Research
    
    /// <summary>
    /// Integration with external systems.
    /// APIs, third-party services, and system connections.
    /// </summary>
    | Integration
    
    /// <summary>
    /// Security and compliance work.
    /// Security features, vulnerability fixes, and compliance requirements.
    /// </summary>
    | Security
    
    /// <summary>
    /// Deployment and operations work.
    /// CI/CD, monitoring, and operational improvements.
    /// </summary>
    | Deployment
    
    /// <summary>
    /// Maintenance and technical debt.
    /// Code cleanup, refactoring, and dependency updates.
    /// </summary>
    | Maintenance

/// <summary>
/// Achievement complexity assessment for effort estimation.
/// Indicates the technical difficulty and skill level required.
/// </summary>
type AchievementComplexity =
    /// <summary>
    /// Simple achievement requiring basic skills.
    /// Straightforward implementation with well-known patterns.
    /// Estimated effort: 1-4 hours.
    /// </summary>
    | Simple = 1
    
    /// <summary>
    /// Moderate achievement requiring intermediate skills.
    /// Some complexity but using familiar technologies and patterns.
    /// Estimated effort: 4-16 hours.
    /// </summary>
    | Moderate = 2
    
    /// <summary>
    /// Complex achievement requiring advanced skills.
    /// Significant complexity with multiple components and dependencies.
    /// Estimated effort: 16-40 hours.
    /// </summary>
    | Complex = 3
    
    /// <summary>
    /// Expert-level achievement requiring specialized knowledge.
    /// High complexity with architectural implications and advanced techniques.
    /// Estimated effort: 40-80 hours.
    /// </summary>
    | Expert = 4
    
    /// <summary>
    /// Research-level achievement requiring investigation and experimentation.
    /// Unknown complexity with potential for significant discovery or innovation.
    /// Estimated effort: 80+ hours with high uncertainty.
    /// </summary>
    | Research = 5

/// <summary>
/// Types of achievement updates for tracking changes.
/// Used in the achievement update event system.
/// </summary>
type AchievementUpdateType =
    /// <summary>
    /// Achievement status has changed (e.g., NotStarted -> InProgress).
    /// Includes automatic timestamp updates and state transition validation.
    /// </summary>
    | StatusChange
    
    /// <summary>
    /// Achievement progress percentage has been updated.
    /// Includes validation to ensure progress is between 0-100%.
    /// </summary>
    | ProgressUpdate
    
    /// <summary>
    /// Achievement priority has been changed.
    /// May trigger re-scheduling and resource reallocation.
    /// </summary>
    | PriorityChange
    
    /// <summary>
    /// Achievement has been assigned to a different agent.
    /// Includes workload balancing and capability matching.
    /// </summary>
    | AssignmentChange
    
    /// <summary>
    /// A new dependency has been added to the achievement.
    /// May affect scheduling and completion estimates.
    /// </summary>
    | DependencyAdded
    
    /// <summary>
    /// A dependency has been removed from the achievement.
    /// May allow earlier scheduling and completion.
    /// </summary>
    | DependencyRemoved
    
    /// <summary>
    /// A new blocker has been identified for the achievement.
    /// Requires immediate attention and resolution planning.
    /// </summary>
    | BlockerAdded
    
    /// <summary>
    /// A blocker has been resolved and removed.
    /// Achievement can now proceed with normal development.
    /// </summary>
    | BlockerRemoved
    
    /// <summary>
    /// Time or effort estimate has been revised.
    /// May affect milestone and roadmap completion dates.
    /// </summary>
    | EstimateRevised
    
    /// <summary>
    /// Expected or target completion date has been changed.
    /// May trigger schedule adjustments and resource planning.
    /// </summary>
    | CompletionDateChanged
    
    /// <summary>
    /// Additional notes or comments have been added.
    /// Provides context and detailed information about progress.
    /// </summary>
    | NotesAdded

/// <summary>
/// Finding severity levels for roadmap analysis results.
/// Indicates the urgency and impact of identified issues.
/// </summary>
type FindingSeverity =
    /// <summary>
    /// Critical finding requiring immediate action.
    /// System-blocking issues or major risks to project success.
    /// </summary>
    | Critical
    
    /// <summary>
    /// High severity finding requiring prompt attention.
    /// Significant issues that may impact timeline or quality.
    /// </summary>
    | High
    
    /// <summary>
    /// Medium severity finding requiring attention.
    /// Moderate issues that should be addressed in normal workflow.
    /// </summary>
    | Medium
    
    /// <summary>
    /// Low severity finding for future consideration.
    /// Minor issues or improvement opportunities.
    /// </summary>
    | Low
    
    /// <summary>
    /// Informational finding for awareness.
    /// Observations or metrics without immediate action required.
    /// </summary>
    | Info

/// <summary>
/// Risk levels for overall roadmap assessment.
/// Used in risk analysis and mitigation planning.
/// </summary>
type RiskLevel =
    /// <summary>
    /// Very high risk - immediate intervention required.
    /// Project success is in serious jeopardy.
    /// </summary>
    | VeryHigh
    
    /// <summary>
    /// High risk - urgent attention needed.
    /// Significant threats to timeline or quality.
    /// </summary>
    | High
    
    /// <summary>
    /// Medium risk - monitoring and mitigation recommended.
    /// Manageable risks with known mitigation strategies.
    /// </summary>
    | Medium
    
    /// <summary>
    /// Low risk - minimal impact expected.
    /// Minor risks that are well-controlled.
    /// </summary>
    | Low
    
    /// <summary>
    /// Very low risk - negligible impact.
    /// Risks are well-managed with minimal probability of impact.
    /// </summary>
    | VeryLow

/// <summary>
/// Trend direction indicators for analysis and reporting.
/// Used to show whether metrics are improving or declining over time.
/// </summary>
type TrendDirection =
    /// <summary>
    /// Trend is improving over time.
    /// Positive direction with measurable improvement.
    /// </summary>
    | Improving
    
    /// <summary>
    /// Trend is stable with minimal change.
    /// Consistent performance within acceptable ranges.
    /// </summary>
    | Stable
    
    /// <summary>
    /// Trend is declining over time.
    /// Negative direction requiring attention and intervention.
    /// </summary>
    | Declining
    
    /// <summary>
    /// Trend is volatile with significant fluctuations.
    /// Inconsistent performance requiring investigation.
    /// </summary>
    | Volatile
    
    /// <summary>
    /// Trend direction cannot be determined.
    /// Insufficient data or unclear patterns.
    /// </summary>
    | Unknown

/// <summary>
/// Recommendation types for autonomous roadmap management.
/// Used by the analysis agent to suggest improvements and actions.
/// </summary>
type RecommendationType =
    /// <summary>
    /// Create a new achievement to address identified needs.
    /// Includes scope definition and priority assignment.
    /// </summary>
    | CreateNew
    
    /// <summary>
    /// Update an existing achievement with new information.
    /// May include scope, timeline, or resource changes.
    /// </summary>
    | UpdateExisting
    
    /// <summary>
    /// Change the priority of an existing achievement.
    /// Based on changing requirements or strategic importance.
    /// </summary>
    | ReprioritizeExisting
    
    /// <summary>
    /// Split a large achievement into smaller, manageable pieces.
    /// Improves tracking granularity and reduces risk.
    /// </summary>
    | SplitAchievement
    
    /// <summary>
    /// Merge related achievements for efficiency.
    /// Reduces overhead and improves coordination.
    /// </summary>
    | MergeAchievements
    
    /// <summary>
    /// Add a dependency relationship between achievements.
    /// Ensures proper sequencing and resource allocation.
    /// </summary>
    | AddDependency
    
    /// <summary>
    /// Remove an unnecessary dependency relationship.
    /// Enables parallel execution and faster completion.
    /// </summary>
    | RemoveDependency
    
    /// <summary>
    /// Reassign achievement to a different agent.
    /// Based on workload balancing or skill matching.
    /// </summary>
    | ReassignAgent
    
    /// <summary>
    /// Adjust time or effort estimates based on new information.
    /// Improves planning accuracy and resource allocation.
    /// </summary>
    | AdjustEstimate
    
    /// <summary>
    /// Set or update deadline for achievement completion.
    /// Ensures alignment with project milestones and commitments.
    /// </summary>
    | SetDeadline
