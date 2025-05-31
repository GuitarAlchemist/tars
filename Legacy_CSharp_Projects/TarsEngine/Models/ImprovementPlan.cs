namespace TarsEngine.Models;

/// <summary>
/// Represents a plan for implementing improvements
/// </summary>
public class ImprovementPlan
{
    /// <summary>
    /// Gets or sets the plan ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the plan name
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the plan description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the plan creation date
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the plan completion date
    /// </summary>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Gets or sets the plan steps
    /// </summary>
    public List<ImprovementStep> Steps { get; set; } = new();

    /// <summary>
    /// Gets or sets the plan status
    /// </summary>
    public ImprovementPlanStatus Status { get; set; } = ImprovementPlanStatus.NotStarted;

    /// <summary>
    /// Gets or sets the plan owner
    /// </summary>
    public string Owner { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the plan tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
}

/// <summary>
/// Represents a step in an improvement plan
/// </summary>
public class ImprovementStep
{
    /// <summary>
    /// Gets or sets the step ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the step name
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the step description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the step dependencies
    /// </summary>
    public List<string> Dependencies { get; set; } = new();

    /// <summary>
    /// Gets or sets the estimated effort to implement the step
    /// </summary>
    public string EstimatedEffort { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the step status
    /// </summary>
    public ImprovementStepStatus Status { get; set; } = ImprovementStepStatus.NotStarted;

    /// <summary>
    /// Gets or sets the step start date
    /// </summary>
    public DateTime? StartedAt { get; set; }

    /// <summary>
    /// Gets or sets the step completion date
    /// </summary>
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Gets or sets the step assignee
    /// </summary>
    public string Assignee { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the opportunity IDs associated with this step
    /// </summary>
    public List<string> OpportunityIds { get; set; } = new();
}

/// <summary>
/// Represents the status of an improvement plan
/// </summary>
public enum ImprovementPlanStatus
{
    /// <summary>
    /// The plan has not been started
    /// </summary>
    NotStarted,

    /// <summary>
    /// The plan is in progress
    /// </summary>
    InProgress,

    /// <summary>
    /// The plan is on hold
    /// </summary>
    OnHold,

    /// <summary>
    /// The plan has been completed
    /// </summary>
    Completed,

    /// <summary>
    /// The plan has been cancelled
    /// </summary>
    Cancelled
}

/// <summary>
/// Represents the status of an improvement step
/// </summary>
public enum ImprovementStepStatus
{
    /// <summary>
    /// The step has not been started
    /// </summary>
    NotStarted,

    /// <summary>
    /// The step is in progress
    /// </summary>
    InProgress,

    /// <summary>
    /// The step is blocked
    /// </summary>
    Blocked,

    /// <summary>
    /// The step has been completed
    /// </summary>
    Completed,

    /// <summary>
    /// The step has been skipped
    /// </summary>
    Skipped
}
