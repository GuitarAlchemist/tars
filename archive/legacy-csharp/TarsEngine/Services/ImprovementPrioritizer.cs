using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for prioritizing improvement opportunities
/// </summary>
public class ImprovementPrioritizer
{
    private readonly ILogger<ImprovementPrioritizer> _logger;

    /// <summary>
    /// Initializes a new instance of the ImprovementPrioritizer class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    public ImprovementPrioritizer(ILogger<ImprovementPrioritizer> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Prioritizes improvement opportunities
    /// </summary>
    /// <param name="opportunities">The improvement opportunities</param>
    /// <param name="options">The prioritization options</param>
    /// <returns>The prioritized improvement opportunities</returns>
    public List<ImprovementOpportunity> PrioritizeOpportunities(List<ImprovementOpportunity> opportunities, PrioritizationOptions options)
    {
        try
        {
            _logger.LogInformation($"Prioritizing {opportunities.Count} improvement opportunities");

            // Create a copy of the opportunities
            var prioritizedOpportunities = new List<ImprovementOpportunity>(opportunities);

            // Apply filters
            if (options.CategoryFilters.Any())
            {
                prioritizedOpportunities = prioritizedOpportunities.Where(o => options.CategoryFilters.Contains(o.Category)).ToList();
            }

            if (options.TagFilters.Any())
            {
                prioritizedOpportunities = prioritizedOpportunities.Where(o => o.Tags.Any(t => options.TagFilters.Contains(t))).ToList();
            }

            if (options.MinPriority > 0)
            {
                prioritizedOpportunities = prioritizedOpportunities.Where(o => o.Priority >= options.MinPriority).ToList();
            }

            if (options.MaxEffort > 0)
            {
                prioritizedOpportunities = prioritizedOpportunities.Where(o => o.EstimatedEffort <= options.MaxEffort).ToList();
            }

            // Calculate scores
            foreach (var opportunity in prioritizedOpportunities)
            {
                var score = CalculateScore(opportunity, options);
                opportunity.Tags.Add($"Score:{score:F2}");
            }

            // Sort by score (descending)
            prioritizedOpportunities = prioritizedOpportunities
                .OrderByDescending(o => CalculateScore(o, options))
                .ToList();

            // Limit the number of opportunities
            if (options.MaxResults > 0 && prioritizedOpportunities.Count > options.MaxResults)
            {
                prioritizedOpportunities = prioritizedOpportunities.Take(options.MaxResults).ToList();
            }

            return prioritizedOpportunities;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error prioritizing improvement opportunities");
            return opportunities;
        }
    }

    /// <summary>
    /// Creates an improvement plan from prioritized opportunities
    /// </summary>
    /// <param name="opportunities">The prioritized improvement opportunities</param>
    /// <param name="planName">The plan name</param>
    /// <param name="planDescription">The plan description</param>
    /// <param name="owner">The plan owner</param>
    /// <returns>The improvement plan</returns>
    public ImprovementPlan CreateImprovementPlan(List<ImprovementOpportunity> opportunities, string planName, string planDescription, string owner)
    {
        try
        {
            _logger.LogInformation($"Creating improvement plan '{planName}' with {opportunities.Count} opportunities");

            // Create the plan
            var plan = new ImprovementPlan
            {
                Name = planName,
                Description = planDescription,
                Owner = owner,
                CreatedAt = DateTime.UtcNow,
                Status = ImprovementPlanStatus.NotStarted,
                Tags = new List<string>()
            };

            // Group opportunities by category
            var opportunitiesByCategory = opportunities.GroupBy(o => o.Category).ToList();

            // Create steps for each category
            foreach (var group in opportunitiesByCategory)
            {
                var categoryName = group.Key;
                var categoryOpportunities = group.ToList();

                var step = new ImprovementStep
                {
                    Name = $"Improve {categoryName}",
                    Description = $"Address {categoryOpportunities.Count} {categoryName} issues",
                    EstimatedEffort = $"{categoryOpportunities.Sum(o => o.EstimatedEffort):F1} hours",
                    Status = ImprovementStepStatus.NotStarted,
                    Assignee = owner,
                    OpportunityIds = categoryOpportunities.Select(o => o.Id).ToList()
                };

                plan.Steps.Add(step);
            }

            // Add tags to the plan
            var categories = opportunities.Select(o => o.Category).Distinct().ToList();
            plan.Tags.AddRange(categories);

            var tags = opportunities.SelectMany(o => o.Tags).Distinct().Take(10).ToList();
            plan.Tags.AddRange(tags);

            return plan;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating improvement plan");
            throw;
        }
    }

    /// <summary>
    /// Calculates a score for an opportunity
    /// </summary>
    /// <param name="opportunity">The opportunity</param>
    /// <param name="options">The prioritization options</param>
    /// <returns>The score</returns>
    private double CalculateScore(ImprovementOpportunity opportunity, PrioritizationOptions options)
    {
        // Calculate the base score
        var priorityScore = opportunity.Priority * options.PriorityWeight;
        var impactScore = opportunity.EstimatedImpact * options.ImpactWeight;
        var effortScore = (1.0 / Math.Max(0.1, opportunity.EstimatedEffort)) * options.EffortWeight;

        var score = priorityScore + impactScore + effortScore;

        // Apply category multipliers
        if (options.CategoryMultipliers.TryGetValue(opportunity.Category, out var categoryMultiplier))
        {
            score *= categoryMultiplier;
        }

        // Apply tag multipliers
        foreach (var tag in opportunity.Tags)
        {
            if (options.TagMultipliers.TryGetValue(tag, out var tagMultiplier))
            {
                score *= tagMultiplier;
            }
        }

        return score;
    }
}

/// <summary>
/// Options for prioritizing improvement opportunities
/// </summary>
public class PrioritizationOptions
{
    /// <summary>
    /// Gets or sets the priority weight
    /// </summary>
    public double PriorityWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the impact weight
    /// </summary>
    public double ImpactWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the effort weight
    /// </summary>
    public double EffortWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the category filters
    /// </summary>
    public List<string> CategoryFilters { get; set; } = new();

    /// <summary>
    /// Gets or sets the tag filters
    /// </summary>
    public List<string> TagFilters { get; set; } = new();

    /// <summary>
    /// Gets or sets the minimum priority
    /// </summary>
    public int MinPriority { get; set; } = 0;

    /// <summary>
    /// Gets or sets the maximum effort
    /// </summary>
    public double MaxEffort { get; set; } = 0;

    /// <summary>
    /// Gets or sets the maximum number of results
    /// </summary>
    public int MaxResults { get; set; } = 0;

    /// <summary>
    /// Gets or sets the category multipliers
    /// </summary>
    public Dictionary<string, double> CategoryMultipliers { get; set; } = new();

    /// <summary>
    /// Gets or sets the tag multipliers
    /// </summary>
    public Dictionary<string, double> TagMultipliers { get; set; } = new();
}
