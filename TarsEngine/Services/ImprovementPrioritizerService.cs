using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for prioritizing improvements
/// </summary>
public class ImprovementPrioritizerService : IImprovementPrioritizerService
{
    private readonly ILogger<ImprovementPrioritizerService> _logger;
    private readonly ImprovementScorer _improvementScorer;
    private readonly StrategicAlignmentService _strategicAlignmentService;
    private readonly DependencyGraphService _dependencyGraphService;
    private readonly ImprovementQueue _improvementQueue;
    private readonly IMetascriptGeneratorService _metascriptGeneratorService;
    private readonly IPatternMatcherService _patternMatcherService;

    /// <summary>
    /// Initializes a new instance of the <see cref="ImprovementPrioritizerService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="improvementScorer">The improvement scorer</param>
    /// <param name="strategicAlignmentService">The strategic alignment service</param>
    /// <param name="dependencyGraphService">The dependency graph service</param>
    /// <param name="improvementQueue">The improvement queue</param>
    /// <param name="metascriptGeneratorService">The metascript generator service</param>
    /// <param name="patternMatcherService">The pattern matcher service</param>
    public ImprovementPrioritizerService(
        ILogger<ImprovementPrioritizerService> logger,
        ImprovementScorer improvementScorer,
        StrategicAlignmentService strategicAlignmentService,
        DependencyGraphService dependencyGraphService,
        ImprovementQueue improvementQueue,
        IMetascriptGeneratorService metascriptGeneratorService,
        IPatternMatcherService patternMatcherService)
    {
        _logger = logger;
        _improvementScorer = improvementScorer;
        _strategicAlignmentService = strategicAlignmentService;
        _dependencyGraphService = dependencyGraphService;
        _improvementQueue = improvementQueue;
        _metascriptGeneratorService = metascriptGeneratorService;
        _patternMatcherService = patternMatcherService;
    }

    /// <inheritdoc/>
    public async Task<PrioritizedImprovement> PrioritizeImprovementAsync(PrioritizedImprovement improvement, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Prioritizing improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);

            // Get strategic goals
            var strategicGoals = await _strategicAlignmentService.GetGoalsAsync();

            // Find strategic goals for the improvement if none are specified
            if (improvement.StrategicGoals.Count == 0)
            {
                var goals = await _strategicAlignmentService.FindGoalsForImprovementAsync(improvement);
                improvement.StrategicGoals = goals.Select(g => g.Id).ToList();
            }

            // Score the improvement
            improvement = _improvementScorer.ScoreImprovement(improvement, strategicGoals, options);

            // Update the improvement
            await _improvementQueue.UpdateImprovementAsync(improvement);

            _logger.LogInformation("Prioritized improvement: {ImprovementName} ({ImprovementId}), Priority: {PriorityScore}",
                improvement.Name, improvement.Id, improvement.PriorityScore);

            return improvement;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error prioritizing improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<List<PrioritizedImprovement>> PrioritizeImprovementsAsync(List<PrioritizedImprovement> improvements, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Prioritizing {ImprovementCount} improvements", improvements.Count);

            // Get strategic goals
            var strategicGoals = await _strategicAlignmentService.GetGoalsAsync();

            // Find strategic goals for improvements if none are specified
            foreach (var improvement in improvements)
            {
                if (improvement.StrategicGoals.Count == 0)
                {
                    var goals = await _strategicAlignmentService.FindGoalsForImprovementAsync(improvement);
                    improvement.StrategicGoals = goals.Select(g => g.Id).ToList();
                }
            }

            // Score improvements
            improvements = _improvementScorer.ScoreImprovements(improvements, strategicGoals, options);

            // Update improvements
            foreach (var improvement in improvements)
            {
                await _improvementQueue.UpdateImprovementAsync(improvement);
            }

            _logger.LogInformation("Prioritized {ImprovementCount} improvements", improvements.Count);
            return improvements;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error prioritizing improvements");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<PrioritizedImprovement> CreateImprovementFromMetascriptAsync(GeneratedMetascript metascript, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Creating improvement from metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);

            // Create improvement
            var improvement = new PrioritizedImprovement
            {
                Name = metascript.Name,
                Description = metascript.Description,
                MetascriptId = metascript.Id,
                PatternMatchId = metascript.PatternId,
                AffectedFiles = metascript.AffectedFiles,
                Tags = metascript.Tags,
                ImpactScore = metascript.ImpactScore,
                EffortScore = metascript.DifficultyScore,
                RiskScore = 0.5, // Default risk score
                Status = ImprovementStatus.Pending
            };

            // Set category based on tags
            improvement.Category = DetermineCategory(metascript.Tags);

            // Set impact based on impact score
            improvement.Impact = DetermineImpact(metascript.ImpactScore);

            // Set effort based on difficulty score
            improvement.Effort = DetermineEffort(metascript.DifficultyScore);

            // Set risk based on risk score
            improvement.Risk = DetermineRisk(improvement.RiskScore);

            // Find strategic goals
            var goals = await _strategicAlignmentService.FindGoalsForImprovementAsync(improvement);
            improvement.StrategicGoals = goals.Select(g => g.Id).ToList();

            // Prioritize improvement
            improvement = await PrioritizeImprovementAsync(improvement, options);

            // Add improvement to queue
            await _improvementQueue.AddImprovementAsync(improvement);

            _logger.LogInformation("Created improvement from metascript: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return improvement;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating improvement from metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<PrioritizedImprovement> CreateImprovementFromPatternMatchAsync(PatternMatch patternMatch, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Creating improvement from pattern match: {PatternName} ({PatternId})", patternMatch.PatternName, patternMatch.PatternId);

            // Generate metascript
            var metascript = await _metascriptGeneratorService.GenerateMetascriptAsync(patternMatch, options);

            // Create improvement from metascript
            var improvement = await CreateImprovementFromMetascriptAsync(metascript, options);

            _logger.LogInformation("Created improvement from pattern match: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return improvement;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating improvement from pattern match: {PatternName} ({PatternId})", patternMatch.PatternName, patternMatch.PatternId);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<List<PrioritizedImprovement>> GetImprovementsAsync(Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting improvements");
            return await _improvementQueue.GetImprovementsAsync(options);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting improvements");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<PrioritizedImprovement?> GetImprovementAsync(string improvementId)
    {
        try
        {
            _logger.LogInformation("Getting improvement by ID: {ImprovementId}", improvementId);
            return await _improvementQueue.GetImprovementAsync(improvementId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting improvement: {ImprovementId}", improvementId);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> UpdateImprovementAsync(PrioritizedImprovement improvement)
    {
        try
        {
            _logger.LogInformation("Updating improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return await _improvementQueue.UpdateImprovementAsync(improvement);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> RemoveImprovementAsync(string improvementId)
    {
        try
        {
            _logger.LogInformation("Removing improvement: {ImprovementId}", improvementId);
            return await _improvementQueue.RemoveImprovementAsync(improvementId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing improvement: {ImprovementId}", improvementId);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<List<PrioritizedImprovement>> GetNextImprovementsAsync(int count, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting next {Count} improvements", count);
            return await _improvementQueue.GetNextImprovementsAsync(count, options);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting next improvements");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<List<StrategicGoal>> GetStrategicGoalsAsync(Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting strategic goals");
            return await _strategicAlignmentService.GetGoalsAsync(options);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting strategic goals");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<StrategicGoal?> GetStrategicGoalAsync(string goalId)
    {
        try
        {
            _logger.LogInformation("Getting strategic goal by ID: {GoalId}", goalId);
            return await _strategicAlignmentService.GetGoalAsync(goalId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting strategic goal: {GoalId}", goalId);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> AddStrategicGoalAsync(StrategicGoal goal)
    {
        try
        {
            _logger.LogInformation("Adding strategic goal: {GoalName} ({GoalId})", goal.Name, goal.Id);
            return await _strategicAlignmentService.AddGoalAsync(goal);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding strategic goal: {GoalName} ({GoalId})", goal.Name, goal.Id);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> UpdateStrategicGoalAsync(StrategicGoal goal)
    {
        try
        {
            _logger.LogInformation("Updating strategic goal: {GoalName} ({GoalId})", goal.Name, goal.Id);
            return await _strategicAlignmentService.UpdateGoalAsync(goal);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating strategic goal: {GoalName} ({GoalId})", goal.Name, goal.Id);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> RemoveStrategicGoalAsync(string goalId)
    {
        try
        {
            _logger.LogInformation("Removing strategic goal: {GoalId}", goalId);
            return await _strategicAlignmentService.RemoveGoalAsync(goalId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing strategic goal: {GoalId}", goalId);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<ImprovementDependencyGraph> GetDependencyGraphAsync(Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting dependency graph");
            return await _improvementQueue.GetDependencyGraphAsync(options);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting dependency graph");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, string>> GetAvailableOptionsAsync()
    {
        var options = new Dictionary<string, string>();

        // Add improvement queue options
        var queueOptions = _improvementQueue.GetAvailableOptions();
        foreach (var option in queueOptions)
        {
            options[option.Key] = option.Value;
        }

        // Add dependency graph options
        var graphOptions = _dependencyGraphService.GetAvailableOptions();
        foreach (var option in graphOptions)
        {
            if (!options.ContainsKey(option.Key))
            {
                options[option.Key] = option.Value;
            }
        }

        // Add strategic alignment options
        var alignmentOptions = _strategicAlignmentService.GetAvailableOptions();
        foreach (var option in alignmentOptions)
        {
            if (!options.ContainsKey(option.Key))
            {
                options[option.Key] = option.Value;
            }
        }

        // Add improvement scorer options
        var scorerOptions = _improvementScorer.GetAvailableOptions();
        foreach (var option in scorerOptions)
        {
            if (!options.ContainsKey(option.Key))
            {
                options[option.Key] = option.Value;
            }
        }

        return options;
    }

    /// <inheritdoc/>
    public async Task<bool> SaveImprovementAsync(PrioritizedImprovement improvement)
    {
        try
        {
            _logger.LogInformation("Saving improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return await _improvementQueue.AddImprovementAsync(improvement);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            throw;
        }
    }

    private ImprovementCategory DetermineCategory(List<string> tags)
    {
        // Map common tags to categories
        var tagCategoryMap = new Dictionary<string, ImprovementCategory>(StringComparer.OrdinalIgnoreCase)
        {
            { "performance", ImprovementCategory.Performance },
            { "security", ImprovementCategory.Security },
            { "maintainability", ImprovementCategory.Maintainability },
            { "reliability", ImprovementCategory.Reliability },
            { "usability", ImprovementCategory.Usability },
            { "functionality", ImprovementCategory.Functionality },
            { "scalability", ImprovementCategory.Scalability },
            { "testability", ImprovementCategory.Testability },
            { "documentation", ImprovementCategory.Documentation },
            { "code-quality", ImprovementCategory.CodeQuality },
            { "architecture", ImprovementCategory.Architecture },
            { "design", ImprovementCategory.Design },
            { "accessibility", ImprovementCategory.Accessibility },
            { "internationalization", ImprovementCategory.Internationalization },
            { "localization", ImprovementCategory.Localization },
            { "compatibility", ImprovementCategory.Compatibility },
            { "portability", ImprovementCategory.Portability },
            { "extensibility", ImprovementCategory.Extensibility },
            { "reusability", ImprovementCategory.Reusability },
            { "modularity", ImprovementCategory.Modularity }
        };

        // Find the first matching tag
        foreach (var tag in tags)
        {
            if (tagCategoryMap.TryGetValue(tag, out var category))
            {
                return category;
            }
        }

        // Default to Other
        return ImprovementCategory.Other;
    }

    private ImprovementImpact DetermineImpact(double impactScore)
    {
        if (impactScore >= 0.9) return ImprovementImpact.Critical;
        if (impactScore >= 0.7) return ImprovementImpact.VeryHigh;
        if (impactScore >= 0.5) return ImprovementImpact.High;
        if (impactScore >= 0.3) return ImprovementImpact.Medium;
        if (impactScore >= 0.1) return ImprovementImpact.Low;
        return ImprovementImpact.VeryLow;
    }

    private ImprovementEffort DetermineEffort(double effortScore)
    {
        if (effortScore >= 0.9) return ImprovementEffort.Extreme;
        if (effortScore >= 0.7) return ImprovementEffort.VeryHigh;
        if (effortScore >= 0.5) return ImprovementEffort.High;
        if (effortScore >= 0.3) return ImprovementEffort.Medium;
        if (effortScore >= 0.1) return ImprovementEffort.Low;
        return ImprovementEffort.VeryLow;
    }

    private ImprovementRisk DetermineRisk(double riskScore)
    {
        if (riskScore >= 0.9) return ImprovementRisk.Critical;
        if (riskScore >= 0.7) return ImprovementRisk.VeryHigh;
        if (riskScore >= 0.5) return ImprovementRisk.High;
        if (riskScore >= 0.3) return ImprovementRisk.Medium;
        if (riskScore >= 0.1) return ImprovementRisk.Low;
        return ImprovementRisk.VeryLow;
    }
}
