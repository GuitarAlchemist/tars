using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for scoring improvements
/// </summary>
public class ImprovementScorer
{
    private readonly ILogger _logger;
    private readonly Dictionary<string, double> _categoryWeights = new();
    private readonly Dictionary<string, double> _goalWeights = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="ImprovementScorer"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public ImprovementScorer(ILogger logger)
    {
        _logger = logger;
        InitializeDefaultWeights();
    }

    /// <summary>
    /// Scores an improvement
    /// </summary>
    /// <param name="improvement">The improvement to score</param>
    /// <param name="strategicGoals">The strategic goals</param>
    /// <param name="options">Optional scoring options</param>
    /// <returns>The scored improvement</returns>
    public PrioritizedImprovement ScoreImprovement(
        PrioritizedImprovement improvement,
        List<StrategicGoal> strategicGoals,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Scoring improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);

            // Calculate impact score
            improvement.ImpactScore = CalculateImpactScore(improvement, options);

            // Calculate effort score
            improvement.EffortScore = CalculateEffortScore(improvement, options);

            // Calculate risk score
            improvement.RiskScore = CalculateRiskScore(improvement, options);

            // Calculate alignment score
            improvement.AlignmentScore = CalculateAlignmentScore(improvement, strategicGoals, options);

            // Calculate priority score
            improvement.PriorityScore = CalculatePriorityScore(improvement, options);

            // Update prioritized timestamp
            improvement.PrioritizedAt = DateTime.UtcNow;

            _logger.LogInformation("Scored improvement: {ImprovementName} ({ImprovementId}), Priority: {PriorityScore}", 
                improvement.Name, improvement.Id, improvement.PriorityScore);

            return improvement;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error scoring improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            throw;
        }
    }

    /// <summary>
    /// Scores a list of improvements
    /// </summary>
    /// <param name="improvements">The improvements to score</param>
    /// <param name="strategicGoals">The strategic goals</param>
    /// <param name="options">Optional scoring options</param>
    /// <returns>The scored improvements</returns>
    public List<PrioritizedImprovement> ScoreImprovements(
        List<PrioritizedImprovement> improvements,
        List<StrategicGoal> strategicGoals,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Scoring {ImprovementCount} improvements", improvements.Count);

            foreach (var improvement in improvements)
            {
                ScoreImprovement(improvement, strategicGoals, options);
            }

            _logger.LogInformation("Scored {ImprovementCount} improvements", improvements.Count);
            return improvements;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error scoring improvements");
            throw;
        }
    }

    /// <summary>
    /// Sets the weight for a category
    /// </summary>
    /// <param name="category">The category</param>
    /// <param name="weight">The weight</param>
    public void SetCategoryWeight(string category, double weight)
    {
        _categoryWeights[category] = weight;
    }

    /// <summary>
    /// Sets the weight for a goal
    /// </summary>
    /// <param name="goalId">The goal ID</param>
    /// <param name="weight">The weight</param>
    public void SetGoalWeight(string goalId, double weight)
    {
        _goalWeights[goalId] = weight;
    }

    /// <summary>
    /// Gets the available scoring options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        return new Dictionary<string, string>
        {
            { "ImpactWeight", "Weight for impact score (default: 0.4)" },
            { "EffortWeight", "Weight for effort score (default: 0.3)" },
            { "RiskWeight", "Weight for risk score (default: 0.2)" },
            { "AlignmentWeight", "Weight for alignment score (default: 0.1)" },
            { "InvertEffort", "Whether to invert effort score (default: true)" },
            { "InvertRisk", "Whether to invert risk score (default: true)" },
            { "NormalizeScores", "Whether to normalize scores (default: true)" },
            { "MinPriorityScore", "Minimum priority score (default: 0.0)" },
            { "MaxPriorityScore", "Maximum priority score (default: 1.0)" },
            { "CategoryWeights", "JSON string of category weights" },
            { "GoalWeights", "JSON string of goal weights" }
        };
    }

    private double CalculateImpactScore(PrioritizedImprovement improvement, Dictionary<string, string>? options)
    {
        try
        {
            // Convert impact enum to score
            var impactScore = improvement.Impact switch
            {
                ImprovementImpact.VeryLow => 0.1,
                ImprovementImpact.Low => 0.3,
                ImprovementImpact.Medium => 0.5,
                ImprovementImpact.High => 0.7,
                ImprovementImpact.VeryHigh => 0.9,
                ImprovementImpact.Critical => 1.0,
                _ => 0.5
            };

            // Apply category weight
            if (_categoryWeights.TryGetValue(improvement.Category.ToString(), out var categoryWeight))
            {
                impactScore *= categoryWeight;
            }

            // Apply custom impact score if provided
            if (improvement.ImpactScore > 0)
            {
                impactScore = improvement.ImpactScore;
            }

            // Normalize score
            if (ParseOption(options, "NormalizeScores", true))
            {
                impactScore = Math.Max(0.0, Math.Min(1.0, impactScore));
            }

            return impactScore;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating impact score for improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return 0.5;
        }
    }

    private double CalculateEffortScore(PrioritizedImprovement improvement, Dictionary<string, string>? options)
    {
        try
        {
            // Convert effort enum to score
            var effortScore = improvement.Effort switch
            {
                ImprovementEffort.VeryLow => 0.1,
                ImprovementEffort.Low => 0.3,
                ImprovementEffort.Medium => 0.5,
                ImprovementEffort.High => 0.7,
                ImprovementEffort.VeryHigh => 0.9,
                ImprovementEffort.Extreme => 1.0,
                _ => 0.5
            };

            // Invert effort score (lower effort = higher score)
            if (ParseOption(options, "InvertEffort", true))
            {
                effortScore = 1.0 - effortScore;
            }

            // Apply custom effort score if provided
            if (improvement.EffortScore > 0)
            {
                effortScore = improvement.EffortScore;
            }

            // Normalize score
            if (ParseOption(options, "NormalizeScores", true))
            {
                effortScore = Math.Max(0.0, Math.Min(1.0, effortScore));
            }

            return effortScore;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating effort score for improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return 0.5;
        }
    }

    private double CalculateRiskScore(PrioritizedImprovement improvement, Dictionary<string, string>? options)
    {
        try
        {
            // Convert risk enum to score
            var riskScore = improvement.Risk switch
            {
                ImprovementRisk.VeryLow => 0.1,
                ImprovementRisk.Low => 0.3,
                ImprovementRisk.Medium => 0.5,
                ImprovementRisk.High => 0.7,
                ImprovementRisk.VeryHigh => 0.9,
                ImprovementRisk.Critical => 1.0,
                _ => 0.5
            };

            // Invert risk score (lower risk = higher score)
            if (ParseOption(options, "InvertRisk", true))
            {
                riskScore = 1.0 - riskScore;
            }

            // Apply custom risk score if provided
            if (improvement.RiskScore > 0)
            {
                riskScore = improvement.RiskScore;
            }

            // Normalize score
            if (ParseOption(options, "NormalizeScores", true))
            {
                riskScore = Math.Max(0.0, Math.Min(1.0, riskScore));
            }

            return riskScore;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating risk score for improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return 0.5;
        }
    }

    private double CalculateAlignmentScore(PrioritizedImprovement improvement, List<StrategicGoal> strategicGoals, Dictionary<string, string>? options)
    {
        try
        {
            if (improvement.StrategicGoals.Count == 0 || strategicGoals.Count == 0)
            {
                return 0.0;
            }

            // Calculate alignment score based on strategic goals
            var alignmentScore = 0.0;
            var totalWeight = 0.0;

            foreach (var goalId in improvement.StrategicGoals)
            {
                var goal = strategicGoals.FirstOrDefault(g => g.Id == goalId);
                if (goal != null && goal.IsActive)
                {
                    var goalWeight = goal.Weight;
                    if (_goalWeights.TryGetValue(goalId, out var customWeight))
                    {
                        goalWeight = customWeight;
                    }

                    alignmentScore += goalWeight;
                    totalWeight += 1.0;
                }
            }

            // Normalize alignment score
            if (totalWeight > 0)
            {
                alignmentScore /= totalWeight;
            }

            // Apply custom alignment score if provided
            if (improvement.AlignmentScore > 0)
            {
                alignmentScore = improvement.AlignmentScore;
            }

            // Normalize score
            if (ParseOption(options, "NormalizeScores", true))
            {
                alignmentScore = Math.Max(0.0, Math.Min(1.0, alignmentScore));
            }

            return alignmentScore;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating alignment score for improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return 0.0;
        }
    }

    private double CalculatePriorityScore(PrioritizedImprovement improvement, Dictionary<string, string>? options)
    {
        try
        {
            // Get weights
            var impactWeight = ParseOption(options, "ImpactWeight", 0.4);
            var effortWeight = ParseOption(options, "EffortWeight", 0.3);
            var riskWeight = ParseOption(options, "RiskWeight", 0.2);
            var alignmentWeight = ParseOption(options, "AlignmentWeight", 0.1);

            // Calculate weighted score
            var priorityScore = (improvement.ImpactScore * impactWeight) +
                               (improvement.EffortScore * effortWeight) +
                               (improvement.RiskScore * riskWeight) +
                               (improvement.AlignmentScore * alignmentWeight);

            // Apply min/max constraints
            var minScore = ParseOption(options, "MinPriorityScore", 0.0);
            var maxScore = ParseOption(options, "MaxPriorityScore", 1.0);
            priorityScore = Math.Max(minScore, Math.Min(maxScore, priorityScore));

            return priorityScore;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating priority score for improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return 0.5;
        }
    }

    private void InitializeDefaultWeights()
    {
        try
        {
            // Initialize category weights
            _categoryWeights[ImprovementCategory.Performance.ToString()] = 0.9;
            _categoryWeights[ImprovementCategory.Security.ToString()] = 1.0;
            _categoryWeights[ImprovementCategory.Maintainability.ToString()] = 0.8;
            _categoryWeights[ImprovementCategory.Reliability.ToString()] = 0.9;
            _categoryWeights[ImprovementCategory.Usability.ToString()] = 0.7;
            _categoryWeights[ImprovementCategory.Functionality.ToString()] = 0.8;
            _categoryWeights[ImprovementCategory.Scalability.ToString()] = 0.8;
            _categoryWeights[ImprovementCategory.Testability.ToString()] = 0.7;
            _categoryWeights[ImprovementCategory.Documentation.ToString()] = 0.6;
            _categoryWeights[ImprovementCategory.CodeQuality.ToString()] = 0.7;
            _categoryWeights[ImprovementCategory.Architecture.ToString()] = 0.8;
            _categoryWeights[ImprovementCategory.Design.ToString()] = 0.7;
            _categoryWeights[ImprovementCategory.Accessibility.ToString()] = 0.6;
            _categoryWeights[ImprovementCategory.Internationalization.ToString()] = 0.6;
            _categoryWeights[ImprovementCategory.Localization.ToString()] = 0.5;
            _categoryWeights[ImprovementCategory.Compatibility.ToString()] = 0.7;
            _categoryWeights[ImprovementCategory.Portability.ToString()] = 0.6;
            _categoryWeights[ImprovementCategory.Extensibility.ToString()] = 0.7;
            _categoryWeights[ImprovementCategory.Reusability.ToString()] = 0.7;
            _categoryWeights[ImprovementCategory.Modularity.ToString()] = 0.7;
            _categoryWeights[ImprovementCategory.Other.ToString()] = 0.5;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing default weights");
        }
    }

    private T ParseOption<T>(Dictionary<string, string>? options, string key, T defaultValue)
    {
        if (options == null || !options.TryGetValue(key, out var value))
        {
            return defaultValue;
        }

        try
        {
            return (T)Convert.ChangeType(value, typeof(T));
        }
        catch
        {
            return defaultValue;
        }
    }
}
