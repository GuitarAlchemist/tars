using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the improvement prioritizer service
/// </summary>
public interface IImprovementPrioritizerService
{
    /// <summary>
    /// Prioritizes an improvement
    /// </summary>
    /// <param name="improvement">The improvement to prioritize</param>
    /// <param name="options">Optional prioritization options</param>
    /// <returns>The prioritized improvement</returns>
    Task<PrioritizedImprovement> PrioritizeImprovementAsync(PrioritizedImprovement improvement, Dictionary<string, string>? options = null);

    /// <summary>
    /// Prioritizes a list of improvements
    /// </summary>
    /// <param name="improvements">The improvements to prioritize</param>
    /// <param name="options">Optional prioritization options</param>
    /// <returns>The prioritized improvements</returns>
    Task<List<PrioritizedImprovement>> PrioritizeImprovementsAsync(List<PrioritizedImprovement> improvements, Dictionary<string, string>? options = null);

    /// <summary>
    /// Creates an improvement from a metascript
    /// </summary>
    /// <param name="metascript">The metascript</param>
    /// <param name="options">Optional creation options</param>
    /// <returns>The created improvement</returns>
    Task<PrioritizedImprovement> CreateImprovementFromMetascriptAsync(GeneratedMetascript metascript, Dictionary<string, string>? options = null);

    /// <summary>
    /// Creates an improvement from a pattern match
    /// </summary>
    /// <param name="patternMatch">The pattern match</param>
    /// <param name="options">Optional creation options</param>
    /// <returns>The created improvement</returns>
    Task<PrioritizedImprovement> CreateImprovementFromPatternMatchAsync(PatternMatch patternMatch, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets all improvements
    /// </summary>
    /// <param name="options">Optional filter options</param>
    /// <returns>The list of improvements</returns>
    Task<List<PrioritizedImprovement>> GetImprovementsAsync(Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets an improvement by ID
    /// </summary>
    /// <param name="improvementId">The improvement ID</param>
    /// <returns>The improvement, or null if not found</returns>
    Task<PrioritizedImprovement?> GetImprovementAsync(string improvementId);

    /// <summary>
    /// Updates an improvement
    /// </summary>
    /// <param name="improvement">The improvement to update</param>
    /// <returns>True if the improvement was updated successfully, false otherwise</returns>
    Task<bool> UpdateImprovementAsync(PrioritizedImprovement improvement);

    /// <summary>
    /// Removes an improvement
    /// </summary>
    /// <param name="improvementId">The ID of the improvement to remove</param>
    /// <returns>True if the improvement was removed successfully, false otherwise</returns>
    Task<bool> RemoveImprovementAsync(string improvementId);

    /// <summary>
    /// Gets the next improvements to implement
    /// </summary>
    /// <param name="count">The number of improvements to get</param>
    /// <param name="options">Optional filter options</param>
    /// <returns>The list of improvements</returns>
    Task<List<PrioritizedImprovement>> GetNextImprovementsAsync(int count, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets all strategic goals
    /// </summary>
    /// <param name="options">Optional filter options</param>
    /// <returns>The list of strategic goals</returns>
    Task<List<StrategicGoal>> GetStrategicGoalsAsync(Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets a strategic goal by ID
    /// </summary>
    /// <param name="goalId">The goal ID</param>
    /// <returns>The strategic goal, or null if not found</returns>
    Task<StrategicGoal?> GetStrategicGoalAsync(string goalId);

    /// <summary>
    /// Adds a strategic goal
    /// </summary>
    /// <param name="goal">The goal to add</param>
    /// <returns>True if the goal was added successfully, false otherwise</returns>
    Task<bool> AddStrategicGoalAsync(StrategicGoal goal);

    /// <summary>
    /// Updates a strategic goal
    /// </summary>
    /// <param name="goal">The goal to update</param>
    /// <returns>True if the goal was updated successfully, false otherwise</returns>
    Task<bool> UpdateStrategicGoalAsync(StrategicGoal goal);

    /// <summary>
    /// Removes a strategic goal
    /// </summary>
    /// <param name="goalId">The ID of the goal to remove</param>
    /// <returns>True if the goal was removed successfully, false otherwise</returns>
    Task<bool> RemoveStrategicGoalAsync(string goalId);

    /// <summary>
    /// Gets the dependency graph for improvements
    /// </summary>
    /// <param name="options">Optional filter options</param>
    /// <returns>The dependency graph</returns>
    Task<ImprovementDependencyGraph> GetDependencyGraphAsync(Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets the available prioritization options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    Task<Dictionary<string, string>> GetAvailableOptionsAsync();

    /// <summary>
    /// Saves an improvement
    /// </summary>
    /// <param name="improvement">The improvement to save</param>
    /// <returns>True if the improvement was saved successfully, false otherwise</returns>
    Task<bool> SaveImprovementAsync(PrioritizedImprovement improvement);
}
