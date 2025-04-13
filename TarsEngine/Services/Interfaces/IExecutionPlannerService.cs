using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the execution planner service
/// </summary>
public interface IExecutionPlannerService
{
    /// <summary>
    /// Creates an execution plan for an improvement
    /// </summary>
    /// <param name="improvement">The improvement</param>
    /// <param name="options">Optional planning options</param>
    /// <returns>The execution plan</returns>
    Task<ExecutionPlan> CreateExecutionPlanAsync(PrioritizedImprovement improvement, Dictionary<string, string>? options = null);

    /// <summary>
    /// Creates an execution plan for a metascript
    /// </summary>
    /// <param name="metascript">The metascript</param>
    /// <param name="options">Optional planning options</param>
    /// <returns>The execution plan</returns>
    Task<ExecutionPlan> CreateExecutionPlanAsync(GeneratedMetascript metascript, Dictionary<string, string>? options = null);

    /// <summary>
    /// Validates an execution plan
    /// </summary>
    /// <param name="plan">The execution plan</param>
    /// <param name="options">Optional validation options</param>
    /// <returns>True if the execution plan is valid, false otherwise</returns>
    Task<bool> ValidateExecutionPlanAsync(ExecutionPlan plan, Dictionary<string, string>? options = null);

    /// <summary>
    /// Executes an execution plan
    /// </summary>
    /// <param name="plan">The execution plan</param>
    /// <param name="options">Optional execution options</param>
    /// <returns>The execution plan result</returns>
    Task<ExecutionPlanResult> ExecuteExecutionPlanAsync(ExecutionPlan plan, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets an execution plan by ID
    /// </summary>
    /// <param name="planId">The execution plan ID</param>
    /// <returns>The execution plan, or null if not found</returns>
    Task<ExecutionPlan?> GetExecutionPlanAsync(string planId);

    /// <summary>
    /// Gets all execution plans
    /// </summary>
    /// <param name="options">Optional filter options</param>
    /// <returns>The list of execution plans</returns>
    Task<List<ExecutionPlan>> GetExecutionPlansAsync(Dictionary<string, string>? options = null);

    /// <summary>
    /// Saves an execution plan
    /// </summary>
    /// <param name="plan">The execution plan</param>
    /// <returns>True if the execution plan was saved successfully, false otherwise</returns>
    Task<bool> SaveExecutionPlanAsync(ExecutionPlan plan);

    /// <summary>
    /// Removes an execution plan
    /// </summary>
    /// <param name="planId">The execution plan ID</param>
    /// <returns>True if the execution plan was removed successfully, false otherwise</returns>
    Task<bool> RemoveExecutionPlanAsync(string planId);

    /// <summary>
    /// Creates an execution context for an execution plan
    /// </summary>
    /// <param name="plan">The execution plan</param>
    /// <param name="options">Optional context options</param>
    /// <returns>The execution context</returns>
    Task<TarsEngine.Models.ExecutionContext> CreateExecutionContextAsync(ExecutionPlan plan, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets an execution context by ID
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <returns>The execution context, or null if not found</returns>
    Task<TarsEngine.Models.ExecutionContext?> GetExecutionContextAsync(string contextId);

    /// <summary>
    /// Saves an execution context
    /// </summary>
    /// <param name="context">The execution context</param>
    /// <returns>True if the execution context was saved successfully, false otherwise</returns>
    Task<bool> SaveExecutionContextAsync(TarsEngine.Models.ExecutionContext context);

    /// <summary>
    /// Gets the execution result for an execution plan
    /// </summary>
    /// <param name="planId">The execution plan ID</param>
    /// <returns>The execution plan result, or null if not found</returns>
    Task<ExecutionPlanResult?> GetExecutionResultAsync(string planId);

    /// <summary>
    /// Saves an execution result
    /// </summary>
    /// <param name="result">The execution plan result</param>
    /// <returns>True if the execution result was saved successfully, false otherwise</returns>
    Task<bool> SaveExecutionResultAsync(ExecutionPlanResult result);

    /// <summary>
    /// Gets the available planning options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    Task<Dictionary<string, string>> GetAvailablePlanningOptionsAsync();

    /// <summary>
    /// Gets the available execution options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    Task<Dictionary<string, string>> GetAvailableExecutionOptionsAsync();
}
