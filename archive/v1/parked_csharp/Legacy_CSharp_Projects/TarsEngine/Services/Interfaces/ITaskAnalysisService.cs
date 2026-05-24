using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the task analysis service
/// </summary>
public interface ITaskAnalysisService
{
    /// <summary>
    /// Analyze a TODO task and generate an implementation plan
    /// </summary>
    /// <param name="taskDescription">The task description</param>
    /// <returns>The implementation plan</returns>
    Task<ImplementationPlan> AnalyzeTaskAsync(string taskDescription);
}
