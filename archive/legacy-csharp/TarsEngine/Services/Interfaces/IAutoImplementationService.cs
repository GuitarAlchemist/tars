using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the auto-implementation service
/// </summary>
public interface IAutoImplementationService
{
    /// <summary>
    /// Implement a TODO task
    /// </summary>
    /// <param name="taskDescription">The task description</param>
    /// <param name="progressCallback">A callback for reporting progress</param>
    /// <returns>The result of the implementation</returns>
    Task<ImplementationResult> ImplementTaskAsync(
        string taskDescription, 
        Action<string, int>? progressCallback = null);
}
