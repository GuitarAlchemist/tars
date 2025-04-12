using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces
{
    /// <summary>
    /// Interface for execution service
    /// </summary>
    public interface IExecutionService
    {
        /// <summary>
        /// Gets all execution plans
        /// </summary>
        /// <returns>A list of execution plans</returns>
        Task<List<ExecutionPlan>> GetExecutionPlansAsync();

        /// <summary>
        /// Gets an execution plan by ID
        /// </summary>
        /// <param name="id">The ID of the execution plan</param>
        /// <returns>The execution plan</returns>
        Task<ExecutionPlan> GetExecutionPlanAsync(string id);

        /// <summary>
        /// Creates a new execution plan
        /// </summary>
        /// <param name="name">The name of the execution plan</param>
        /// <param name="description">The description of the execution plan</param>
        /// <param name="tags">The tags for the execution plan</param>
        /// <returns>The created execution plan</returns>
        Task<ExecutionPlan> CreateExecutionPlanAsync(string name, string description, List<string> tags);

        /// <summary>
        /// Starts an execution plan
        /// </summary>
        /// <param name="id">The ID of the execution plan</param>
        /// <returns>The updated execution plan</returns>
        Task<ExecutionPlan> StartExecutionAsync(string id);

        /// <summary>
        /// Stops an execution plan
        /// </summary>
        /// <param name="id">The ID of the execution plan</param>
        /// <returns>The updated execution plan</returns>
        Task<ExecutionPlan> StopExecutionAsync(string id);

        /// <summary>
        /// Deletes an execution plan
        /// </summary>
        /// <param name="id">The ID of the execution plan</param>
        /// <returns>True if the execution plan was deleted, false otherwise</returns>
        Task<bool> DeleteExecutionPlanAsync(string id);

        /// <summary>
        /// Gets the logs for an execution plan
        /// </summary>
        /// <param name="id">The ID of the execution plan</param>
        /// <returns>A list of log entries</returns>
        Task<List<LogEntry>> GetExecutionLogsAsync(string id);
    }
}
