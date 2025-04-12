using System.Collections.Generic;
using System.Threading.Tasks;

namespace TarsEngine.Services.Interfaces
{
    /// <summary>
    /// Interface for improvement service
    /// </summary>
    public interface IImprovementService
    {
        /// <summary>
        /// Gets all improvement categories
        /// </summary>
        /// <returns>A list of improvement categories</returns>
        Task<List<string>> GetImprovementCategoriesAsync();

        /// <summary>
        /// Gets all improvement priorities
        /// </summary>
        /// <returns>A list of improvement priorities</returns>
        Task<List<string>> GetImprovementPrioritiesAsync();

        /// <summary>
        /// Gets all improvement tags
        /// </summary>
        /// <returns>A list of improvement tags</returns>
        Task<List<string>> GetImprovementTagsAsync();

        /// <summary>
        /// Prioritizes an improvement
        /// </summary>
        /// <param name="executionId">The ID of the execution plan</param>
        /// <param name="improvementId">The ID of the improvement</param>
        /// <param name="priority">The priority value</param>
        /// <returns>True if the improvement was prioritized, false otherwise</returns>
        Task<bool> PrioritizeImprovementAsync(string executionId, string improvementId, int priority);

        /// <summary>
        /// Tags an improvement
        /// </summary>
        /// <param name="executionId">The ID of the execution plan</param>
        /// <param name="improvementId">The ID of the improvement</param>
        /// <param name="tag">The tag to add</param>
        /// <returns>True if the improvement was tagged, false otherwise</returns>
        Task<bool> TagImprovementAsync(string executionId, string improvementId, string tag);

        /// <summary>
        /// Removes a tag from an improvement
        /// </summary>
        /// <param name="executionId">The ID of the execution plan</param>
        /// <param name="improvementId">The ID of the improvement</param>
        /// <param name="tag">The tag to remove</param>
        /// <returns>True if the tag was removed, false otherwise</returns>
        Task<bool> RemoveTagFromImprovementAsync(string executionId, string improvementId, string tag);
    }
}
