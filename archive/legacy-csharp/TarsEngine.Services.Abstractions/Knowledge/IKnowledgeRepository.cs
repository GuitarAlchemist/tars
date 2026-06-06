using TarsEngine.Services.Abstractions.Common;
using TarsEngine.Services.Abstractions.Models.Knowledge;

namespace TarsEngine.Services.Abstractions.Knowledge
{
    /// <summary>
    /// Interface for a repository that stores and retrieves knowledge items.
    /// </summary>
    public interface IKnowledgeRepository : IService
    {
        /// <summary>
        /// Adds a knowledge item to the repository.
        /// </summary>
        /// <param name="item">The knowledge item to add.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task AddItemAsync(KnowledgeItem item);

        /// <summary>
        /// Retrieves a knowledge item by its ID.
        /// </summary>
        /// <param name="id">The ID of the knowledge item to retrieve.</param>
        /// <returns>The knowledge item, or null if not found.</returns>
        Task<KnowledgeItem?> GetItemByIdAsync(string id);

        /// <summary>
        /// Searches for knowledge items based on the provided query.
        /// </summary>
        /// <param name="query">The search query.</param>
        /// <param name="limit">The maximum number of items to return.</param>
        /// <returns>A collection of knowledge items matching the query.</returns>
        Task<IEnumerable<KnowledgeItem>> SearchAsync(string query, int limit = 10);

        /// <summary>
        /// Retrieves knowledge items of the specified type.
        /// </summary>
        /// <param name="type">The type of knowledge items to retrieve.</param>
        /// <param name="limit">The maximum number of items to return.</param>
        /// <returns>A collection of knowledge items of the specified type.</returns>
        Task<IEnumerable<KnowledgeItem>> GetItemsByTypeAsync(KnowledgeType type, int limit = 10);

        /// <summary>
        /// Updates an existing knowledge item.
        /// </summary>
        /// <param name="item">The knowledge item to update.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task UpdateItemAsync(KnowledgeItem item);

        /// <summary>
        /// Deletes a knowledge item by its ID.
        /// </summary>
        /// <param name="id">The ID of the knowledge item to delete.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task DeleteItemAsync(string id);
    }
}
