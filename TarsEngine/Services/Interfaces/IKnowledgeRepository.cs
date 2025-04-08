using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the knowledge repository
/// </summary>
public interface IKnowledgeRepository
{
    /// <summary>
    /// Adds a knowledge item to the repository
    /// </summary>
    /// <param name="item">The knowledge item to add</param>
    /// <returns>The added knowledge item</returns>
    Task<TarsEngine.Models.KnowledgeItem> AddItemAsync(TarsEngine.Models.KnowledgeItem item);

    /// <summary>
    /// Adds multiple knowledge items to the repository
    /// </summary>
    /// <param name="items">The knowledge items to add</param>
    /// <returns>The added knowledge items</returns>
    Task<IEnumerable<TarsEngine.Models.KnowledgeItem>> AddItemsAsync(IEnumerable<TarsEngine.Models.KnowledgeItem> items);

    /// <summary>
    /// Gets a knowledge item by ID
    /// </summary>
    /// <param name="id">The ID of the knowledge item to get</param>
    /// <returns>The knowledge item, or null if not found</returns>
    Task<TarsEngine.Models.KnowledgeItem?> GetItemAsync(string id);

    /// <summary>
    /// Updates a knowledge item in the repository
    /// </summary>
    /// <param name="item">The knowledge item to update</param>
    /// <returns>The updated knowledge item</returns>
    Task<TarsEngine.Models.KnowledgeItem> UpdateItemAsync(TarsEngine.Models.KnowledgeItem item);

    /// <summary>
    /// Deletes a knowledge item from the repository
    /// </summary>
    /// <param name="id">The ID of the knowledge item to delete</param>
    /// <returns>True if the item was deleted, false otherwise</returns>
    Task<bool> DeleteItemAsync(string id);

    /// <summary>
    /// Searches for knowledge items in the repository
    /// </summary>
    /// <param name="query">The search query</param>
    /// <param name="options">Optional search options</param>
    /// <returns>The matching knowledge items</returns>
    Task<IEnumerable<TarsEngine.Models.KnowledgeItem>> SearchItemsAsync(string query, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets all knowledge items of a specific type
    /// </summary>
    /// <param name="type">The type of knowledge items to get</param>
    /// <param name="options">Optional retrieval options</param>
    /// <returns>The knowledge items of the specified type</returns>
    Task<IEnumerable<TarsEngine.Models.KnowledgeItem>> GetItemsByTypeAsync(TarsEngine.Models.KnowledgeType type, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets all knowledge items with a specific tag
    /// </summary>
    /// <param name="tag">The tag to search for</param>
    /// <param name="options">Optional retrieval options</param>
    /// <returns>The knowledge items with the specified tag</returns>
    Task<IEnumerable<TarsEngine.Models.KnowledgeItem>> GetItemsByTagAsync(string tag, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets all knowledge items from a specific source
    /// </summary>
    /// <param name="source">The source to search for</param>
    /// <param name="options">Optional retrieval options</param>
    /// <returns>The knowledge items from the specified source</returns>
    Task<IEnumerable<TarsEngine.Models.KnowledgeItem>> GetItemsBySourceAsync(string source, Dictionary<string, string>? options = null);

    /// <summary>
    /// Adds a relationship between knowledge items
    /// </summary>
    /// <param name="relationship">The relationship to add</param>
    /// <returns>The added relationship</returns>
    Task<TarsEngine.Models.KnowledgeRelationship> AddRelationshipAsync(TarsEngine.Models.KnowledgeRelationship relationship);

    /// <summary>
    /// Gets all relationships for a knowledge item
    /// </summary>
    /// <param name="itemId">The ID of the knowledge item</param>
    /// <returns>The relationships for the knowledge item</returns>
    Task<IEnumerable<TarsEngine.Models.KnowledgeRelationship>> GetRelationshipsForItemAsync(string itemId);

    /// <summary>
    /// Deletes a relationship between knowledge items
    /// </summary>
    /// <param name="relationshipId">The ID of the relationship to delete</param>
    /// <returns>True if the relationship was deleted, false otherwise</returns>
    Task<bool> DeleteRelationshipAsync(string relationshipId);

    /// <summary>
    /// Gets statistics about the knowledge repository
    /// </summary>
    /// <returns>The repository statistics</returns>
    Task<KnowledgeRepositoryStats> GetStatisticsAsync();
}

/// <summary>
/// Represents statistics about the knowledge repository
/// </summary>
public class KnowledgeRepositoryStats
{
    /// <summary>
    /// Gets or sets the total number of knowledge items
    /// </summary>
    public int TotalItems { get; set; }

    /// <summary>
    /// Gets or sets the number of items by type
    /// </summary>
    public Dictionary<KnowledgeType, int> ItemsByType { get; set; } = new Dictionary<KnowledgeType, int>();

    /// <summary>
    /// Gets or sets the number of items by tag
    /// </summary>
    public Dictionary<string, int> ItemsByTag { get; set; } = new Dictionary<string, int>();

    /// <summary>
    /// Gets or sets the number of items by source
    /// </summary>
    public Dictionary<string, int> ItemsBySource { get; set; } = new Dictionary<string, int>();

    /// <summary>
    /// Gets or sets the total number of relationships
    /// </summary>
    public int TotalRelationships { get; set; }

    /// <summary>
    /// Gets or sets the number of relationships by type
    /// </summary>
    public Dictionary<RelationshipType, int> RelationshipsByType { get; set; } = new Dictionary<RelationshipType, int>();

    /// <summary>
    /// Gets or sets the average confidence score
    /// </summary>
    public double AverageConfidence { get; set; }

    /// <summary>
    /// Gets or sets the average relevance score
    /// </summary>
    public double AverageRelevance { get; set; }
}
