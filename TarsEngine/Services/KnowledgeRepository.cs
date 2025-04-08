using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Repository for storing and retrieving knowledge items
/// </summary>
public class KnowledgeRepository : IKnowledgeRepository
{
    private readonly ILogger<KnowledgeRepository> _logger;
    private readonly string _dataDirectory;
    private readonly Dictionary<string, TarsEngine.Models.KnowledgeItem> _itemsCache = new();
    private readonly Dictionary<string, List<TarsEngine.Models.KnowledgeRelationship>> _relationshipsCache = new();
    private readonly object _lockObject = new();
    private bool _isInitialized = false;

    /// <summary>
    /// Initializes a new instance of the <see cref="KnowledgeRepository"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public KnowledgeRepository(ILogger<KnowledgeRepository> logger)
    {
        _logger = logger;
        _dataDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "Knowledge");
    }

    /// <inheritdoc/>
    public async Task<TarsEngine.Models.KnowledgeItem> AddItemAsync(TarsEngine.Models.KnowledgeItem item)
    {
        try
        {
            _logger.LogInformation("Adding knowledge item of type {Type}", item.Type);

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Set created/updated timestamps
            item.CreatedAt = DateTime.UtcNow;
            item.UpdatedAt = DateTime.UtcNow;

            // Add to cache
            lock (_lockObject)
            {
                _itemsCache[item.Id] = item;
            }

            // Save to disk
            await SaveItemToDiskAsync(item);

            _logger.LogInformation("Added knowledge item: {ItemId}", item.Id);
            return item;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding knowledge item");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<TarsEngine.Models.KnowledgeItem>> AddItemsAsync(IEnumerable<TarsEngine.Models.KnowledgeItem> items)
    {
        try
        {
            var itemsList = items.ToList();
            _logger.LogInformation("Adding {Count} knowledge items", itemsList.Count);

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Add each item
            var addedItems = new List<TarsEngine.Models.KnowledgeItem>();
            foreach (var item in itemsList)
            {
                var addedItem = await AddItemAsync(item);
                addedItems.Add(addedItem);
            }

            _logger.LogInformation("Added {Count} knowledge items", addedItems.Count);
            return addedItems;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding knowledge items");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<TarsEngine.Models.KnowledgeItem?> GetItemAsync(string id)
    {
        try
        {
            _logger.LogInformation("Getting knowledge item: {ItemId}", id);

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Check cache
            lock (_lockObject)
            {
                if (_itemsCache.TryGetValue(id, out var cachedItem))
                {
                    return cachedItem;
                }
            }

            // Not found
            _logger.LogInformation("Knowledge item not found: {ItemId}", id);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting knowledge item: {ItemId}", id);
            return null;
        }
    }

    /// <inheritdoc/>
    public async Task<TarsEngine.Models.KnowledgeItem> UpdateItemAsync(TarsEngine.Models.KnowledgeItem item)
    {
        try
        {
            _logger.LogInformation("Updating knowledge item: {ItemId}", item.Id);

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Check if item exists
            var existingItem = await GetItemAsync(item.Id);
            if (existingItem == null)
            {
                throw new ArgumentException($"Knowledge item not found: {item.Id}");
            }

            // Update timestamp
            item.UpdatedAt = DateTime.UtcNow;
            item.CreatedAt = existingItem.CreatedAt; // Preserve original creation time

            // Update cache
            lock (_lockObject)
            {
                _itemsCache[item.Id] = item;
            }

            // Save to disk
            await SaveItemToDiskAsync(item);

            _logger.LogInformation("Updated knowledge item: {ItemId}", item.Id);
            return item;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating knowledge item: {ItemId}", item.Id);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> DeleteItemAsync(string id)
    {
        try
        {
            _logger.LogInformation("Deleting knowledge item: {ItemId}", id);

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Check if item exists
            var existingItem = await GetItemAsync(id);
            if (existingItem == null)
            {
                _logger.LogInformation("Knowledge item not found: {ItemId}", id);
                return false;
            }

            // Remove from cache
            lock (_lockObject)
            {
                _itemsCache.Remove(id);
            }

            // Delete from disk
            var filePath = GetItemFilePath(id);
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }

            // Delete relationships
            await DeleteRelationshipsForItemAsync(id);

            _logger.LogInformation("Deleted knowledge item: {ItemId}", id);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deleting knowledge item: {ItemId}", id);
            return false;
        }
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<TarsEngine.Models.KnowledgeItem>> SearchItemsAsync(string query, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Searching for knowledge items with query: {Query}", query);

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Parse options
            var maxResults = options != null && options.TryGetValue("MaxResults", out var maxResultsStr) && int.TryParse(maxResultsStr, out var maxResultsInt)
                ? maxResultsInt
                : 100;
            var typeFilter = options != null && options.TryGetValue("Type", out var typeStr) && Enum.TryParse<KnowledgeType>(typeStr, out var typeEnum)
                ? typeEnum
                : (KnowledgeType?)null;
            var minConfidence = options != null && options.TryGetValue("MinConfidence", out var minConfidenceStr) && double.TryParse(minConfidenceStr, out var minConfidenceDouble)
                ? minConfidenceDouble
                : 0.0;
            var minRelevance = options != null && options.TryGetValue("MinRelevance", out var minRelevanceStr) && double.TryParse(minRelevanceStr, out var minRelevanceDouble)
                ? minRelevanceDouble
                : 0.0;

            // Prepare search regex
            var searchRegex = new Regex(query, RegexOptions.IgnoreCase);

            // Search in cache
            List<TarsEngine.Models.KnowledgeItem> results;
            lock (_lockObject)
            {
                results = _itemsCache.Values
                    .Where(item =>
                        (searchRegex.IsMatch(item.Content) ||
                         searchRegex.IsMatch(item.Source) ||
                         item.Tags.Any(tag => searchRegex.IsMatch(tag))) &&
                        (typeFilter == null || item.Type.Equals(typeFilter.Value)) &&
                        item.Confidence >= minConfidence &&
                        item.Relevance >= minRelevance)
                    .OrderByDescending(item => item.Relevance)
                    .Take(maxResults)
                    .ToList();
            }

            _logger.LogInformation("Found {Count} knowledge items matching query: {Query}", results.Count, query);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error searching for knowledge items with query: {Query}", query);
            return Enumerable.Empty<TarsEngine.Models.KnowledgeItem>();
        }
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<TarsEngine.Models.KnowledgeItem>> GetItemsByTypeAsync(TarsEngine.Models.KnowledgeType type, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting knowledge items of type: {Type}", type);

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Parse options
            var maxResults = options != null && options.TryGetValue("MaxResults", out var maxResultsStr) && int.TryParse(maxResultsStr, out var maxResultsInt)
                ? maxResultsInt
                : 100;
            var minConfidence = options != null && options.TryGetValue("MinConfidence", out var minConfidenceStr) && double.TryParse(minConfidenceStr, out var minConfidenceDouble)
                ? minConfidenceDouble
                : 0.0;
            var minRelevance = options != null && options.TryGetValue("MinRelevance", out var minRelevanceStr) && double.TryParse(minRelevanceStr, out var minRelevanceDouble)
                ? minRelevanceDouble
                : 0.0;

            // Get items from cache
            List<TarsEngine.Models.KnowledgeItem> results;
            lock (_lockObject)
            {
                results = _itemsCache.Values
                    .Where(item =>
                        item.Type.Equals(type) &&
                        item.Confidence >= minConfidence &&
                        item.Relevance >= minRelevance)
                    .OrderByDescending(item => item.Relevance)
                    .Take(maxResults)
                    .ToList();
            }

            _logger.LogInformation("Found {Count} knowledge items of type: {Type}", results.Count, type);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting knowledge items of type: {Type}", type);
            return Enumerable.Empty<TarsEngine.Models.KnowledgeItem>();
        }
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<TarsEngine.Models.KnowledgeItem>> GetItemsByTagAsync(string tag, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting knowledge items with tag: {Tag}", tag);

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Parse options
            var maxResults = options != null && options.TryGetValue("MaxResults", out var maxResultsStr) && int.TryParse(maxResultsStr, out var maxResultsInt)
                ? maxResultsInt
                : 100;
            var typeFilter = options != null && options.TryGetValue("Type", out var typeStr) && Enum.TryParse<TarsEngine.Services.Interfaces.KnowledgeType>(typeStr, out var typeEnum)
                ? typeEnum
                : (TarsEngine.Services.Interfaces.KnowledgeType?)null;
            var minConfidence = options != null && options.TryGetValue("MinConfidence", out var minConfidenceStr) && double.TryParse(minConfidenceStr, out var minConfidenceDouble)
                ? minConfidenceDouble
                : 0.0;
            var minRelevance = options != null && options.TryGetValue("MinRelevance", out var minRelevanceStr) && double.TryParse(minRelevanceStr, out var minRelevanceDouble)
                ? minRelevanceDouble
                : 0.0;

            // Get items from cache
            List<TarsEngine.Models.KnowledgeItem> results;
            lock (_lockObject)
            {
                results = _itemsCache.Values
                    .Where(item =>
                        item.Tags.Contains(tag) &&
                        (typeFilter == null || item.Type.Equals(typeFilter.Value)) &&
                        item.Confidence >= minConfidence &&
                        item.Relevance >= minRelevance)
                    .OrderByDescending(item => item.Relevance)
                    .Take(maxResults)
                    .ToList();
            }

            _logger.LogInformation("Found {Count} knowledge items with tag: {Tag}", results.Count, tag);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting knowledge items with tag: {Tag}", tag);
            return Enumerable.Empty<TarsEngine.Models.KnowledgeItem>();
        }
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<TarsEngine.Models.KnowledgeItem>> GetItemsBySourceAsync(string source, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Getting knowledge items from source: {Source}", source);

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Parse options
            var maxResults = options != null && options.TryGetValue("MaxResults", out var maxResultsStr) && int.TryParse(maxResultsStr, out var maxResultsInt)
                ? maxResultsInt
                : 100;
            var typeFilter = options != null && options.TryGetValue("Type", out var typeStr) && Enum.TryParse<TarsEngine.Services.Interfaces.KnowledgeType>(typeStr, out var typeEnum)
                ? typeEnum
                : (TarsEngine.Services.Interfaces.KnowledgeType?)null;
            var minConfidence = options != null && options.TryGetValue("MinConfidence", out var minConfidenceStr) && double.TryParse(minConfidenceStr, out var minConfidenceDouble)
                ? minConfidenceDouble
                : 0.0;
            var minRelevance = options != null && options.TryGetValue("MinRelevance", out var minRelevanceStr) && double.TryParse(minRelevanceStr, out var minRelevanceDouble)
                ? minRelevanceDouble
                : 0.0;

            // Get items from cache
            List<TarsEngine.Models.KnowledgeItem> results;
            lock (_lockObject)
            {
                results = _itemsCache.Values
                    .Where(item =>
                        item.Source.Equals(source, StringComparison.OrdinalIgnoreCase) &&
                        (typeFilter == null || item.Type.Equals(typeFilter.Value)) &&
                        item.Confidence >= minConfidence &&
                        item.Relevance >= minRelevance)
                    .OrderByDescending(item => item.Relevance)
                    .Take(maxResults)
                    .ToList();
            }

            _logger.LogInformation("Found {Count} knowledge items from source: {Source}", results.Count, source);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting knowledge items from source: {Source}", source);
            return Enumerable.Empty<TarsEngine.Models.KnowledgeItem>();
        }
    }

    /// <inheritdoc/>
    public async Task<KnowledgeRelationship> AddRelationshipAsync(KnowledgeRelationship relationship)
    {
        try
        {
            _logger.LogInformation("Adding relationship between {SourceId} and {TargetId}", relationship.SourceId, relationship.TargetId);

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Set created timestamp
            relationship.CreatedAt = DateTime.UtcNow;

            // Add to cache
            lock (_lockObject)
            {
                if (!_relationshipsCache.ContainsKey(relationship.SourceId))
                {
                    _relationshipsCache[relationship.SourceId] = new List<KnowledgeRelationship>();
                }
                _relationshipsCache[relationship.SourceId].Add(relationship);

                if (!_relationshipsCache.ContainsKey(relationship.TargetId))
                {
                    _relationshipsCache[relationship.TargetId] = new List<KnowledgeRelationship>();
                }
                _relationshipsCache[relationship.TargetId].Add(relationship);
            }

            // Save to disk
            await SaveRelationshipToDiskAsync(relationship);

            _logger.LogInformation("Added relationship: {RelationshipId}", relationship.Id);
            return relationship;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding relationship");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<IEnumerable<KnowledgeRelationship>> GetRelationshipsForItemAsync(string itemId)
    {
        try
        {
            _logger.LogInformation("Getting relationships for item: {ItemId}", itemId);

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Get relationships from cache
            List<KnowledgeRelationship> results;
            lock (_lockObject)
            {
                if (_relationshipsCache.TryGetValue(itemId, out var relationships))
                {
                    results = relationships.ToList();
                }
                else
                {
                    results = new List<KnowledgeRelationship>();
                }
            }

            _logger.LogInformation("Found {Count} relationships for item: {ItemId}", results.Count, itemId);
            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting relationships for item: {ItemId}", itemId);
            return Enumerable.Empty<KnowledgeRelationship>();
        }
    }

    /// <inheritdoc/>
    public async Task<bool> DeleteRelationshipAsync(string relationshipId)
    {
        try
        {
            _logger.LogInformation("Deleting relationship: {RelationshipId}", relationshipId);

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Find the relationship
            KnowledgeRelationship? relationship = null;
            lock (_lockObject)
            {
                foreach (var relationships in _relationshipsCache.Values)
                {
                    relationship = relationships.FirstOrDefault(r => r.Id == relationshipId);
                    if (relationship != null)
                    {
                        break;
                    }
                }
            }

            if (relationship == null)
            {
                _logger.LogInformation("Relationship not found: {RelationshipId}", relationshipId);
                return false;
            }

            // Remove from cache
            lock (_lockObject)
            {
                if (_relationshipsCache.TryGetValue(relationship.SourceId, out var sourceRelationships))
                {
                    sourceRelationships.RemoveAll(r => r.Id == relationshipId);
                }
                if (_relationshipsCache.TryGetValue(relationship.TargetId, out var targetRelationships))
                {
                    targetRelationships.RemoveAll(r => r.Id == relationshipId);
                }
            }

            // Delete from disk
            var filePath = GetRelationshipFilePath(relationshipId);
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }

            _logger.LogInformation("Deleted relationship: {RelationshipId}", relationshipId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deleting relationship: {RelationshipId}", relationshipId);
            return false;
        }
    }

    /// <inheritdoc/>
    public async Task<KnowledgeRepositoryStats> GetStatisticsAsync()
    {
        try
        {
            _logger.LogInformation("Getting knowledge repository statistics");

            // Ensure repository is initialized
            await EnsureInitializedAsync();

            // Calculate statistics
            var stats = new KnowledgeRepositoryStats();
            lock (_lockObject)
            {
                // Items statistics
                stats.TotalItems = _itemsCache.Count;

                // Items by type
                foreach (var type in Enum.GetValues<KnowledgeType>())
                {
                    var count = _itemsCache.Values.Count(item => item.Type.Equals(type));
                    if (count > 0)
                    {
                        stats.ItemsByType[type] = count;
                    }
                }

                // Items by tag
                var allTags = _itemsCache.Values.SelectMany(item => item.Tags).Distinct();
                foreach (var tag in allTags)
                {
                    var count = _itemsCache.Values.Count(item => item.Tags.Contains(tag));
                    stats.ItemsByTag[tag] = count;
                }

                // Items by source
                var allSources = _itemsCache.Values.Select(item => item.Source).Distinct();
                foreach (var source in allSources)
                {
                    var count = _itemsCache.Values.Count(item => item.Source == source);
                    stats.ItemsBySource[source] = count;
                }

                // Relationships statistics
                var allRelationships = _relationshipsCache.Values.SelectMany(r => r).Distinct().ToList();
                stats.TotalRelationships = allRelationships.Count;

                // Relationships by type
                foreach (var type in Enum.GetValues<RelationshipType>())
                {
                    var count = allRelationships.Count(r => r.Type.Equals(type));
                    if (count > 0)
                    {
                        stats.RelationshipsByType[type] = count;
                    }
                }

                // Average scores
                if (_itemsCache.Count > 0)
                {
                    stats.AverageConfidence = _itemsCache.Values.Average(item => item.Confidence);
                    stats.AverageRelevance = _itemsCache.Values.Average(item => item.Relevance);
                }
            }

            _logger.LogInformation("Retrieved knowledge repository statistics: {TotalItems} items, {TotalRelationships} relationships",
                stats.TotalItems, stats.TotalRelationships);
            return stats;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting knowledge repository statistics");
            return new KnowledgeRepositoryStats();
        }
    }

    private async Task EnsureInitializedAsync()
    {
        if (_isInitialized)
        {
            return;
        }

        lock (_lockObject)
        {
            if (_isInitialized)
            {
                return;
            }

            _logger.LogInformation("Initializing knowledge repository");

            // Create directories if they don't exist
            var itemsDirectory = Path.Combine(_dataDirectory, "Items");
            var relationshipsDirectory = Path.Combine(_dataDirectory, "Relationships");

            Directory.CreateDirectory(itemsDirectory);
            Directory.CreateDirectory(relationshipsDirectory);

            // Load items from disk
            _itemsCache.Clear();
            foreach (var filePath in Directory.GetFiles(itemsDirectory, "*.json"))
            {
                try
                {
                    var json = File.ReadAllText(filePath);
                    var item = JsonSerializer.Deserialize<TarsEngine.Models.KnowledgeItem>(json);
                    if (item != null)
                    {
                        _itemsCache[item.Id] = item;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error loading knowledge item from file: {FilePath}", filePath);
                }
            }

            // Load relationships from disk
            _relationshipsCache.Clear();
            foreach (var filePath in Directory.GetFiles(relationshipsDirectory, "*.json"))
            {
                try
                {
                    var json = File.ReadAllText(filePath);
                    var relationship = JsonSerializer.Deserialize<KnowledgeRelationship>(json);
                    if (relationship != null)
                    {
                        if (!_relationshipsCache.ContainsKey(relationship.SourceId))
                        {
                            _relationshipsCache[relationship.SourceId] = new List<KnowledgeRelationship>();
                        }
                        _relationshipsCache[relationship.SourceId].Add(relationship);

                        if (!_relationshipsCache.ContainsKey(relationship.TargetId))
                        {
                            _relationshipsCache[relationship.TargetId] = new List<KnowledgeRelationship>();
                        }
                        _relationshipsCache[relationship.TargetId].Add(relationship);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error loading relationship from file: {FilePath}", filePath);
                }
            }

            _logger.LogInformation("Initialized knowledge repository with {ItemCount} items and {RelationshipCount} relationships",
                _itemsCache.Count, _relationshipsCache.Values.SelectMany(r => r).Distinct().Count());

            _isInitialized = true;
        }
    }

    private async Task SaveItemToDiskAsync(TarsEngine.Models.KnowledgeItem item)
    {
        var filePath = GetItemFilePath(item.Id);
        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var json = JsonSerializer.Serialize(item, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(filePath, json);
    }

    private async Task SaveRelationshipToDiskAsync(KnowledgeRelationship relationship)
    {
        var filePath = GetRelationshipFilePath(relationship.Id);
        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var json = JsonSerializer.Serialize(relationship, new JsonSerializerOptions { WriteIndented = true });
        await File.WriteAllTextAsync(filePath, json);
    }

    private string GetItemFilePath(string id)
    {
        return Path.Combine(_dataDirectory, "Items", $"{id}.json");
    }

    private string GetRelationshipFilePath(string id)
    {
        return Path.Combine(_dataDirectory, "Relationships", $"{id}.json");
    }

    private async Task DeleteRelationshipsForItemAsync(string itemId)
    {
        // Get all relationships for the item
        var relationships = await GetRelationshipsForItemAsync(itemId);

        // Delete each relationship
        foreach (var relationship in relationships)
        {
            await DeleteRelationshipAsync(relationship.Id);
        }
    }
}
