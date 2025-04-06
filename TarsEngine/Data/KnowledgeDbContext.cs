using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Data;

/// <summary>
/// Database context for knowledge items and relationships
/// </summary>
public class KnowledgeDbContext
{
    private readonly ILogger<KnowledgeDbContext> _logger;
    private readonly string _dataDirectory;

    /// <summary>
    /// Initializes a new instance of the <see cref="KnowledgeDbContext"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public KnowledgeDbContext(ILogger<KnowledgeDbContext> logger)
    {
        _logger = logger;
        _dataDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data", "Knowledge");
        EnsureDirectoriesExist();
    }

    /// <summary>
    /// Gets the path to the items directory
    /// </summary>
    public string ItemsDirectory => Path.Combine(_dataDirectory, "Items");

    /// <summary>
    /// Gets the path to the relationships directory
    /// </summary>
    public string RelationshipsDirectory => Path.Combine(_dataDirectory, "Relationships");

    /// <summary>
    /// Saves a knowledge item to disk
    /// </summary>
    /// <param name="item">The knowledge item to save</param>
    public void SaveItem(KnowledgeItem item)
    {
        try
        {
            var filePath = GetItemFilePath(item.Id);
            var json = JsonSerializer.Serialize(item, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filePath, json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving knowledge item: {ItemId}", item.Id);
            throw;
        }
    }

    /// <summary>
    /// Loads a knowledge item from disk
    /// </summary>
    /// <param name="id">The ID of the knowledge item to load</param>
    /// <returns>The knowledge item, or null if not found</returns>
    public KnowledgeItem? LoadItem(string id)
    {
        try
        {
            var filePath = GetItemFilePath(id);
            if (!File.Exists(filePath))
            {
                return null;
            }

            var json = File.ReadAllText(filePath);
            return JsonSerializer.Deserialize<KnowledgeItem>(json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading knowledge item: {ItemId}", id);
            return null;
        }
    }

    /// <summary>
    /// Loads all knowledge items from disk
    /// </summary>
    /// <returns>The knowledge items</returns>
    public IEnumerable<KnowledgeItem> LoadAllItems()
    {
        try
        {
            var items = new List<KnowledgeItem>();
            foreach (var filePath in Directory.GetFiles(ItemsDirectory, "*.json"))
            {
                try
                {
                    var json = File.ReadAllText(filePath);
                    var item = JsonSerializer.Deserialize<KnowledgeItem>(json);
                    if (item != null)
                    {
                        items.Add(item);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error loading knowledge item from file: {FilePath}", filePath);
                }
            }
            return items;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading all knowledge items");
            return new List<KnowledgeItem>();
        }
    }

    /// <summary>
    /// Deletes a knowledge item from disk
    /// </summary>
    /// <param name="id">The ID of the knowledge item to delete</param>
    /// <returns>True if the item was deleted, false otherwise</returns>
    public bool DeleteItem(string id)
    {
        try
        {
            var filePath = GetItemFilePath(id);
            if (!File.Exists(filePath))
            {
                return false;
            }

            File.Delete(filePath);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deleting knowledge item: {ItemId}", id);
            return false;
        }
    }

    /// <summary>
    /// Saves a knowledge relationship to disk
    /// </summary>
    /// <param name="relationship">The knowledge relationship to save</param>
    public void SaveRelationship(KnowledgeRelationship relationship)
    {
        try
        {
            var filePath = GetRelationshipFilePath(relationship.Id);
            var json = JsonSerializer.Serialize(relationship, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filePath, json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving knowledge relationship: {RelationshipId}", relationship.Id);
            throw;
        }
    }

    /// <summary>
    /// Loads a knowledge relationship from disk
    /// </summary>
    /// <param name="id">The ID of the knowledge relationship to load</param>
    /// <returns>The knowledge relationship, or null if not found</returns>
    public KnowledgeRelationship? LoadRelationship(string id)
    {
        try
        {
            var filePath = GetRelationshipFilePath(id);
            if (!File.Exists(filePath))
            {
                return null;
            }

            var json = File.ReadAllText(filePath);
            return JsonSerializer.Deserialize<KnowledgeRelationship>(json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading knowledge relationship: {RelationshipId}", id);
            return null;
        }
    }

    /// <summary>
    /// Loads all knowledge relationships from disk
    /// </summary>
    /// <returns>The knowledge relationships</returns>
    public IEnumerable<KnowledgeRelationship> LoadAllRelationships()
    {
        try
        {
            var relationships = new List<KnowledgeRelationship>();
            foreach (var filePath in Directory.GetFiles(RelationshipsDirectory, "*.json"))
            {
                try
                {
                    var json = File.ReadAllText(filePath);
                    var relationship = JsonSerializer.Deserialize<KnowledgeRelationship>(json);
                    if (relationship != null)
                    {
                        relationships.Add(relationship);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error loading knowledge relationship from file: {FilePath}", filePath);
                }
            }
            return relationships;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading all knowledge relationships");
            return new List<KnowledgeRelationship>();
        }
    }

    /// <summary>
    /// Deletes a knowledge relationship from disk
    /// </summary>
    /// <param name="id">The ID of the knowledge relationship to delete</param>
    /// <returns>True if the relationship was deleted, false otherwise</returns>
    public bool DeleteRelationship(string id)
    {
        try
        {
            var filePath = GetRelationshipFilePath(id);
            if (!File.Exists(filePath))
            {
                return false;
            }

            File.Delete(filePath);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deleting knowledge relationship: {RelationshipId}", id);
            return false;
        }
    }

    private string GetItemFilePath(string id)
    {
        return Path.Combine(ItemsDirectory, $"{id}.json");
    }

    private string GetRelationshipFilePath(string id)
    {
        return Path.Combine(RelationshipsDirectory, $"{id}.json");
    }

    private void EnsureDirectoriesExist()
    {
        try
        {
            Directory.CreateDirectory(_dataDirectory);
            Directory.CreateDirectory(ItemsDirectory);
            Directory.CreateDirectory(RelationshipsDirectory);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating knowledge database directories");
            throw;
        }
    }
}
