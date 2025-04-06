using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using TarsEngine.Models;
using TarsEngine.Services;
using Xunit;

namespace TarsEngine.Tests.Services;

public class KnowledgeRepositoryTests
{
    private readonly Mock<ILogger<KnowledgeRepository>> _loggerMock;
    private readonly KnowledgeRepository _repository;
    private readonly string _testDataDirectory;

    public KnowledgeRepositoryTests()
    {
        _loggerMock = new Mock<ILogger<KnowledgeRepository>>();
        _repository = new KnowledgeRepository(_loggerMock.Object);

        // Use reflection to set the data directory to a test directory
        _testDataDirectory = Path.Combine(Path.GetTempPath(), "TarsEngineTests", "Knowledge");
        var fieldInfo = typeof(KnowledgeRepository).GetField("_dataDirectory", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        fieldInfo?.SetValue(_repository, _testDataDirectory);

        // Clean up test directory if it exists
        if (Directory.Exists(_testDataDirectory))
        {
            Directory.Delete(_testDataDirectory, true);
        }
    }

    [Fact]
    public async Task AddItemAsync_WithValidItem_AddsItemToRepository()
    {
        // Arrange
        var item = new KnowledgeItem
        {
            Type = KnowledgeType.Concept,
            Content = "Test concept",
            Confidence = 0.9,
            Relevance = 0.8,
            Tags = new List<string> { "test", "concept" }
        };

        // Act
        var result = await _repository.AddItemAsync(item);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(item.Id, result.Id);
        Assert.Equal(item.Type, result.Type);
        Assert.Equal(item.Content, result.Content);

        // Verify item was saved
        var retrievedItem = await _repository.GetItemAsync(item.Id);
        Assert.NotNull(retrievedItem);
        Assert.Equal(item.Id, retrievedItem.Id);
    }

    [Fact]
    public async Task AddItemsAsync_WithValidItems_AddsItemsToRepository()
    {
        // Arrange
        var items = new List<KnowledgeItem>
        {
            new KnowledgeItem
            {
                Type = KnowledgeType.Concept,
                Content = "Test concept 1",
                Confidence = 0.9,
                Relevance = 0.8,
                Tags = new List<string> { "test", "concept" }
            },
            new KnowledgeItem
            {
                Type = KnowledgeType.CodePattern,
                Content = "Test code pattern",
                Confidence = 0.8,
                Relevance = 0.7,
                Tags = new List<string> { "test", "code" }
            }
        };

        // Act
        var results = await _repository.AddItemsAsync(items);

        // Assert
        Assert.NotNull(results);
        Assert.Equal(items.Count, results.Count());

        // Verify items were saved
        foreach (var item in items)
        {
            var retrievedItem = await _repository.GetItemAsync(item.Id);
            Assert.NotNull(retrievedItem);
            Assert.Equal(item.Id, retrievedItem.Id);
        }
    }

    [Fact]
    public async Task UpdateItemAsync_WithValidItem_UpdatesItemInRepository()
    {
        // Arrange
        var item = new KnowledgeItem
        {
            Type = KnowledgeType.Concept,
            Content = "Test concept",
            Confidence = 0.9,
            Relevance = 0.8,
            Tags = new List<string> { "test", "concept" }
        };
        await _repository.AddItemAsync(item);

        // Update the item
        item.Content = "Updated test concept";
        item.Confidence = 0.95;
        item.Tags.Add("updated");

        // Act
        var result = await _repository.UpdateItemAsync(item);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(item.Id, result.Id);
        Assert.Equal(item.Content, result.Content);
        Assert.Equal(item.Confidence, result.Confidence);
        Assert.Contains("updated", result.Tags);

        // Verify item was updated
        var retrievedItem = await _repository.GetItemAsync(item.Id);
        Assert.NotNull(retrievedItem);
        Assert.Equal(item.Content, retrievedItem.Content);
        Assert.Equal(item.Confidence, retrievedItem.Confidence);
        Assert.Contains("updated", retrievedItem.Tags);
    }

    [Fact]
    public async Task DeleteItemAsync_WithValidId_DeletesItemFromRepository()
    {
        // Arrange
        var item = new KnowledgeItem
        {
            Type = KnowledgeType.Concept,
            Content = "Test concept",
            Confidence = 0.9,
            Relevance = 0.8,
            Tags = new List<string> { "test", "concept" }
        };
        await _repository.AddItemAsync(item);

        // Act
        var result = await _repository.DeleteItemAsync(item.Id);

        // Assert
        Assert.True(result);

        // Verify item was deleted
        var retrievedItem = await _repository.GetItemAsync(item.Id);
        Assert.Null(retrievedItem);
    }

    [Fact]
    public async Task SearchItemsAsync_WithValidQuery_ReturnsMatchingItems()
    {
        // Arrange
        var items = new List<KnowledgeItem>
        {
            new KnowledgeItem
            {
                Type = KnowledgeType.Concept,
                Content = "Knowledge extraction is important",
                Confidence = 0.9,
                Relevance = 0.8,
                Tags = new List<string> { "knowledge", "extraction" }
            },
            new KnowledgeItem
            {
                Type = KnowledgeType.CodePattern,
                Content = "public class KnowledgeExtractor { }",
                Confidence = 0.8,
                Relevance = 0.7,
                Tags = new List<string> { "code", "extractor" }
            },
            new KnowledgeItem
            {
                Type = KnowledgeType.Insight,
                Content = "Autonomous systems can improve over time",
                Confidence = 0.7,
                Relevance = 0.6,
                Tags = new List<string> { "autonomous", "improvement" }
            }
        };
        await _repository.AddItemsAsync(items);

        // Act
        var results = await _repository.SearchItemsAsync("knowledge");

        // Assert
        Assert.NotNull(results);
        Assert.NotEmpty(results);
        Assert.Equal(2, results.Count()); // Should match the first two items
        Assert.Contains(results, item => item.Content.Contains("Knowledge extraction"));
        Assert.Contains(results, item => item.Content.Contains("KnowledgeExtractor"));
    }

    [Fact]
    public async Task GetItemsByTypeAsync_WithValidType_ReturnsMatchingItems()
    {
        // Arrange
        var items = new List<KnowledgeItem>
        {
            new KnowledgeItem
            {
                Type = KnowledgeType.Concept,
                Content = "Knowledge extraction is important",
                Confidence = 0.9,
                Relevance = 0.8,
                Tags = new List<string> { "knowledge", "extraction" }
            },
            new KnowledgeItem
            {
                Type = KnowledgeType.CodePattern,
                Content = "public class KnowledgeExtractor { }",
                Confidence = 0.8,
                Relevance = 0.7,
                Tags = new List<string> { "code", "extractor" }
            },
            new KnowledgeItem
            {
                Type = KnowledgeType.Concept,
                Content = "Autonomous systems can improve over time",
                Confidence = 0.7,
                Relevance = 0.6,
                Tags = new List<string> { "autonomous", "improvement" }
            }
        };
        await _repository.AddItemsAsync(items);

        // Act
        var results = await _repository.GetItemsByTypeAsync(KnowledgeType.Concept);

        // Assert
        Assert.NotNull(results);
        Assert.NotEmpty(results);
        Assert.Equal(2, results.Count()); // Should match the two Concept items
        Assert.All(results, item => Assert.Equal(KnowledgeType.Concept, item.Type));
    }

    [Fact]
    public async Task GetItemsByTagAsync_WithValidTag_ReturnsMatchingItems()
    {
        // Arrange
        var items = new List<KnowledgeItem>
        {
            new KnowledgeItem
            {
                Type = KnowledgeType.Concept,
                Content = "Knowledge extraction is important",
                Confidence = 0.9,
                Relevance = 0.8,
                Tags = new List<string> { "knowledge", "extraction" }
            },
            new KnowledgeItem
            {
                Type = KnowledgeType.CodePattern,
                Content = "public class KnowledgeExtractor { }",
                Confidence = 0.8,
                Relevance = 0.7,
                Tags = new List<string> { "code", "extractor" }
            },
            new KnowledgeItem
            {
                Type = KnowledgeType.Insight,
                Content = "Autonomous systems can improve over time",
                Confidence = 0.7,
                Relevance = 0.6,
                Tags = new List<string> { "autonomous", "improvement" }
            }
        };
        await _repository.AddItemsAsync(items);

        // Act
        var results = await _repository.GetItemsByTagAsync("extraction");

        // Assert
        Assert.NotNull(results);
        Assert.NotEmpty(results);
        Assert.Single(results); // Should match only the first item
        Assert.Contains(results, item => item.Content.Contains("Knowledge extraction"));
    }

    [Fact]
    public async Task AddRelationshipAsync_WithValidRelationship_AddsRelationshipToRepository()
    {
        // Arrange
        var item1 = new KnowledgeItem
        {
            Type = KnowledgeType.Concept,
            Content = "Knowledge extraction is important",
            Confidence = 0.9,
            Relevance = 0.8,
            Tags = new List<string> { "knowledge", "extraction" }
        };
        var item2 = new KnowledgeItem
        {
            Type = KnowledgeType.CodePattern,
            Content = "public class KnowledgeExtractor { }",
            Confidence = 0.8,
            Relevance = 0.7,
            Tags = new List<string> { "code", "extractor" }
        };
        await _repository.AddItemsAsync(new[] { item1, item2 });

        var relationship = new KnowledgeRelationship
        {
            SourceId = item1.Id,
            TargetId = item2.Id,
            Type = RelationshipType.Implements,
            Strength = 0.9,
            Description = "Code implements concept"
        };

        // Act
        var result = await _repository.AddRelationshipAsync(relationship);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(relationship.Id, result.Id);
        Assert.Equal(relationship.SourceId, result.SourceId);
        Assert.Equal(relationship.TargetId, result.TargetId);
        Assert.Equal(relationship.Type, result.Type);

        // Verify relationship was saved
        var relationships = await _repository.GetRelationshipsForItemAsync(item1.Id);
        Assert.NotNull(relationships);
        Assert.NotEmpty(relationships);
        Assert.Contains(relationships, r => r.Id == relationship.Id);
    }

    [Fact]
    public async Task GetStatisticsAsync_WithItems_ReturnsValidStatistics()
    {
        // Arrange
        var items = new List<KnowledgeItem>
        {
            new KnowledgeItem
            {
                Type = KnowledgeType.Concept,
                Content = "Knowledge extraction is important",
                Confidence = 0.9,
                Relevance = 0.8,
                Tags = new List<string> { "knowledge", "extraction" },
                Source = "test1.md"
            },
            new KnowledgeItem
            {
                Type = KnowledgeType.CodePattern,
                Content = "public class KnowledgeExtractor { }",
                Confidence = 0.8,
                Relevance = 0.7,
                Tags = new List<string> { "code", "extractor" },
                Source = "test1.md"
            },
            new KnowledgeItem
            {
                Type = KnowledgeType.Insight,
                Content = "Autonomous systems can improve over time",
                Confidence = 0.7,
                Relevance = 0.6,
                Tags = new List<string> { "autonomous", "improvement" },
                Source = "test2.md"
            }
        };
        await _repository.AddItemsAsync(items);

        // Add a relationship
        var relationship = new KnowledgeRelationship
        {
            SourceId = items[0].Id,
            TargetId = items[1].Id,
            Type = RelationshipType.Implements,
            Strength = 0.9,
            Description = "Code implements concept"
        };
        await _repository.AddRelationshipAsync(relationship);

        // Act
        var stats = await _repository.GetStatisticsAsync();

        // Assert
        Assert.NotNull(stats);
        Assert.Equal(3, stats.TotalItems);
        Assert.Equal(1, stats.TotalRelationships);
        
        // Check items by type
        Assert.Equal(1, stats.ItemsByType[KnowledgeType.Concept]);
        Assert.Equal(1, stats.ItemsByType[KnowledgeType.CodePattern]);
        Assert.Equal(1, stats.ItemsByType[KnowledgeType.Insight]);
        
        // Check items by tag
        Assert.Equal(1, stats.ItemsByTag["knowledge"]);
        Assert.Equal(1, stats.ItemsByTag["extraction"]);
        Assert.Equal(1, stats.ItemsByTag["code"]);
        Assert.Equal(1, stats.ItemsByTag["extractor"]);
        Assert.Equal(1, stats.ItemsByTag["autonomous"]);
        Assert.Equal(1, stats.ItemsByTag["improvement"]);
        
        // Check items by source
        Assert.Equal(2, stats.ItemsBySource["test1.md"]);
        Assert.Equal(1, stats.ItemsBySource["test2.md"]);
        
        // Check relationships by type
        Assert.Equal(1, stats.RelationshipsByType[RelationshipType.Implements]);
        
        // Check average scores
        Assert.Equal(0.8, stats.AverageConfidence, 1);
        Assert.Equal(0.7, stats.AverageRelevance, 1);
    }
}
