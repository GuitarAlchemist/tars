using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using TarsEngine.Models;
using TarsEngine.Services;
using TarsEngine.Services.Interfaces;
using Xunit;

namespace TarsEngine.Tests.Services;

public class KnowledgeExtractorServiceTests
{
    private readonly Mock<ILogger<KnowledgeExtractorService>> _loggerMock;
    private readonly Mock<IDocumentParserService> _documentParserServiceMock;
    private readonly Mock<IContentClassifierService> _contentClassifierServiceMock;
    private readonly KnowledgeExtractorService _service;

    public KnowledgeExtractorServiceTests()
    {
        _loggerMock = new Mock<ILogger<KnowledgeExtractorService>>();
        _documentParserServiceMock = new Mock<IDocumentParserService>();
        _contentClassifierServiceMock = new Mock<IContentClassifierService>();
        _service = new KnowledgeExtractorService(
            _loggerMock.Object,
            _documentParserServiceMock.Object,
            _contentClassifierServiceMock.Object);
    }

    [Fact]
    public async Task ExtractFromTextAsync_WithValidContent_ReturnsKnowledgeItems()
    {
        // Arrange
        var content = "The concept is defined as a fundamental unit of knowledge. I realized that knowledge extraction is important.";

        // Act
        var result = await _service.ExtractFromTextAsync(content);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result.Items);
        Assert.Contains(result.Items, item => item.Type == KnowledgeType.Concept);
        Assert.Contains(result.Items, item => item.Type == KnowledgeType.Insight);
        Assert.Equal("text", result.Source);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task ExtractFromTextAsync_WithEmptyContent_ReturnsError()
    {
        // Arrange
        var content = string.Empty;

        // Act
        var result = await _service.ExtractFromTextAsync(content);

        // Assert
        Assert.NotNull(result);
        Assert.Empty(result.Items);
        Assert.NotEmpty(result.Errors);
        Assert.Contains("Content is empty or whitespace", result.Errors[0]);
    }

    [Fact]
    public async Task ExtractFromCodeAsync_WithValidCode_ReturnsKnowledgeItems()
    {
        // Arrange
        var code = @"
public class TestClass
{
    public void TestMethod()
    {
        // Test method implementation
    }
}";
        var language = "csharp";

        // Act
        var result = await _service.ExtractFromCodeAsync(code, language);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result.Items);
        Assert.Contains(result.Items, item => item.Type == KnowledgeType.CodePattern);
        Assert.Contains(result.Items, item => item.Content.Contains("TestClass"));
        Assert.Contains(result.Items, item => item.Content.Contains("TestMethod"));
        Assert.Equal("code", result.Source);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task ExtractFromDocumentAsync_WithValidDocument_ReturnsKnowledgeItems()
    {
        // Arrange
        var document = new DocumentParsingResult
        {
            DocumentType = DocumentType.Markdown,
            Title = "Test Document",
            DocumentPath = "test.md",
            Sections = new List<ContentSection>
            {
                new ContentSection
                {
                    Heading = "Concepts",
                    ContentType = ContentType.Concept,
                    RawContent = "The concept is defined as a fundamental unit of knowledge.",
                    Order = 0
                },
                new ContentSection
                {
                    Heading = "Code Example",
                    ContentType = ContentType.Code,
                    RawContent = "public class TestClass { }",
                    Order = 1,
                    CodeBlocks = new List<CodeBlock>
                    {
                        new CodeBlock
                        {
                            Language = "csharp",
                            Code = "public class TestClass { }",
                            IsExecutable = true
                        }
                    }
                }
            }
        };

        // Act
        var result = await _service.ExtractFromDocumentAsync(document);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result.Items);
        Assert.Equal(document.DocumentPath, result.Source);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task ExtractFromClassificationAsync_WithValidClassification_ReturnsKnowledgeItem()
    {
        // Arrange
        var classification = new ContentClassification
        {
            PrimaryCategory = ContentCategory.Concept,
            Content = "The concept is defined as a fundamental unit of knowledge.",
            ConfidenceScore = 0.9,
            RelevanceScore = 0.8,
            QualityScore = 0.7,
            Tags = new List<string> { "concept", "knowledge" }
        };

        // Act
        var result = await _service.ExtractFromClassificationAsync(classification);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result.Items);
        Assert.Equal(KnowledgeType.Concept, result.Items[0].Type);
        Assert.Equal(classification.Content, result.Items[0].Content);
        Assert.Equal(classification.ConfidenceScore, result.Items[0].Confidence);
        Assert.Equal(classification.RelevanceScore, result.Items[0].Relevance);
        Assert.Contains("concept", result.Items[0].Tags);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task DetectRelationshipsAsync_WithRelatedItems_ReturnsRelationships()
    {
        // Arrange
        var items = new List<KnowledgeItem>
        {
            new KnowledgeItem
            {
                Id = "1",
                Type = KnowledgeType.Concept,
                Content = "Knowledge extraction is the process of deriving structured information from unstructured data."
            },
            new KnowledgeItem
            {
                Id = "2",
                Type = KnowledgeType.CodePattern,
                Content = "public class KnowledgeExtractor { }"
            }
        };

        // Act
        var relationships = await _service.DetectRelationshipsAsync(items);

        // Assert
        Assert.NotNull(relationships);
        Assert.NotEmpty(relationships);
        Assert.Contains(relationships, r => r.SourceId == "1" && r.TargetId == "2");
    }

    [Fact]
    public async Task ValidateKnowledgeItemAsync_WithValidItem_ReturnsValidResult()
    {
        // Arrange
        var item = new KnowledgeItem
        {
            Type = KnowledgeType.Concept,
            Content = "Knowledge extraction is the process of deriving structured information from unstructured data.",
            Confidence = 0.9,
            Relevance = 0.8
        };

        // Act
        var result = await _service.ValidateKnowledgeItemAsync(item);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.IsValid);
        Assert.Empty(result.Issues);
    }

    [Fact]
    public async Task ValidateKnowledgeItemAsync_WithInvalidItem_ReturnsInvalidResult()
    {
        // Arrange
        var item = new KnowledgeItem
        {
            Type = KnowledgeType.Concept,
            Content = "", // Empty content should fail validation
            Confidence = 0.3, // Low confidence should fail validation
            Relevance = 0.2 // Low relevance should fail validation
        };

        // Act
        var result = await _service.ValidateKnowledgeItemAsync(item);

        // Assert
        Assert.NotNull(result);
        Assert.False(result.IsValid);
        Assert.NotEmpty(result.Issues);
    }
}
