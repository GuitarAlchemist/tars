using Microsoft.Extensions.Logging;
using Moq;
using TarsEngine.Models;
using TarsEngine.Services;
using TarsEngine.Services.Interfaces;
using Xunit;

// Use aliases to avoid ambiguity
using ModelKnowledgeItem = TarsEngine.Models.KnowledgeItem;
using ModelKnowledgeType = TarsEngine.Models.KnowledgeType;
using ServicesModelKnowledgeItem = TarsEngine.Services.Models.KnowledgeItem;
using ServicesModelKnowledgeType = TarsEngine.Services.Models.KnowledgeType;

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

        // Check that we have at least one item with content containing "fundamental unit of knowledge"
        Assert.Contains(result.Items, item => item.Content.Contains("fundamental unit of knowledge"));

        // Check that we have at least one item with content containing "knowledge extraction is important"
        Assert.Contains(result.Items, item => item.Content.Contains("knowledge extraction is important"));

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
        Assert.Contains(result.Items, item => item.Type.ToString() == ServicesModelKnowledgeType.CodePattern.ToString());
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
            Sections =
            [
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
                    CodeBlocks =
                    [
                        new CodeBlock
                        {
                            Language = "csharp",
                            Code = "public class TestClass { }",
                            IsExecutable = true
                        }
                    ]
                }
            ]
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
            Tags = ["concept", "knowledge"]
        };

        // Act
        var result = await _service.ExtractFromClassificationAsync(classification);

        // Assert
        Assert.NotNull(result);
        Assert.NotEmpty(result.Items);
        Assert.Equal(ServicesModelKnowledgeType.Concept.ToString(), result.Items[0].Type.ToString());
        Assert.Equal(classification.Content, result.Items[0].Content);
        Assert.Equal(classification.ConfidenceScore, result.Items[0].Confidence);
        // Note: Relevance is not available in the ServicesModelKnowledgeItem class
        // Assert.Equal(classification.RelevanceScore, result.Items[0].Relevance);
        Assert.Contains("concept", result.Items[0].Tags);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public async Task DetectRelationshipsAsync_WithRelatedItems_ReturnsRelationships()
    {
        // Arrange
        var items = new List<ServicesModelKnowledgeItem>
        {
            new()
            {
                Id = "1",
                Type = ServicesModelKnowledgeType.Concept,
                Content = "Knowledge extraction is the process of deriving structured information from unstructured data."
            },
            new()
            {
                Id = "2",
                Type = ServicesModelKnowledgeType.CodePattern,
                Content = "public class KnowledgeExtractor { }"
            }
        };

        // Act
        // Convert ServicesModelKnowledgeItem to ModelKnowledgeItem for the method call
        var modelItems = items.Select(item => new ModelKnowledgeItem
        {
            Id = item.Id,
            Type = (ModelKnowledgeType)(int)item.Type,
            Content = item.Content,
            Confidence = item.Confidence
        }).ToList();
        var relationships = await _service.DetectRelationshipsAsync(modelItems);

        // Assert
        Assert.NotNull(relationships);
        Assert.NotEmpty(relationships);
        Assert.Contains(relationships, r => r.SourceId == "1" && r.TargetId == "2");
    }

    [Fact]
    public async Task ValidateKnowledgeItemAsync_WithValidItem_ReturnsValidResult()
    {
        // Arrange
        var item = new ServicesModelKnowledgeItem
        {
            Type = ServicesModelKnowledgeType.Concept,
            Content = "Knowledge extraction is the process of deriving structured information from unstructured data.",
            Confidence = 0.9,
            // Relevance is not available in ServicesModelKnowledgeItem
        };

        // Act
        // Convert ServicesModelKnowledgeItem to ModelKnowledgeItem for the method call
        var modelItem = new ModelKnowledgeItem
        {
            Id = item.Id,
            Type = (ModelKnowledgeType)(int)item.Type,
            Content = item.Content,
            Confidence = item.Confidence,
            Relevance = 0.7 // Add relevance to pass the MinRelevance validation rule
        };
        var result = await _service.ValidateKnowledgeItemAsync(modelItem);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.IsValid);
        Assert.Empty(result.Issues);
    }

    [Fact]
    public async Task ValidateKnowledgeItemAsync_WithInvalidItem_ReturnsInvalidResult()
    {
        // Arrange
        var item = new ServicesModelKnowledgeItem
        {
            Type = ServicesModelKnowledgeType.Concept,
            Content = "", // Empty content should fail validation
            Confidence = 0.3, // Low confidence should fail validation
            // Relevance is not available in ServicesModelKnowledgeItem
        };

        // Act
        // Convert ServicesModelKnowledgeItem to ModelKnowledgeItem for the method call
        var modelItem = new ModelKnowledgeItem
        {
            Id = item.Id,
            Type = (ModelKnowledgeType)(int)item.Type,
            Content = item.Content,
            Confidence = item.Confidence,
            // Relevance is not available in ServicesModelKnowledgeItem
        };
        var result = await _service.ValidateKnowledgeItemAsync(modelItem);

        // Assert
        Assert.NotNull(result);
        Assert.False(result.IsValid);
        Assert.NotEmpty(result.Issues);
    }
}
