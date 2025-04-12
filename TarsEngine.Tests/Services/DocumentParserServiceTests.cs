using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using TarsEngine.Models;
using TarsEngine.Services;
using Xunit;

namespace TarsEngine.Tests.Services;

public class DocumentParserServiceTests
{
    private readonly Mock<ILogger<DocumentParserService>> _loggerMock;
    private readonly DocumentParserService _documentParserService;

    public DocumentParserServiceTests()
    {
        _loggerMock = new Mock<ILogger<DocumentParserService>>();
        _documentParserService = new DocumentParserService(_loggerMock.Object);
    }

    [Fact]
    public void GetDocumentTypeFromPath_ShouldReturnCorrectType()
    {
        // Arrange
        var markdownPath = "path/to/document.md";
        var chatPath = "path/to/chats/conversation.md";
        var reflectionPath = "path/to/reflections/insight.md";
        var csharpPath = "path/to/code.cs";
        var fsharpPath = "path/to/code.fs";
        var unknownPath = "path/to/unknown.xyz";

        // Act
        var markdownType = _documentParserService.GetDocumentTypeFromPath(markdownPath);
        var chatType = _documentParserService.GetDocumentTypeFromPath(chatPath);
        var reflectionType = _documentParserService.GetDocumentTypeFromPath(reflectionPath);
        var csharpType = _documentParserService.GetDocumentTypeFromPath(csharpPath);
        var fsharpType = _documentParserService.GetDocumentTypeFromPath(fsharpPath);
        var unknownType = _documentParserService.GetDocumentTypeFromPath(unknownPath);

        // Assert
        Assert.Equal(DocumentType.Markdown, markdownType);
        Assert.Equal(DocumentType.ChatTranscript, chatType);
        Assert.Equal(DocumentType.Reflection, reflectionType);
        Assert.Equal(DocumentType.CodeFile, csharpType);
        Assert.Equal(DocumentType.CodeFile, fsharpType);
        Assert.Equal(DocumentType.Unknown, unknownType);
    }

    [Fact]
    public void GetDocumentTypeFromContent_ShouldReturnCorrectType()
    {
        // Arrange
        var markdownContent = "# Heading\n\nThis is a markdown document.";
        var chatContent = "Human: What is TARS?\nAssistant: TARS is a software project.";
        var codeContent = "using System;\nnamespace Test { class Program { } }";
        var unknownContent = "This is just plain text.";

        // Act
        var markdownType = _documentParserService.GetDocumentTypeFromContent(markdownContent);
        var chatType = _documentParserService.GetDocumentTypeFromContent(chatContent);
        var codeType = _documentParserService.GetDocumentTypeFromContent(codeContent);
        var unknownType = _documentParserService.GetDocumentTypeFromContent(unknownContent);

        // Assert
        Assert.Equal(DocumentType.Markdown, markdownType);
        Assert.Equal(DocumentType.ChatTranscript, chatType);
        Assert.Equal(DocumentType.CodeFile, codeType);
        Assert.Equal(DocumentType.Unknown, unknownType);
    }

    [Fact]
    public async Task ParseMarkdownAsync_ShouldExtractSections()
    {
        // Arrange
        var markdownContent = @"# Test Document

This is a test document.

## Section 1

This is section 1.

```csharp
Console.WriteLine(""Hello, World!"");
```

## Section 2

This is section 2.";

        // Act
        var result = await _documentParserService.ParseMarkdownAsync(markdownContent);

        // Assert
        Assert.Equal(DocumentType.Markdown, result.DocumentType);
        Assert.Equal("Test Document", result.Title);
        Assert.Equal(3, result.Sections.Count);
        Assert.Single(result.Sections[1].CodeBlocks);
        Assert.Equal("csharp", result.Sections[1].CodeBlocks[0].Language);
        Assert.Equal("Console.WriteLine(\"Hello, World!\");".TrimEnd('\r'), result.Sections[1].CodeBlocks[0].Code.TrimEnd('\r'));
    }

    [Fact]
    public async Task ParseChatTranscriptAsync_ShouldExtractQuestionsAndAnswers()
    {
        // Arrange
        var chatContent = @"Human: What is TARS?
Assistant: TARS is a software project with CLI components in C# and engine components in F#.

Human: How can I use it?
Assistant: You can use the CLI commands to interact with TARS.";

        // Act
        var result = await _documentParserService.ParseChatTranscriptAsync(chatContent);

        // Assert
        Assert.Equal(DocumentType.ChatTranscript, result.DocumentType);
        Assert.Equal("Chat Transcript", result.Title);
        Assert.Equal(4, result.Sections.Count);
        Assert.Equal(ContentType.Question, result.Sections[0].ContentType);
        Assert.Equal(ContentType.Answer, result.Sections[1].ContentType);
        Assert.Equal(ContentType.Question, result.Sections[2].ContentType);
        Assert.Equal(ContentType.Answer, result.Sections[3].ContentType);
    }

    [Fact]
    public async Task ExtractCodeBlocksAsync_ShouldExtractCodeBlocks()
    {
        // Arrange
        var content = "```csharp\nConsole.WriteLine(\"Hello, World!\");\n```\n\n```fsharp\nprintfn \"Hello, World!\"\n```";

        // Act
        var codeBlocks = await _documentParserService.ExtractCodeBlocksAsync(content);

        // Assert
        Assert.Equal(2, codeBlocks.Count);
        Assert.Equal("csharp", codeBlocks[0].Language);
        Assert.Equal("Console.WriteLine(\"Hello, World!\");".TrimEnd('\r'), codeBlocks[0].Code.TrimEnd('\r'));
        Assert.Equal("fsharp", codeBlocks[1].Language);
        Assert.Equal("printfn \"Hello, World!\"".TrimEnd('\r'), codeBlocks[1].Code.TrimEnd('\r'));
    }
}