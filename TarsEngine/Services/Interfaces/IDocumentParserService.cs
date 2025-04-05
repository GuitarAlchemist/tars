using System.Collections.Generic;
using System.Threading.Tasks;
using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Service for parsing documents in various formats
/// </summary>
public interface IDocumentParserService
{
    /// <summary>
    /// Parses a document from a file path
    /// </summary>
    /// <param name="filePath">The path to the document file</param>
    /// <param name="options">Optional parsing options</param>
    /// <returns>The parsing result</returns>
    Task<DocumentParsingResult> ParseDocumentAsync(string filePath, Dictionary<string, string>? options = null);

    /// <summary>
    /// Parses a document from text content
    /// </summary>
    /// <param name="content">The document content</param>
    /// <param name="documentType">The type of document</param>
    /// <param name="options">Optional parsing options</param>
    /// <returns>The parsing result</returns>
    Task<DocumentParsingResult> ParseDocumentContentAsync(string content, DocumentType documentType, Dictionary<string, string>? options = null);

    /// <summary>
    /// Parses a Markdown document
    /// </summary>
    /// <param name="content">The Markdown content</param>
    /// <param name="options">Optional parsing options</param>
    /// <returns>The parsing result</returns>
    Task<DocumentParsingResult> ParseMarkdownAsync(string content, Dictionary<string, string>? options = null);

    /// <summary>
    /// Parses a chat transcript document
    /// </summary>
    /// <param name="content">The chat transcript content</param>
    /// <param name="options">Optional parsing options</param>
    /// <returns>The parsing result</returns>
    Task<DocumentParsingResult> ParseChatTranscriptAsync(string content, Dictionary<string, string>? options = null);

    /// <summary>
    /// Parses a reflection document
    /// </summary>
    /// <param name="content">The reflection document content</param>
    /// <param name="options">Optional parsing options</param>
    /// <returns>The parsing result</returns>
    Task<DocumentParsingResult> ParseReflectionAsync(string content, Dictionary<string, string>? options = null);

    /// <summary>
    /// Parses a code file
    /// </summary>
    /// <param name="content">The code file content</param>
    /// <param name="language">The programming language</param>
    /// <param name="options">Optional parsing options</param>
    /// <returns>The parsing result</returns>
    Task<DocumentParsingResult> ParseCodeFileAsync(string content, string language, Dictionary<string, string>? options = null);

    /// <summary>
    /// Extracts code blocks from a document
    /// </summary>
    /// <param name="content">The document content</param>
    /// <param name="options">Optional extraction options</param>
    /// <returns>The extracted code blocks</returns>
    Task<List<CodeBlock>> ExtractCodeBlocksAsync(string content, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets the document type from a file path
    /// </summary>
    /// <param name="filePath">The path to the document file</param>
    /// <returns>The detected document type</returns>
    DocumentType GetDocumentTypeFromPath(string filePath);

    /// <summary>
    /// Gets the document type from content
    /// </summary>
    /// <param name="content">The document content</param>
    /// <returns>The detected document type</returns>
    DocumentType GetDocumentTypeFromContent(string content);
}
