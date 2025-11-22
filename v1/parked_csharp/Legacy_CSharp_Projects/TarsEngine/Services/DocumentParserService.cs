using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for parsing documents in various formats
/// </summary>
public class DocumentParserService : IDocumentParserService
{
    private readonly ILogger<DocumentParserService> _logger;
    private readonly Dictionary<string, DocumentParsingResult> _parsingCache = new();

    // Regular expressions for parsing
    private static readonly Regex _markdownHeadingRegex = new(@"^(#{1,6})\s+(.+)$", RegexOptions.Multiline);
    private static readonly Regex _markdownCodeBlockRegex = new(@"```([a-zA-Z0-9]*)\n([\s\S]*?)\n```", RegexOptions.Multiline);
    private static readonly Regex _chatQuestionRegex = new(@"^(Human|User):\s*(.+)$", RegexOptions.Multiline);
    private static readonly Regex _chatAnswerRegex = new(@"^(Assistant|AI|Bot):\s*(.+)$", RegexOptions.Multiline);

    /// <summary>
    /// Initializes a new instance of the <see cref="DocumentParserService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public DocumentParserService(ILogger<DocumentParserService> logger)
    {
        _logger = logger;
    }

    /// <inheritdoc/>
    public async Task<DocumentParsingResult> ParseDocumentAsync(string filePath, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Parsing document: {FilePath}", filePath);

            // Check if the file exists
            if (!File.Exists(filePath))
            {
                _logger.LogError("File not found: {FilePath}", filePath);
                return new DocumentParsingResult
                {
                    DocumentPath = filePath,
                    Errors = { "File not found" }
                };
            }

            // Check if we have a cached result
            var cacheKey = $"{filePath}_{File.GetLastWriteTimeUtc(filePath).Ticks}";
            if (_parsingCache.TryGetValue(cacheKey, out var cachedResult))
            {
                _logger.LogInformation("Using cached parsing result for: {FilePath}", filePath);
                return cachedResult;
            }

            // Read the file content
            var content = await File.ReadAllTextAsync(filePath);

            // Determine the document type
            var documentType = GetDocumentTypeFromPath(filePath);

            // Parse the document based on its type
            var result = await ParseDocumentContentAsync(content, documentType, options);
            result.DocumentPath = filePath;
            result.Title = Path.GetFileNameWithoutExtension(filePath);

            // Cache the result
            _parsingCache[cacheKey] = result;

            _logger.LogInformation("Successfully parsed document: {FilePath}", filePath);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing document: {FilePath}", filePath);
            return new DocumentParsingResult
            {
                DocumentPath = filePath,
                Errors = { $"Error parsing document: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<DocumentParsingResult> ParseDocumentContentAsync(string content, DocumentType documentType, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Parsing document content of type: {DocumentType}", documentType);

            // Parse the document based on its type
            return documentType switch
            {
                DocumentType.Markdown => await ParseMarkdownAsync(content, options),
                DocumentType.ChatTranscript => await ParseChatTranscriptAsync(content, options),
                DocumentType.Reflection => await ParseReflectionAsync(content, options),
                DocumentType.CodeFile => await ParseCodeFileAsync(content, options?.GetValueOrDefault("language") ?? "unknown", options),
                _ => await ParseGenericDocumentAsync(content, documentType, options)
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing document content of type: {DocumentType}", documentType);
            return new DocumentParsingResult
            {
                DocumentType = documentType,
                Errors = { $"Error parsing document content: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<DocumentParsingResult> ParseMarkdownAsync(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Parsing Markdown document");

            var result = new DocumentParsingResult
            {
                DocumentType = DocumentType.Markdown
            };

            // Extract title from the first heading
            var titleMatch = _markdownHeadingRegex.Match(content);
            if (titleMatch.Success)
            {
                result.Title = titleMatch.Groups[2].Value.Trim();
            }

            // Extract sections based on headings
            var sections = new List<ContentSection>();
            var currentSection = new ContentSection
            {
                ContentType = ContentType.Text,
                Order = 0
            };

            var lines = content.Split('\n');
            var currentContent = new List<string>();
            var inCodeBlock = false;
            var codeBlockLanguage = string.Empty;
            var codeBlockContent = new List<string>();
            var codeBlockStartLine = 0;

            for (var i = 0; i < lines.Length; i++)
            {
                var line = lines[i];

                // Check for code block start/end
                if (line.StartsWith("```"))
                {
                    if (!inCodeBlock)
                    {
                        // Start of code block
                        inCodeBlock = true;
                        codeBlockLanguage = line.Substring(3).Trim();
                        codeBlockContent.Clear();
                        codeBlockStartLine = i;
                    }
                    else
                    {
                        // End of code block
                        inCodeBlock = false;
                        var codeBlock = new CodeBlock
                        {
                            Language = codeBlockLanguage,
                            Code = string.Join("\n", codeBlockContent),
                            StartLine = codeBlockStartLine,
                            EndLine = i,
                            IsExecutable = IsExecutableLanguage(codeBlockLanguage)
                        };
                        currentSection.CodeBlocks.Add(codeBlock);
                    }
                    continue;
                }

                if (inCodeBlock)
                {
                    // Inside code block
                    codeBlockContent.Add(line);
                    continue;
                }

                // Check for heading
                var headingMatch = _markdownHeadingRegex.Match(line);
                if (headingMatch.Success)
                {
                    // Save the current section if it has content
                    if (currentContent.Count > 0)
                    {
                        currentSection.RawContent = string.Join("\n", currentContent);
                        currentSection.ProcessedContent = ProcessContent(currentSection.RawContent);
                        sections.Add(currentSection);
                    }

                    // Start a new section
                    currentSection = new ContentSection
                    {
                        Heading = headingMatch.Groups[2].Value.Trim(),
                        ContentType = ContentType.Text,
                        Order = sections.Count
                    };
                    currentContent.Clear();
                    continue;
                }

                // Add line to current content
                currentContent.Add(line);
            }

            // Add the last section
            if (currentContent.Count > 0)
            {
                currentSection.RawContent = string.Join("\n", currentContent);
                currentSection.ProcessedContent = ProcessContent(currentSection.RawContent);
                sections.Add(currentSection);
            }

            result.Sections = sections;

            // Extract code blocks from the entire document
            result.Metadata["CodeBlockCount"] = (await ExtractCodeBlocksAsync(content, options)).Count.ToString();

            _logger.LogInformation("Successfully parsed Markdown document with {SectionCount} sections", sections.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing Markdown document");
            return new DocumentParsingResult
            {
                DocumentType = DocumentType.Markdown,
                Errors = { $"Error parsing Markdown document: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<DocumentParsingResult> ParseChatTranscriptAsync(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Parsing chat transcript document");

            var result = new DocumentParsingResult
            {
                DocumentType = DocumentType.ChatTranscript,
                Title = "Chat Transcript"
            };

            // Extract metadata from the content
            var metadataMatch = Regex.Match(content, @"---\s*\n([\s\S]*?)\n---");
            if (metadataMatch.Success)
            {
                var metadataContent = metadataMatch.Groups[1].Value;
                var metadataLines = metadataContent.Split('\n');
                foreach (var line in metadataLines)
                {
                    var parts = line.Split(':', 2);
                    if (parts.Length == 2)
                    {
                        var key = parts[0].Trim();
                        var value = parts[1].Trim();
                        result.Metadata[key] = value;

                        // Use title from metadata if available
                        if (key.Equals("title", StringComparison.OrdinalIgnoreCase))
                        {
                            result.Title = value;
                        }
                    }
                }

                // Remove metadata from content
                content = content.Replace(metadataMatch.Value, "").Trim();
            }

            // Extract sections based on questions and answers
            var sections = new List<ContentSection>();
            var questionMatches = _chatQuestionRegex.Matches(content);
            var answerMatches = _chatAnswerRegex.Matches(content);

            // Combine questions and answers into a conversation
            var conversationParts = new List<(int Index, string Type, string Content)>();
            foreach (Match match in questionMatches)
            {
                conversationParts.Add((match.Index, "Question", match.Groups[2].Value));
            }
            foreach (Match match in answerMatches)
            {
                conversationParts.Add((match.Index, "Answer", match.Groups[2].Value));
            }

            // Sort by position in the document
            conversationParts = conversationParts.OrderBy(p => p.Index).ToList();

            // Create sections for each part
            for (var i = 0; i < conversationParts.Count; i++)
            {
                var part = conversationParts[i];
                var contentType = part.Type == "Question" ? ContentType.Question : ContentType.Answer;
                
                var section = new ContentSection
                {
                    Heading = $"{part.Type} {i + 1}",
                    ContentType = contentType,
                    RawContent = part.Content,
                    ProcessedContent = ProcessContent(part.Content),
                    Order = i
                };

                // Extract code blocks from this section
                var codeBlocks = await ExtractCodeBlocksAsync(part.Content, options);
                section.CodeBlocks.AddRange(codeBlocks);

                sections.Add(section);
            }

            result.Sections = sections;

            _logger.LogInformation("Successfully parsed chat transcript with {SectionCount} sections", sections.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing chat transcript document");
            return new DocumentParsingResult
            {
                DocumentType = DocumentType.ChatTranscript,
                Errors = { $"Error parsing chat transcript document: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<DocumentParsingResult> ParseReflectionAsync(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Parsing reflection document");

            var result = new DocumentParsingResult
            {
                DocumentType = DocumentType.Reflection
            };

            // Extract title from the first heading
            var titleMatch = _markdownHeadingRegex.Match(content);
            if (titleMatch.Success)
            {
                result.Title = titleMatch.Groups[2].Value.Trim();
            }

            // Parse as Markdown first
            var markdownResult = await ParseMarkdownAsync(content, options);
            result.Sections = markdownResult.Sections;

            // Identify insight sections
            foreach (var section in result.Sections)
            {
                // Check if the section heading or content suggests it's an insight
                if (section.Heading.Contains("insight", StringComparison.OrdinalIgnoreCase) ||
                    section.Heading.Contains("reflection", StringComparison.OrdinalIgnoreCase) ||
                    section.Heading.Contains("lesson", StringComparison.OrdinalIgnoreCase) ||
                    section.RawContent.Contains("I realized", StringComparison.OrdinalIgnoreCase) ||
                    section.RawContent.Contains("key takeaway", StringComparison.OrdinalIgnoreCase))
                {
                    section.ContentType = ContentType.Insight;
                }
            }

            // Add reflection-specific metadata
            result.Metadata["InsightCount"] = result.Sections.Count(s => s.ContentType == ContentType.Insight).ToString();

            _logger.LogInformation("Successfully parsed reflection document with {InsightCount} insights", 
                result.Metadata["InsightCount"]);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing reflection document");
            return new DocumentParsingResult
            {
                DocumentType = DocumentType.Reflection,
                Errors = { $"Error parsing reflection document: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<DocumentParsingResult> ParseCodeFileAsync(string content, string language, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Parsing code file with language: {Language}", language);

            // Move CPU-intensive parsing operations to a background thread
            return await Task.Run(async () =>
            {
                var result = new DocumentParsingResult
                {
                    DocumentType = DocumentType.CodeFile,
                    Title = $"Code File ({language})"
                };

                // Create a single section for the code
                var section = new ContentSection
                {
                    ContentType = ContentType.Code,
                    RawContent = content,
                    ProcessedContent = content,
                    Order = 0
                };

                // Create a code block for the entire file
                var codeBlock = new CodeBlock
                {
                    Language = language,
                    Code = content,
                    StartLine = 0,
                    EndLine = content.Split('\n').Length,
                    IsExecutable = IsExecutableLanguage(language)
                };
                section.CodeBlocks.Add(codeBlock);

                result.Sections = [section];

                // Extract comments from the code
                var comments = ExtractCommentsFromCode(content, language);
                if (comments.Any())
                {
                    var commentSection = new ContentSection
                    {
                        Heading = "Comments",
                        ContentType = ContentType.Text,
                        RawContent = string.Join("\n", comments),
                        ProcessedContent = string.Join("\n", comments),
                        Order = 1
                    };
                    result.Sections.Add(commentSection);
                }

                // Extract code blocks if any special processing is needed
                var codeBlocks = await ExtractCodeBlocksAsync(content, options);
                if (codeBlocks.Any())
                {
                    section.CodeBlocks.AddRange(codeBlocks);
                }

                _logger.LogInformation("Successfully parsed code file with {LineCount} lines", codeBlock.EndLine);
                return result;
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing code file");
            return new DocumentParsingResult
            {
                DocumentType = DocumentType.CodeFile,
                Errors = { $"Error parsing code file: {ex.Message}" }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<List<CodeBlock>> ExtractCodeBlocksAsync(string content, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Extracting code blocks from content");

            // Move regex matching and code block extraction to a background thread
            // since it could be CPU-intensive for large documents
            return await Task.Run(() =>
            {
                var codeBlocks = new List<CodeBlock>();
                var matches = _markdownCodeBlockRegex.Matches(content);

                foreach (Match match in matches)
                {
                    var language = match.Groups[1].Value.Trim();
                    var code = match.Groups[2].Value;

                    var codeBlock = new CodeBlock
                    {
                        Language = language,
                        Code = code,
                        StartLine = content.Substring(0, match.Index).Count(c => c == '\n'),
                        EndLine = content.Substring(0, match.Index + match.Length).Count(c => c == '\n'),
                        IsExecutable = IsExecutableLanguage(language)
                    };

                    codeBlocks.Add(codeBlock);
                }

                _logger.LogInformation("Extracted {CodeBlockCount} code blocks", codeBlocks.Count);
                return codeBlocks;
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting code blocks");
            return [];
        }
    }

    /// <inheritdoc/>
    public DocumentType GetDocumentTypeFromPath(string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        var fileName = Path.GetFileNameWithoutExtension(filePath).ToLowerInvariant();
        var directoryName = Path.GetDirectoryName(filePath)?.ToLowerInvariant() ?? string.Empty;

        // Check for chat transcripts
        if (directoryName.Contains("chat") || fileName.Contains("chat") || fileName.Contains("conversation"))
        {
            return DocumentType.ChatTranscript;
        }

        // Check for reflections
        if (directoryName.Contains("reflection") || fileName.Contains("reflection") || fileName.Contains("insight"))
        {
            return DocumentType.Reflection;
        }

        // Check file extension
        return extension switch
        {
            ".md" => DocumentType.Markdown,
            ".markdown" => DocumentType.Markdown,
            ".cs" => DocumentType.CodeFile,
            ".fs" => DocumentType.CodeFile,
            ".js" => DocumentType.CodeFile,
            ".ts" => DocumentType.CodeFile,
            ".py" => DocumentType.CodeFile,
            ".java" => DocumentType.CodeFile,
            ".cpp" => DocumentType.CodeFile,
            ".c" => DocumentType.CodeFile,
            ".h" => DocumentType.CodeFile,
            ".json" => DocumentType.CodeFile,
            ".xml" => DocumentType.CodeFile,
            ".yaml" => DocumentType.CodeFile,
            ".yml" => DocumentType.CodeFile,
            _ => DocumentType.Unknown
        };
    }

    /// <inheritdoc/>
    public DocumentType GetDocumentTypeFromContent(string content)
    {
        // Check for Markdown
        if (content.Contains("# ") || content.Contains("## ") || _markdownCodeBlockRegex.IsMatch(content))
        {
            return DocumentType.Markdown;
        }

        // Check for chat transcript
        if (_chatQuestionRegex.IsMatch(content) && _chatAnswerRegex.IsMatch(content))
        {
            return DocumentType.ChatTranscript;
        }

        // Check for code file
        if (content.Contains("class ") || content.Contains("function ") || content.Contains("def ") ||
            content.Contains("import ") || content.Contains("using ") || content.Contains("namespace "))
        {
            return DocumentType.CodeFile;
        }

        // Default to unknown
        return DocumentType.Unknown;
    }

    private async Task<DocumentParsingResult> ParseGenericDocumentAsync(string content, DocumentType documentType, Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Parsing generic document of type: {DocumentType}", documentType);

            var result = new DocumentParsingResult
            {
                DocumentType = documentType
            };

            // Create a single section for the content
            var section = new ContentSection
            {
                ContentType = ContentType.Text,
                RawContent = content,
                ProcessedContent = ProcessContent(content),
                Order = 0
            };

            // Extract code blocks
            var codeBlocks = await ExtractCodeBlocksAsync(content, options);
            section.CodeBlocks.AddRange(codeBlocks);

            result.Sections = [section];

            _logger.LogInformation("Successfully parsed generic document with {CodeBlockCount} code blocks", codeBlocks.Count);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing generic document");
            return new DocumentParsingResult
            {
                DocumentType = documentType,
                Errors = { $"Error parsing generic document: {ex.Message}" }
            };
        }
    }

    private string ProcessContent(string content)
    {
        // Remove code blocks
        content = _markdownCodeBlockRegex.Replace(content, "");

        // Remove Markdown formatting
        content = Regex.Replace(content, @"\*\*(.*?)\*\*", "$1"); // Bold
        content = Regex.Replace(content, @"\*(.*?)\*", "$1"); // Italic
        content = Regex.Replace(content, @"__(.*?)__", "$1"); // Bold
        content = Regex.Replace(content, @"_(.*?)_", "$1"); // Italic
        content = Regex.Replace(content, @"~~(.*?)~~", "$1"); // Strikethrough
        content = Regex.Replace(content, @"\[(.*?)\]\((.*?)\)", "$1"); // Links

        // Trim and normalize whitespace
        content = Regex.Replace(content, @"\s+", " ").Trim();

        return content;
    }

    private bool IsExecutableLanguage(string language)
    {
        var executableLanguages = new[] { "csharp", "cs", "fsharp", "fs", "javascript", "js", "typescript", "ts", "python", "py", "java" };
        return executableLanguages.Contains(language.ToLowerInvariant());
    }

    private List<string> ExtractCommentsFromCode(string code, string language)
    {
        var comments = new List<string>();
        var lines = code.Split('\n');

        switch (language.ToLowerInvariant())
        {
            case "csharp":
            case "cs":
            case "java":
            case "javascript":
            case "js":
            case "typescript":
            case "ts":
                // Single-line comments
                foreach (var line in lines)
                {
                    var commentIndex = line.IndexOf("//");
                    if (commentIndex >= 0)
                    {
                        var comment = line.Substring(commentIndex + 2).Trim();
                        if (!string.IsNullOrWhiteSpace(comment))
                        {
                            comments.Add(comment);
                        }
                    }
                }

                // Multi-line comments
                var multiLineCommentRegex = new Regex(@"/\*([\s\S]*?)\*/");
                var matches = multiLineCommentRegex.Matches(code);
                foreach (Match match in matches)
                {
                    var comment = match.Groups[1].Value.Trim();
                    if (!string.IsNullOrWhiteSpace(comment))
                    {
                        comments.Add(comment);
                    }
                }
                break;

            case "python":
            case "py":
                // Single-line comments
                foreach (var line in lines)
                {
                    var commentIndex = line.IndexOf("#");
                    if (commentIndex >= 0)
                    {
                        var comment = line.Substring(commentIndex + 1).Trim();
                        if (!string.IsNullOrWhiteSpace(comment))
                        {
                            comments.Add(comment);
                        }
                    }
                }

                // Multi-line comments (docstrings)
                var docstringRegex = new Regex(@"(?:'''|"""""")([\s\S]*?)(?:'''|"""""")");
                var docMatches = docstringRegex.Matches(code);
                foreach (Match match in docMatches)
                {
                    var comment = match.Groups[1].Value.Trim();
                    if (!string.IsNullOrWhiteSpace(comment))
                    {
                        comments.Add(comment);
                    }
                }
                break;

            case "fsharp":
            case "fs":
                // Single-line comments
                foreach (var line in lines)
                {
                    var commentIndex = line.IndexOf("//");
                    if (commentIndex >= 0)
                    {
                        var comment = line.Substring(commentIndex + 2).Trim();
                        if (!string.IsNullOrWhiteSpace(comment))
                        {
                            comments.Add(comment);
                        }
                    }
                }

                // Multi-line comments
                var fsMultiLineCommentRegex = new Regex(@"\(\*([\s\S]*?)\*\)");
                var fsMatches = fsMultiLineCommentRegex.Matches(code);
                foreach (Match match in fsMatches)
                {
                    var comment = match.Groups[1].Value.Trim();
                    if (!string.IsNullOrWhiteSpace(comment))
                    {
                        comments.Add(comment);
                    }
                }
                break;
        }

        return comments;
    }
}
