using System.CommandLine;
using System.CommandLine.Invocation;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsCli.Commands;

/// <summary>
/// Command for knowledge extraction and processing
/// </summary>
public class KnowledgeExtractionCommand : Command
{
    private readonly ILogger<KnowledgeExtractionCommand> _logger;
    private readonly IDocumentParserService _documentParserService;
    private readonly IContentClassifierService _contentClassifierService;
    private readonly IKnowledgeExtractorService _knowledgeExtractorService;
    private readonly IKnowledgeRepository _knowledgeRepository;

    /// <summary>
    /// Initializes a new instance of the <see cref="KnowledgeExtractionCommand"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="documentParserService">The document parser service</param>
    /// <param name="contentClassifierService">The content classifier service</param>
    /// <param name="knowledgeExtractorService">The knowledge extractor service</param>
    /// <param name="knowledgeRepository">The knowledge repository</param>
    public KnowledgeExtractionCommand(
        ILogger<KnowledgeExtractionCommand> logger,
        IDocumentParserService documentParserService,
        IContentClassifierService contentClassifierService,
        IKnowledgeExtractorService knowledgeExtractorService,
        IKnowledgeRepository knowledgeRepository)
        : base("knowledge", "Extract and process knowledge from documents")
    {
        _logger = logger;
        _documentParserService = documentParserService;
        _contentClassifierService = contentClassifierService;
        _knowledgeExtractorService = knowledgeExtractorService;
        _knowledgeRepository = knowledgeRepository;

        // Add subcommands
        AddCommand(CreateExtractCommand());
        AddCommand(CreateSearchCommand());
        AddCommand(CreateStatsCommand());
    }

    private Command CreateExtractCommand()
    {
        var command = new Command("extract", "Extract knowledge from documents");

        // Add options
        var pathOption = new Option<string>(
            name: "--path",
            description: "Path to the document or directory to extract knowledge from");
        pathOption.IsRequired = true;
        command.AddOption(pathOption);

        var recursiveOption = new Option<bool>(
            name: "--recursive",
            description: "Whether to recursively process directories");
        command.AddOption(recursiveOption);

        var typeOption = new Option<string>(
            name: "--type",
            description: "The type of document to extract knowledge from (markdown, chat, code, reflection)");
        command.AddOption(typeOption);

        var saveOption = new Option<bool>(
            name: "--save",
            description: "Whether to save the extracted knowledge to the repository");
        command.AddOption(saveOption);

        // Set handler
        command.SetHandler(async (string path, bool recursive, string? type, bool save) =>
        {
            await ExtractKnowledgeAsync(path, recursive, type, save);
        }, pathOption, recursiveOption, typeOption, saveOption);

        return command;
    }

    private Command CreateSearchCommand()
    {
        var command = new Command("search", "Search for knowledge items");

        // Add options
        var queryOption = new Option<string>(
            name: "--query",
            description: "The search query");
        queryOption.IsRequired = true;
        command.AddOption(queryOption);

        var typeOption = new Option<string>(
            name: "--type",
            description: "The type of knowledge to search for");
        command.AddOption(typeOption);

        var maxResultsOption = new Option<int>(
            name: "--max-results",
            description: "The maximum number of results to return",
            getDefaultValue: () => 10);
        command.AddOption(maxResultsOption);

        // Set handler
        command.SetHandler(async (string query, string? type, int maxResults) =>
        {
            await SearchKnowledgeAsync(query, type, maxResults);
        }, queryOption, typeOption, maxResultsOption);

        return command;
    }

    private Command CreateStatsCommand()
    {
        var command = new Command("stats", "Get statistics about the knowledge repository");

        // Set handler
        command.SetHandler(async () =>
        {
            await GetRepositoryStatsAsync();
        });

        return command;
    }

    private async Task ExtractKnowledgeAsync(string path, bool recursive, string? type, bool save)
    {
        try
        {
            _logger.LogInformation("Extracting knowledge from {Path}", path);

            // Check if path exists
            if (!File.Exists(path) && !Directory.Exists(path))
            {
                _logger.LogError("Path not found: {Path}", path);
                Console.WriteLine($"Error: Path not found: {path}");
                return;
            }

            // Process file or directory
            if (File.Exists(path))
            {
                await ProcessFileAsync(path, type, save);
            }
            else
            {
                await ProcessDirectoryAsync(path, recursive, type, save);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting knowledge from {Path}", path);
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    private async Task ProcessFileAsync(string filePath, string? type, bool save)
    {
        try
        {
            _logger.LogInformation("Processing file: {FilePath}", filePath);
            Console.WriteLine($"Processing file: {filePath}");

            // Determine document type
            DocumentType documentType;
            if (!string.IsNullOrEmpty(type))
            {
                documentType = type.ToLowerInvariant() switch
                {
                    "markdown" => DocumentType.Markdown,
                    "chat" => DocumentType.ChatTranscript,
                    "code" => DocumentType.CodeFile,
                    "reflection" => DocumentType.Reflection,
                    _ => _documentParserService.GetDocumentTypeFromPath(filePath)
                };
            }
            else
            {
                documentType = _documentParserService.GetDocumentTypeFromPath(filePath);
            }

            // Parse document
            var document = await _documentParserService.ParseDocumentAsync(filePath);
            if (!document.IsSuccessful)
            {
                _logger.LogError("Error parsing document: {FilePath}", filePath);
                foreach (var error in document.Errors)
                {
                    Console.WriteLine($"  Error: {error}");
                }
                return;
            }

            // Classify content
            var classification = await _contentClassifierService.ClassifyDocumentAsync(document);
            Console.WriteLine($"Classified document with {classification.Classifications.Count} classifications");

            // Extract knowledge
            var extractionResult = await _knowledgeExtractorService.ExtractFromClassificationBatchAsync(classification);
            Console.WriteLine($"Extracted {extractionResult.Items.Count} knowledge items");

            // Save to repository if requested
            if (save && extractionResult.Items.Any())
            {
                var savedItems = await _knowledgeRepository.AddItemsAsync(extractionResult.Items);
                Console.WriteLine($"Saved {savedItems.Count()} knowledge items to repository");

                // Detect and save relationships
                var relationships = await _knowledgeExtractorService.DetectRelationshipsAsync(extractionResult.Items.ToList());
                if (relationships.Any())
                {
                    foreach (var relationship in relationships)
                    {
                        await _knowledgeRepository.AddRelationshipAsync(relationship);
                    }
                    Console.WriteLine($"Saved {relationships.Count} relationships to repository");
                }
            }

            // Display summary
            Console.WriteLine();
            Console.WriteLine("Knowledge Extraction Summary:");
            Console.WriteLine($"  Document: {document.Title}");
            Console.WriteLine($"  Type: {document.DocumentType}");
            Console.WriteLine($"  Sections: {document.Sections.Count}");
            Console.WriteLine($"  Knowledge Items: {extractionResult.Items.Count}");
            Console.WriteLine();

            // Display knowledge items by type
            var itemsByType = extractionResult.Items.GroupBy(i => i.Type).OrderByDescending(g => g.Count());
            Console.WriteLine("Knowledge Items by Type:");
            foreach (var group in itemsByType)
            {
                Console.WriteLine($"  {group.Key}: {group.Count()}");
            }
            Console.WriteLine();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing file: {FilePath}", filePath);
            Console.WriteLine($"Error processing file {filePath}: {ex.Message}");
        }
    }

    private async Task ProcessDirectoryAsync(string directoryPath, bool recursive, string? type, bool save)
    {
        try
        {
            _logger.LogInformation("Processing directory: {DirectoryPath}", directoryPath);
            Console.WriteLine($"Processing directory: {directoryPath}");

            // Get files
            var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
            var files = Directory.GetFiles(directoryPath, "*.*", searchOption)
                .Where(f => IsProcessableFile(f))
                .ToList();

            Console.WriteLine($"Found {files.Count} files to process");

            // Process each file
            int processedCount = 0;
            foreach (var file in files)
            {
                await ProcessFileAsync(file, type, save);
                processedCount++;
                Console.WriteLine($"Processed {processedCount} of {files.Count} files");
            }

            Console.WriteLine($"Completed processing {processedCount} files in {directoryPath}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing directory: {DirectoryPath}", directoryPath);
            Console.WriteLine($"Error processing directory {directoryPath}: {ex.Message}");
        }
    }

    private bool IsProcessableFile(string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();
        return extension switch
        {
            ".md" => true,
            ".markdown" => true,
            ".txt" => true,
            ".cs" => true,
            ".fs" => true,
            ".js" => true,
            ".ts" => true,
            ".py" => true,
            ".java" => true,
            ".json" => true,
            ".xml" => true,
            ".yaml" => true,
            ".yml" => true,
            _ => false
        };
    }

    private async Task SearchKnowledgeAsync(string query, string? type, int maxResults)
    {
        try
        {
            _logger.LogInformation("Searching for knowledge items with query: {Query}", query);
            Console.WriteLine($"Searching for knowledge items with query: {query}");

            // Prepare search options
            var options = new Dictionary<string, string>
            {
                { "MaxResults", maxResults.ToString() }
            };

            // Add type filter if specified
            if (!string.IsNullOrEmpty(type) && Enum.TryParse<KnowledgeType>(type, true, out var knowledgeType))
            {
                options["Type"] = knowledgeType.ToString();
            }

            // Search repository
            var results = await _knowledgeRepository.SearchItemsAsync(query, options);
            var resultsList = results.ToList();

            Console.WriteLine($"Found {resultsList.Count} knowledge items");
            Console.WriteLine();

            // Display results
            for (int i = 0; i < resultsList.Count; i++)
            {
                var item = resultsList[i];
                Console.WriteLine($"[{i + 1}] {item.Type}: {TruncateContent(item.Content, 100)}");
                Console.WriteLine($"    Source: {item.Source}");
                Console.WriteLine($"    Confidence: {item.Confidence:P0}, Relevance: {item.Relevance:P0}");
                Console.WriteLine($"    Tags: {string.Join(", ", item.Tags)}");
                Console.WriteLine();
            }

            // Prompt for details
            if (resultsList.Any())
            {
                Console.Write("Enter item number to view details (or press Enter to exit): ");
                var input = Console.ReadLine();
                if (!string.IsNullOrEmpty(input) && int.TryParse(input, out var itemNumber) && itemNumber > 0 && itemNumber <= resultsList.Count)
                {
                    var selectedItem = resultsList[itemNumber - 1];
                    DisplayKnowledgeItemDetails(selectedItem);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error searching for knowledge items");
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    private async Task GetRepositoryStatsAsync()
    {
        try
        {
            _logger.LogInformation("Getting knowledge repository statistics");
            Console.WriteLine("Getting knowledge repository statistics...");

            // Get statistics
            var stats = await _knowledgeRepository.GetStatisticsAsync();

            // Display statistics
            Console.WriteLine();
            Console.WriteLine("Knowledge Repository Statistics:");
            Console.WriteLine($"  Total Items: {stats.TotalItems}");
            Console.WriteLine($"  Total Relationships: {stats.TotalRelationships}");
            Console.WriteLine($"  Average Confidence: {stats.AverageConfidence:P0}");
            Console.WriteLine($"  Average Relevance: {stats.AverageRelevance:P0}");
            Console.WriteLine();

            // Display items by type
            Console.WriteLine("Items by Type:");
            foreach (var type in stats.ItemsByType.OrderByDescending(kv => kv.Value))
            {
                Console.WriteLine($"  {type.Key}: {type.Value}");
            }
            Console.WriteLine();

            // Display items by tag (top 10)
            Console.WriteLine("Top 10 Tags:");
            foreach (var tag in stats.ItemsByTag.OrderByDescending(kv => kv.Value).Take(10))
            {
                Console.WriteLine($"  {tag.Key}: {tag.Value}");
            }
            Console.WriteLine();

            // Display relationships by type
            Console.WriteLine("Relationships by Type:");
            foreach (var type in stats.RelationshipsByType.OrderByDescending(kv => kv.Value))
            {
                Console.WriteLine($"  {type.Key}: {type.Value}");
            }
            Console.WriteLine();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting knowledge repository statistics");
            Console.WriteLine($"Error: {ex.Message}");
        }
    }

    private void DisplayKnowledgeItemDetails(KnowledgeItem item)
    {
        Console.WriteLine();
        Console.WriteLine("Knowledge Item Details:");
        Console.WriteLine($"  ID: {item.Id}");
        Console.WriteLine($"  Type: {item.Type}");
        Console.WriteLine($"  Content: {item.Content}");
        Console.WriteLine($"  Source: {item.Source}");
        Console.WriteLine($"  Context: {item.Context}");
        Console.WriteLine($"  Confidence: {item.Confidence:P0}");
        Console.WriteLine($"  Relevance: {item.Relevance:P0}");
        Console.WriteLine($"  Created: {item.CreatedAt}");
        Console.WriteLine($"  Updated: {item.UpdatedAt}");
        Console.WriteLine($"  Tags: {string.Join(", ", item.Tags)}");
        Console.WriteLine($"  Validation Status: {item.ValidationStatus}");
        if (!string.IsNullOrEmpty(item.ValidationNotes))
        {
            Console.WriteLine($"  Validation Notes: {item.ValidationNotes}");
        }
        Console.WriteLine();

        // Display metadata
        if (item.Metadata.Any())
        {
            Console.WriteLine("  Metadata:");
            foreach (var meta in item.Metadata)
            {
                Console.WriteLine($"    {meta.Key}: {meta.Value}");
            }
            Console.WriteLine();
        }
    }

    private string TruncateContent(string content, int maxLength)
    {
        if (string.IsNullOrEmpty(content))
        {
            return string.Empty;
        }

        if (content.Length <= maxLength)
        {
            return content;
        }

        return content.Substring(0, maxLength - 3) + "...";
    }
}
