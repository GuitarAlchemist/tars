using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Configuration;
using TarsCli.Models;

namespace TarsCli.Services;

/// <summary>
/// Service for extracting and applying knowledge from documentation
/// </summary>
public class KnowledgeApplicationService
{
    private readonly ILogger<KnowledgeApplicationService> _logger;
    private readonly OllamaService _ollamaService;
    private readonly DocumentationService _documentationService;
    private readonly string _knowledgeBaseDirectory;

    public KnowledgeApplicationService(
        ILogger<KnowledgeApplicationService> logger,
        OllamaService ollamaService,
        DocumentationService documentationService,
        IConfiguration configuration)
    {
        _logger = logger;
        _ollamaService = ollamaService;
        _documentationService = documentationService;

        // Set up the knowledge base directory
        var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        if (string.IsNullOrEmpty(appDataPath))
        {
            appDataPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars");
        }
        else
        {
            appDataPath = Path.Combine(appDataPath, "TARS");
        }

        _knowledgeBaseDirectory = Path.Combine(appDataPath, "KnowledgeBase");

        if (!Directory.Exists(_knowledgeBaseDirectory))
        {
            Directory.CreateDirectory(_knowledgeBaseDirectory);
        }
    }

    /// <summary>
    /// Extract knowledge from a markdown file
    /// </summary>
    /// <param name="filePath">Path to the markdown file</param>
    /// <param name="model">Model to use for extraction</param>
    /// <returns>The extracted knowledge</returns>
    public async Task<DocumentationKnowledge> ExtractKnowledgeAsync(string filePath, string model = "llama3")
    {
        _logger.LogInformation($"Extracting knowledge from: {Path.GetFullPath(filePath)}");

        try
        {
            // Read the file content
            var content = await File.ReadAllTextAsync(filePath);

            // Create a prompt for knowledge extraction
            var prompt = $@"You are an expert at extracting structured knowledge from documentation.

I'll provide you with the content of a markdown file from the TARS project. Please extract key knowledge, concepts, and insights from this document.

Focus on:
1. Key concepts and their definitions
2. Important insights and conclusions
3. Technical details and specifications
4. Design decisions and rationales
5. Relationships between concepts

Here's the content:

{content}

Please provide your extracted knowledge in the following JSON format:
{{
  ""title"": ""The title of the document"",
  ""summary"": ""A concise summary of the document"",
  ""key_concepts"": [
    {{
      ""name"": ""Concept name"",
      ""definition"": ""Concept definition"",
      ""related_concepts"": [""Related concept 1"", ""Related concept 2""]
    }}
  ],
  ""insights"": [
    {{
      ""description"": ""Description of the insight"",
      ""importance"": ""Why this insight is important"",
      ""applications"": [""Application 1"", ""Application 2""]
    }}
  ],
  ""technical_details"": [
    {{
      ""topic"": ""Topic name"",
      ""details"": ""Technical details"",
      ""code_examples"": [""Example 1"", ""Example 2""]
    }}
  ],
  ""design_decisions"": [
    {{
      ""decision"": ""The decision made"",
      ""rationale"": ""Why this decision was made"",
      ""alternatives"": [""Alternative 1"", ""Alternative 2""]
    }}
  ],
  ""relationships"": [
    {{
      ""from"": ""Concept A"",
      ""to"": ""Concept B"",
      ""relationship"": ""How A relates to B""
    }}
  ]
}}

Focus on extracting the most valuable knowledge from this document.";

            // Get the extracted knowledge from the LLM
            var response = await _ollamaService.GenerateAsync(prompt, model);

            // Parse the JSON response
            var jsonMatch = Regex.Match(response, @"\{[\s\S]*\}");
            if (!jsonMatch.Success)
            {
                _logger.LogWarning($"Failed to extract knowledge from {filePath}: No JSON found in response");
                return new DocumentationKnowledge
                {
                    Title = Path.GetFileNameWithoutExtension(filePath),
                    Summary = "Failed to extract knowledge",
                    SourceFile = filePath,
                    ExtractionDate = DateTime.Now
                };
            }

            var jsonString = jsonMatch.Value;
            var knowledge = JsonSerializer.Deserialize<DocumentationKnowledge>(jsonString, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            // Add metadata
            knowledge.SourceFile = filePath;
            knowledge.ExtractionDate = DateTime.Now;

            // Save the knowledge to the knowledge base
            await SaveKnowledgeAsync(knowledge);

            _logger.LogInformation($"Successfully extracted knowledge from {filePath}");
            return knowledge;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error extracting knowledge from {filePath}");
            throw;
        }
    }

    /// <summary>
    /// Save knowledge to the knowledge base
    /// </summary>
    /// <param name="knowledge">The knowledge to save</param>
    private async Task SaveKnowledgeAsync(DocumentationKnowledge knowledge)
    {
        try
        {
            // Create a file name based on the title
            var fileName = SanitizeFileName(knowledge.Title) + ".json";
            var filePath = Path.Combine(_knowledgeBaseDirectory, fileName);

            // Serialize the knowledge to JSON
            var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(knowledge, options);

            // Save the knowledge to a file
            await File.WriteAllTextAsync(filePath, json);

            _logger.LogInformation($"Knowledge saved to: {Path.GetFullPath(filePath)}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving knowledge to knowledge base");
            throw;
        }
    }

    /// <summary>
    /// Get all knowledge from the knowledge base
    /// </summary>
    /// <returns>All knowledge in the knowledge base</returns>
    public async Task<List<DocumentationKnowledge>> GetAllKnowledgeAsync()
    {
        try
        {
            var knowledge = new List<DocumentationKnowledge>();

            foreach (var file in Directory.GetFiles(_knowledgeBaseDirectory, "*.json"))
            {
                try
                {
                    var json = await File.ReadAllTextAsync(file);
                    var item = JsonSerializer.Deserialize<DocumentationKnowledge>(json, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });

                    knowledge.Add(item);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"Error loading knowledge from {file}");
                }
            }

            return knowledge.OrderByDescending(k => k.ExtractionDate).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting knowledge from knowledge base");
            throw;
        }
    }

    /// <summary>
    /// Apply knowledge to improve a file
    /// </summary>
    /// <param name="filePath">Path to the file to improve</param>
    /// <param name="model">Model to use for improvement</param>
    /// <returns>True if the file was improved, false otherwise</returns>
    public async Task<bool> ApplyKnowledgeToFileAsync(string filePath, string model = "llama3")
    {
        _logger.LogInformation($"Applying knowledge to improve: {Path.GetFullPath(filePath)}");

        try
        {
            // Get all knowledge from the knowledge base
            var allKnowledge = await GetAllKnowledgeAsync();
            if (allKnowledge.Count == 0)
            {
                _logger.LogWarning("No knowledge available in the knowledge base");
                return false;
            }

            // Read the file content
            var content = await File.ReadAllTextAsync(filePath);

            // Create a summary of the knowledge
            var knowledgeSummary = new System.Text.StringBuilder();
            knowledgeSummary.AppendLine("# Knowledge Base Summary");
            knowledgeSummary.AppendLine();

            // Add key concepts
            knowledgeSummary.AppendLine("## Key Concepts");
            knowledgeSummary.AppendLine();
            foreach (var knowledge in allKnowledge.Take(5))
            {
                if (knowledge.KeyConcepts != null)
                {
                    foreach (var concept in knowledge.KeyConcepts.Take(3))
                    {
                        knowledgeSummary.AppendLine($"- **{concept.Name}**: {concept.Definition}");
                    }
                }
            }
            knowledgeSummary.AppendLine();

            // Add insights
            knowledgeSummary.AppendLine("## Key Insights");
            knowledgeSummary.AppendLine();
            foreach (var knowledge in allKnowledge.Take(5))
            {
                if (knowledge.Insights != null)
                {
                    foreach (var insight in knowledge.Insights.Take(3))
                    {
                        knowledgeSummary.AppendLine($"- {insight.Description}");
                    }
                }
            }
            knowledgeSummary.AppendLine();

            // Add design decisions
            knowledgeSummary.AppendLine("## Design Decisions");
            knowledgeSummary.AppendLine();
            foreach (var knowledge in allKnowledge.Take(5))
            {
                if (knowledge.DesignDecisions != null)
                {
                    foreach (var decision in knowledge.DesignDecisions.Take(3))
                    {
                        knowledgeSummary.AppendLine($"- **{decision.Decision}**: {decision.Rationale}");
                    }
                }
            }

            // Create a prompt for applying knowledge
            var prompt = $@"You are an expert at improving code and documentation by applying knowledge from a knowledge base.

I'll provide you with:
1. The content of a file from the TARS project
2. A summary of knowledge extracted from TARS documentation

Your task is to improve the file by applying relevant knowledge from the knowledge base. This could involve:
- Adding comments to explain concepts or design decisions
- Improving variable or function names based on standard terminology
- Adding documentation that references key concepts
- Restructuring code to better align with design principles
- Adding references to related concepts or components

Here's the file content:

```
{content}
```

Here's the knowledge base summary:

{knowledgeSummary}

Please provide your improved version of the file. If you make changes, explain each change and how it applies knowledge from the knowledge base.

If the file doesn't need improvement or the knowledge isn't relevant, just say so.";

            // Get the improved content from the LLM
            var response = await _ollamaService.GenerateAsync(prompt, model);

            // Extract the improved content
            var codeBlockMatch = Regex.Match(response, @"```(?:[\w]*)\s*([\s\S]*?)```");
            if (!codeBlockMatch.Success)
            {
                _logger.LogWarning($"Failed to extract improved content for {filePath}");
                return false;
            }

            var improvedContent = codeBlockMatch.Groups[1].Value.Trim();

            // Check if the content was actually improved
            if (string.Equals(content.Trim(), improvedContent.Trim(), StringComparison.Ordinal))
            {
                _logger.LogInformation($"No improvements needed for {filePath}");
                return false;
            }

            // Create a backup of the original file
            var backupPath = filePath + ".bak";
            await File.WriteAllTextAsync(backupPath, content);

            // Write the improved content back to the file
            await File.WriteAllTextAsync(filePath, improvedContent);

            _logger.LogInformation($"Successfully improved {filePath} using knowledge from the knowledge base");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error applying knowledge to {filePath}");
            return false;
        }
    }

    /// <summary>
    /// Generate a knowledge report
    /// </summary>
    /// <returns>Path to the generated report</returns>
    public async Task<string> GenerateKnowledgeReportAsync()
    {
        _logger.LogInformation("Generating knowledge report");

        try
        {
            // Get all knowledge from the knowledge base
            var allKnowledge = await GetAllKnowledgeAsync();
            if (allKnowledge.Count == 0)
            {
                _logger.LogWarning("No knowledge available in the knowledge base");
                return null;
            }

            // Create the report
            var report = new System.Text.StringBuilder();
            report.AppendLine("# TARS Knowledge Base Report");
            report.AppendLine();
            report.AppendLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            report.AppendLine();
            report.AppendLine($"Total knowledge items: {allKnowledge.Count}");
            report.AppendLine();

            // Add a summary of each knowledge item
            report.AppendLine("## Knowledge Items");
            report.AppendLine();
            foreach (var knowledge in allKnowledge)
            {
                report.AppendLine($"### {knowledge.Title}");
                report.AppendLine();
                report.AppendLine($"Source: {knowledge.SourceFile}");
                report.AppendLine($"Extracted: {knowledge.ExtractionDate:yyyy-MM-dd HH:mm:ss}");
                report.AppendLine();
                report.AppendLine(knowledge.Summary);
                report.AppendLine();

                // Add key concepts
                if (knowledge.KeyConcepts != null && knowledge.KeyConcepts.Count > 0)
                {
                    report.AppendLine("#### Key Concepts");
                    report.AppendLine();
                    foreach (var concept in knowledge.KeyConcepts)
                    {
                        report.AppendLine($"- **{concept.Name}**: {concept.Definition}");
                    }
                    report.AppendLine();
                }

                // Add insights
                if (knowledge.Insights != null && knowledge.Insights.Count > 0)
                {
                    report.AppendLine("#### Insights");
                    report.AppendLine();
                    foreach (var insight in knowledge.Insights)
                    {
                        report.AppendLine($"- {insight.Description}");
                    }
                    report.AppendLine();
                }
            }

            // Save the report
            var reportPath = Path.Combine(_knowledgeBaseDirectory, $"KnowledgeReport_{DateTime.Now:yyyyMMdd_HHmmss}.md");
            await File.WriteAllTextAsync(reportPath, report.ToString());

            _logger.LogInformation($"Knowledge report generated: {Path.GetFullPath(reportPath)}");
            return reportPath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating knowledge report");
            return null;
        }
    }

    /// <summary>
    /// Sanitize a string for use as a file name
    /// </summary>
    /// <param name="input">The input string</param>
    /// <returns>A sanitized string</returns>
    private string SanitizeFileName(string input)
    {
        if (string.IsNullOrEmpty(input))
        {
            return "unknown";
        }

        // Replace invalid characters with underscores
        var invalidChars = Path.GetInvalidFileNameChars();
        var sanitized = new string(input.Select(c => invalidChars.Contains(c) ? '_' : c).ToArray());

        // Trim and replace spaces with underscores
        sanitized = sanitized.Trim().Replace(' ', '_');

        // Ensure the file name is not too long
        if (sanitized.Length > 50)
        {
            sanitized = sanitized.Substring(0, 50);
        }

        return sanitized;
    }
}
