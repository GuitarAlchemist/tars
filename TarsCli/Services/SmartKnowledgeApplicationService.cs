using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TarsCli.Models;

namespace TarsCli.Services;

/// <summary>
/// Enhanced service for applying knowledge with validation and incremental improvements
/// </summary>
public class SmartKnowledgeApplicationService
{
    private readonly ILogger<SmartKnowledgeApplicationService> _logger;
    private readonly KnowledgeApplicationService _baseKnowledgeService;
    private readonly OllamaService _ollamaService;
    private readonly CompilationService _compilationService;
    private readonly TestRunnerService _testRunnerService;
    private readonly ConsoleService _consoleService;

    public SmartKnowledgeApplicationService(
        ILogger<SmartKnowledgeApplicationService> logger,
        KnowledgeApplicationService baseKnowledgeService,
        OllamaService ollamaService,
        CompilationService compilationService,
        TestRunnerService testRunnerService,
        ConsoleService consoleService)
    {
        _logger = logger;
        _baseKnowledgeService = baseKnowledgeService;
        _ollamaService = ollamaService;
        _compilationService = compilationService;
        _testRunnerService = testRunnerService;
        _consoleService = consoleService;
    }

    /// <summary>
    /// Apply knowledge to improve a file with validation and incremental improvements
    /// </summary>
    /// <param name="filePath">Path to the file to improve</param>
    /// <param name="model">Model to use for improvement</param>
    /// <param name="validateCompilation">Whether to validate that the improved file compiles</param>
    /// <param name="runTests">Whether to run tests after improvement</param>
    /// <returns>True if the file was improved, false otherwise</returns>
    public async Task<bool> ApplyKnowledgeWithValidationAsync(
        string filePath,
        string model = "llama3",
        bool validateCompilation = true,
        bool runTests = true)
    {
        _logger.LogInformation($"Applying knowledge with validation to: {Path.GetFullPath(filePath)}");

        try
        {
            // Read the original file content
            var originalContent = await File.ReadAllTextAsync(filePath);

            // Create a backup of the original file
            var backupPath = filePath + ".bak";
            await File.WriteAllTextAsync(backupPath, originalContent);

            // Get all knowledge from the knowledge base
            var allKnowledge = await _baseKnowledgeService.GetAllKnowledgeAsync();
            if (allKnowledge.Count == 0)
            {
                _logger.LogWarning("No knowledge available in the knowledge base");
                return false;
            }

            // Filter knowledge relevant to this file
            var relevantKnowledge = await FilterRelevantKnowledgeAsync(filePath, allKnowledge, model);
            if (relevantKnowledge.Count == 0)
            {
                _logger.LogInformation($"No relevant knowledge found for {filePath}");
                return false;
            }

            _logger.LogInformation($"Found {relevantKnowledge.Count} relevant knowledge items for {filePath}");

            // Apply improvements incrementally
            var currentContent = originalContent;
            var improvementsMade = false;

            foreach (var knowledge in relevantKnowledge)
            {
                var improvedContent = await ApplySingleKnowledgeItemAsync(filePath, currentContent, knowledge, model);

                // If no changes were made, continue to the next knowledge item
                if (string.Equals(currentContent.Trim(), improvedContent.Trim(), StringComparison.Ordinal))
                {
                    continue;
                }

                // Validate the improved content
                if (validateCompilation)
                {
                    var tempPath = filePath + ".tmp";
                    await File.WriteAllTextAsync(tempPath, improvedContent);

                    var compilationResult = await _compilationService.ValidateCompilationAsync(tempPath);
                    if (!compilationResult.Success)
                    {
                        _logger.LogWarning($"Improved content failed compilation: {compilationResult.ErrorMessage}");
                        File.Delete(tempPath);
                        continue;
                    }

                    File.Delete(tempPath);
                }

                // Update the current content with the improved content
                currentContent = improvedContent;
                improvementsMade = true;

                _logger.LogInformation($"Applied knowledge item: {knowledge.Title}");
            }

            // If no improvements were made, return false
            if (!improvementsMade)
            {
                _logger.LogInformation($"No improvements made to {filePath}");
                return false;
            }

            // Write the improved content back to the file
            await File.WriteAllTextAsync(filePath, currentContent);

            // Run tests if requested
            if (runTests)
            {
                var testResult = await _testRunnerService.RunTestsForFileAsync(filePath);
                if (!testResult.Success)
                {
                    _logger.LogWarning($"Tests failed after improvement: {testResult.ErrorMessage}");

                    // Restore the original file
                    await File.WriteAllTextAsync(filePath, originalContent);

                    _logger.LogInformation($"Restored original file due to test failures");
                    return false;
                }
            }

            _logger.LogInformation($"Successfully improved {filePath}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error applying knowledge to {filePath}");
            return false;
        }
    }

    /// <summary>
    /// Filter knowledge items relevant to a specific file
    /// </summary>
    private async Task<List<DocumentationKnowledge>> FilterRelevantKnowledgeAsync(
        string filePath,
        List<DocumentationKnowledge> allKnowledge,
        string model)
    {
        try
        {
            // Read the file content
            var content = await File.ReadAllTextAsync(filePath);

            // Create a prompt for relevance filtering
            var prompt = $@"You are an expert at determining which knowledge is relevant to a specific code file.

I'll provide you with:
1. The content of a code file
2. A list of knowledge items extracted from documentation

Your task is to identify which knowledge items are most relevant to the code file. Consider:
- Technical concepts mentioned in both the code and knowledge
- Design patterns or architectural approaches
- Domain-specific terminology
- Implementation details

Here's the code file:

```
{content}
```

Here are the knowledge items:

{string.Join("\n\n", allKnowledge.Select((k, i) => $"Item {i+1}: {k.Title}\n{k.Summary}"))}

Please provide the indices of the relevant knowledge items in JSON format:
{{
  ""relevant_indices"": [1, 3, 5]
}}

Only include indices of items that are truly relevant to the code file.";

            // Get the relevant indices from the LLM
            var response = await _ollamaService.GenerateAsync(prompt, model);

            // Parse the JSON response
            var jsonMatch = Regex.Match(response, @"\{[\s\S]*\}");
            if (!jsonMatch.Success)
            {
                _logger.LogWarning($"Failed to extract relevant indices from response");
                return new List<DocumentationKnowledge>();
            }

            var jsonString = jsonMatch.Value;
            var relevanceResult = JsonSerializer.Deserialize<RelevanceResult>(jsonString, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            // Filter the knowledge items by the relevant indices
            var relevantKnowledge = new List<DocumentationKnowledge>();
            foreach (var index in relevanceResult.RelevantIndices)
            {
                if (index > 0 && index <= allKnowledge.Count)
                {
                    relevantKnowledge.Add(allKnowledge[index - 1]);
                }
            }

            return relevantKnowledge;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error filtering relevant knowledge for {filePath}");
            return new List<DocumentationKnowledge>();
        }
    }

    /// <summary>
    /// Apply a single knowledge item to improve a file
    /// </summary>
    private async Task<string> ApplySingleKnowledgeItemAsync(
        string filePath,
        string currentContent,
        DocumentationKnowledge knowledge,
        string model)
    {
        try
        {
            // Create a prompt for applying the knowledge
            var prompt = $@"You are an expert at improving code by applying specific knowledge.

I'll provide you with:
1. The content of a code file
2. A specific knowledge item extracted from documentation

Your task is to improve the code file by applying the knowledge. Make SMALL, INCREMENTAL improvements only.
Focus on:
- Adding or improving comments
- Clarifying variable or function names
- Adding documentation references
- Small refactorings that preserve functionality
- Performance optimizations based on the knowledge

DO NOT:
- Completely rewrite the file
- Change the overall structure
- Add new features
- Remove existing functionality

Here's the code file:

```
{currentContent}
```

Here's the knowledge item:

Title: {knowledge.Title}
Summary: {knowledge.Summary}

Key Concepts:
{string.Join("\n", knowledge.KeyConcepts?.Select(c => $"- {c.Name}: {c.Definition}") ?? Array.Empty<string>())}

Insights:
{string.Join("\n", knowledge.Insights?.Select(i => $"- {i.Description}") ?? Array.Empty<string>())}

Design Decisions:
{string.Join("\n", knowledge.DesignDecisions?.Select(d => $"- {d.Decision}: {d.Rationale}") ?? Array.Empty<string>())}

Please provide your improved version of the file. If you make changes, explain each change in a comment.
If the file doesn't need improvement based on this knowledge, return it unchanged.";

            // Get the improved content from the LLM
            var response = await _ollamaService.GenerateAsync(prompt, model);

            // Extract the improved content
            var codeBlockMatch = Regex.Match(response, @"```(?:[\w]*)\s*([\s\S]*?)```");
            if (!codeBlockMatch.Success)
            {
                _logger.LogWarning($"Failed to extract improved content");
                return currentContent;
            }

            var improvedContent = codeBlockMatch.Groups[1].Value.Trim();

            return improvedContent;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error applying knowledge item to {filePath}");
            return currentContent;
        }
    }

    /// <summary>
    /// Apply knowledge to improve multiple files with validation
    /// </summary>
    public async Task<int> ApplyKnowledgeToDirectoryAsync(
        string directoryPath,
        string pattern = "*.cs",
        string model = "llama3",
        bool validateCompilation = true,
        bool runTests = true)
    {
        _logger.LogInformation($"Applying knowledge to directory: {Path.GetFullPath(directoryPath)}");

        try
        {
            // Get all files matching the pattern
            var files = Directory.GetFiles(directoryPath, pattern, SearchOption.AllDirectories);
            _logger.LogInformation($"Found {files.Length} files matching pattern {pattern}");

            // Apply knowledge to each file
            var improvedCount = 0;
            foreach (var file in files)
            {
                _consoleService.WriteInfo($"Processing: {Path.GetFileName(file)}");
                var result = await ApplyKnowledgeWithValidationAsync(file, model, validateCompilation, runTests);

                if (result)
                {
                    improvedCount++;
                    _consoleService.WriteSuccess($"Improved: {Path.GetFileName(file)}");
                }
            }

            _logger.LogInformation($"Improved {improvedCount} out of {files.Length} files");
            return improvedCount;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error applying knowledge to directory {directoryPath}");
            return 0;
        }
    }

    /// <summary>
    /// Helper class for parsing relevance results
    /// </summary>
    private class RelevanceResult
    {
        [JsonPropertyName("relevant_indices")]
        public List<int> RelevantIndices { get; set; } = new List<int>();
    }
}
