using Microsoft.Extensions.Logging;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using TarsCli.Models;

namespace TarsCli.Services;

/// <summary>
/// Service for prioritizing knowledge application to files
/// </summary>
public class KnowledgePrioritizationService
{
    private readonly ILogger<KnowledgePrioritizationService> _logger;
    private readonly KnowledgeApplicationService _knowledgeApplicationService;
    private readonly OllamaService _ollamaService;
    private readonly GitService _gitService;

    public KnowledgePrioritizationService(
        ILogger<KnowledgePrioritizationService> logger,
        KnowledgeApplicationService knowledgeApplicationService,
        OllamaService ollamaService,
        GitService gitService)
    {
        _logger = logger;
        _knowledgeApplicationService = knowledgeApplicationService;
        _ollamaService = ollamaService;
        _gitService = gitService;
    }

    /// <summary>
    /// Prioritize files for knowledge application
    /// </summary>
    /// <param name="directoryPath">Directory containing files to prioritize</param>
    /// <param name="pattern">File pattern to match</param>
    /// <param name="model">Model to use for prioritization</param>
    /// <param name="maxFiles">Maximum number of files to return</param>
    /// <returns>Prioritized list of files</returns>
    public async Task<List<PrioritizedFile>> PrioritizeFilesAsync(
        string directoryPath,
        string pattern = "*.cs",
        string model = "llama3",
        int maxFiles = 10)
    {
        _logger.LogInformation($"Prioritizing files in {directoryPath} with pattern {pattern}");

        try
        {
            // Get all files matching the pattern
            var files = Directory.GetFiles(directoryPath, pattern, SearchOption.AllDirectories);
            _logger.LogInformation($"Found {files.Length} files matching pattern {pattern}");

            // Get all knowledge from the knowledge base
            var allKnowledge = await _knowledgeApplicationService.GetAllKnowledgeAsync();
            if (allKnowledge.Count == 0)
            {
                _logger.LogWarning("No knowledge available in the knowledge base");
                return new List<PrioritizedFile>();
            }

            // Get file metrics
            var fileMetrics = await GetFileMetricsAsync(files);

            // Create a prompt for prioritization
            var knowledgeSummary = CreateKnowledgeSummary(allKnowledge);
            var filesSummary = CreateFilesSummary(files, fileMetrics);

            var prompt = $@"You are an expert at prioritizing code files for improvement based on knowledge.

I'll provide you with:
1. A summary of knowledge extracted from documentation
2. A list of code files with metrics

Your task is to prioritize the files for improvement based on:
- Relevance to the knowledge base
- Complexity and maintainability
- Recent changes and activity
- Potential impact of improvements

Knowledge Summary:
{knowledgeSummary}

Files Summary:
{filesSummary}

Please provide a prioritized list of files in JSON format:
{{
  ""prioritized_files"": [
    {{
      ""file_path"": ""path/to/file.cs"",
      ""priority_score"": 95,
      ""rationale"": ""This file is highly relevant to the knowledge about X and has high complexity.""
    }},
    ...
  ]
}}

Limit your response to the top {maxFiles} files.";

            // Get the prioritized files from the LLM
            var response = await _ollamaService.GenerateAsync(prompt, model);

            // Parse the JSON response
            var jsonMatch = Regex.Match(response, @"\{[\s\S]*\}");
            if (!jsonMatch.Success)
            {
                _logger.LogWarning($"Failed to extract prioritized files from response");
                return new List<PrioritizedFile>();
            }

            var jsonString = jsonMatch.Value;
            var prioritizationResult = JsonSerializer.Deserialize<PrioritizationResult>(jsonString, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });

            // Sort the prioritized files by priority score
            var prioritizedFiles = prioritizationResult.PrioritizedFiles
                .OrderByDescending(f => f.PriorityScore)
                .Take(maxFiles)
                .ToList();

            _logger.LogInformation($"Prioritized {prioritizedFiles.Count} files for improvement");
            return prioritizedFiles;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error prioritizing files in {directoryPath}");
            return new List<PrioritizedFile>();
        }
    }

    /// <summary>
    /// Get metrics for a list of files
    /// </summary>
    private async Task<Dictionary<string, FileMetrics>> GetFileMetricsAsync(string[] files)
    {
        var metrics = new Dictionary<string, FileMetrics>();

        foreach (var file in files)
        {
            try
            {
                // Get file content
                var content = await File.ReadAllTextAsync(file);

                // Calculate metrics
                var lineCount = content.Split('\n').Length;
                var complexity = CalculateComplexity(content);
                var lastModified = File.GetLastWriteTime(file);
                var gitHistory = await _gitService.GetFileHistoryAsync(file, 10);

                metrics[file] = new FileMetrics
                {
                    LineCount = lineCount,
                    Complexity = complexity,
                    LastModified = lastModified,
                    CommitCount = gitHistory?.Count ?? 0,
                    RecentCommits = gitHistory?.Count(c => c.Date > DateTime.Now.AddDays(-30)) ?? 0
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error getting metrics for {file}");
            }
        }

        return metrics;
    }

    /// <summary>
    /// Calculate complexity of a file
    /// </summary>
    private int CalculateComplexity(string content)
    {
        try
        {
            // Simple complexity metric based on control flow statements
            var ifCount = Regex.Matches(content, @"\bif\s*\(").Count;
            var forCount = Regex.Matches(content, @"\bfor\s*\(").Count;
            var foreachCount = Regex.Matches(content, @"\bforeach\s*\(").Count;
            var whileCount = Regex.Matches(content, @"\bwhile\s*\(").Count;
            var switchCount = Regex.Matches(content, @"\bswitch\s*\(").Count;
            var catchCount = Regex.Matches(content, @"\bcatch\s*\(").Count;

            return ifCount + forCount + foreachCount + whileCount + switchCount + catchCount;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating complexity");
            return 0;
        }
    }

    /// <summary>
    /// Create a summary of knowledge items
    /// </summary>
    private string CreateKnowledgeSummary(List<DocumentationKnowledge> knowledgeItems)
    {
        var summary = new StringBuilder();

        foreach (var item in knowledgeItems.Take(10))
        {
            summary.AppendLine($"Title: {item.Title}");
            summary.AppendLine($"Summary: {item.Summary}");

            if (item.KeyConcepts?.Count > 0)
            {
                summary.AppendLine("Key Concepts:");
                foreach (var concept in item.KeyConcepts.Take(3))
                {
                    summary.AppendLine($"- {concept.Name}: {concept.Definition}");
                }
            }

            summary.AppendLine();
        }

        return summary.ToString();
    }

    /// <summary>
    /// Create a summary of files with metrics
    /// </summary>
    private string CreateFilesSummary(string[] files, Dictionary<string, FileMetrics> metrics)
    {
        var summary = new StringBuilder();

        foreach (var file in files.Take(50))
        {
            if (metrics.TryGetValue(file, out var fileMetrics))
            {
                summary.AppendLine($"File: {file}");
                summary.AppendLine($"Lines: {fileMetrics.LineCount}");
                summary.AppendLine($"Complexity: {fileMetrics.Complexity}");
                summary.AppendLine($"Last Modified: {fileMetrics.LastModified}");
                summary.AppendLine($"Commit Count: {fileMetrics.CommitCount}");
                summary.AppendLine($"Recent Commits: {fileMetrics.RecentCommits}");
                summary.AppendLine();
            }
        }

        return summary.ToString();
    }

    /// <summary>
    /// File metrics
    /// </summary>
    private class FileMetrics
    {
        /// <summary>
        /// Number of lines in the file
        /// </summary>
        public int LineCount { get; set; }

        /// <summary>
        /// Complexity of the file
        /// </summary>
        public int Complexity { get; set; }

        /// <summary>
        /// Last modified date of the file
        /// </summary>
        public DateTime LastModified { get; set; }

        /// <summary>
        /// Number of commits to the file
        /// </summary>
        public int CommitCount { get; set; }

        /// <summary>
        /// Number of recent commits to the file
        /// </summary>
        public int RecentCommits { get; set; }
    }

    /// <summary>
    /// Result of prioritization
    /// </summary>
    private class PrioritizationResult
    {
        [JsonPropertyName("prioritized_files")]
        public List<PrioritizedFile> PrioritizedFiles { get; set; } = new List<PrioritizedFile>();
    }
}

/// <summary>
/// Prioritized file for knowledge application
/// </summary>
public class PrioritizedFile
{
    /// <summary>
    /// Path to the file
    /// </summary>
    [JsonPropertyName("file_path")]
    public string FilePath { get; set; }

    /// <summary>
    /// Priority score (0-100)
    /// </summary>
    [JsonPropertyName("priority_score")]
    public int PriorityScore { get; set; }

    /// <summary>
    /// Rationale for the priority
    /// </summary>
    [JsonPropertyName("rationale")]
    public string Rationale { get; set; }
}
