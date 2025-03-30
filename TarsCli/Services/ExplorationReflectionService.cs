using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for analyzing and reflecting on TARS exploration files
    /// </summary>
    public class ExplorationReflectionService
    {
        private readonly ILogger<ExplorationReflectionService> _logger;
        private readonly IConfiguration _configuration;
        private readonly OllamaService _ollamaService;
        private readonly string _explorationsDirectory;

        public ExplorationReflectionService(
            ILogger<ExplorationReflectionService> logger,
            IConfiguration configuration,
            OllamaService ollamaService)
        {
            _logger = logger;
            _configuration = configuration;
            _ollamaService = ollamaService;

            // Get explorations directory from configuration or use default
            _explorationsDirectory = _configuration.GetValue<string>("Tars:Explorations:Directory",
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "docs", "Explorations"));

            _logger.LogInformation($"ExplorationReflectionService initialized with explorations directory: {_explorationsDirectory}");
        }

        /// <summary>
        /// Get all exploration files in the specified directory
        /// </summary>
        public List<ExplorationFileOld> GetExplorationFiles(string subDirectory = "v1/Chats")
        {
            try
            {
                var directory = Path.Combine(_explorationsDirectory, subDirectory);

                if (!Directory.Exists(directory))
                {
                    _logger.LogWarning($"Explorations directory not found: {directory}");
                    return new List<ExplorationFileOld>();
                }

                var files = Directory.GetFiles(directory, "*.md")
                    .Select(f => new ExplorationFileOld
                    {
                        FilePath = f,
                        FileName = Path.GetFileName(f),
                        Title = Path.GetFileNameWithoutExtension(f).Replace("ChatGPT-", ""),
                        LastModified = File.GetLastWriteTime(f),
                        SizeInBytes = new FileInfo(f).Length
                    })
                    .OrderByDescending(f => f.LastModified)
                    .ToList();

                _logger.LogInformation($"Found {files.Count} exploration files in {directory}");
                return files;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error getting exploration files from {_explorationsDirectory}");
                return new List<ExplorationFileOld>();
            }
        }

        /// <summary>
        /// Parse an exploration file to extract its content
        /// </summary>
        public async Task<ExplorationContent> ParseExplorationFileAsync(string filePath)
        {
            try
            {
                if (!File.Exists(filePath))
                {
                    _logger.LogWarning($"Exploration file not found: {filePath}");
                    return null;
                }

                var content = await File.ReadAllTextAsync(filePath);
                var exploration = new ExplorationContent
                {
                    FilePath = filePath,
                    FileName = Path.GetFileName(filePath),
                    Title = Path.GetFileNameWithoutExtension(filePath).Replace("ChatGPT-", ""),
                    FullContent = content
                };

                // Extract metadata
                var metadataRegex = new Regex(@"\*\*User:\*\* (.*?)\s+\*\*Created:\*\* (.*?)\s+\*\*Updated:\*\* (.*?)\s+\*\*Exported:\*\* (.*?)\s+", RegexOptions.Singleline);
                var metadataMatch = metadataRegex.Match(content);

                if (metadataMatch.Success)
                {
                    exploration.User = metadataMatch.Groups[1].Value.Trim();
                    exploration.Created = ParseDateTime(metadataMatch.Groups[2].Value.Trim());
                    exploration.Updated = ParseDateTime(metadataMatch.Groups[3].Value.Trim());
                    exploration.Exported = ParseDateTime(metadataMatch.Groups[4].Value.Trim());
                }

                // Extract prompt
                var promptRegex = new Regex(@"## Prompt:\s+(.*?)(?=## Response:|$)", RegexOptions.Singleline);
                var promptMatch = promptRegex.Match(content);

                if (promptMatch.Success)
                {
                    exploration.Prompt = promptMatch.Groups[1].Value.Trim();
                }

                // Extract response
                var responseRegex = new Regex(@"## Response:\s+(.*?)(?=## |$)", RegexOptions.Singleline);
                var responseMatch = responseRegex.Match(content);

                if (responseMatch.Success)
                {
                    exploration.Response = responseMatch.Groups[1].Value.Trim();
                }

                _logger.LogInformation($"Parsed exploration file: {exploration.Title}");
                return exploration;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error parsing exploration file: {filePath}");
                return null;
            }
        }

        /// <summary>
        /// Generate a reflection on a single exploration file
        /// </summary>
        public async Task<string> GenerateReflectionAsync(string filePath, string model = "llama3")
        {
            try
            {
                var exploration = await ParseExplorationFileAsync(filePath);

                if (exploration == null)
                {
                    return "Error: Could not parse exploration file.";
                }

                var prompt = $@"You are TARS, an AI assistant that is reflecting on past explorations and conversations.
Please analyze the following exploration and provide a thoughtful reflection on its content, insights, and implications for the TARS project.

EXPLORATION TITLE: {exploration.Title}
CREATED: {exploration.Created}

PROMPT:
{exploration.Prompt}

RESPONSE SUMMARY:
{TruncateText(exploration.Response, 2000)}

Based on this exploration, please provide:
1. A brief summary of the key points discussed
2. The most valuable insights gained from this exploration
3. How these insights could be applied to improve or extend TARS
4. Any potential future directions or follow-up explorations that might be valuable

Your reflection should be concise but insightful, focusing on the most important aspects of this exploration.";

                _logger.LogInformation($"Generating reflection for: {exploration.Title}");
                var reflection = await _ollamaService.GenerateCompletion(prompt, model);

                return reflection;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating reflection for: {filePath}");
                return $"Error generating reflection: {ex.Message}";
            }
        }

        /// <summary>
        /// Generate a meta-reflection on multiple exploration files
        /// </summary>
        public async Task<string> GenerateMetaReflectionAsync(List<string> filePaths, string model = "llama3")
        {
            try
            {
                if (filePaths == null || filePaths.Count == 0)
                {
                    return "Error: No exploration files provided.";
                }

                var explorations = new List<ExplorationContent>();

                foreach (var filePath in filePaths)
                {
                    var exploration = await ParseExplorationFileAsync(filePath);
                    if (exploration != null)
                    {
                        explorations.Add(exploration);
                    }
                }

                if (explorations.Count == 0)
                {
                    return "Error: Could not parse any exploration files.";
                }

                var explorationsText = string.Join("\n\n", explorations.Select(e =>
                    $"TITLE: {e.Title}\nCREATED: {e.Created}\nPROMPT: {TruncateText(e.Prompt, 200)}\n"));

                var prompt = $@"You are TARS, an AI assistant that is reflecting on past explorations and conversations.
Please analyze the following set of {explorations.Count} explorations and provide a thoughtful meta-reflection on the patterns, themes, and insights across these explorations.

EXPLORATIONS:
{explorationsText}

Based on these explorations, please provide:
1. The main themes and patterns you observe across these explorations
2. How these explorations relate to each other and build upon each other
3. The most valuable insights gained from these explorations collectively
4. How these insights could be synthesized to improve or extend TARS
5. Potential future directions or areas of exploration that might be valuable based on these past explorations

Your meta-reflection should identify connections between explorations and synthesize the knowledge gained across them.";

                _logger.LogInformation($"Generating meta-reflection for {explorations.Count} explorations");
                var reflection = await _ollamaService.GenerateCompletion(prompt, model);

                return reflection;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating meta-reflection");
                return $"Error generating meta-reflection: {ex.Message}";
            }
        }

        /// <summary>
        /// Generate a comprehensive reflection report on all explorations
        /// </summary>
        public async Task<string> GenerateReflectionReportAsync(string subDirectory = "v1/Chats", string model = "llama3")
        {
            try
            {
                var files = GetExplorationFiles(subDirectory);

                if (files.Count == 0)
                {
                    return "Error: No exploration files found.";
                }

                var sb = new StringBuilder();
                sb.AppendLine("# TARS Exploration Reflections");
                sb.AppendLine();
                sb.AppendLine($"Generated: {DateTime.Now}");
                sb.AppendLine();
                sb.AppendLine("## Overview");
                sb.AppendLine();
                sb.AppendLine($"This report contains reflections on {files.Count} explorations in the TARS project.");
                sb.AppendLine();

                // Generate meta-reflection on all explorations
                sb.AppendLine("## Meta-Reflection");
                sb.AppendLine();
                var metaReflection = await GenerateMetaReflectionAsync(files.Take(10).Select(f => f.FilePath).ToList(), model);
                sb.AppendLine(metaReflection);
                sb.AppendLine();

                // Generate reflections on individual explorations
                sb.AppendLine("## Individual Reflections");
                sb.AppendLine();

                // Limit to 5 most recent explorations for the report
                foreach (var file in files.Take(5))
                {
                    sb.AppendLine($"### {file.Title}");
                    sb.AppendLine();
                    var reflection = await GenerateReflectionAsync(file.FilePath, model);
                    sb.AppendLine(reflection);
                    sb.AppendLine();
                    sb.AppendLine("---");
                    sb.AppendLine();
                }

                // Add conclusion
                sb.AppendLine("## Conclusion");
                sb.AppendLine();
                sb.AppendLine("These reflections provide insights into the TARS project's explorations and potential future directions.");
                sb.AppendLine("By analyzing past conversations and explorations, TARS can build upon existing knowledge and identify new areas for growth.");

                return sb.ToString();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating reflection report");
                return $"Error generating reflection report: {ex.Message}";
            }
        }

        /// <summary>
        /// Save a reflection report to a file
        /// </summary>
        public async Task<string> SaveReflectionReportAsync(string report, string fileName = "reflection_report.md")
        {
            try
            {
                var reportsDirectory = Path.Combine(_explorationsDirectory, "Reflections");

                if (!Directory.Exists(reportsDirectory))
                {
                    Directory.CreateDirectory(reportsDirectory);
                }

                var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                var filePath = Path.Combine(reportsDirectory, $"{timestamp}_{fileName}");

                await File.WriteAllTextAsync(filePath, report);

                _logger.LogInformation($"Saved reflection report to: {filePath}");
                return filePath;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving reflection report");
                return null;
            }
        }

        /// <summary>
        /// Parse a date time string
        /// </summary>
        private DateTime ParseDateTime(string dateTimeString)
        {
            if (DateTime.TryParse(dateTimeString, out var dateTime))
            {
                return dateTime;
            }

            return DateTime.MinValue;
        }

        /// <summary>
        /// Truncate text to a maximum length
        /// </summary>
        private string TruncateText(string text, int maxLength)
        {
            if (string.IsNullOrEmpty(text) || text.Length <= maxLength)
            {
                return text;
            }

            return text.Substring(0, maxLength) + "...";
        }
    }

    /// <summary>
    /// Represents an exploration file
    /// </summary>
    public class ExplorationFileOld
    {
        /// <summary>
        /// The full path to the file
        /// </summary>
        public string FilePath { get; set; }

        /// <summary>
        /// The file name
        /// </summary>
        public string FileName { get; set; }

        /// <summary>
        /// The title of the exploration
        /// </summary>
        public string Title { get; set; }

        /// <summary>
        /// The last modified date of the file
        /// </summary>
        public DateTime LastModified { get; set; }

        /// <summary>
        /// The size of the file in bytes
        /// </summary>
        public long SizeInBytes { get; set; }
    }

    /// <summary>
    /// Represents the content of an exploration file
    /// </summary>
    public class ExplorationContent
    {
        /// <summary>
        /// The full path to the file
        /// </summary>
        public string FilePath { get; set; }

        /// <summary>
        /// The file name
        /// </summary>
        public string FileName { get; set; }

        /// <summary>
        /// The title of the exploration
        /// </summary>
        public string Title { get; set; }

        /// <summary>
        /// The user who created the exploration
        /// </summary>
        public string User { get; set; }

        /// <summary>
        /// The date the exploration was created
        /// </summary>
        public DateTime Created { get; set; }

        /// <summary>
        /// The date the exploration was updated
        /// </summary>
        public DateTime Updated { get; set; }

        /// <summary>
        /// The date the exploration was exported
        /// </summary>
        public DateTime Exported { get; set; }

        /// <summary>
        /// The prompt used for the exploration
        /// </summary>
        public string Prompt { get; set; }

        /// <summary>
        /// The response from the exploration
        /// </summary>
        public string Response { get; set; }

        /// <summary>
        /// The full content of the exploration file
        /// </summary>
        public string FullContent { get; set; }
    }
}
