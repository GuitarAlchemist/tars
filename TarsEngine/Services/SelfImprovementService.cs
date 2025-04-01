using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;
using TarsEngine.Services.Models;

namespace TarsEngine.Services
{
    /// <summary>
    /// Service for self-improvement capabilities
    /// </summary>
    public class SelfImprovementService : ISelfImprovementService
    {
        private readonly ILogger<SelfImprovementService> _logger;
        private readonly IProjectAnalysisService _projectAnalysisService;
        private readonly ICodeAnalysisService _codeAnalysisService;
        private readonly ICodeGenerationService _codeGenerationService;
        private readonly ILlmService _llmService;

        public SelfImprovementService(
            ILogger<SelfImprovementService> logger,
            IProjectAnalysisService projectAnalysisService,
            ICodeAnalysisService codeAnalysisService,
            ICodeGenerationService codeGenerationService,
            ILlmService llmService)
        {
            _logger = logger;
            _projectAnalysisService = projectAnalysisService;
            _codeAnalysisService = codeAnalysisService;
            _codeGenerationService = codeGenerationService;
            _llmService = llmService;
        }

        /// <summary>
        /// Analyzes a file and suggests improvements
        /// </summary>
        /// <param name="filePath">Path to the file to analyze</param>
        /// <param name="projectPath">Path to the project</param>
        /// <returns>A list of improvement suggestions</returns>
        public virtual async Task<List<ImprovementSuggestion>> AnalyzeFileForImprovementsAsync(
            string filePath,
            string projectPath)
        {
            try
            {
                _logger.LogInformation($"Analyzing file for improvements: {filePath}");

                // Ensure the file exists
                if (!File.Exists(filePath))
                {
                    _logger.LogError($"File not found: {filePath}");
                    throw new FileNotFoundException($"File not found: {filePath}");
                }

                // Read the file content
                string fileContent = await File.ReadAllTextAsync(filePath);

                // Analyze the file
                var codeAnalysis = await _codeAnalysisService.AnalyzeFileAsync(filePath);
                if (!codeAnalysis.Success)
                {
                    _logger.LogError($"Failed to analyze file: {codeAnalysis.ErrorMessage}");
                    throw new Exception($"Failed to analyze file: {codeAnalysis.ErrorMessage}");
                }

                // Analyze the project to get context
                var projectAnalysis = await _projectAnalysisService.AnalyzeProjectAsync(projectPath);
                if (!projectAnalysis.Success)
                {
                    _logger.LogError($"Failed to analyze project: {projectAnalysis.ErrorMessage}");
                    throw new Exception($"Failed to analyze project: {projectAnalysis.ErrorMessage}");
                }

                // Create a prompt for the LLM
                string prompt = CreateImprovementAnalysisPrompt(fileContent, filePath, codeAnalysis, projectAnalysis);

                // Get suggestions from the LLM
                string llmResponse = await _llmService.GetCompletionAsync(prompt, temperature: 0.2, maxTokens: 2000);

                // Parse the suggestions
                var suggestions = ParseImprovementSuggestions(llmResponse, filePath);

                return suggestions;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error analyzing file for improvements: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Applies suggested improvements to a file
        /// </summary>
        /// <param name="filePath">Path to the file to improve</param>
        /// <param name="suggestions">List of improvement suggestions to apply</param>
        /// <param name="createBackup">Whether to create a backup of the original file</param>
        /// <returns>The path to the improved file</returns>
        public virtual async Task<string> ApplyImprovementsAsync(
            string filePath,
            List<ImprovementSuggestion> suggestions,
            bool createBackup = true)
        {
            try
            {
                _logger.LogInformation($"Applying improvements to file: {filePath}");

                // Ensure the file exists
                if (!File.Exists(filePath))
                {
                    _logger.LogError($"File not found: {filePath}");
                    throw new FileNotFoundException($"File not found: {filePath}");
                }

                // Create a backup if requested
                if (createBackup)
                {
                    string backupPath = $"{filePath}.bak.{DateTime.Now:yyyyMMddHHmmss}";
                    File.Copy(filePath, backupPath);
                    _logger.LogInformation($"Created backup at: {backupPath}");
                }

                // Read the file content
                string fileContent = await File.ReadAllTextAsync(filePath);

                // Apply each suggestion
                foreach (var suggestion in suggestions.OrderByDescending(s => s.LineNumber))
                {
                    if (suggestion.ReplacementCode != null)
                    {
                        // Apply the replacement
                        fileContent = ApplyReplacement(fileContent, suggestion);
                    }
                }

                // Write the improved content back to the file
                await File.WriteAllTextAsync(filePath, fileContent);

                _logger.LogInformation($"Applied {suggestions.Count} improvements to file: {filePath}");

                return filePath;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error applying improvements: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Generates a complete implementation for a file based on its interface or requirements
        /// </summary>
        /// <param name="filePath">Path to the file to implement</param>
        /// <param name="projectPath">Path to the project</param>
        /// <param name="requirements">Requirements for the implementation</param>
        /// <returns>The path to the implemented file</returns>
        public virtual async Task<string> GenerateImplementationAsync(
            string filePath,
            string projectPath,
            string requirements)
        {
            try
            {
                _logger.LogInformation($"Generating implementation for: {filePath}");

                // Determine the language from the file extension
                string extension = Path.GetExtension(filePath).ToLowerInvariant();
                var language = extension switch
                {
                    ".cs" => ProgrammingLanguage.CSharp,
                    ".fs" => ProgrammingLanguage.FSharp,
                    ".js" => ProgrammingLanguage.JavaScript,
                    ".ts" => ProgrammingLanguage.TypeScript,
                    ".py" => ProgrammingLanguage.Python,
                    ".java" => ProgrammingLanguage.Java,
                    ".cpp" or ".h" or ".hpp" => ProgrammingLanguage.Cpp,
                    _ => ProgrammingLanguage.Unknown
                };

                // Generate the implementation
                var result = await _codeGenerationService.GenerateCodeAsync(
                    requirements,
                    projectPath,
                    language,
                    filePath);

                if (!result.Success)
                {
                    _logger.LogError($"Failed to generate implementation: {result.ErrorMessage}");
                    throw new Exception($"Failed to generate implementation: {result.ErrorMessage}");
                }

                _logger.LogInformation($"Generated implementation saved to: {result.OutputPath}");

                return result.OutputPath;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating implementation: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Runs a self-improvement cycle on a project
        /// </summary>
        /// <param name="projectPath">Path to the project</param>
        /// <param name="maxFiles">Maximum number of files to improve</param>
        /// <param name="createBackups">Whether to create backups of original files</param>
        /// <returns>A summary of the improvements made</returns>
        public virtual async Task<SelfImprovementSummary> RunSelfImprovementCycleAsync(
            string projectPath,
            int maxFiles = 10,
            bool createBackups = true)
        {
            try
            {
                _logger.LogInformation($"Running self-improvement cycle on project: {projectPath}");

                var summary = new SelfImprovementSummary
                {
                    ProjectPath = projectPath,
                    StartTime = DateTime.Now
                };

                // Analyze the project
                var projectAnalysis = await _projectAnalysisService.AnalyzeProjectAsync(projectPath);
                if (!projectAnalysis.Success)
                {
                    _logger.LogError($"Failed to analyze project: {projectAnalysis.ErrorMessage}");
                    throw new Exception($"Failed to analyze project: {projectAnalysis.ErrorMessage}");
                }

                // Get all code files in the project
                var codeFiles = projectAnalysis.CodeAnalysisResults
                    .Where(r => r.Success)
                    .Select(r => r.FilePath)
                    .ToList();

                // Prioritize files for improvement
                var prioritizedFiles = await PrioritizeFilesForImprovementAsync(codeFiles, projectAnalysis);

                // Limit the number of files to improve
                var filesToImprove = prioritizedFiles.Take(maxFiles).ToList();

                // Improve each file
                foreach (var file in filesToImprove)
                {
                    try
                    {
                        // Analyze the file for improvements
                        var suggestions = await AnalyzeFileForImprovementsAsync(file, projectPath);

                        if (suggestions.Any())
                        {
                            // Apply the improvements
                            string improvedFilePath = await ApplyImprovementsAsync(file, suggestions, createBackups);

                            // Add to the summary
                            summary.ImprovedFiles.Add(new ImprovedFile
                            {
                                FilePath = improvedFilePath,
                                SuggestionsApplied = suggestions.Count
                            });
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, $"Error improving file {file}: {ex.Message}");
                        summary.Errors.Add($"Error improving file {file}: {ex.Message}");
                    }
                }

                summary.EndTime = DateTime.Now;
                summary.Duration = summary.EndTime - summary.StartTime;

                _logger.LogInformation($"Self-improvement cycle completed. Improved {summary.ImprovedFiles.Count} files.");

                return summary;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error running self-improvement cycle: {ex.Message}");
                throw;
            }
        }

        /// <summary>
        /// Creates a prompt for improvement analysis
        /// </summary>
        private string CreateImprovementAnalysisPrompt(
            string fileContent,
            string filePath,
            CodeAnalysisResult codeAnalysis,
            ProjectAnalysisResult projectAnalysis)
        {
            var sb = new StringBuilder();

            // Add system instructions
            sb.AppendLine("You are an expert software developer tasked with analyzing code for potential improvements.");
            sb.AppendLine("Analyze the following code and suggest specific improvements.");
            sb.AppendLine();

            // Add the file content
            sb.AppendLine("# File Content");
            sb.AppendLine("```");
            sb.AppendLine(fileContent);
            sb.AppendLine("```");
            sb.AppendLine();

            // Add file information
            sb.AppendLine("# File Information");
            sb.AppendLine($"File Path: {filePath}");
            sb.AppendLine($"Language: {codeAnalysis.Language}");
            sb.AppendLine();

            // Add project context
            sb.AppendLine("# Project Context");
            sb.AppendLine($"Project Name: {projectAnalysis.ProjectName}");
            sb.AppendLine($"Project Type: {projectAnalysis.ProjectType}");
            sb.AppendLine();

            // Add output format instructions
            sb.AppendLine("# Output Format");
            sb.AppendLine("Provide a list of specific improvement suggestions in the following format:");
            sb.AppendLine();
            sb.AppendLine("SUGGESTION:");
            sb.AppendLine("Line: [line number]");
            sb.AppendLine("Issue: [description of the issue]");
            sb.AppendLine("Improvement: [description of the suggested improvement]");
            sb.AppendLine("Code: [replacement code]");
            sb.AppendLine();
            sb.AppendLine("Focus on the following types of improvements:");
            sb.AppendLine("1. Code quality and readability");
            sb.AppendLine("2. Performance optimizations");
            sb.AppendLine("3. Bug fixes and error handling");
            sb.AppendLine("4. Security improvements");
            sb.AppendLine("5. Best practices and design patterns");

            return sb.ToString();
        }

        /// <summary>
        /// Parses improvement suggestions from an LLM response
        /// </summary>
        private List<ImprovementSuggestion> ParseImprovementSuggestions(string llmResponse, string filePath)
        {
            var suggestions = new List<ImprovementSuggestion>();

            // Split the response into suggestion blocks
            var suggestionBlocks = llmResponse.Split("SUGGESTION:", StringSplitOptions.RemoveEmptyEntries);

            foreach (var block in suggestionBlocks.Skip(1)) // Skip the first block (it's usually empty or contains intro text)
            {
                try
                {
                    var suggestion = new ImprovementSuggestion
                    {
                        FilePath = filePath
                    };

                    // Parse line number
                    var lineMatch = System.Text.RegularExpressions.Regex.Match(block, @"Line:\s*(\d+)");
                    if (lineMatch.Success)
                    {
                        suggestion.LineNumber = int.Parse(lineMatch.Groups[1].Value);
                    }

                    // Parse issue
                    var issueMatch = System.Text.RegularExpressions.Regex.Match(block, @"Issue:\s*(.+?)(?=\r?\nImprovement:|$)", System.Text.RegularExpressions.RegexOptions.Singleline);
                    if (issueMatch.Success)
                    {
                        suggestion.Issue = issueMatch.Groups[1].Value.Trim();
                    }

                    // Parse improvement
                    var improvementMatch = System.Text.RegularExpressions.Regex.Match(block, @"Improvement:\s*(.+?)(?=\r?\nCode:|$)", System.Text.RegularExpressions.RegexOptions.Singleline);
                    if (improvementMatch.Success)
                    {
                        suggestion.Improvement = improvementMatch.Groups[1].Value.Trim();
                    }

                    // Parse replacement code
                    var codeMatch = System.Text.RegularExpressions.Regex.Match(block, @"Code:\s*(.+?)(?=\r?\nSUGGESTION:|$)", System.Text.RegularExpressions.RegexOptions.Singleline);
                    if (codeMatch.Success)
                    {
                        suggestion.ReplacementCode = codeMatch.Groups[1].Value.Trim();
                    }

                    // Add the suggestion if it has at least an issue and improvement
                    if (!string.IsNullOrEmpty(suggestion.Issue) && !string.IsNullOrEmpty(suggestion.Improvement))
                    {
                        suggestions.Add(suggestion);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"Error parsing suggestion block: {ex.Message}");
                }
            }

            return suggestions;
        }

        /// <summary>
        /// Applies a replacement to the file content
        /// </summary>
        private string ApplyReplacement(string fileContent, ImprovementSuggestion suggestion)
        {
            try
            {
                // Split the content into lines
                var lines = fileContent.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

                // Ensure the line number is valid
                if (suggestion.LineNumber < 1 || suggestion.LineNumber > lines.Length)
                {
                    _logger.LogWarning($"Invalid line number {suggestion.LineNumber} for file with {lines.Length} lines");
                    return fileContent;
                }

                // Replace the line
                lines[suggestion.LineNumber - 1] = suggestion.ReplacementCode;

                // Join the lines back together
                return string.Join(Environment.NewLine, lines);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error applying replacement: {ex.Message}");
                return fileContent;
            }
        }

        /// <summary>
        /// Prioritizes files for improvement
        /// </summary>
        private async Task<List<string>> PrioritizeFilesForImprovementAsync(
            List<string> files,
            ProjectAnalysisResult projectAnalysis)
        {
            try
            {
                // Create a prompt for the LLM
                var sb = new StringBuilder();
                sb.AppendLine("You are an expert software developer tasked with prioritizing files for improvement.");
                sb.AppendLine("Given the following list of files in a project, prioritize them for improvement based on their importance and potential impact.");
                sb.AppendLine();

                // Add project context
                sb.AppendLine("# Project Context");
                sb.AppendLine($"Project Name: {projectAnalysis.ProjectName}");
                sb.AppendLine($"Project Type: {projectAnalysis.ProjectType}");
                sb.AppendLine();

                // Add the list of files
                sb.AppendLine("# Files");
                foreach (var file in files)
                {
                    sb.AppendLine(file);
                }
                sb.AppendLine();

                // Add output format instructions
                sb.AppendLine("# Output Format");
                sb.AppendLine("Provide a prioritized list of file paths, one per line, with the most important files first.");
                sb.AppendLine("Do not include any explanations or additional text.");

                // Get prioritization from the LLM
                string llmResponse = await _llmService.GetCompletionAsync(sb.ToString(), temperature: 0.2, maxTokens: 2000);

                // Parse the response
                var prioritizedFiles = llmResponse
                    .Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.RemoveEmptyEntries)
                    .Where(f => files.Contains(f))
                    .ToList();

                // Add any files that weren't included in the response
                prioritizedFiles.AddRange(files.Except(prioritizedFiles));

                return prioritizedFiles;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error prioritizing files: {ex.Message}");
                return files; // Return the original list if prioritization fails
            }
        }
    }

    /// <summary>
    /// Represents an improvement suggestion
    /// </summary>
    public class ImprovementSuggestion
    {
        public string FilePath { get; set; }
        public int LineNumber { get; set; }
        public string Issue { get; set; }
        public string Improvement { get; set; }
        public string ReplacementCode { get; set; }
    }

    /// <summary>
    /// Represents a file that has been improved
    /// </summary>
    public class ImprovedFile
    {
        public string FilePath { get; set; }
        public int SuggestionsApplied { get; set; }
    }

    /// <summary>
    /// Represents a summary of a self-improvement cycle
    /// </summary>
    public class SelfImprovementSummary
    {
        public string ProjectPath { get; set; }
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public TimeSpan Duration { get; set; }
        public List<ImprovedFile> ImprovedFiles { get; set; } = new List<ImprovedFile>();
        public List<string> Errors { get; set; } = new List<string>();
    }
}
