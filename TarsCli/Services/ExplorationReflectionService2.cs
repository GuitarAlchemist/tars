using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Configuration;

namespace TarsCli.Services;

/// <summary>
/// Service for analyzing and reflecting on TARS exploration chats
/// </summary>
public class ExplorationReflectionService2
{
    private readonly ILogger<ExplorationReflectionService2> _logger;
    private readonly IConfiguration _configuration;
    private readonly OllamaService _ollamaService;
    private readonly string _explorationsDirectory;

    public ExplorationReflectionService2(
        ILogger<ExplorationReflectionService2> logger,
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
    /// Get all exploration chat files in the specified directory
    /// </summary>
    public List<ExplorationFile> GetExplorationFiles(string directoryPath)
    {
        try
        {
            var fullPath = Path.Combine(_explorationsDirectory, directoryPath);
                
            if (!Directory.Exists(fullPath))
            {
                _logger.LogWarning($"Directory not found: {fullPath}");
                return [];
            }

            var files = Directory.GetFiles(fullPath, "*.md")
                .Select(f => new ExplorationFile(f))
                .ToList();

            _logger.LogInformation($"Found {files.Count} exploration files in {fullPath}");
            return files;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting exploration files from {directoryPath}");
            return [];
        }
    }

    /// <summary>
    /// Parse an exploration chat file
    /// </summary>
    public async Task<ExplorationChat> ParseExplorationFileAsync(string filePath)
    {
        try
        {
            var content = await File.ReadAllTextAsync(filePath);
            var fileName = Path.GetFileNameWithoutExtension(filePath);
                
            // Extract title (remove "ChatGPT-" prefix)
            var title = fileName.StartsWith("ChatGPT-") ? fileName.Substring(8) : fileName;
                
            // Extract metadata
            var createdMatch = Regex.Match(content, @"\*\*Created:\*\* (.*?)\s+");
            var updatedMatch = Regex.Match(content, @"\*\*Updated:\*\* (.*?)\s+");
            var exportedMatch = Regex.Match(content, @"\*\*Exported:\*\* (.*?)\s+");
                
            var created = createdMatch.Success ? createdMatch.Groups[1].Value : string.Empty;
            var updated = updatedMatch.Success ? updatedMatch.Groups[1].Value : string.Empty;
            var exported = exportedMatch.Success ? exportedMatch.Groups[1].Value : string.Empty;
                
            // Extract prompt and response
            var promptMatch = Regex.Match(content, @"## Prompt:\s+(.*?)(?=##|\Z)", RegexOptions.Singleline);
            var responseMatch = Regex.Match(content, @"## Response:\s+(.*?)(?=##|\Z)", RegexOptions.Singleline);
                
            var prompt = promptMatch.Success ? promptMatch.Groups[1].Value.Trim() : string.Empty;
            var response = responseMatch.Success ? responseMatch.Groups[1].Value.Trim() : string.Empty;
                
            // Create exploration chat object
            var chat = new ExplorationChat
            {
                Title = title,
                FilePath = filePath,
                Created = created,
                Updated = updated,
                Exported = exported,
                Prompt = prompt,
                Response = response
            };
                
            _logger.LogInformation($"Parsed exploration file: {fileName}");
            return chat;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error parsing exploration file: {filePath}");
            return new ExplorationChat
            {
                Title = Path.GetFileNameWithoutExtension(filePath),
                FilePath = filePath,
                Error = ex.Message
            };
        }
    }

    /// <summary>
    /// Generate a reflection on a single exploration chat
    /// </summary>
    public async Task<string> GenerateReflectionAsync(ExplorationChat chat, string model = "llama3")
    {
        try
        {
            var prompt = $@"You are TARS, an AI assistant that is reflecting on past explorations and conversations. 
Please analyze the following exploration chat and provide a thoughtful reflection:

TITLE: {chat.Title}
DATE: {chat.Created}

PROMPT:
{chat.Prompt}

RESPONSE SUMMARY:
{TruncateText(chat.Response, 2000)}

Based on this exploration, please provide:
1. A brief summary of what was explored
2. Key insights or knowledge gained
3. Potential applications or relevance to the TARS project
4. Connections to other explorations or topics
5. Suggestions for further exploration

Your reflection should be concise but insightful, focusing on the value of this exploration for the TARS project.";

            _logger.LogInformation($"Generating reflection for: {chat.Title}");
            var reflection = await _ollamaService.GenerateCompletion(prompt, model);
                
            return reflection;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating reflection for: {chat.Title}");
            return $"Error generating reflection: {ex.Message}";
        }
    }

    /// <summary>
    /// Generate a meta-reflection on multiple exploration chats
    /// </summary>
    public async Task<string> GenerateMetaReflectionAsync(List<ExplorationChat> chats, string model = "llama3")
    {
        try
        {
            // Create a summary of all chats
            var chatSummaries = new StringBuilder();
            foreach (var chat in chats)
            {
                chatSummaries.AppendLine($"TITLE: {chat.Title}");
                chatSummaries.AppendLine($"DATE: {chat.Created}");
                chatSummaries.AppendLine($"PROMPT: {TruncateText(chat.Prompt, 200)}");
                chatSummaries.AppendLine();
            }
                
            var prompt = $@"You are TARS, an AI assistant that is reflecting on past explorations and conversations.
Please analyze the following list of {chats.Count} exploration chats and provide a meta-reflection:

{chatSummaries}

Based on these explorations, please provide:
1. An overview of the main themes and topics explored
2. Patterns or trends in the explorations
3. Key insights across multiple explorations
4. How these explorations contribute to the TARS project
5. Recommendations for future exploration directions
6. Potential connections or synergies between different explorations

Your meta-reflection should identify the big picture and strategic insights from these explorations.";

            _logger.LogInformation($"Generating meta-reflection for {chats.Count} chats");
            var reflection = await _ollamaService.GenerateCompletion(prompt, model);
                
            return reflection;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating meta-reflection");
            return $"Error generating meta-reflection: {ex.Message}";
        }
    }

    /// <summary>
    /// Generate a comprehensive reflection report on all explorations
    /// </summary>
    public async Task<string> GenerateReflectionReportAsync(string directoryPath, string model = "llama3")
    {
        try
        {
            // Get all exploration files
            var files = GetExplorationFiles(directoryPath);
            if (files.Count == 0)
            {
                return "No exploration files found.";
            }
                
            // Parse all files
            var chats = new List<ExplorationChat>();
            foreach (var file in files)
            {
                var chat = await ParseExplorationFileAsync(file.FilePath);
                chats.Add(chat);
            }
                
            // Group chats by themes
            var themes = GroupChatsByThemes(chats);
                
            // Generate meta-reflection
            var metaReflection = await GenerateMetaReflectionAsync(chats, model);
                
            // Build the report
            var report = new StringBuilder();
            report.AppendLine("# TARS Exploration Reflections");
            report.AppendLine();
            report.AppendLine("## Overview");
            report.AppendLine();
            report.AppendLine($"This report contains reflections on {chats.Count} explorations conducted as part of the TARS project.");
            report.AppendLine();
            report.AppendLine("## Meta-Reflection");
            report.AppendLine();
            report.AppendLine(metaReflection);
            report.AppendLine();
            report.AppendLine("## Exploration Themes");
            report.AppendLine();
                
            foreach (var theme in themes)
            {
                report.AppendLine($"### {theme.Key}");
                report.AppendLine();
                    
                foreach (var chat in theme.Value)
                {
                    report.AppendLine($"- [{chat.Title}]({Path.GetFileName(chat.FilePath)}) - {chat.Created}");
                }
                    
                report.AppendLine();
            }
                
            report.AppendLine("## Individual Reflections");
            report.AppendLine();
                
            // Generate individual reflections for each chat
            foreach (var chat in chats)
            {
                report.AppendLine($"### {chat.Title}");
                report.AppendLine();
                report.AppendLine($"**Date:** {chat.Created}");
                report.AppendLine();
                report.AppendLine($"**Prompt:** {TruncateText(chat.Prompt, 200)}");
                report.AppendLine();
                    
                var reflection = await GenerateReflectionAsync(chat, model);
                report.AppendLine(reflection);
                report.AppendLine();
                report.AppendLine("---");
                report.AppendLine();
            }
                
            return report.ToString();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating reflection report");
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
    /// Group chats by themes based on their titles and content
    /// </summary>
    private Dictionary<string, List<ExplorationChat>> GroupChatsByThemes(List<ExplorationChat> chats)
    {
        var themes = new Dictionary<string, List<ExplorationChat>>();
            
        // Define theme keywords
        var themeKeywords = new Dictionary<string, List<string>>
        {
            { "AI and Machine Learning",
                ["AI", "Machine Learning", "Neural", "Model", "Fine-tuning", "LlamaIndex", "Inference"]
            },
            { "Mathematics and Algorithms", ["Math", "Algorithm", "Markov", "Nash", "Quaternion", "Geometry"] },
            { "Software Development", ["Code", "C#", ".NET", "F#", "GitHub", "Repo", "DSL", "BNF", "Parsing"] },
            { "TARS Architecture", ["TARS", "Architecture", "Workflow", "Auto Improvement", "Self-Improvement"] },
            { "Integration and APIs", ["API", "Integration", "MCP", "Protocol", "Redis", "Vector Store"] },
            { "Speech and Language", ["TTS", "Speech", "Coqui", "Language"] },
            { "Visualization and UI", ["Three.js", "Visualization", "Blazor", "MudBlazor", "UI"] },
            { "Ethics and Implications", ["Ethics", "Legal", "Political", "Implications", "Business Value"] },
            { "Performance and Optimization", ["GPU", "Optimization", "Speed", "ONNX", "Ollama"] }
        };
            
        // Initialize theme lists
        foreach (var theme in themeKeywords.Keys)
        {
            themes[theme] = [];
        }
            
        // Add "Other" theme for unclassified chats
        themes["Other"] = [];
            
        // Classify each chat
        foreach (var chat in chats)
        {
            var classified = false;
                
            // Check title and prompt against theme keywords
            var textToCheck = chat.Title + " " + chat.Prompt;
                
            foreach (var theme in themeKeywords)
            {
                foreach (var keyword in theme.Value)
                {
                    if (textToCheck.Contains(keyword, StringComparison.OrdinalIgnoreCase))
                    {
                        themes[theme.Key].Add(chat);
                        classified = true;
                        break;
                    }
                }
                    
                if (classified)
                {
                    break;
                }
            }
                
            // Add to "Other" if not classified
            if (!classified)
            {
                themes["Other"].Add(chat);
            }
        }
            
        // Remove empty themes
        var emptyThemes = themes.Where(t => t.Value.Count == 0).Select(t => t.Key).ToList();
        foreach (var theme in emptyThemes)
        {
            themes.Remove(theme);
        }
            
        return themes;
    }

    /// <summary>
    /// Truncate text to a maximum length
    /// </summary>
    private string TruncateText(string text, int maxLength)
    {
        if (string.IsNullOrEmpty(text))
        {
            return string.Empty;
        }
            
        if (text.Length <= maxLength)
        {
            return text;
        }
            
        return text.Substring(0, maxLength) + "...";
    }
}

/// <summary>
/// Represents an exploration chat
/// </summary>
public class ExplorationChat
{
    /// <summary>
    /// The title of the exploration
    /// </summary>
    public string Title { get; set; } = string.Empty;
        
    /// <summary>
    /// The file path of the exploration
    /// </summary>
    public string FilePath { get; set; } = string.Empty;
        
    /// <summary>
    /// The creation date of the exploration
    /// </summary>
    public string Created { get; set; } = string.Empty;
        
    /// <summary>
    /// The update date of the exploration
    /// </summary>
    public string Updated { get; set; } = string.Empty;
        
    /// <summary>
    /// The export date of the exploration
    /// </summary>
    public string Exported { get; set; } = string.Empty;
        
    /// <summary>
    /// The prompt of the exploration
    /// </summary>
    public string Prompt { get; set; } = string.Empty;
        
    /// <summary>
    /// The response of the exploration
    /// </summary>
    public string Response { get; set; } = string.Empty;
        
    /// <summary>
    /// Any error that occurred during parsing
    /// </summary>
    public string Error { get; set; } = string.Empty;
}