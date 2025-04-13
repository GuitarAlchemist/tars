using Microsoft.Extensions.Logging;
using TarsCli.Services.CodeAnalysis;
using TarsCli.Services.Adapters;

namespace TarsCli.Services;

/// <summary>
/// Service that integrates with the F# code analysis and metascript engine
/// </summary>
public class FSharpIntegrationService
{
    private readonly ILogger<FSharpIntegrationService> _logger;

    public FSharpIntegrationService(ILogger<FSharpIntegrationService> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Analyzes a file using the F# code analysis module
    /// </summary>
    public async Task<AnalysisResult> AnalyzeFileAsync(string filePath)
    {
        try
        {
            _logger.LogInformation($"Analyzing file: {filePath}");

            // Call the F# module
            var result = FSharpAnalysisWrapper.analyzeFile(filePath);

            return new AnalysisResult
            {
                FilePath = result.FilePath,
                Issues = ConvertIssues(result.Issues),
                SuggestedFixes = result.SuggestedFixes.Select(f => new CodeFix
                {
                    Original = f.Item1,
                    Replacement = f.Item2
                }).ToList()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing file {filePath}");
            return new AnalysisResult
            {
                FilePath = filePath,
                Issues = new List<CodeIssue>(),
                SuggestedFixes = new List<CodeFix>()
            };
        }
    }

    /// <summary>
    /// Analyzes a project using the F# code analysis module
    /// </summary>
    public async Task<List<AnalysisResult>> AnalyzeProjectAsync(string projectPath, int maxFiles = 50)
    {
        try
        {
            _logger.LogInformation($"Analyzing project: {projectPath}");

            // Call the F# module
            var results = FSharpAnalysisWrapper.analyzeProject(projectPath, maxFiles);

            return results.Select(r => new AnalysisResult
            {
                FilePath = r.FilePath,
                Issues = ConvertIssues(r.Issues),
                SuggestedFixes = r.SuggestedFixes.Select(f => new CodeFix
                {
                    Original = f.Item1,
                    Replacement = f.Item2
                }).ToList()
            }).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing project {projectPath}");
            return new List<AnalysisResult>();
        }
    }

    /// <summary>
    /// Applies metascript rules to a file
    /// </summary>
    public async Task<string> ApplyMetascriptRulesToFileAsync(string filePath, string metascriptPath)
    {
        try
        {
            _logger.LogInformation($"Applying metascript rules from {metascriptPath} to {filePath}");

            // Load the rules
            var rules = TarsEngineFSharp.MetascriptEngine.loadRules(metascriptPath);

            // Apply the rules
            var transformedCode = TarsEngineFSharp.MetascriptEngine.applyRulesToFile(filePath, rules);

            return transformedCode;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error applying metascript rules to {filePath}");
            return string.Empty;
        }
    }

    /// <summary>
    /// Applies metascript rules to a project
    /// </summary>
    public async Task<Dictionary<string, string>> ApplyMetascriptRulesToProjectAsync(
        string projectPath, string metascriptPath, int maxFiles = 50)
    {
        try
        {
            _logger.LogInformation($"Applying metascript rules from {metascriptPath} to project {projectPath}");

            var results = new Dictionary<string, string>();

            // Load the rules
            var rules = TarsEngineFSharp.MetascriptEngine.loadRules(metascriptPath);

            // Get all C# files
            var files = Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories)
                .Take(maxFiles);

            foreach (var file in files)
            {
                // Apply the rules
                var transformedCode = TarsEngineFSharp.MetascriptEngine.applyRulesToFile(file, rules);
                results[file] = transformedCode;
            }

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error applying metascript rules to project {projectPath}");
            return new Dictionary<string, string>();
        }
    }

    // Helper method to convert F# issues to C# issues
    private List<CodeIssue> ConvertIssues(Microsoft.FSharp.Collections.FSharpList<CodeAnalysis.CodeIssue> issues)
    {
        var result = new List<CodeIssue>();

        foreach (var issue in issues)
        {
            if (issue.Type == CodeIssueType.Security)
            {
                result.Add(new CodeIssue
                {
                    Type = IssueType.MissingExceptionHandling,
                    Location = issue.LineNumber + ":" + issue.ColumnNumber,
                    Description = issue.Description
                });
            }
            else if (issue.Type == CodeIssueType.Performance)
            {
                result.Add(new CodeIssue
                {
                    Type = IssueType.IneffectiveCode,
                    Location = issue.LineNumber + ":" + issue.ColumnNumber,
                    Description = issue.Description,
                    Suggestion = issue.SuggestedFix
                });
            }
            else if (issue.Type == CodeIssueType.Style)
            {
                result.Add(new CodeIssue
                {
                    Type = IssueType.StyleViolation,
                    Location = issue.LineNumber + ":" + issue.ColumnNumber,
                    Description = issue.Description
                });
            }
            else if (issue.Type == CodeIssueType.Documentation)
            {
                result.Add(new CodeIssue
                {
                    Type = IssueType.DocumentationIssue,
                    Location = issue.LineNumber + ":" + issue.ColumnNumber,
                    Description = $"Missing documentation: {issue.Description}"
                });
            }
        }

        return result;
    }
}

// C# models to match the F# types
public enum IssueType
{
    MissingExceptionHandling,
    IneffectiveCode,
    StyleViolation,
    DocumentationIssue
}

public class CodeIssue
{
    public IssueType Type { get; set; }
    public string Location { get; set; }
    public string Description { get; set; }
    public string Suggestion { get; set; }
}

public class CodeFix
{
    public string Original { get; set; }
    public string Replacement { get; set; }
}

public class AnalysisResult
{
    public string FilePath { get; set; }
    public List<CodeIssue> Issues { get; set; } = new();
    public List<CodeFix> SuggestedFixes { get; set; } = new();
}