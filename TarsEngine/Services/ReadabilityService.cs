using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for analyzing code readability
/// </summary>
public class ReadabilityService : IReadabilityService
{
    private readonly ILogger<ReadabilityService> _logger;
    private readonly IMetascriptService _metascriptService;

    /// <summary>
    /// Initializes a new instance of the <see cref="ReadabilityService"/> class
    /// </summary>
    public ReadabilityService(ILogger<ReadabilityService> logger, IMetascriptService metascriptService)
    {
        _logger = logger;
        _metascriptService = metascriptService;
    }

    /// <inheritdoc/>
    public async Task<ReadabilityAnalysisResult> AnalyzeReadabilityAsync(string filePath, string language)
    {
        try
        {
            _logger.LogInformation("Analyzing readability for file: {FilePath}", filePath);

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            var fileContent = await File.ReadAllTextAsync(filePath);

            // Create a metascript for readability analysis
            var metascript = $@"
// Readability analysis metascript
// Language: {language}

// Read the source file
let sourceCode = `{fileContent.Replace("`", "\\`")}`;

// Analyze readability
let readabilityResult = analyzeReadability(sourceCode, '{language}');

// Return the readability result
return JSON.stringify(readabilityResult);

// Helper function to analyze readability
function analyzeReadability(code, language) {{
    // This would be implemented with a more sophisticated analysis
    // For now, we'll use a simple placeholder
    
    // Calculate readability scores
    const overallScore = 75;
    const namingConventionScore = 80;
    const commentQualityScore = 70;
    const codeFormattingScore = 85;
    const documentationScore = 65;
    
    // Identify readability issues
    const issues = [
        {{
            description: 'Variable name is too short',
            location: 'Line 42',
            suggestedFix: 'Use a more descriptive name',
            scoreImpact: -5
        }},
        {{
            description: 'Method is missing XML documentation',
            location: 'Line 57',
            suggestedFix: 'Add XML documentation for the method',
            scoreImpact: -3
        }}
    ];
    
    // Calculate readability metrics
    const metrics = {{
        averageIdentifierLength: 12.3,
        commentDensity: 0.15,
        documentationCoverage: 0.75,
        averageParameterCount: 2.5,
        maxParameterCount: 5,
        readabilityIssues: [
            {{
                description: 'Identifier name is too short',
                location: 'Line 63',
                suggestedFix: 'Use a more descriptive name',
                scoreImpact: -2
            }}
        ]
    }};
    
    return {{
        overallScore,
        namingConventionScore,
        commentQualityScore,
        codeFormattingScore,
        documentationScore,
        issues,
        metrics
    }};
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            
            // Parse the result as JSON
            var readabilityResult = System.Text.Json.JsonSerializer.Deserialize<ReadabilityAnalysisResult>(result.ToString());
            return readabilityResult ?? new ReadabilityAnalysisResult();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing readability for file: {FilePath}", filePath);
            return new ReadabilityAnalysisResult
            {
                OverallScore = 0,
                NamingConventionScore = 0,
                CommentQualityScore = 0,
                CodeFormattingScore = 0,
                DocumentationScore = 0,
                Issues = new List<ReadabilityIssue>
                {
                    new ReadabilityIssue
                    {
                        Description = $"Error analyzing readability: {ex.Message}",
                        Location = filePath
                    }
                },
                Metrics = new ReadabilityMetrics()
            };
        }
    }

    /// <inheritdoc/>
    public async Task<ReadabilityAnalysisResult> AnalyzeProjectReadabilityAsync(string projectPath)
    {
        try
        {
            _logger.LogInformation("Analyzing readability for project: {ProjectPath}", projectPath);

            if (!Directory.Exists(projectPath) && !File.Exists(projectPath))
            {
                throw new DirectoryNotFoundException($"Project not found: {projectPath}");
            }

            // Find all source files in the project
            var sourceFiles = FindSourceFiles(projectPath);
            if (sourceFiles.Count == 0)
            {
                _logger.LogWarning("No source files found in project: {ProjectPath}", projectPath);
                return new ReadabilityAnalysisResult
                {
                    OverallScore = 0,
                    NamingConventionScore = 0,
                    CommentQualityScore = 0,
                    CodeFormattingScore = 0,
                    DocumentationScore = 0,
                    Issues = new List<ReadabilityIssue>(),
                    Metrics = new ReadabilityMetrics()
                };
            }

            // Analyze each source file
            var readabilityResults = new List<ReadabilityAnalysisResult>();
            foreach (var sourceFile in sourceFiles)
            {
                var fileExtension = Path.GetExtension(sourceFile).ToLowerInvariant();
                var language = GetLanguageFromExtension(fileExtension);
                var readabilityResult = await AnalyzeReadabilityAsync(sourceFile, language);
                readabilityResults.Add(readabilityResult);
            }

            // Calculate overall readability
            return CalculateOverallReadability(readabilityResults);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing readability for project: {ProjectPath}", projectPath);
            return new ReadabilityAnalysisResult
            {
                OverallScore = 0,
                NamingConventionScore = 0,
                CommentQualityScore = 0,
                CodeFormattingScore = 0,
                DocumentationScore = 0,
                Issues = new List<ReadabilityIssue>
                {
                    new ReadabilityIssue
                    {
                        Description = $"Error analyzing readability: {ex.Message}",
                        Location = projectPath
                    }
                },
                Metrics = new ReadabilityMetrics()
            };
        }
    }

    /// <inheritdoc/>
    public async Task<ReadabilityAnalysisResult> AnalyzeSolutionReadabilityAsync(string solutionPath)
    {
        try
        {
            _logger.LogInformation("Analyzing readability for solution: {SolutionPath}", solutionPath);

            if (!File.Exists(solutionPath))
            {
                throw new FileNotFoundException($"Solution file not found: {solutionPath}");
            }

            // Find all projects in the solution
            var projectPaths = FindProjectsInSolution(solutionPath);
            if (projectPaths.Count == 0)
            {
                _logger.LogWarning("No projects found in solution: {SolutionPath}", solutionPath);
                return new ReadabilityAnalysisResult
                {
                    OverallScore = 0,
                    NamingConventionScore = 0,
                    CommentQualityScore = 0,
                    CodeFormattingScore = 0,
                    DocumentationScore = 0,
                    Issues = new List<ReadabilityIssue>(),
                    Metrics = new ReadabilityMetrics()
                };
            }

            // Analyze each project
            var readabilityResults = new List<ReadabilityAnalysisResult>();
            foreach (var projectPath in projectPaths)
            {
                var readabilityResult = await AnalyzeProjectReadabilityAsync(projectPath);
                readabilityResults.Add(readabilityResult);
            }

            // Calculate overall readability
            return CalculateOverallReadability(readabilityResults);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing readability for solution: {SolutionPath}", solutionPath);
            return new ReadabilityAnalysisResult
            {
                OverallScore = 0,
                NamingConventionScore = 0,
                CommentQualityScore = 0,
                CodeFormattingScore = 0,
                DocumentationScore = 0,
                Issues = new List<ReadabilityIssue>
                {
                    new ReadabilityIssue
                    {
                        Description = $"Error analyzing readability: {ex.Message}",
                        Location = solutionPath
                    }
                },
                Metrics = new ReadabilityMetrics()
            };
        }
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityIssue>> IdentifyReadabilityIssuesAsync(string filePath, string language)
    {
        try
        {
            _logger.LogInformation("Identifying readability issues in file: {FilePath}", filePath);

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            var fileContent = await File.ReadAllTextAsync(filePath);

            // Create a metascript for identifying readability issues
            var metascript = $@"
// Readability issue identification metascript
// Language: {language}

// Read the source file
let sourceCode = `{fileContent.Replace("`", "\\`")}`;

// Identify readability issues
let readabilityIssues = identifyReadabilityIssues(sourceCode, '{language}');

// Return the readability issues
return JSON.stringify(readabilityIssues);

// Helper function to identify readability issues
function identifyReadabilityIssues(code, language) {{
    // This would be implemented with a more sophisticated analysis
    // For now, we'll use a simple placeholder
    
    const issues = [
        {{
            description: 'Variable name is too short',
            location: 'Line 42',
            suggestedFix: 'Use a more descriptive name',
            scoreImpact: -5
        }},
        {{
            description: 'Method is missing XML documentation',
            location: 'Line 57',
            suggestedFix: 'Add XML documentation for the method',
            scoreImpact: -3
        }},
        {{
            description: 'Too many parameters in method',
            location: 'Line 75',
            suggestedFix: 'Consider using a parameter object',
            scoreImpact: -4
        }}
    ];
    
    return issues;
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            
            // Parse the result as JSON
            var readabilityIssues = System.Text.Json.JsonSerializer.Deserialize<List<ReadabilityIssue>>(result.ToString());
            return readabilityIssues ?? new List<ReadabilityIssue>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error identifying readability issues in file: {FilePath}", filePath);
            return new List<ReadabilityIssue>
            {
                new ReadabilityIssue
                {
                    Description = $"Error identifying readability issues: {ex.Message}",
                    Location = filePath
                }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityImprovement>> SuggestReadabilityImprovementsAsync(ReadabilityIssue readabilityIssue)
    {
        try
        {
            _logger.LogInformation("Suggesting readability improvements for issue: {Description}", readabilityIssue.Description);

            // Create a metascript for suggesting readability improvements
            var metascript = $@"
// Readability improvement suggestion metascript

// Readability issue
let readabilityIssue = {System.Text.Json.JsonSerializer.Serialize(readabilityIssue)};

// Suggest readability improvements
let improvements = suggestReadabilityImprovements(readabilityIssue);

// Return the suggested improvements
return JSON.stringify(improvements);

// Helper function to suggest readability improvements
function suggestReadabilityImprovements(issue) {{
    // This would be implemented with a more sophisticated analysis
    // For now, we'll use a simple placeholder
    
    const improvements = [
        {{
            description: `Improvement for: ${{issue.description}}`,
            improvedCode: issue.suggestedFix ? `// ${{issue.suggestedFix}}` : '// Improved code',
            readabilityScore: Math.abs(issue.scoreImpact || 3),
            confidence: 0.8,
            category: 'Naming'
        }}
    ];
    
    // Add more specific improvements based on the issue
    if (issue.description.includes('name')) {{
        improvements.push({{
            description: 'Use more descriptive names',
            improvedCode: '// Example of descriptive naming',
            readabilityScore: 4,
            confidence: 0.9,
            category: 'Naming'
        }});
    }} else if (issue.description.includes('documentation')) {{
        improvements.push({{
            description: 'Add XML documentation',
            improvedCode: '/// <summary>\\n/// Description of the method\\n/// </summary>',
            readabilityScore: 3,
            confidence: 0.85,
            category: 'Documentation'
        }});
    }} else if (issue.description.includes('parameters')) {{
        improvements.push({{
            description: 'Use parameter object',
            improvedCode: '// Example of using a parameter object',
            readabilityScore: 4,
            confidence: 0.75,
            category: 'MethodStructure'
        }});
    }}
    
    return improvements;
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            
            // Parse the result as JSON
            var improvements = System.Text.Json.JsonSerializer.Deserialize<List<ReadabilityImprovement>>(result.ToString());
            return improvements ?? new List<ReadabilityImprovement>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error suggesting readability improvements for issue: {Description}", readabilityIssue.Description);
            return new List<ReadabilityImprovement>();
        }
    }

    private List<string> FindSourceFiles(string projectPath)
    {
        try
        {
            var sourceFiles = new List<string>();

            if (File.Exists(projectPath))
            {
                // If projectPath is a file, assume it's a project file
                var projectDirectory = Path.GetDirectoryName(projectPath) ?? string.Empty;
                sourceFiles.AddRange(Directory.GetFiles(projectDirectory, "*.cs", SearchOption.AllDirectories));
                sourceFiles.AddRange(Directory.GetFiles(projectDirectory, "*.fs", SearchOption.AllDirectories));
                sourceFiles.AddRange(Directory.GetFiles(projectDirectory, "*.js", SearchOption.AllDirectories));
                sourceFiles.AddRange(Directory.GetFiles(projectDirectory, "*.ts", SearchOption.AllDirectories));
            }
            else if (Directory.Exists(projectPath))
            {
                // If projectPath is a directory, search for source files
                sourceFiles.AddRange(Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories));
                sourceFiles.AddRange(Directory.GetFiles(projectPath, "*.fs", SearchOption.AllDirectories));
                sourceFiles.AddRange(Directory.GetFiles(projectPath, "*.js", SearchOption.AllDirectories));
                sourceFiles.AddRange(Directory.GetFiles(projectPath, "*.ts", SearchOption.AllDirectories));
            }

            // Filter out test files, generated files, etc.
            sourceFiles = sourceFiles.Where(file =>
                !file.Contains("\\obj\\") &&
                !file.Contains("\\bin\\") &&
                !file.Contains("\\node_modules\\") &&
                !file.Contains("\\dist\\") &&
                !file.Contains("\\test\\") &&
                !file.Contains("\\tests\\") &&
                !file.Contains(".Test.") &&
                !file.Contains(".Tests.") &&
                !file.Contains(".Generated.")
            ).ToList();

            return sourceFiles;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding source files in project: {ProjectPath}", projectPath);
            return new List<string>();
        }
    }

    private List<string> FindProjectsInSolution(string solutionPath)
    {
        try
        {
            var projectPaths = new List<string>();

            // Read the solution file
            var solutionContent = File.ReadAllText(solutionPath);
            var projectLines = solutionContent.Split('\n').Where(line => line.Contains("Project(")).ToList();

            foreach (var line in projectLines)
            {
                // Extract the project path
                var match = System.Text.RegularExpressions.Regex.Match(line, @"Project\([^)]+\)\s*=\s*""[^""]*""\s*,\s*""([^""]*)""\s*,");
                if (match.Success)
                {
                    var relativePath = match.Groups[1].Value;
                    var absolutePath = Path.Combine(Path.GetDirectoryName(solutionPath) ?? string.Empty, relativePath);
                    if (File.Exists(absolutePath))
                    {
                        projectPaths.Add(absolutePath);
                    }
                }
            }

            return projectPaths;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding projects in solution: {SolutionPath}", solutionPath);
            return new List<string>();
        }
    }

    private ReadabilityAnalysisResult CalculateOverallReadability(List<ReadabilityAnalysisResult> readabilityResults)
    {
        if (readabilityResults.Count == 0)
        {
            return new ReadabilityAnalysisResult();
        }

        // Calculate average scores
        var overallScore = readabilityResults.Average(r => r.OverallScore);
        var namingConventionScore = readabilityResults.Average(r => r.NamingConventionScore);
        var commentQualityScore = readabilityResults.Average(r => r.CommentQualityScore);
        var codeFormattingScore = readabilityResults.Average(r => r.CodeFormattingScore);
        var documentationScore = readabilityResults.Average(r => r.DocumentationScore);

        // Combine issues
        var issues = readabilityResults.SelectMany(r => r.Issues).ToList();

        // Combine metrics
        var readabilityIssues = readabilityResults.SelectMany(r => r.Metrics?.ReadabilityIssues ?? new List<ReadabilityIssue>()).ToList();
        var averageIdentifierLength = readabilityResults.Average(r => r.Metrics?.AverageIdentifierLength ?? 0);
        var commentDensity = readabilityResults.Average(r => r.Metrics?.CommentDensity ?? 0);
        var documentationCoverage = readabilityResults.Average(r => r.Metrics?.DocumentationCoverage ?? 0);
        var averageParameterCount = readabilityResults.Average(r => r.Metrics?.AverageParameterCount ?? 0);
        var maxParameterCount = readabilityResults.Max(r => r.Metrics?.MaxParameterCount ?? 0);

        return new ReadabilityAnalysisResult
        {
            OverallScore = overallScore,
            NamingConventionScore = namingConventionScore,
            CommentQualityScore = commentQualityScore,
            CodeFormattingScore = codeFormattingScore,
            DocumentationScore = documentationScore,
            Issues = issues,
            Metrics = new ReadabilityMetrics
            {
                AverageIdentifierLength = averageIdentifierLength,
                CommentDensity = commentDensity,
                DocumentationCoverage = documentationCoverage,
                AverageParameterCount = averageParameterCount,
                MaxParameterCount = maxParameterCount,
                ReadabilityIssues = readabilityIssues
            }
        };
    }

    private string GetLanguageFromExtension(string extension)
    {
        return extension.ToLowerInvariant() switch
        {
            ".cs" => "csharp",
            ".fs" => "fsharp",
            ".js" => "javascript",
            ".ts" => "typescript",
            ".py" => "python",
            ".java" => "java",
            _ => "unknown"
        };
    }
}
