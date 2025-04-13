using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for analyzing code quality
/// </summary>
public class CodeQualityService : ICodeQualityService
{
    private readonly ILogger<CodeQualityService> _logger;
    private readonly IMetascriptService _metascriptService;

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeQualityService"/> class
    /// </summary>
    public CodeQualityService(ILogger<CodeQualityService> logger, IMetascriptService metascriptService)
    {
        _logger = logger;
        _metascriptService = metascriptService;
    }

    /// <inheritdoc/>
    public async Task<CodeQualityResult> AnalyzeCodeQualityAsync(string filePath, string language)
    {
        try
        {
            _logger.LogInformation("Analyzing code quality for file: {FilePath}", filePath);

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            var fileContent = await File.ReadAllTextAsync(filePath);

            // Create a metascript for code quality analysis
            var metascript = $@"
// Code quality analysis metascript
// Language: {language}

// Read the source file
let sourceCode = `{fileContent.Replace("`", "\\`")}`;

// Analyze code quality
let qualityResult = analyzeCodeQuality(sourceCode, '{language}');

// Return the quality result
return JSON.stringify(qualityResult);

// Helper function to analyze code quality
function analyzeCodeQuality(code, language) {{
    // This would be implemented with a more sophisticated analysis
    // For now, we'll use a simple placeholder
    
    // Calculate quality scores
    const overallScore = 75;
    const maintainabilityScore = 80;
    const reliabilityScore = 70;
    const securityScore = 65;
    const performanceScore = 85;
    
    // Identify quality issues
    const issues = [
        {{
            description: 'Method is too long',
            severity: 'Warning',
            category: 'Maintainability',
            location: 'Line 42',
            suggestedFix: 'Extract parts of the method into smaller methods',
            scoreImpact: -5
        }},
        {{
            description: 'Unused variable',
            severity: 'Info',
            category: 'Maintainability',
            location: 'Line 57',
            suggestedFix: 'Remove the unused variable',
            scoreImpact: -2
        }}
    ];
    
    // Calculate complexity metrics
    const complexityMetrics = {{
        averageCyclomaticComplexity: 3.5,
        maxCyclomaticComplexity: 8,
        averageCognitiveComplexity: 2.8,
        maxCognitiveComplexity: 6,
        averageMethodLength: 15.2,
        maxMethodLength: 42,
        averageClassLength: 120.5,
        maxClassLength: 250,
        complexMethods: [
            {{
                methodName: 'ProcessData',
                className: 'DataProcessor',
                filePath: 'DataProcessor.cs',
                lineNumber: 42,
                cyclomaticComplexity: 8,
                cognitiveComplexity: 6,
                methodLength: 42
            }}
        ]
    }};
    
    // Calculate readability metrics
    const readabilityMetrics = {{
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
    
    // Calculate duplication metrics
    const duplicationMetrics = {{
        duplicationPercentage: 0.05,
        duplicatedBlocks: 2,
        duplicatedLines: 15,
        duplicatedBlocksList: [
            {{
                filePath: 'DataProcessor.cs',
                startLine: 100,
                endLine: 107,
                content: '// Duplicated code block',
                duplicateLocations: [
                    {{
                        filePath: 'DataProcessor.cs',
                        startLine: 200,
                        endLine: 207
                    }}
                ]
            }}
        ]
    }};
    
    return {{
        overallScore,
        maintainabilityScore,
        reliabilityScore,
        securityScore,
        performanceScore,
        issues,
        complexityMetrics,
        readabilityMetrics,
        duplicationMetrics
    }};
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            
            // Parse the result as JSON
            var qualityResult = System.Text.Json.JsonSerializer.Deserialize<CodeQualityResult>(result.ToString());
            return qualityResult ?? new CodeQualityResult();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing code quality for file: {FilePath}", filePath);
            return new CodeQualityResult
            {
                OverallScore = 0,
                MaintainabilityScore = 0,
                ReliabilityScore = 0,
                SecurityScore = 0,
                PerformanceScore = 0,
                Issues =
                [
                    new QualityIssue
                    {
                        Description = $"Error analyzing code quality: {ex.Message}",
                        Severity = IssueSeverity.Error,
                        Category = "Error",
                        Location = filePath
                    }
                ]
            };
        }
    }

    /// <inheritdoc/>
    public async Task<CodeQualityResult> AnalyzeProjectQualityAsync(string projectPath)
    {
        try
        {
            _logger.LogInformation("Analyzing code quality for project: {ProjectPath}", projectPath);

            if (!Directory.Exists(projectPath) && !File.Exists(projectPath))
            {
                throw new DirectoryNotFoundException($"Project not found: {projectPath}");
            }

            // Find all source files in the project
            var sourceFiles = FindSourceFiles(projectPath);
            if (sourceFiles.Count == 0)
            {
                _logger.LogWarning("No source files found in project: {ProjectPath}", projectPath);
                return new CodeQualityResult
                {
                    OverallScore = 0,
                    MaintainabilityScore = 0,
                    ReliabilityScore = 0,
                    SecurityScore = 0,
                    PerformanceScore = 0
                };
            }

            // Analyze each source file
            var qualityResults = new List<CodeQualityResult>();
            foreach (var sourceFile in sourceFiles)
            {
                var fileExtension = Path.GetExtension(sourceFile).ToLowerInvariant();
                var language = GetLanguageFromExtension(fileExtension);
                var qualityResult = await AnalyzeCodeQualityAsync(sourceFile, language);
                qualityResults.Add(qualityResult);
            }

            // Calculate overall quality
            return CalculateOverallQuality(qualityResults);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing code quality for project: {ProjectPath}", projectPath);
            return new CodeQualityResult
            {
                OverallScore = 0,
                MaintainabilityScore = 0,
                ReliabilityScore = 0,
                SecurityScore = 0,
                PerformanceScore = 0,
                Issues =
                [
                    new QualityIssue
                    {
                        Description = $"Error analyzing code quality: {ex.Message}",
                        Severity = IssueSeverity.Error,
                        Category = "Error",
                        Location = projectPath
                    }
                ]
            };
        }
    }

    /// <inheritdoc/>
    public async Task<CodeQualityResult> AnalyzeSolutionQualityAsync(string solutionPath)
    {
        try
        {
            _logger.LogInformation("Analyzing code quality for solution: {SolutionPath}", solutionPath);

            if (!File.Exists(solutionPath))
            {
                throw new FileNotFoundException($"Solution file not found: {solutionPath}");
            }

            // Find all projects in the solution
            var projectPaths = FindProjectsInSolution(solutionPath);
            if (projectPaths.Count == 0)
            {
                _logger.LogWarning("No projects found in solution: {SolutionPath}", solutionPath);
                return new CodeQualityResult
                {
                    OverallScore = 0,
                    MaintainabilityScore = 0,
                    ReliabilityScore = 0,
                    SecurityScore = 0,
                    PerformanceScore = 0
                };
            }

            // Analyze each project
            var qualityResults = new List<CodeQualityResult>();
            foreach (var projectPath in projectPaths)
            {
                var qualityResult = await AnalyzeProjectQualityAsync(projectPath);
                qualityResults.Add(qualityResult);
            }

            // Calculate overall quality
            return CalculateOverallQuality(qualityResults);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing code quality for solution: {SolutionPath}", solutionPath);
            return new CodeQualityResult
            {
                OverallScore = 0,
                MaintainabilityScore = 0,
                ReliabilityScore = 0,
                SecurityScore = 0,
                PerformanceScore = 0,
                Issues =
                [
                    new QualityIssue
                    {
                        Description = $"Error analyzing code quality: {ex.Message}",
                        Severity = IssueSeverity.Error,
                        Category = "Error",
                        Location = solutionPath
                    }
                ]
            };
        }
    }

    /// <inheritdoc/>
    public async Task<QualityTrendResult> TrackQualityScoresAsync(string projectPath)
    {
        try
        {
            _logger.LogInformation("Tracking quality scores for project: {ProjectPath}", projectPath);

            // This would be implemented with a real quality tracking system
            // For now, we'll return a placeholder result
            await Task.Delay(100); // Simulate work

            return new QualityTrendResult
            {
                Snapshots =
                [
                    new QualitySnapshot
                    {
                        Timestamp = DateTime.Now.AddDays(-30),
                        OverallScore = 70,
                        MaintainabilityScore = 75,
                        ReliabilityScore = 65,
                        SecurityScore = 60,
                        PerformanceScore = 80,
                        IssueCount = 15,
                        CommitHash = "abc123"
                    },

                    new QualitySnapshot
                    {
                        Timestamp = DateTime.Now.AddDays(-15),
                        OverallScore = 72,
                        MaintainabilityScore = 77,
                        ReliabilityScore = 67,
                        SecurityScore = 62,
                        PerformanceScore = 82,
                        IssueCount = 12,
                        CommitHash = "def456"
                    },

                    new QualitySnapshot
                    {
                        Timestamp = DateTime.Now,
                        OverallScore = 75,
                        MaintainabilityScore = 80,
                        ReliabilityScore = 70,
                        SecurityScore = 65,
                        PerformanceScore = 85,
                        IssueCount = 10,
                        CommitHash = "ghi789"
                    }
                ],
                OverallTrend = TrendDirection.Improving,
                MaintainabilityTrend = TrendDirection.Improving,
                ReliabilityTrend = TrendDirection.Improving,
                SecurityTrend = TrendDirection.Improving,
                PerformanceTrend = TrendDirection.Improving
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error tracking quality scores for project: {ProjectPath}", projectPath);
            return new QualityTrendResult
            {
                Snapshots = [],
                OverallTrend = TrendDirection.Stable,
                MaintainabilityTrend = TrendDirection.Stable,
                ReliabilityTrend = TrendDirection.Stable,
                SecurityTrend = TrendDirection.Stable,
                PerformanceTrend = TrendDirection.Stable
            };
        }
    }

    /// <inheritdoc/>
    public async Task<List<QualityImprovement>> SuggestQualityImprovementsAsync(CodeQualityResult qualityResult)
    {
        try
        {
            _logger.LogInformation("Suggesting quality improvements");

            // Create a metascript for suggesting quality improvements
            var metascript = $@"
// Quality improvement suggestion metascript

// Quality result
let qualityResult = {System.Text.Json.JsonSerializer.Serialize(qualityResult)};

// Suggest quality improvements
let improvements = suggestQualityImprovements(qualityResult);

// Return the suggested improvements
return JSON.stringify(improvements);

// Helper function to suggest quality improvements
function suggestQualityImprovements(qualityResult) {{
    // This would be implemented with a more sophisticated analysis
    // For now, we'll use a simple placeholder
    
    const improvements = [];
    
    // Suggest improvements based on issues
    for (const issue of qualityResult.issues || []) {{
        improvements.push({{
            description: `Fix issue: ${{issue.description}}`,
            category: issue.category || 'General',
            priority: issue.severity === 'Critical' ? 'Critical' : (issue.severity === 'Error' ? 'High' : (issue.severity === 'Warning' ? 'Medium' : 'Low')),
            estimatedEffort: 1.0,
            estimatedScoreImpact: Math.abs(issue.scoreImpact || 1.0),
            affectedFiles: [issue.location ? issue.location.split(':')[0] : ''],
            implementationSteps: [
                issue.suggestedFix || `Implement a fix for: ${{issue.description}}`
            ]
        }});
    }}
    
    // Suggest improvements based on complexity metrics
    if (qualityResult.complexityMetrics && qualityResult.complexityMetrics.complexMethods && qualityResult.complexityMetrics.complexMethods.length > 0) {{
        improvements.push({{
            description: 'Reduce complexity in complex methods',
            category: 'Complexity',
            priority: 'High',
            estimatedEffort: 4.0,
            estimatedScoreImpact: 5.0,
            affectedFiles: qualityResult.complexityMetrics.complexMethods.map(m => m.filePath),
            implementationSteps: [
                'Identify the most complex methods',
                'Extract parts of the methods into smaller methods',
                'Simplify conditional logic',
                'Remove duplication'
            ]
        }});
    }}
    
    // Suggest improvements based on readability metrics
    if (qualityResult.readabilityMetrics && qualityResult.readabilityMetrics.documentationCoverage < 0.8) {{
        improvements.push({{
            description: 'Improve documentation coverage',
            category: 'Readability',
            priority: 'Medium',
            estimatedEffort: 3.0,
            estimatedScoreImpact: 3.0,
            affectedFiles: [],
            implementationSteps: [
                'Add XML documentation to public methods and classes',
                'Add comments to complex code sections',
                'Improve existing documentation for clarity'
            ]
        }});
    }}
    
    // Suggest improvements based on duplication metrics
    if (qualityResult.duplicationMetrics && qualityResult.duplicationMetrics.duplicationPercentage > 0.03) {{
        improvements.push({{
            description: 'Reduce code duplication',
            category: 'Duplication',
            priority: 'Medium',
            estimatedEffort: 2.0,
            estimatedScoreImpact: 2.0,
            affectedFiles: qualityResult.duplicationMetrics.duplicatedBlocksList ? qualityResult.duplicationMetrics.duplicatedBlocksList.map(b => b.filePath) : [],
            implementationSteps: [
                'Identify duplicated code blocks',
                'Extract duplicated code into reusable methods',
                'Create utility classes for common functionality'
            ]
        }});
    }}
    
    return improvements;
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            
            // Parse the result as JSON
            var improvements = System.Text.Json.JsonSerializer.Deserialize<List<QualityImprovement>>(result.ToString());
            return improvements ?? [];
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error suggesting quality improvements");
            return [];
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
            return [];
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
            return [];
        }
    }

    private CodeQualityResult CalculateOverallQuality(List<CodeQualityResult> qualityResults)
    {
        if (qualityResults.Count == 0)
        {
            return new CodeQualityResult();
        }

        // Calculate average scores
        var overallScore = qualityResults.Average(r => r.OverallScore);
        var maintainabilityScore = qualityResults.Average(r => r.MaintainabilityScore);
        var reliabilityScore = qualityResults.Average(r => r.ReliabilityScore);
        var securityScore = qualityResults.Average(r => r.SecurityScore);
        var performanceScore = qualityResults.Average(r => r.PerformanceScore);

        // Combine issues
        var issues = qualityResults.SelectMany(r => r.Issues).ToList();

        // Combine complexity metrics
        var complexMethods = qualityResults.SelectMany(r => r.ComplexityMetrics?.ComplexMethods ?? []).ToList();
        var averageCyclomaticComplexity = qualityResults.Average(r => r.ComplexityMetrics?.AverageCyclomaticComplexity ?? 0);
        var maxCyclomaticComplexity = qualityResults.Max(r => r.ComplexityMetrics?.MaxCyclomaticComplexity ?? 0);
        var averageCognitiveComplexity = qualityResults.Average(r => r.ComplexityMetrics?.AverageCognitiveComplexity ?? 0);
        var maxCognitiveComplexity = qualityResults.Max(r => r.ComplexityMetrics?.MaxCognitiveComplexity ?? 0);
        var averageMethodLength = qualityResults.Average(r => r.ComplexityMetrics?.AverageMethodLength ?? 0);
        var maxMethodLength = qualityResults.Max(r => r.ComplexityMetrics?.MaxMethodLength ?? 0);
        var averageClassLength = qualityResults.Average(r => r.ComplexityMetrics?.AverageClassLength ?? 0);
        var maxClassLength = qualityResults.Max(r => r.ComplexityMetrics?.MaxClassLength ?? 0);

        // Combine readability metrics
        var readabilityIssues = qualityResults.SelectMany(r => r.ReadabilityMetrics?.ReadabilityIssues ?? []).ToList();
        var averageIdentifierLength = qualityResults.Average(r => r.ReadabilityMetrics?.AverageIdentifierLength ?? 0);
        var commentDensity = qualityResults.Average(r => r.ReadabilityMetrics?.CommentDensity ?? 0);
        var documentationCoverage = qualityResults.Average(r => r.ReadabilityMetrics?.DocumentationCoverage ?? 0);
        var averageParameterCount = qualityResults.Average(r => r.ReadabilityMetrics?.AverageParameterCount ?? 0);
        var maxParameterCount = qualityResults.Max(r => r.ReadabilityMetrics?.MaxParameterCount ?? 0);

        // Combine duplication metrics
        var duplicatedBlocksList = qualityResults.SelectMany(r => r.DuplicationMetrics?.DuplicatedBlocksList ?? []).ToList();
        var duplicationPercentage = qualityResults.Average(r => r.DuplicationMetrics?.DuplicationPercentage ?? 0);
        var duplicatedBlocks = qualityResults.Sum(r => r.DuplicationMetrics?.DuplicatedBlocks ?? 0);
        var duplicatedLines = qualityResults.Sum(r => r.DuplicationMetrics?.DuplicatedLines ?? 0);

        return new CodeQualityResult
        {
            OverallScore = overallScore,
            MaintainabilityScore = maintainabilityScore,
            ReliabilityScore = reliabilityScore,
            SecurityScore = securityScore,
            PerformanceScore = performanceScore,
            Issues = issues,
            ComplexityMetrics = new ComplexityMetrics
            {
                AverageCyclomaticComplexity = averageCyclomaticComplexity,
                MaxCyclomaticComplexity = maxCyclomaticComplexity,
                AverageCognitiveComplexity = averageCognitiveComplexity,
                MaxCognitiveComplexity = maxCognitiveComplexity,
                AverageMethodLength = averageMethodLength,
                MaxMethodLength = maxMethodLength,
                AverageClassLength = averageClassLength,
                MaxClassLength = maxClassLength,
                ComplexMethods = complexMethods
            },
            ReadabilityMetrics = new ReadabilityMetrics
            {
                AverageIdentifierLength = averageIdentifierLength,
                CommentDensity = commentDensity,
                DocumentationCoverage = documentationCoverage,
                AverageParameterCount = averageParameterCount,
                MaxParameterCount = maxParameterCount,
                ReadabilityIssues = readabilityIssues
            },
            DuplicationMetrics = new DuplicationMetrics
            {
                DuplicationPercentage = duplicationPercentage,
                DuplicatedBlocks = duplicatedBlocks,
                DuplicatedLines = duplicatedLines,
                DuplicatedBlocksList = duplicatedBlocksList
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
