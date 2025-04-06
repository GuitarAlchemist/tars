using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for analyzing code complexity
/// </summary>
public class ComplexityAnalysisService : IComplexityAnalysisService
{
    private readonly ILogger<ComplexityAnalysisService> _logger;
    private readonly IMetascriptService _metascriptService;

    /// <summary>
    /// Initializes a new instance of the <see cref="ComplexityAnalysisService"/> class
    /// </summary>
    public ComplexityAnalysisService(ILogger<ComplexityAnalysisService> logger, IMetascriptService metascriptService)
    {
        _logger = logger;
        _metascriptService = metascriptService;
    }

    /// <inheritdoc/>
    public async Task<ComplexityAnalysisResult> AnalyzeComplexityAsync(string filePath, string language)
    {
        try
        {
            _logger.LogInformation("Analyzing complexity for file: {FilePath}", filePath);

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            var fileContent = await File.ReadAllTextAsync(filePath);

            // Create a metascript for complexity analysis
            var metascript = $@"
// Complexity analysis metascript
// Language: {language}

// Read the source file
let sourceCode = `{fileContent.Replace("`", "\\`")}`;

// Analyze complexity
let complexityResult = analyzeComplexity(sourceCode, '{language}');

// Return the complexity result
return JSON.stringify(complexityResult);

// Helper function to analyze complexity
function analyzeComplexity(code, language) {{
    // This would be implemented with a more sophisticated analysis
    // For now, we'll use a simple placeholder
    
    // Calculate complexity metrics
    const averageCyclomaticComplexity = 3.5;
    const maxCyclomaticComplexity = 8;
    const averageCognitiveComplexity = 2.8;
    const maxCognitiveComplexity = 6;
    const averageHalsteadComplexity = 25.3;
    const maxHalsteadComplexity = 45.7;
    const averageMaintainabilityIndex = 65.2;
    const minMaintainabilityIndex = 48.5;
    
    // Identify complex methods
    const complexMethods = [
        {{
            methodName: 'ProcessData',
            className: 'DataProcessor',
            filePath: 'DataProcessor.cs',
            lineNumber: 42,
            cyclomaticComplexity: 8,
            cognitiveComplexity: 6,
            methodLength: 42
        }},
        {{
            methodName: 'ValidateInput',
            className: 'InputValidator',
            filePath: 'InputValidator.cs',
            lineNumber: 25,
            cyclomaticComplexity: 6,
            cognitiveComplexity: 4,
            methodLength: 30
        }}
    ];
    
    // Identify complex classes
    const complexClasses = [
        {{
            className: 'DataProcessor',
            filePath: 'DataProcessor.cs',
            lineNumber: 10,
            classLength: 250,
            methodCount: 15,
            propertyCount: 8,
            fieldCount: 5,
            weightedMethodCount: 45,
            inheritanceDepth: 2,
            childrenCount: 0
        }}
    ];
    
    // Calculate complexity distribution
    const cyclomaticComplexityDistribution = {{
        '1-5': 20,
        '6-10': 5,
        '11-15': 2,
        '16-20': 1,
        '21+': 0
    }};
    
    const cognitiveComplexityDistribution = {{
        '1-5': 22,
        '6-10': 4,
        '11-15': 1,
        '16-20': 1,
        '21+': 0
    }};
    
    const maintainabilityIndexDistribution = {{
        '0-19': 0,
        '20-39': 2,
        '40-59': 5,
        '60-79': 15,
        '80-100': 6
    }};
    
    const methodLengthDistribution = {{
        '1-10': 15,
        '11-20': 8,
        '21-30': 3,
        '31-40': 1,
        '41+': 1
    }};
    
    const classLengthDistribution = {{
        '1-100': 3,
        '101-200': 1,
        '201-300': 1,
        '301-400': 0,
        '401+': 0
    }};
    
    return {{
        averageCyclomaticComplexity,
        maxCyclomaticComplexity,
        averageCognitiveComplexity,
        maxCognitiveComplexity,
        averageHalsteadComplexity,
        maxHalsteadComplexity,
        averageMaintainabilityIndex,
        minMaintainabilityIndex,
        complexMethods,
        complexClasses,
        complexityDistribution: {{
            cyclomaticComplexityDistribution,
            cognitiveComplexityDistribution,
            maintainabilityIndexDistribution,
            methodLengthDistribution,
            classLengthDistribution
        }}
    }};
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            
            // Parse the result as JSON
            var complexityResult = System.Text.Json.JsonSerializer.Deserialize<ComplexityAnalysisResult>(result.ToString());
            return complexityResult ?? new ComplexityAnalysisResult();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing complexity for file: {FilePath}", filePath);
            return new ComplexityAnalysisResult
            {
                AverageCyclomaticComplexity = 0,
                MaxCyclomaticComplexity = 0,
                AverageCognitiveComplexity = 0,
                MaxCognitiveComplexity = 0,
                AverageHalsteadComplexity = 0,
                MaxHalsteadComplexity = 0,
                AverageMaintainabilityIndex = 0,
                MinMaintainabilityIndex = 0,
                ComplexMethods = new List<ComplexMethod>(),
                ComplexClasses = new List<ComplexClass>(),
                ComplexityDistribution = new ComplexityDistribution()
            };
        }
    }

    /// <inheritdoc/>
    public async Task<ComplexityAnalysisResult> AnalyzeProjectComplexityAsync(string projectPath)
    {
        try
        {
            _logger.LogInformation("Analyzing complexity for project: {ProjectPath}", projectPath);

            if (!Directory.Exists(projectPath) && !File.Exists(projectPath))
            {
                throw new DirectoryNotFoundException($"Project not found: {projectPath}");
            }

            // Find all source files in the project
            var sourceFiles = FindSourceFiles(projectPath);
            if (sourceFiles.Count == 0)
            {
                _logger.LogWarning("No source files found in project: {ProjectPath}", projectPath);
                return new ComplexityAnalysisResult
                {
                    AverageCyclomaticComplexity = 0,
                    MaxCyclomaticComplexity = 0,
                    AverageCognitiveComplexity = 0,
                    MaxCognitiveComplexity = 0,
                    AverageHalsteadComplexity = 0,
                    MaxHalsteadComplexity = 0,
                    AverageMaintainabilityIndex = 0,
                    MinMaintainabilityIndex = 0,
                    ComplexMethods = new List<ComplexMethod>(),
                    ComplexClasses = new List<ComplexClass>(),
                    ComplexityDistribution = new ComplexityDistribution()
                };
            }

            // Analyze each source file
            var complexityResults = new List<ComplexityAnalysisResult>();
            foreach (var sourceFile in sourceFiles)
            {
                var fileExtension = Path.GetExtension(sourceFile).ToLowerInvariant();
                var language = GetLanguageFromExtension(fileExtension);
                var complexityResult = await AnalyzeComplexityAsync(sourceFile, language);
                complexityResults.Add(complexityResult);
            }

            // Calculate overall complexity
            return CalculateOverallComplexity(complexityResults);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing complexity for project: {ProjectPath}", projectPath);
            return new ComplexityAnalysisResult
            {
                AverageCyclomaticComplexity = 0,
                MaxCyclomaticComplexity = 0,
                AverageCognitiveComplexity = 0,
                MaxCognitiveComplexity = 0,
                AverageHalsteadComplexity = 0,
                MaxHalsteadComplexity = 0,
                AverageMaintainabilityIndex = 0,
                MinMaintainabilityIndex = 0,
                ComplexMethods = new List<ComplexMethod>(),
                ComplexClasses = new List<ComplexClass>(),
                ComplexityDistribution = new ComplexityDistribution()
            };
        }
    }

    /// <inheritdoc/>
    public async Task<ComplexityAnalysisResult> AnalyzeSolutionComplexityAsync(string solutionPath)
    {
        try
        {
            _logger.LogInformation("Analyzing complexity for solution: {SolutionPath}", solutionPath);

            if (!File.Exists(solutionPath))
            {
                throw new FileNotFoundException($"Solution file not found: {solutionPath}");
            }

            // Find all projects in the solution
            var projectPaths = FindProjectsInSolution(solutionPath);
            if (projectPaths.Count == 0)
            {
                _logger.LogWarning("No projects found in solution: {SolutionPath}", solutionPath);
                return new ComplexityAnalysisResult
                {
                    AverageCyclomaticComplexity = 0,
                    MaxCyclomaticComplexity = 0,
                    AverageCognitiveComplexity = 0,
                    MaxCognitiveComplexity = 0,
                    AverageHalsteadComplexity = 0,
                    MaxHalsteadComplexity = 0,
                    AverageMaintainabilityIndex = 0,
                    MinMaintainabilityIndex = 0,
                    ComplexMethods = new List<ComplexMethod>(),
                    ComplexClasses = new List<ComplexClass>(),
                    ComplexityDistribution = new ComplexityDistribution()
                };
            }

            // Analyze each project
            var complexityResults = new List<ComplexityAnalysisResult>();
            foreach (var projectPath in projectPaths)
            {
                var complexityResult = await AnalyzeProjectComplexityAsync(projectPath);
                complexityResults.Add(complexityResult);
            }

            // Calculate overall complexity
            return CalculateOverallComplexity(complexityResults);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing complexity for solution: {SolutionPath}", solutionPath);
            return new ComplexityAnalysisResult
            {
                AverageCyclomaticComplexity = 0,
                MaxCyclomaticComplexity = 0,
                AverageCognitiveComplexity = 0,
                MaxCognitiveComplexity = 0,
                AverageHalsteadComplexity = 0,
                MaxHalsteadComplexity = 0,
                AverageMaintainabilityIndex = 0,
                MinMaintainabilityIndex = 0,
                ComplexMethods = new List<ComplexMethod>(),
                ComplexClasses = new List<ComplexClass>(),
                ComplexityDistribution = new ComplexityDistribution()
            };
        }
    }

    /// <inheritdoc/>
    public async Task<List<ComplexCodeSection>> IdentifyComplexCodeAsync(string filePath, string language, int threshold = 10)
    {
        try
        {
            _logger.LogInformation("Identifying complex code in file: {FilePath}", filePath);

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            var fileContent = await File.ReadAllTextAsync(filePath);

            // Create a metascript for identifying complex code
            var metascript = $@"
// Complex code identification metascript
// Language: {language}
// Threshold: {threshold}

// Read the source file
let sourceCode = `{fileContent.Replace("`", "\\`")}`;

// Identify complex code
let complexSections = identifyComplexCode(sourceCode, '{language}', {threshold});

// Return the complex sections
return JSON.stringify(complexSections);

// Helper function to identify complex code
function identifyComplexCode(code, language, threshold) {{
    // This would be implemented with a more sophisticated analysis
    // For now, we'll use a simple placeholder
    
    const complexSections = [
        {{
            filePath: 'DataProcessor.cs',
            startLine: 42,
            endLine: 84,
            content: '// Complex code section',
            complexityType: 'Cyclomatic',
            complexityValue: 12,
            methodName: 'ProcessData',
            className: 'DataProcessor'
        }},
        {{
            filePath: 'InputValidator.cs',
            startLine: 25,
            endLine: 55,
            content: '// Complex code section',
            complexityType: 'Cognitive',
            complexityValue: 8,
            methodName: 'ValidateInput',
            className: 'InputValidator'
        }}
    ];
    
    return complexSections;
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            
            // Parse the result as JSON
            var complexSections = System.Text.Json.JsonSerializer.Deserialize<List<ComplexCodeSection>>(result.ToString());
            return complexSections ?? new List<ComplexCodeSection>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error identifying complex code in file: {FilePath}", filePath);
            return new List<ComplexCodeSection>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<CodeSimplification>> SuggestSimplificationsAsync(ComplexCodeSection complexCodeSection)
    {
        try
        {
            _logger.LogInformation("Suggesting simplifications for complex code section");

            // Create a metascript for suggesting simplifications
            var metascript = $@"
// Simplification suggestion metascript

// Complex code section
let complexSection = {System.Text.Json.JsonSerializer.Serialize(complexCodeSection)};

// Suggest simplifications
let simplifications = suggestSimplifications(complexSection);

// Return the suggested simplifications
return JSON.stringify(simplifications);

// Helper function to suggest simplifications
function suggestSimplifications(complexSection) {{
    // This would be implemented with a more sophisticated analysis
    // For now, we'll use a simple placeholder
    
    const simplifications = [
        {{
            description: 'Extract method for nested conditional',
            simplifiedCode: '// Simplified code',
            complexityReduction: 3,
            confidence: 0.8,
            potentialRisks: [
                'May affect performance slightly'
            ]
        }},
        {{
            description: 'Use LINQ instead of loops',
            simplifiedCode: '// Simplified code using LINQ',
            complexityReduction: 2,
            confidence: 0.7,
            potentialRisks: [
                'May be less readable for developers unfamiliar with LINQ'
            ]
        }}
    ];
    
    return simplifications;
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            
            // Parse the result as JSON
            var simplifications = System.Text.Json.JsonSerializer.Deserialize<List<CodeSimplification>>(result.ToString());
            return simplifications ?? new List<CodeSimplification>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error suggesting simplifications for complex code section");
            return new List<CodeSimplification>();
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

    private ComplexityAnalysisResult CalculateOverallComplexity(List<ComplexityAnalysisResult> complexityResults)
    {
        if (complexityResults.Count == 0)
        {
            return new ComplexityAnalysisResult();
        }

        // Calculate average metrics
        var averageCyclomaticComplexity = complexityResults.Average(r => r.AverageCyclomaticComplexity);
        var maxCyclomaticComplexity = complexityResults.Max(r => r.MaxCyclomaticComplexity);
        var averageCognitiveComplexity = complexityResults.Average(r => r.AverageCognitiveComplexity);
        var maxCognitiveComplexity = complexityResults.Max(r => r.MaxCognitiveComplexity);
        var averageHalsteadComplexity = complexityResults.Average(r => r.AverageHalsteadComplexity);
        var maxHalsteadComplexity = complexityResults.Max(r => r.MaxHalsteadComplexity);
        var averageMaintainabilityIndex = complexityResults.Average(r => r.AverageMaintainabilityIndex);
        var minMaintainabilityIndex = complexityResults.Min(r => r.MinMaintainabilityIndex);

        // Combine complex methods and classes
        var complexMethods = complexityResults.SelectMany(r => r.ComplexMethods).ToList();
        var complexClasses = complexityResults.SelectMany(r => r.ComplexClasses).ToList();

        // Create a new complexity distribution
        var complexityDistribution = new ComplexityDistribution();

        return new ComplexityAnalysisResult
        {
            AverageCyclomaticComplexity = averageCyclomaticComplexity,
            MaxCyclomaticComplexity = maxCyclomaticComplexity,
            AverageCognitiveComplexity = averageCognitiveComplexity,
            MaxCognitiveComplexity = maxCognitiveComplexity,
            AverageHalsteadComplexity = averageHalsteadComplexity,
            MaxHalsteadComplexity = maxHalsteadComplexity,
            AverageMaintainabilityIndex = averageMaintainabilityIndex,
            MinMaintainabilityIndex = minMaintainabilityIndex,
            ComplexMethods = complexMethods,
            ComplexClasses = complexClasses,
            ComplexityDistribution = complexityDistribution
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
