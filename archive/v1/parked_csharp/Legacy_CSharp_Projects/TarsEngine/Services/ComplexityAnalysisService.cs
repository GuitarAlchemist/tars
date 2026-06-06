using System.Text.Json;
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
    
    return {{
        averageCyclomaticComplexity,
        maxCyclomaticComplexity,
        averageCognitiveComplexity,
        maxCognitiveComplexity,
        averageHalsteadComplexity,
        maxHalsteadComplexity,
        averageMaintainabilityIndex,
        minMaintainabilityIndex,
        complexMethods: [],
        complexClasses: [],
        complexityDistribution: {{
            cyclomaticComplexityDistribution: [],
            cognitiveComplexityDistribution: [],
            maintainabilityIndexDistribution: [],
            methodLengthDistribution: [],
            classLengthDistribution: []
        }}
    }};
}}";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            
            if (result == null)
            {
                _logger.LogWarning("Metascript execution returned null result for file: {FilePath}", filePath);
                return CreateEmptyComplexityResult();
            }

            var resultString = result.ToString();
            if (string.IsNullOrEmpty(resultString))
            {
                _logger.LogWarning("Metascript execution returned empty result for file: {FilePath}", filePath);
                return CreateEmptyComplexityResult();
            }

            try
            {
                // Parse the result as JSON
                var complexityResult = JsonSerializer.Deserialize<ComplexityAnalysisResult>(
                    resultString,
                    new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });

                return complexityResult ?? CreateEmptyComplexityResult();
            }
            catch (JsonException ex)
            {
                _logger.LogError(ex, "Error deserializing complexity result for file: {FilePath}", filePath);
                return CreateEmptyComplexityResult();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing complexity for file: {FilePath}", filePath);
            return CreateEmptyComplexityResult();
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
return JSON.stringify(complexSections);";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            
            if (result == null)
            {
                _logger.LogWarning("Metascript execution returned null result for file: {FilePath}", filePath);
                return new List<ComplexCodeSection>();
            }

            var resultString = result.ToString();
            if (string.IsNullOrEmpty(resultString))
            {
                _logger.LogWarning("Metascript execution returned empty result for file: {FilePath}", filePath);
                return new List<ComplexCodeSection>();
            }

            // Parse the result as JSON
            var complexSections = JsonSerializer.Deserialize<List<ComplexCodeSection>>(
                resultString,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true }
            );
            
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
let complexSection = {JsonSerializer.Serialize(complexCodeSection)};

// Suggest simplifications
let simplifications = suggestSimplifications(complexSection);

// Return the suggested simplifications
return JSON.stringify(simplifications);";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            
            if (result == null)
            {
                _logger.LogWarning("Metascript execution returned null result for code section");
                return new List<CodeSimplification>();
            }

            var resultString = result.ToString();
            if (string.IsNullOrEmpty(resultString))
            {
                _logger.LogWarning("Metascript execution returned empty result for code section");
                return new List<CodeSimplification>();
            }

            // Parse the result as JSON
            var simplifications = JsonSerializer.Deserialize<List<CodeSimplification>>(
                resultString,
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true }
            );
            
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

    /// <summary>
    /// Creates an empty complexity analysis result
    /// </summary>
    private static ComplexityAnalysisResult CreateEmptyComplexityResult()
    {
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
            ComplexMethods = [],
            ComplexClasses = [],
            ComplexityDistribution = new ComplexityDistribution
            {
                CyclomaticComplexityDistribution = [],
                CognitiveComplexityDistribution = [],
                MaintainabilityIndexDistribution = [],
                MethodLengthDistribution = [],
                ClassLengthDistribution = []
            }
        };
    }
}
