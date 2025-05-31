using System.Text.RegularExpressions;
using DuplicationAnalyzerTests.Models;
using DuplicationAnalyzerTests.Services.Interfaces;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.Extensions.Logging;

namespace DuplicationAnalyzerTests.Services;

/// <summary>
/// Analyzer for C# code duplication
/// </summary>
public class CSharpDuplicationAnalyzer : IDuplicationAnalyzer
{
    private readonly ILogger<CSharpDuplicationAnalyzer> _logger;
    private readonly Dictionary<string, Dictionary<DuplicationType, Dictionary<string, double>>> _thresholds;

    // Minimum sequence length to consider as duplication (in tokens)
    private const int MinimumDuplicateTokens = 100;

    // Minimum sequence length to consider as duplication (in lines)
    private const int MinimumDuplicateLines = 5;

    /// <summary>
    /// Initializes a new instance of the <see cref="CSharpDuplicationAnalyzer"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    public CSharpDuplicationAnalyzer(ILogger<CSharpDuplicationAnalyzer> logger)
    {
        _logger = logger;
        _thresholds = new Dictionary<string, Dictionary<DuplicationType, Dictionary<string, double>>>
        {
            ["C#"] = new Dictionary<DuplicationType, Dictionary<string, double>>
            {
                [DuplicationType.TokenBased] = new Dictionary<string, double>
                {
                    ["Method"] = 5, // 5% duplication threshold for methods
                    ["Class"] = 10, // 10% duplication threshold for classes
                    ["File"] = 15   // 15% duplication threshold for files
                },
                [DuplicationType.Semantic] = new Dictionary<string, double>
                {
                    ["Method"] = 10, // 10% semantic duplication threshold for methods
                    ["Class"] = 15,  // 15% semantic duplication threshold for classes
                    ["File"] = 20    // 20% semantic duplication threshold for files
                },
                [DuplicationType.Overall] = new Dictionary<string, double>
                {
                    ["Method"] = 7,  // 7% overall duplication threshold for methods
                    ["Class"] = 12,  // 12% overall duplication threshold for classes
                    ["File"] = 17    // 17% overall duplication threshold for files
                }
            }
        };
    }

    /// <inheritdoc/>
    public async Task<List<DuplicationMetric>> AnalyzeTokenBasedDuplicationAsync(string filePath, string language)
    {
        if (language != "C#")
        {
            _logger.LogWarning("Language {Language} not supported by CSharpDuplicationAnalyzer", language);
            return new List<DuplicationMetric>();
        }

        try
        {
            _logger.LogInformation("Analyzing token-based duplication for file {FilePath}", filePath);

            // Read the file content
            var sourceCode = await File.ReadAllTextAsync(filePath);

            // Create the syntax tree
            var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
            var root = await syntaxTree.GetRootAsync();

            // Create the result list
            var metrics = new List<DuplicationMetric>();

            // Analyze file-level duplication
            var fileMetric = await AnalyzeFileDuplicationAsync(filePath, root);
            metrics.Add(fileMetric);

            // Analyze class-level duplication
            var classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
            foreach (var classDecl in classDeclarations)
            {
                var className = classDecl.Identifier.Text;
                var namespaceName = GetNamespace(classDecl);
                var fullClassName = string.IsNullOrEmpty(namespaceName) ? className : $"{namespaceName}.{className}";

                var classMetric = await AnalyzeClassDuplicationAsync(filePath, classDecl, fullClassName);
                metrics.Add(classMetric);

                // Analyze method-level duplication
                var methodDeclarations = classDecl.DescendantNodes().OfType<MethodDeclarationSyntax>();
                foreach (var methodDecl in methodDeclarations)
                {
                    var methodName = methodDecl.Identifier.Text;
                    var fullMethodName = $"{fullClassName}.{methodName}";

                    var methodMetric = await AnalyzeMethodDuplicationAsync(filePath, methodDecl, fullMethodName);
                    metrics.Add(methodMetric);
                }
            }

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing token-based duplication for file {FilePath}", filePath);
            return new List<DuplicationMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<DuplicationMetric>> AnalyzeSemanticDuplicationAsync(string filePath, string language)
    {
        if (language != "C#")
        {
            _logger.LogWarning("Language {Language} not supported by CSharpDuplicationAnalyzer", language);
            return new List<DuplicationMetric>();
        }

        try
        {
            _logger.LogInformation("Analyzing semantic duplication for file {FilePath}", filePath);

            // Read the file content
            var sourceCode = await File.ReadAllTextAsync(filePath);

            // Create the syntax tree
            var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
            var root = await syntaxTree.GetRootAsync();

            // Create the result list
            var metrics = new List<DuplicationMetric>();

            // Analyze file-level semantic duplication
            var fileMetric = await AnalyzeFileSemanticDuplicationAsync(filePath, root);
            metrics.Add(fileMetric);

            // Analyze class-level semantic duplication
            var classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
            foreach (var classDecl in classDeclarations)
            {
                var className = classDecl.Identifier.Text;
                var namespaceName = GetNamespace(classDecl);
                var fullClassName = string.IsNullOrEmpty(namespaceName) ? className : $"{namespaceName}.{className}";

                var classMetric = await AnalyzeClassSemanticDuplicationAsync(filePath, classDecl, fullClassName);
                metrics.Add(classMetric);

                // Analyze method-level semantic duplication
                var methodDeclarations = classDecl.DescendantNodes().OfType<MethodDeclarationSyntax>();
                foreach (var methodDecl in methodDeclarations)
                {
                    var methodName = methodDecl.Identifier.Text;
                    var fullMethodName = $"{fullClassName}.{methodName}";

                    var methodMetric = await AnalyzeMethodSemanticDuplicationAsync(filePath, methodDecl, fullMethodName);
                    metrics.Add(methodMetric);
                }
            }

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing semantic duplication for file {FilePath}", filePath);
            return new List<DuplicationMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<DuplicationMetric>> AnalyzeAllDuplicationMetricsAsync(string filePath, string language)
    {
        var metrics = new List<DuplicationMetric>();

        // Get token-based duplication metrics
        metrics.AddRange(await AnalyzeTokenBasedDuplicationAsync(filePath, language));

        // Get semantic duplication metrics
        metrics.AddRange(await AnalyzeSemanticDuplicationAsync(filePath, language));

        return metrics;
    }

    /// <inheritdoc/>
    public async Task<List<DuplicationMetric>> AnalyzeProjectDuplicationAsync(string projectPath)
    {
        try
        {
            _logger.LogInformation("Analyzing duplication for project {ProjectPath}", projectPath);

            var metrics = new List<DuplicationMetric>();

            // Get all C# files in the project
            var csharpFiles = Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories);

            // Analyze each file
            foreach (var file in csharpFiles)
            {
                metrics.AddRange(await AnalyzeAllDuplicationMetricsAsync(file, "C#"));
            }

            // Calculate project-level duplication
            var projectName = Path.GetFileName(projectPath);
            var projectMetric = await CalculateProjectDuplicationAsync(projectPath, projectName, metrics);
            metrics.Add(projectMetric);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing duplication for project {ProjectPath}", projectPath);
            return new List<DuplicationMetric>();
        }
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, double>> GetDuplicationThresholdsAsync(string language, DuplicationType duplicationType)
    {
        if (_thresholds.TryGetValue(language, out var languageThresholds) &&
            languageThresholds.TryGetValue(duplicationType, out var typeThresholds))
        {
            return Task.FromResult(typeThresholds);
        }

        return Task.FromResult(new Dictionary<string, double>());
    }

    /// <inheritdoc/>
    public Task<bool> SetDuplicationThresholdAsync(string language, DuplicationType duplicationType, string targetType, double threshold)
    {
        try
        {
            if (!_thresholds.ContainsKey(language))
            {
                _thresholds[language] = new Dictionary<DuplicationType, Dictionary<string, double>>();
            }

            if (!_thresholds[language].ContainsKey(duplicationType))
            {
                _thresholds[language][duplicationType] = new Dictionary<string, double>();
            }

            _thresholds[language][duplicationType][targetType] = threshold;
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting duplication threshold for {Language}, {DuplicationType}, {TargetType}",
                language, duplicationType, targetType);
            return Task.FromResult(false);
        }
    }

    /// <inheritdoc/>
    public Task<bool> VisualizeDuplicationAsync(string path, string language, string outputPath)
    {
        // This is a simplified implementation for testing
        return Task.FromResult(true);
    }

    /// <summary>
    /// Analyzes file-level duplication
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="root">Syntax tree root</param>
    /// <returns>Duplication metric for the file</returns>
    private Task<DuplicationMetric> AnalyzeFileDuplicationAsync(string filePath, SyntaxNode root)
    {
        return Task.Run(() => {
            var fileName = Path.GetFileName(filePath);
            var sourceCode = root.ToFullString();
            var lines = sourceCode.Split('\n');

            // For testing purposes, we'll create a simple duplication metric
            var duplicatedLines = lines.Length > 10 ? 2 : 0;
            var duplicationPercentage = lines.Length > 0 ? (double)duplicatedLines / lines.Length * 100 : 0;

            // Create a duplication metric
            var metric = new DuplicationMetric
            {
                Name = $"Token-Based Duplication - {fileName}",
                Description = $"Token-based duplication for file {fileName}",
                Type = DuplicationType.TokenBased,
                FilePath = filePath,
                Language = "C#",
                Target = fileName,
                TargetType = TargetType.File,
                TotalLinesOfCode = lines.Length,
                DuplicatedLinesOfCode = duplicatedLines,
                DuplicationPercentage = duplicationPercentage,
                DuplicatedBlockCount = duplicatedLines > 0 ? 1 : 0,
                Timestamp = DateTime.UtcNow,
                ThresholdValue = GetThreshold("C#", DuplicationType.TokenBased, "File")
            };

            // Add a duplicated block if there are duplicated lines
            if (duplicatedLines > 0)
            {
                var duplicatedBlock = new DuplicatedBlock
                {
                    SourceFilePath = filePath,
                    SourceStartLine = 1,
                    SourceEndLine = duplicatedLines,
                    TargetFilePath = filePath,
                    TargetStartLine = lines.Length - duplicatedLines + 1,
                    TargetEndLine = lines.Length,
                    DuplicatedCode = string.Join("\n", lines.Take(duplicatedLines)),
                    SimilarityPercentage = 100.0
                };

                metric.DuplicatedBlocks.Add(duplicatedBlock);
            }

            // Set the value to the duplication percentage
            metric.Value = metric.DuplicationPercentage;

            return metric;
        });
    }

    /// <summary>
    /// Analyzes class-level duplication
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="classDecl">Class declaration syntax</param>
    /// <param name="fullClassName">Full class name</param>
    /// <returns>Duplication metric for the class</returns>
    private Task<DuplicationMetric> AnalyzeClassDuplicationAsync(string filePath, ClassDeclarationSyntax classDecl, string fullClassName)
    {
        return Task.Run(() => {
            var sourceCode = classDecl.ToFullString();
            var lines = sourceCode.Split('\n');

            // For testing purposes, we'll create a simple duplication metric
            var duplicatedLines = lines.Length > 20 ? 4 : 0;
            var duplicationPercentage = lines.Length > 0 ? (double)duplicatedLines / lines.Length * 100 : 0;

            // Create a duplication metric
            var metric = new DuplicationMetric
            {
                Name = $"Token-Based Duplication - {fullClassName}",
                Description = $"Token-based duplication for class {fullClassName}",
                Type = DuplicationType.TokenBased,
                FilePath = filePath,
                Language = "C#",
                Target = fullClassName,
                TargetType = TargetType.Class,
                TotalLinesOfCode = lines.Length,
                DuplicatedLinesOfCode = duplicatedLines,
                DuplicationPercentage = duplicationPercentage,
                DuplicatedBlockCount = duplicatedLines > 0 ? 1 : 0,
                Timestamp = DateTime.UtcNow,
                ThresholdValue = GetThreshold("C#", DuplicationType.TokenBased, "Class")
            };

            // Add a duplicated block if there are duplicated lines
            if (duplicatedLines > 0)
            {
                var duplicatedBlock = new DuplicatedBlock
                {
                    SourceFilePath = filePath,
                    SourceStartLine = 1,
                    SourceEndLine = duplicatedLines,
                    TargetFilePath = filePath,
                    TargetStartLine = lines.Length - duplicatedLines + 1,
                    TargetEndLine = lines.Length,
                    DuplicatedCode = string.Join("\n", lines.Take(duplicatedLines)),
                    SimilarityPercentage = 100.0
                };

                metric.DuplicatedBlocks.Add(duplicatedBlock);
            }

            // Set the value to the duplication percentage
            metric.Value = metric.DuplicationPercentage;

            return metric;
        });
    }

    /// <summary>
    /// Analyzes method-level duplication
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="methodDecl">Method declaration syntax</param>
    /// <param name="fullMethodName">Full method name</param>
    /// <returns>Duplication metric for the method</returns>
    private Task<DuplicationMetric> AnalyzeMethodDuplicationAsync(string filePath, MethodDeclarationSyntax methodDecl, string fullMethodName)
    {
        return Task.Run(() => {
            var sourceCode = methodDecl.ToFullString();
            var lines = sourceCode.Split('\n');

            // For testing purposes, we'll create a simple duplication metric
            var duplicatedLines = lines.Length > 5 ? 2 : 0;
            var duplicationPercentage = lines.Length > 0 ? (double)duplicatedLines / lines.Length * 100 : 0;

            // Create a duplication metric
            var metric = new DuplicationMetric
            {
                Name = $"Token-Based Duplication - {fullMethodName}",
                Description = $"Token-based duplication for method {fullMethodName}",
                Type = DuplicationType.TokenBased,
                FilePath = filePath,
                Language = "C#",
                Target = fullMethodName,
                TargetType = TargetType.Method,
                TotalLinesOfCode = lines.Length,
                DuplicatedLinesOfCode = duplicatedLines,
                DuplicationPercentage = duplicationPercentage,
                DuplicatedBlockCount = duplicatedLines > 0 ? 1 : 0,
                Timestamp = DateTime.UtcNow,
                ThresholdValue = GetThreshold("C#", DuplicationType.TokenBased, "Method")
            };

            // Add a duplicated block if there are duplicated lines
            if (duplicatedLines > 0)
            {
                var duplicatedBlock = new DuplicatedBlock
                {
                    SourceFilePath = filePath,
                    SourceStartLine = 1,
                    SourceEndLine = duplicatedLines,
                    TargetFilePath = filePath,
                    TargetStartLine = lines.Length - duplicatedLines + 1,
                    TargetEndLine = lines.Length,
                    DuplicatedCode = string.Join("\n", lines.Take(duplicatedLines)),
                    SimilarityPercentage = 100.0
                };

                metric.DuplicatedBlocks.Add(duplicatedBlock);
            }

            // Set the value to the duplication percentage
            metric.Value = metric.DuplicationPercentage;

            return metric;
        });
    }

    /// <summary>
    /// Analyzes file-level semantic duplication
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="root">Syntax tree root</param>
    /// <returns>Duplication metric for the file</returns>
    private Task<DuplicationMetric> AnalyzeFileSemanticDuplicationAsync(string filePath, SyntaxNode root)
    {
        return Task.Run(() => {
            var fileName = Path.GetFileName(filePath);
            var sourceCode = root.ToFullString();
            var lines = sourceCode.Split('\n');

            // For testing purposes, we'll create a simple duplication metric
            var duplicatedLines = lines.Length > 15 ? 3 : 0;
            var duplicationPercentage = lines.Length > 0 ? (double)duplicatedLines / lines.Length * 100 : 0;

            // Create a duplication metric
            var metric = new DuplicationMetric
            {
                Name = $"Semantic Duplication - {fileName}",
                Description = $"Semantic duplication for file {fileName}",
                Type = DuplicationType.Semantic,
                FilePath = filePath,
                Language = "C#",
                Target = fileName,
                TargetType = TargetType.File,
                TotalLinesOfCode = lines.Length,
                DuplicatedLinesOfCode = duplicatedLines,
                DuplicationPercentage = duplicationPercentage,
                DuplicatedBlockCount = duplicatedLines > 0 ? 1 : 0,
                Timestamp = DateTime.UtcNow,
                ThresholdValue = GetThreshold("C#", DuplicationType.Semantic, "File")
            };

            // Add a duplicated block if there are duplicated lines
            if (duplicatedLines > 0)
            {
                var duplicatedBlock = new DuplicatedBlock
                {
                    SourceFilePath = filePath,
                    SourceStartLine = 1,
                    SourceEndLine = duplicatedLines,
                    TargetFilePath = filePath,
                    TargetStartLine = lines.Length - duplicatedLines + 1,
                    TargetEndLine = lines.Length,
                    DuplicatedCode = string.Join("\n", lines.Take(duplicatedLines)),
                    SimilarityPercentage = 85.0
                };

                metric.DuplicatedBlocks.Add(duplicatedBlock);
            }

            // Set the value to the duplication percentage
            metric.Value = metric.DuplicationPercentage;

            return metric;
        });
    }

    /// <summary>
    /// Analyzes class-level semantic duplication
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="classDecl">Class declaration syntax</param>
    /// <param name="fullClassName">Full class name</param>
    /// <returns>Duplication metric for the class</returns>
    private Task<DuplicationMetric> AnalyzeClassSemanticDuplicationAsync(string filePath, ClassDeclarationSyntax classDecl, string fullClassName)
    {
        return Task.Run(() => {
            var sourceCode = classDecl.ToFullString();
            var lines = sourceCode.Split('\n');

            // For testing purposes, we'll create a simple duplication metric
            var duplicatedLines = lines.Length > 25 ? 5 : 0;
            var duplicationPercentage = lines.Length > 0 ? (double)duplicatedLines / lines.Length * 100 : 0;

            // Create a duplication metric
            var metric = new DuplicationMetric
            {
                Name = $"Semantic Duplication - {fullClassName}",
                Description = $"Semantic duplication for class {fullClassName}",
                Type = DuplicationType.Semantic,
                FilePath = filePath,
                Language = "C#",
                Target = fullClassName,
                TargetType = TargetType.Class,
                TotalLinesOfCode = lines.Length,
                DuplicatedLinesOfCode = duplicatedLines,
                DuplicationPercentage = duplicationPercentage,
                DuplicatedBlockCount = duplicatedLines > 0 ? 1 : 0,
                Timestamp = DateTime.UtcNow,
                ThresholdValue = GetThreshold("C#", DuplicationType.Semantic, "Class")
            };

            // Add a duplicated block if there are duplicated lines
            if (duplicatedLines > 0)
            {
                var duplicatedBlock = new DuplicatedBlock
                {
                    SourceFilePath = filePath,
                    SourceStartLine = 1,
                    SourceEndLine = duplicatedLines,
                    TargetFilePath = filePath,
                    TargetStartLine = lines.Length - duplicatedLines + 1,
                    TargetEndLine = lines.Length,
                    DuplicatedCode = string.Join("\n", lines.Take(duplicatedLines)),
                    SimilarityPercentage = 85.0
                };

                metric.DuplicatedBlocks.Add(duplicatedBlock);
            }

            // Set the value to the duplication percentage
            metric.Value = metric.DuplicationPercentage;

            return metric;
        });
    }

    /// <summary>
    /// Analyzes method-level semantic duplication
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="methodDecl">Method declaration syntax</param>
    /// <param name="fullMethodName">Full method name</param>
    /// <returns>Duplication metric for the method</returns>
    private Task<DuplicationMetric> AnalyzeMethodSemanticDuplicationAsync(string filePath, MethodDeclarationSyntax methodDecl, string fullMethodName)
    {
        return Task.Run(() => {
            var sourceCode = methodDecl.ToFullString();
            var lines = sourceCode.Split('\n');

            // For testing purposes, we'll create a simple duplication metric
            var duplicatedLines = lines.Length > 8 ? 3 : 0;
            var duplicationPercentage = lines.Length > 0 ? (double)duplicatedLines / lines.Length * 100 : 0;

            // Create a duplication metric
            var metric = new DuplicationMetric
            {
                Name = $"Semantic Duplication - {fullMethodName}",
                Description = $"Semantic duplication for method {fullMethodName}",
                Type = DuplicationType.Semantic,
                FilePath = filePath,
                Language = "C#",
                Target = fullMethodName,
                TargetType = TargetType.Method,
                TotalLinesOfCode = lines.Length,
                DuplicatedLinesOfCode = duplicatedLines,
                DuplicationPercentage = duplicationPercentage,
                DuplicatedBlockCount = duplicatedLines > 0 ? 1 : 0,
                Timestamp = DateTime.UtcNow,
                ThresholdValue = GetThreshold("C#", DuplicationType.Semantic, "Method")
            };

            // Add a duplicated block if there are duplicated lines
            if (duplicatedLines > 0)
            {
                var duplicatedBlock = new DuplicatedBlock
                {
                    SourceFilePath = filePath,
                    SourceStartLine = 1,
                    SourceEndLine = duplicatedLines,
                    TargetFilePath = filePath,
                    TargetStartLine = lines.Length - duplicatedLines + 1,
                    TargetEndLine = lines.Length,
                    DuplicatedCode = string.Join("\n", lines.Take(duplicatedLines)),
                    SimilarityPercentage = 85.0
                };

                metric.DuplicatedBlocks.Add(duplicatedBlock);
            }

            // Set the value to the duplication percentage
            metric.Value = metric.DuplicationPercentage;

            return metric;
        });
    }

    /// <summary>
    /// Calculates project-level duplication
    /// </summary>
    /// <param name="projectPath">Project path</param>
    /// <param name="projectName">Project name</param>
    /// <param name="fileMetrics">File metrics</param>
    /// <returns>Duplication metric for the project</returns>
    private Task<DuplicationMetric> CalculateProjectDuplicationAsync(string projectPath, string projectName, List<DuplicationMetric> fileMetrics)
    {
        return Task.Run(() => {
            // Get file-level metrics
            var fileLevelMetrics = fileMetrics.Where(m => m.TargetType == TargetType.File && m.Type == DuplicationType.TokenBased).ToList();

            // Calculate total lines of code
            var totalLinesOfCode = fileLevelMetrics.Sum(m => m.TotalLinesOfCode);

            // Calculate duplicated lines of code
            var duplicatedLinesOfCode = fileLevelMetrics.Sum(m => m.DuplicatedLinesOfCode);

            // Calculate duplication percentage
            var duplicationPercentage = totalLinesOfCode > 0 ? (double)duplicatedLinesOfCode / totalLinesOfCode * 100 : 0;

            // Create a duplication metric
            var metric = new DuplicationMetric
            {
                Name = $"Token-Based Duplication - {projectName}",
                Description = $"Token-based duplication for project {projectName}",
                Type = DuplicationType.TokenBased,
                FilePath = projectPath,
                Language = "C#",
                Target = projectName,
                TargetType = TargetType.Project,
                TotalLinesOfCode = totalLinesOfCode,
                DuplicatedLinesOfCode = duplicatedLinesOfCode,
                DuplicationPercentage = duplicationPercentage,
                DuplicatedBlockCount = fileLevelMetrics.Sum(m => m.DuplicatedBlockCount),
                Timestamp = DateTime.UtcNow,
                ThresholdValue = GetThreshold("C#", DuplicationType.TokenBased, "Project")
            };

            // Set the value to the duplication percentage
            metric.Value = metric.DuplicationPercentage;

            return metric;
        });
    }

    /// <summary>
    /// Gets the namespace for a class
    /// </summary>
    /// <param name="classDecl">Class declaration syntax</param>
    /// <returns>Namespace name</returns>
    private string GetNamespace(ClassDeclarationSyntax classDecl)
    {
        var namespaceDecl = classDecl.Ancestors().OfType<NamespaceDeclarationSyntax>().FirstOrDefault();
        return namespaceDecl?.Name.ToString() ?? string.Empty;
    }

    /// <summary>
    /// Gets the threshold value for a specific duplication type and target type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="duplicationType">Type of duplication</param>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <returns>Threshold value</returns>
    private double GetThreshold(string language, DuplicationType duplicationType, string targetType)
    {
        if (_thresholds.TryGetValue(language, out var languageThresholds) &&
            languageThresholds.TryGetValue(duplicationType, out var typeThresholds) &&
            typeThresholds.TryGetValue(targetType, out var threshold))
        {
            return threshold;
        }

        // Default thresholds if not configured
        return duplicationType switch
        {
            DuplicationType.TokenBased => targetType switch
            {
                "Method" => 5,
                "Class" => 10,
                "File" => 15,
                "Project" => 15,
                _ => 10
            },
            DuplicationType.Semantic => targetType switch
            {
                "Method" => 10,
                "Class" => 15,
                "File" => 20,
                "Project" => 20,
                _ => 15
            },
            DuplicationType.Overall => targetType switch
            {
                "Method" => 7,
                "Class" => 12,
                "File" => 17,
                "Project" => 17,
                _ => 12
            },
            _ => 10
        };
    }
}
