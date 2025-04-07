using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

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
        // This will be implemented in a future step
        _logger.LogInformation("Duplication visualization not yet implemented");
        return Task.FromResult(false);
    }

    /// <summary>
    /// Analyzes file-level duplication
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="root">Syntax tree root</param>
    /// <returns>Duplication metric for the file</returns>
    private async Task<DuplicationMetric> AnalyzeFileDuplicationAsync(string filePath, SyntaxNode root)
    {
        var fileName = Path.GetFileName(filePath);
        var sourceCode = root.ToFullString();
        var lines = sourceCode.Split('\n');

        // Create a duplication detector
        var detector = new TokenBasedDuplicationDetector(_logger, MinimumDuplicateTokens, MinimumDuplicateLines);

        // Detect duplicated blocks
        var duplicatedBlocks = detector.DetectDuplication(filePath, sourceCode);

        // Calculate duplicated lines of code
        var duplicatedLinesOfCode = duplicatedBlocks.Sum(b => b.DuplicatedLines);

        // Calculate duplication percentage
        var duplicationPercentage = lines.Length > 0 ? (double)duplicatedLinesOfCode / lines.Length * 100 : 0;

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
            DuplicatedLinesOfCode = duplicatedLinesOfCode,
            DuplicationPercentage = duplicationPercentage,
            DuplicatedBlockCount = duplicatedBlocks.Count,
            DuplicatedBlocks = duplicatedBlocks,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold("C#", DuplicationType.TokenBased, "File")
        };

        // Set the value to the duplication percentage
        metric.Value = metric.DuplicationPercentage;

        return metric;
    }

    /// <summary>
    /// Analyzes class-level duplication
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="classDecl">Class declaration syntax</param>
    /// <param name="fullClassName">Full class name</param>
    /// <returns>Duplication metric for the class</returns>
    private async Task<DuplicationMetric> AnalyzeClassDuplicationAsync(string filePath, ClassDeclarationSyntax classDecl, string fullClassName)
    {
        var sourceCode = classDecl.ToFullString();
        var lines = sourceCode.Split('\n');

        // Create a duplication detector
        var detector = new TokenBasedDuplicationDetector(_logger, MinimumDuplicateTokens, MinimumDuplicateLines);

        // Detect duplicated blocks
        var duplicatedBlocks = detector.DetectDuplication(filePath, sourceCode);

        // Calculate duplicated lines of code
        var duplicatedLinesOfCode = duplicatedBlocks.Sum(b => b.DuplicatedLines);

        // Calculate duplication percentage
        var duplicationPercentage = lines.Length > 0 ? (double)duplicatedLinesOfCode / lines.Length * 100 : 0;

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
            DuplicatedLinesOfCode = duplicatedLinesOfCode,
            DuplicationPercentage = duplicationPercentage,
            DuplicatedBlockCount = duplicatedBlocks.Count,
            DuplicatedBlocks = duplicatedBlocks,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold("C#", DuplicationType.TokenBased, "Class")
        };

        // Set the value to the duplication percentage
        metric.Value = metric.DuplicationPercentage;

        return metric;
    }

    /// <summary>
    /// Analyzes method-level duplication
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="methodDecl">Method declaration syntax</param>
    /// <param name="fullMethodName">Full method name</param>
    /// <returns>Duplication metric for the method</returns>
    private async Task<DuplicationMetric> AnalyzeMethodDuplicationAsync(string filePath, MethodDeclarationSyntax methodDecl, string fullMethodName)
    {
        var sourceCode = methodDecl.ToFullString();
        var lines = sourceCode.Split('\n');

        // Create a duplication detector
        var detector = new TokenBasedDuplicationDetector(_logger, MinimumDuplicateTokens, MinimumDuplicateLines);

        // Detect duplicated blocks
        var duplicatedBlocks = detector.DetectDuplication(filePath, sourceCode);

        // Calculate duplicated lines of code
        var duplicatedLinesOfCode = duplicatedBlocks.Sum(b => b.DuplicatedLines);

        // Calculate duplication percentage
        var duplicationPercentage = lines.Length > 0 ? (double)duplicatedLinesOfCode / lines.Length * 100 : 0;

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
            DuplicatedLinesOfCode = duplicatedLinesOfCode,
            DuplicationPercentage = duplicationPercentage,
            DuplicatedBlockCount = duplicatedBlocks.Count,
            DuplicatedBlocks = duplicatedBlocks,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold("C#", DuplicationType.TokenBased, "Method")
        };

        // Set the value to the duplication percentage
        metric.Value = metric.DuplicationPercentage;

        return metric;
    }

    /// <summary>
    /// Calculates project-level duplication
    /// </summary>
    /// <param name="projectPath">Project path</param>
    /// <param name="projectName">Project name</param>
    /// <param name="fileMetrics">File metrics</param>
    /// <returns>Duplication metric for the project</returns>
    private async Task<DuplicationMetric> CalculateProjectDuplicationAsync(string projectPath, string projectName, List<DuplicationMetric> fileMetrics)
    {
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
    }

    /// <summary>
    /// Analyzes file-level semantic duplication
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="root">Syntax tree root</param>
    /// <returns>Duplication metric for the file</returns>
    private async Task<DuplicationMetric> AnalyzeFileSemanticDuplicationAsync(string filePath, SyntaxNode root)
    {
        var fileName = Path.GetFileName(filePath);
        var sourceCode = root.ToFullString();
        var lines = sourceCode.Split('\n');

        // Create a semantic similarity analyzer
        var analyzer = new SemanticSimilarityAnalyzer(_logger, 80, MinimumDuplicateLines);

        // Find similar code blocks
        var similarBlocks = analyzer.FindSimilarCodeBlocks(filePath, sourceCode);

        // Convert to duplicated blocks
        var duplicatedBlocks = new List<DuplicatedBlock>();
        foreach (var block in similarBlocks)
        {
            var duplicatedCode = GetCodeBetweenLines(sourceCode, block.StartLine, block.EndLine);

            var duplicatedBlock = new DuplicatedBlock
            {
                SourceFilePath = filePath,
                SourceStartLine = block.StartLine,
                SourceEndLine = block.EndLine,
                TargetFilePath = filePath,
                TargetStartLine = block.SimilarStartLine,
                TargetEndLine = block.SimilarEndLine,
                DuplicatedCode = duplicatedCode,
                SimilarityPercentage = block.SimilarityPercentage
            };

            duplicatedBlocks.Add(duplicatedBlock);
        }

        // Calculate duplicated lines of code
        var duplicatedLinesOfCode = duplicatedBlocks.Sum(b => b.DuplicatedLines);

        // Calculate duplication percentage
        var duplicationPercentage = lines.Length > 0 ? (double)duplicatedLinesOfCode / lines.Length * 100 : 0;

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
            DuplicatedLinesOfCode = duplicatedLinesOfCode,
            DuplicationPercentage = duplicationPercentage,
            DuplicatedBlockCount = duplicatedBlocks.Count,
            DuplicatedBlocks = duplicatedBlocks,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold("C#", DuplicationType.Semantic, "File")
        };

        // Set the value to the duplication percentage
        metric.Value = metric.DuplicationPercentage;

        return metric;
    }

    /// <summary>
    /// Analyzes class-level semantic duplication
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="classDecl">Class declaration syntax</param>
    /// <param name="fullClassName">Full class name</param>
    /// <returns>Duplication metric for the class</returns>
    private async Task<DuplicationMetric> AnalyzeClassSemanticDuplicationAsync(string filePath, ClassDeclarationSyntax classDecl, string fullClassName)
    {
        var sourceCode = classDecl.ToFullString();
        var lines = sourceCode.Split('\n');

        // Create a semantic similarity analyzer
        var analyzer = new SemanticSimilarityAnalyzer(_logger, 80, MinimumDuplicateLines);

        // Find similar code blocks
        var similarBlocks = analyzer.FindSimilarCodeBlocks(filePath, sourceCode);

        // Convert to duplicated blocks
        var duplicatedBlocks = new List<DuplicatedBlock>();
        foreach (var block in similarBlocks)
        {
            var duplicatedCode = GetCodeBetweenLines(sourceCode, block.StartLine, block.EndLine);

            var duplicatedBlock = new DuplicatedBlock
            {
                SourceFilePath = filePath,
                SourceStartLine = block.StartLine,
                SourceEndLine = block.EndLine,
                TargetFilePath = filePath,
                TargetStartLine = block.SimilarStartLine,
                TargetEndLine = block.SimilarEndLine,
                DuplicatedCode = duplicatedCode,
                SimilarityPercentage = block.SimilarityPercentage
            };

            duplicatedBlocks.Add(duplicatedBlock);
        }

        // Calculate duplicated lines of code
        var duplicatedLinesOfCode = duplicatedBlocks.Sum(b => b.DuplicatedLines);

        // Calculate duplication percentage
        var duplicationPercentage = lines.Length > 0 ? (double)duplicatedLinesOfCode / lines.Length * 100 : 0;

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
            DuplicatedLinesOfCode = duplicatedLinesOfCode,
            DuplicationPercentage = duplicationPercentage,
            DuplicatedBlockCount = duplicatedBlocks.Count,
            DuplicatedBlocks = duplicatedBlocks,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold("C#", DuplicationType.Semantic, "Class")
        };

        // Set the value to the duplication percentage
        metric.Value = metric.DuplicationPercentage;

        return metric;
    }

    /// <summary>
    /// Analyzes method-level semantic duplication
    /// </summary>
    /// <param name="filePath">File path</param>
    /// <param name="methodDecl">Method declaration syntax</param>
    /// <param name="fullMethodName">Full method name</param>
    /// <returns>Duplication metric for the method</returns>
    private async Task<DuplicationMetric> AnalyzeMethodSemanticDuplicationAsync(string filePath, MethodDeclarationSyntax methodDecl, string fullMethodName)
    {
        var sourceCode = methodDecl.ToFullString();
        var lines = sourceCode.Split('\n');

        // Create a semantic similarity analyzer
        var analyzer = new SemanticSimilarityAnalyzer(_logger, 80, MinimumDuplicateLines);

        // Find similar code blocks
        var similarBlocks = analyzer.FindSimilarCodeBlocks(filePath, sourceCode);

        // Convert to duplicated blocks
        var duplicatedBlocks = new List<DuplicatedBlock>();
        foreach (var block in similarBlocks)
        {
            var duplicatedCode = GetCodeBetweenLines(sourceCode, block.StartLine, block.EndLine);

            var duplicatedBlock = new DuplicatedBlock
            {
                SourceFilePath = filePath,
                SourceStartLine = block.StartLine,
                SourceEndLine = block.EndLine,
                TargetFilePath = filePath,
                TargetStartLine = block.SimilarStartLine,
                TargetEndLine = block.SimilarEndLine,
                DuplicatedCode = duplicatedCode,
                SimilarityPercentage = block.SimilarityPercentage
            };

            duplicatedBlocks.Add(duplicatedBlock);
        }

        // Calculate duplicated lines of code
        var duplicatedLinesOfCode = duplicatedBlocks.Sum(b => b.DuplicatedLines);

        // Calculate duplication percentage
        var duplicationPercentage = lines.Length > 0 ? (double)duplicatedLinesOfCode / lines.Length * 100 : 0;

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
            DuplicatedLinesOfCode = duplicatedLinesOfCode,
            DuplicationPercentage = duplicationPercentage,
            DuplicatedBlockCount = duplicatedBlocks.Count,
            DuplicatedBlocks = duplicatedBlocks,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold("C#", DuplicationType.Semantic, "Method")
        };

        // Set the value to the duplication percentage
        metric.Value = metric.DuplicationPercentage;

        return metric;
    }

    /// <summary>
    /// Gets the code between two line numbers
    /// </summary>
    /// <param name="sourceCode">Source code</param>
    /// <param name="startLine">Start line</param>
    /// <param name="endLine">End line</param>
    /// <returns>Code between the lines</returns>
    private string GetCodeBetweenLines(string sourceCode, int startLine, int endLine)
    {
        var lines = sourceCode.Split('\n');

        // Adjust line numbers to be 0-based
        startLine = Math.Max(0, startLine - 1);
        endLine = Math.Min(lines.Length - 1, endLine - 1);

        // Extract the lines
        var codeLines = new List<string>();
        for (int i = startLine; i <= endLine; i++)
        {
            codeLines.Add(lines[i]);
        }

        return string.Join("\n", codeLines);
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
