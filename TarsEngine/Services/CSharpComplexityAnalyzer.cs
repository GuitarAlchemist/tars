using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TarsEngine.Models.Metrics;
using TarsEngine.Services.Interfaces;
using UnifiedComplexityType = TarsEngine.Models.Unified.ComplexityTypeUnified;
using ModelComplexityType = TarsEngine.Models.Metrics.ComplexityType;

namespace TarsEngine.Services;

/// <summary>
/// Analyzer for C# code complexity metrics
/// </summary>
public class CSharpComplexityAnalyzer : ICodeComplexityAnalyzer
{
    private readonly ILogger<CSharpComplexityAnalyzer> _logger;
    private readonly Dictionary<string, Dictionary<TarsEngine.Models.Metrics.ComplexityType, Dictionary<string, double>>> _thresholds;

    /// <summary>
    /// Initializes a new instance of the <see cref="CSharpComplexityAnalyzer"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    public CSharpComplexityAnalyzer(ILogger<CSharpComplexityAnalyzer> logger)
    {
        _logger = logger;
        _thresholds = new Dictionary<string, Dictionary<TarsEngine.Models.Metrics.ComplexityType, Dictionary<string, double>>>
        {
            ["C#"] = new()
            {
                [TarsEngine.Models.Metrics.ComplexityType.Cyclomatic] = new Dictionary<string, double>
                {
                    ["Method"] = 10,
                    ["Class"] = 20,
                    ["File"] = 50
                },
                [TarsEngine.Models.Metrics.ComplexityType.Cognitive] = new Dictionary<string, double>
                {
                    ["Method"] = 15,
                    ["Class"] = 30,
                    ["File"] = 75
                }
            }
        };
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeCyclomaticComplexityAsync(string filePath, string language)
    {
        try
        {
            if (language != "C#")
            {
                _logger.LogWarning("Language {Language} not supported by CSharpComplexityAnalyzer", language);
                return new List<ComplexityMetric>();
            }

            var sourceCode = await File.ReadAllTextAsync(filePath);
            var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
            var root = await syntaxTree.GetRootAsync();

            var metrics = new List<ComplexityMetric>();

            // Analyze methods
            var methodDeclarations = root.DescendantNodes().OfType<MethodDeclarationSyntax>();
            foreach (var method in methodDeclarations)
            {
                var complexity = CalculateCyclomaticComplexity(method);
                var className = GetClassName(method);
                var methodName = method.Identifier.Text;
                var fullMethodName = $"{className}.{methodName}";

                var metric = new ComplexityMetric
                {
                    Name = $"Cyclomatic Complexity - {fullMethodName}",
                    Description = $"McCabe's cyclomatic complexity for method {fullMethodName}",
                    Value = complexity,
                    Type = TarsEngine.Models.Metrics.ComplexityType.Cyclomatic,
                    FilePath = filePath,
                    Language = language,
                    Target = fullMethodName,
                    TargetType = TargetType.Method,
                    Timestamp = DateTime.UtcNow,
                    ThresholdValue = GetThreshold(language, "Cyclomatic", "Method")
                };

                metrics.Add(metric);
            }

            // Analyze classes
            var classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
            foreach (var classDecl in classDeclarations)
            {
                var className = classDecl.Identifier.Text;
                var namespaceName = GetNamespace(classDecl);
                var fullClassName = string.IsNullOrEmpty(namespaceName) ? className : $"{namespaceName}.{className}";

                // Sum complexity of all methods in the class
                var classMethods = classDecl.DescendantNodes().OfType<MethodDeclarationSyntax>();
                var classComplexity = classMethods.Sum(m => CalculateCyclomaticComplexity(m));

                var metric = new ComplexityMetric
                {
                    Name = $"Cyclomatic Complexity - {fullClassName}",
                    Description = $"McCabe's cyclomatic complexity for class {fullClassName}",
                    Value = classComplexity,
                    Type = TarsEngine.Models.Metrics.ComplexityType.Cyclomatic,
                    FilePath = filePath,
                    Language = language,
                    Target = fullClassName,
                    TargetType = TargetType.Class,
                    Timestamp = DateTime.UtcNow,
                    ThresholdValue = GetThreshold(language, "Cyclomatic", "Class")
                };

                metrics.Add(metric);
            }

            // Calculate file complexity
            var fileComplexity = methodDeclarations.Sum(m => CalculateCyclomaticComplexity(m));
            var fileName = Path.GetFileName(filePath);

            var fileMetric = new ComplexityMetric
            {
                Name = $"Cyclomatic Complexity - {fileName}",
                Description = $"McCabe's cyclomatic complexity for file {fileName}",
                Value = fileComplexity,
                Type = TarsEngine.Models.Metrics.ComplexityType.Cyclomatic,
                FilePath = filePath,
                Language = language,
                Target = fileName,
                TargetType = TargetType.File,
                Timestamp = DateTime.UtcNow,
                ThresholdValue = GetThreshold(language, "Cyclomatic", "File")
            };

            metrics.Add(fileMetric);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing cyclomatic complexity for file {FilePath}", filePath);
            return new List<ComplexityMetric>();
        }
    }

    /// <inheritdoc/>
    public Task<List<ComplexityMetric>> AnalyzeCognitiveComplexityAsync(string filePath, string language)
    {
        // Implementation will be added in a future task
        _logger.LogInformation("Cognitive complexity analysis not yet implemented");
        return Task.FromResult(new List<ComplexityMetric>());
    }

    /// <inheritdoc/>
    public async Task<List<MaintainabilityMetric>> AnalyzeMaintainabilityIndexAsync(string filePath, string language)
    {
        try
        {
            if (language != "C#")
            {
                _logger.LogWarning("Language {Language} not supported by CSharpComplexityAnalyzer", language);
                return new List<MaintainabilityMetric>();
            }

            // Create maintainability analyzer
            var halsteadAnalyzer = new CSharpHalsteadAnalyzer(null);
            var maintainabilityAnalyzer = new CSharpMaintainabilityAnalyzer(null, this, halsteadAnalyzer);

            // Analyze maintainability index
            return await maintainabilityAnalyzer.AnalyzeMaintainabilityIndexAsync(filePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing maintainability index for file {FilePath}", filePath);
            return new List<MaintainabilityMetric>();
        }
    }

    /// <inheritdoc/>
    public async Task<List<HalsteadMetric>> AnalyzeHalsteadComplexityAsync(string filePath, string language)
    {
        try
        {
            if (language != "C#")
            {
                _logger.LogWarning("Language {Language} not supported by CSharpComplexityAnalyzer", language);
                return new List<HalsteadMetric>();
            }

            // Create Halstead analyzer
            var halsteadAnalyzer = new CSharpHalsteadAnalyzer(null);

            // Analyze Halstead complexity
            return await halsteadAnalyzer.AnalyzeHalsteadComplexityAsync(filePath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing Halstead complexity for file {FilePath}", filePath);
            return new List<HalsteadMetric>();
        }
    }

    /// <inheritdoc/>
    public Task<List<ReadabilityMetric>> AnalyzeReadabilityAsync(string filePath, string language, ReadabilityType readabilityType)
    {
        if (language != "C#")
        {
            _logger.LogWarning("Language {Language} not supported by CSharpComplexityAnalyzer", language);
            return Task.FromResult(new List<ReadabilityMetric>());
        }

        try
        {
            // Use the readability analyzer service
            var readabilityAnalyzer = new CSharpReadabilityAnalyzer(null);

            return readabilityType switch
            {
                ReadabilityType.IdentifierQuality => readabilityAnalyzer.AnalyzeIdentifierQualityAsync(filePath, language),
                ReadabilityType.CommentQuality => readabilityAnalyzer.AnalyzeCommentQualityAsync(filePath, language),
                ReadabilityType.CodeStructure => readabilityAnalyzer.AnalyzeCodeStructureAsync(filePath, language),
                ReadabilityType.Overall => readabilityAnalyzer.AnalyzeOverallReadabilityAsync(filePath, language),
                _ => readabilityAnalyzer.AnalyzeAllReadabilityMetricsAsync(filePath, language)
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing readability for file {FilePath}", filePath);
            return Task.FromResult(new List<ReadabilityMetric>());
        }
    }

    /// <inheritdoc/>
    public async Task<(List<ComplexityMetric> ComplexityMetrics, List<HalsteadMetric> HalsteadMetrics, List<MaintainabilityMetric> MaintainabilityMetrics, List<ReadabilityMetric> ReadabilityMetrics)> AnalyzeAllComplexityMetricsAsync(string filePath, string language)
    {
        var complexityMetrics = new List<ComplexityMetric>();
        var halsteadMetrics = new List<HalsteadMetric>();
        var maintainabilityMetrics = new List<MaintainabilityMetric>();
        var readabilityMetrics = new List<ReadabilityMetric>();

        // Get cyclomatic complexity metrics
        complexityMetrics.AddRange(await AnalyzeCyclomaticComplexityAsync(filePath, language));

        // Get cognitive complexity metrics
        complexityMetrics.AddRange(await AnalyzeCognitiveComplexityAsync(filePath, language));

        // Get Halstead complexity metrics
        halsteadMetrics.AddRange(await AnalyzeHalsteadComplexityAsync(filePath, language));

        // Get maintainability index metrics
        maintainabilityMetrics.AddRange(await AnalyzeMaintainabilityIndexAsync(filePath, language));

        // Get readability metrics
        var readabilityAnalyzer = new CSharpReadabilityAnalyzer(null);
        readabilityMetrics.AddRange(await readabilityAnalyzer.AnalyzeAllReadabilityMetricsAsync(filePath, language));

        return (ComplexityMetrics: complexityMetrics, HalsteadMetrics: halsteadMetrics, MaintainabilityMetrics: maintainabilityMetrics, ReadabilityMetrics: readabilityMetrics);
    }

    /// <inheritdoc/>
    public async Task<(List<ComplexityMetric> ComplexityMetrics, List<HalsteadMetric> HalsteadMetrics, List<MaintainabilityMetric> MaintainabilityMetrics, List<ReadabilityMetric> ReadabilityMetrics)> AnalyzeProjectComplexityAsync(string projectPath)
    {
        var complexityMetrics = new List<ComplexityMetric>();
        var halsteadMetrics = new List<HalsteadMetric>();
        var maintainabilityMetrics = new List<MaintainabilityMetric>();
        var readabilityMetrics = new List<ReadabilityMetric>();

        try
        {
            var csharpFiles = Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories);

            foreach (var file in csharpFiles)
            {
                // Analyze all metrics for each file
                var fileMetrics = await AnalyzeAllComplexityMetricsAsync(file, "C#");

                // Add file metrics to project metrics
                complexityMetrics.AddRange(fileMetrics.ComplexityMetrics);
                halsteadMetrics.AddRange(fileMetrics.HalsteadMetrics);
                maintainabilityMetrics.AddRange(fileMetrics.MaintainabilityMetrics);
                readabilityMetrics.AddRange(fileMetrics.ReadabilityMetrics);
            }

            // Calculate project-level metrics
            var projectName = Path.GetFileName(projectPath);

            // Calculate project-level cyclomatic complexity
            var projectCyclomaticComplexity = complexityMetrics
                .Where(m => m.Type == TarsEngine.Models.Metrics.ComplexityType.Cyclomatic && m.TargetType == TargetType.File)
                .Sum(m => m.Value);

            var cyclomaticMetric = new ComplexityMetric
            {
                Name = $"Cyclomatic Complexity - {projectName}",
                Description = $"McCabe's cyclomatic complexity for project {projectName}",
                Value = projectCyclomaticComplexity,
                Type = TarsEngine.Models.Metrics.ComplexityType.Cyclomatic,
                FilePath = projectPath,
                Language = "C#",
                Target = projectName,
                TargetType = TargetType.Project,
                Timestamp = DateTime.UtcNow
            };

            complexityMetrics.Add(cyclomaticMetric);

            // Calculate project-level Halstead volume
            var projectHalsteadVolume = halsteadMetrics
                .Where(m => m.Type == HalsteadType.Volume && m.TargetType == TargetType.File)
                .Sum(m => m.Value);

            var halsteadMetric = new HalsteadMetric
            {
                Name = $"Halstead Volume - {projectName}",
                Description = $"Halstead volume for project {projectName}",
                Value = projectHalsteadVolume,
                Type = HalsteadType.Volume,
                FilePath = projectPath,
                Language = "C#",
                Target = projectName,
                TargetType = TargetType.Project,
                Timestamp = DateTime.UtcNow
            };

            halsteadMetrics.Add(halsteadMetric);

            // Calculate project-level maintainability index
            // Use average of file maintainability indices
            var fileMaintenanceIndices = maintainabilityMetrics
                .Where(m => m.TargetType == TargetType.File)
                .ToList();

            if (fileMaintenanceIndices.Any())
            {
                var averageMaintainabilityIndex = fileMaintenanceIndices.Average(m => m.Value);

                var maintainabilityMetric = new MaintainabilityMetric
                {
                    Name = $"Maintainability Index - {projectName}",
                    Description = $"Maintainability index for project {projectName}",
                    Value = averageMaintainabilityIndex,
                    HalsteadVolume = projectHalsteadVolume,
                    CyclomaticComplexity = projectCyclomaticComplexity,
                    LinesOfCode = fileMaintenanceIndices.Sum(m => m.LinesOfCode),
                    CommentPercentage = fileMaintenanceIndices.Average(m => m.CommentPercentage),
                    FilePath = projectPath,
                    Language = "C#",
                    Target = projectName,
                    TargetType = TargetType.Project,
                    Timestamp = DateTime.UtcNow,
                    UseMicrosoftFormula = true
                };

                maintainabilityMetrics.Add(maintainabilityMetric);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing project complexity for {ProjectPath}", projectPath);
        }

        return (ComplexityMetrics: complexityMetrics, HalsteadMetrics: halsteadMetrics, MaintainabilityMetrics: maintainabilityMetrics, ReadabilityMetrics: readabilityMetrics);
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, double>> GetComplexityThresholdsAsync(string language, TarsEngine.Services.Interfaces.UnifiedComplexityType complexityType)
    {
        if (!_thresholds.ContainsKey(language))
        {
            _logger.LogWarning("Language {Language} not supported for complexity thresholds", language);
            return Task.FromResult(new Dictionary<string, double>());
        }

        // Convert from interface type to model type
        var modelType = complexityType switch
        {
            TarsEngine.Services.Interfaces.UnifiedComplexityType.Cyclomatic => TarsEngine.Models.Metrics.ComplexityType.Cyclomatic,
            TarsEngine.Services.Interfaces.UnifiedComplexityType.Cognitive => TarsEngine.Models.Metrics.ComplexityType.Cognitive,
            TarsEngine.Services.Interfaces.UnifiedComplexityType.Halstead => TarsEngine.Models.Metrics.ComplexityType.Halstead,
            TarsEngine.Services.Interfaces.UnifiedComplexityType.Maintainability => TarsEngine.Models.Metrics.ComplexityType.MaintainabilityIndex,
            _ => TarsEngine.Models.Metrics.ComplexityType.Cyclomatic
        };

        if (!_thresholds[language].ContainsKey(modelType))
        {
            _logger.LogWarning("Complexity type {ComplexityType} not supported for language {Language}", complexityType, language);
            return Task.FromResult(new Dictionary<string, double>());
        }

        return Task.FromResult(_thresholds[language][modelType]);
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, double>> GetHalsteadThresholdsAsync(string language, HalsteadType halsteadType)
    {
        if (language != "C#")
        {
            _logger.LogWarning("Language {Language} not supported by CSharpComplexityAnalyzer", language);
            return Task.FromResult(new Dictionary<string, double>());
        }

        // Create Halstead analyzer
        var halsteadAnalyzer = new CSharpHalsteadAnalyzer(null);

        // Get Halstead thresholds
        return Task.FromResult(halsteadAnalyzer.GetHalsteadThresholds(halsteadType));
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, double>> GetMaintainabilityThresholdsAsync(string language)
    {
        if (language != "C#")
        {
            _logger.LogWarning("Language {Language} not supported by CSharpComplexityAnalyzer", language);
            return Task.FromResult(new Dictionary<string, double>());
        }

        // Create maintainability analyzer with null logger instances
        var halsteadLogger = NullLoggerFactory.Instance.CreateLogger<CSharpHalsteadAnalyzer>();
        var maintainabilityLogger = NullLoggerFactory.Instance.CreateLogger<CSharpMaintainabilityAnalyzer>();
        var halsteadAnalyzer = new CSharpHalsteadAnalyzer(halsteadLogger);
        var maintainabilityAnalyzer = new CSharpMaintainabilityAnalyzer(maintainabilityLogger, this, halsteadAnalyzer);

        // Get maintainability thresholds
        return Task.FromResult(maintainabilityAnalyzer.GetMaintainabilityThresholds());
    }

    /// <inheritdoc/>
    public Task<bool> SetComplexityThresholdAsync(string language, TarsEngine.Services.Interfaces.UnifiedComplexityType complexityType, string targetType, double threshold)
    {
        try
        {
            // Convert from interface type to model type
            var modelType = complexityType switch
            {
                TarsEngine.Services.Interfaces.UnifiedComplexityType.Cyclomatic => TarsEngine.Models.Metrics.ComplexityType.Cyclomatic,
                TarsEngine.Services.Interfaces.UnifiedComplexityType.Cognitive => TarsEngine.Models.Metrics.ComplexityType.Cognitive,
                TarsEngine.Services.Interfaces.UnifiedComplexityType.Halstead => TarsEngine.Models.Metrics.ComplexityType.Halstead,
                TarsEngine.Services.Interfaces.UnifiedComplexityType.Maintainability => TarsEngine.Models.Metrics.ComplexityType.MaintainabilityIndex,
                _ => TarsEngine.Models.Metrics.ComplexityType.Cyclomatic
            };

            if (!_thresholds.ContainsKey(language))
            {
                _thresholds[language] = new Dictionary<TarsEngine.Models.Metrics.ComplexityType, Dictionary<string, double>>();
            }

            if (!_thresholds[language].ContainsKey(modelType))
            {
                _thresholds[language][modelType] = new Dictionary<string, double>();
            }

            _thresholds[language][modelType][targetType] = threshold;
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting complexity threshold for {Language}, {ComplexityType}, {TargetType}",
                language, complexityType, targetType);
            return Task.FromResult(false);
        }
    }

    /// <inheritdoc/>
    public Task<bool> SetHalsteadThresholdAsync(string language, HalsteadType halsteadType, string targetType, double threshold)
    {
        if (language != "C#")
        {
            _logger.LogWarning("Language {Language} not supported by CSharpComplexityAnalyzer", language);
            return Task.FromResult(false);
        }

        try
        {
            // Create Halstead analyzer with a null logger instance
            var halsteadLogger = NullLoggerFactory.Instance.CreateLogger<CSharpHalsteadAnalyzer>();
            var halsteadAnalyzer = new CSharpHalsteadAnalyzer(halsteadLogger);

            // Set Halstead threshold
            return Task.FromResult(halsteadAnalyzer.SetHalsteadThreshold(halsteadType, targetType, threshold));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting Halstead threshold for {HalsteadType}, {TargetType}",
                halsteadType, targetType);
            return Task.FromResult(false);
        }
    }

    /// <inheritdoc/>
    public Task<bool> SetMaintainabilityThresholdAsync(string language, string targetType, double threshold)
    {
        if (language != "C#")
        {
            _logger.LogWarning("Language {Language} not supported by CSharpComplexityAnalyzer", language);
            return Task.FromResult(false);
        }

        try
        {
            // Create maintainability analyzer with null logger instances
            var halsteadLogger = NullLoggerFactory.Instance.CreateLogger<CSharpHalsteadAnalyzer>();
            var maintainabilityLogger = NullLoggerFactory.Instance.CreateLogger<CSharpMaintainabilityAnalyzer>();
            var halsteadAnalyzer = new CSharpHalsteadAnalyzer(halsteadLogger);
            var maintainabilityAnalyzer = new CSharpMaintainabilityAnalyzer(maintainabilityLogger, this, halsteadAnalyzer);

            // Set maintainability threshold
            return Task.FromResult(maintainabilityAnalyzer.SetMaintainabilityThreshold(targetType, threshold));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting maintainability threshold for {TargetType}", targetType);
            return Task.FromResult(false);
        }
    }

    /// <summary>
    /// Calculates the cyclomatic complexity of a method
    /// </summary>
    /// <param name="method">Method declaration syntax</param>
    /// <returns>Cyclomatic complexity value</returns>
    private int CalculateCyclomaticComplexity(MethodDeclarationSyntax method)
    {
        // McCabe's cyclomatic complexity: E - N + 2P
        // Where:
        // E = number of edges in the control flow graph
        // N = number of nodes in the control flow graph
        // P = number of connected components (usually 1 for a method)

        // For practical purposes, we can count:
        // 1 (base complexity) +
        // number of decision points (if, while, for, foreach, case, &&, ||, ?:, etc.)

        // Start with base complexity of 1
        var complexity = 1;

        // Count if statements
        complexity += method.DescendantNodes().OfType<IfStatementSyntax>().Count();

        // Count switch sections
        complexity += method.DescendantNodes().OfType<SwitchSectionSyntax>().Count();

        // Count loops
        complexity += method.DescendantNodes().OfType<WhileStatementSyntax>().Count();
        complexity += method.DescendantNodes().OfType<ForStatementSyntax>().Count();
        complexity += method.DescendantNodes().OfType<ForEachStatementSyntax>().Count();
        complexity += method.DescendantNodes().OfType<DoStatementSyntax>().Count();

        // Count conditional expressions
        complexity += method.DescendantNodes().OfType<ConditionalExpressionSyntax>().Count();

        // Count logical operators (&&, ||)
        complexity += method.DescendantNodes().OfType<BinaryExpressionSyntax>()
            .Count(b => b.Kind() == SyntaxKind.LogicalAndExpression || b.Kind() == SyntaxKind.LogicalOrExpression);

        // Count catch clauses
        complexity += method.DescendantNodes().OfType<CatchClauseSyntax>().Count();

        return complexity;
    }

    /// <summary>
    /// Gets the class name for a method
    /// </summary>
    /// <param name="method">Method declaration syntax</param>
    /// <returns>Class name</returns>
    private string GetClassName(MethodDeclarationSyntax method)
    {
        var classDecl = method.Ancestors().OfType<ClassDeclarationSyntax>().FirstOrDefault();
        return classDecl?.Identifier.Text ?? "Unknown";
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
    /// Gets the threshold value for a specific language, complexity type, and target type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="complexityType">Type of complexity</param>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <returns>Threshold value</returns>
    private double GetThreshold(string language, string complexityTypeStr, string targetType)
    {
        // Convert string to ComplexityType
        var modelType = complexityTypeStr switch
        {
            "Cyclomatic" => TarsEngine.Models.Metrics.ComplexityType.Cyclomatic,
            "Cognitive" => TarsEngine.Models.Metrics.ComplexityType.Cognitive,
            "Halstead" => TarsEngine.Models.Metrics.ComplexityType.Halstead,
            "Maintainability" => TarsEngine.Models.Metrics.ComplexityType.MaintainabilityIndex,
            _ => TarsEngine.Models.Metrics.ComplexityType.Cyclomatic
        };

        if (_thresholds.TryGetValue(language, out var languageThresholds) &&
            languageThresholds.TryGetValue(modelType, out var typeThresholds) &&
            typeThresholds.TryGetValue(targetType, out var threshold))
        {
            return threshold;
        }

        // Default thresholds if not configured
        var modelComplexityType = modelType;

        return modelComplexityType switch
        {
            TarsEngine.Models.Metrics.ComplexityType.Cyclomatic => targetType switch
            {
                "Method" => 10,
                "Class" => 20,
                "File" => 50,
                _ => 10
            },
            TarsEngine.Models.Metrics.ComplexityType.Cognitive => targetType switch
            {
                "Method" => 15,
                "Class" => 30,
                "File" => 75,
                _ => 15
            },
            TarsEngine.Models.Metrics.ComplexityType.MaintainabilityIndex => 20,
            _ => 10
        };
    }


}
