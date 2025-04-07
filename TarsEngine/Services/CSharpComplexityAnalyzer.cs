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
/// Analyzer for C# code complexity metrics
/// </summary>
public class CSharpComplexityAnalyzer : ICodeComplexityAnalyzer
{
    private readonly ILogger<CSharpComplexityAnalyzer> _logger;
    private readonly Dictionary<string, Dictionary<ComplexityType, Dictionary<string, double>>> _thresholds;

    /// <summary>
    /// Initializes a new instance of the <see cref="CSharpComplexityAnalyzer"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    public CSharpComplexityAnalyzer(ILogger<CSharpComplexityAnalyzer> logger)
    {
        _logger = logger;
        _thresholds = new Dictionary<string, Dictionary<ComplexityType, Dictionary<string, double>>>
        {
            ["C#"] = new Dictionary<ComplexityType, Dictionary<string, double>>
            {
                [ComplexityType.Cyclomatic] = new Dictionary<string, double>
                {
                    ["Method"] = 10,
                    ["Class"] = 20,
                    ["File"] = 50
                },
                [ComplexityType.Cognitive] = new Dictionary<string, double>
                {
                    ["Method"] = 15,
                    ["Class"] = 30,
                    ["File"] = 75
                },
                [ComplexityType.MaintainabilityIndex] = new Dictionary<string, double>
                {
                    ["Method"] = 20,
                    ["Class"] = 20,
                    ["File"] = 20
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
                    Type = ComplexityType.Cyclomatic,
                    FilePath = filePath,
                    Language = language,
                    Target = fullMethodName,
                    TargetType = TargetType.Method,
                    Timestamp = DateTime.UtcNow,
                    ThresholdValue = GetThreshold(language, ComplexityType.Cyclomatic, "Method")
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
                    Type = ComplexityType.Cyclomatic,
                    FilePath = filePath,
                    Language = language,
                    Target = fullClassName,
                    TargetType = TargetType.Class,
                    Timestamp = DateTime.UtcNow,
                    ThresholdValue = GetThreshold(language, ComplexityType.Cyclomatic, "Class")
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
                Type = ComplexityType.Cyclomatic,
                FilePath = filePath,
                Language = language,
                Target = fileName,
                TargetType = TargetType.File,
                Timestamp = DateTime.UtcNow,
                ThresholdValue = GetThreshold(language, ComplexityType.Cyclomatic, "File")
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
    public async Task<List<ComplexityMetric>> AnalyzeCognitiveComplexityAsync(string filePath, string language)
    {
        // Implementation will be added in a future task
        _logger.LogInformation("Cognitive complexity analysis not yet implemented");
        return new List<ComplexityMetric>();
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeMaintainabilityIndexAsync(string filePath, string language)
    {
        // Implementation will be added in a future task
        _logger.LogInformation("Maintainability index analysis not yet implemented");
        return new List<ComplexityMetric>();
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeHalsteadComplexityAsync(string filePath, string language)
    {
        // Implementation will be added in a future task
        _logger.LogInformation("Halstead complexity analysis not yet implemented");
        return new List<ComplexityMetric>();
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeAllComplexityMetricsAsync(string filePath, string language)
    {
        var metrics = new List<ComplexityMetric>();
        
        metrics.AddRange(await AnalyzeCyclomaticComplexityAsync(filePath, language));
        metrics.AddRange(await AnalyzeCognitiveComplexityAsync(filePath, language));
        metrics.AddRange(await AnalyzeMaintainabilityIndexAsync(filePath, language));
        metrics.AddRange(await AnalyzeHalsteadComplexityAsync(filePath, language));
        
        return metrics;
    }

    /// <inheritdoc/>
    public async Task<List<ComplexityMetric>> AnalyzeProjectComplexityAsync(string projectPath)
    {
        var metrics = new List<ComplexityMetric>();
        
        try
        {
            var csharpFiles = Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories);
            
            foreach (var file in csharpFiles)
            {
                metrics.AddRange(await AnalyzeCyclomaticComplexityAsync(file, "C#"));
            }
            
            // Calculate project-level metrics
            var projectName = Path.GetFileName(projectPath);
            
            var projectCyclomaticComplexity = metrics
                .Where(m => m.Type == ComplexityType.Cyclomatic && m.TargetType == TargetType.File)
                .Sum(m => m.Value);
            
            var projectMetric = new ComplexityMetric
            {
                Name = $"Cyclomatic Complexity - {projectName}",
                Description = $"McCabe's cyclomatic complexity for project {projectName}",
                Value = projectCyclomaticComplexity,
                Type = ComplexityType.Cyclomatic,
                FilePath = projectPath,
                Language = "C#",
                Target = projectName,
                TargetType = TargetType.Project,
                Timestamp = DateTime.UtcNow
            };
            
            metrics.Add(projectMetric);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing project complexity for {ProjectPath}", projectPath);
        }
        
        return metrics;
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, double>> GetComplexityThresholdsAsync(string language, ComplexityType complexityType)
    {
        if (_thresholds.TryGetValue(language, out var languageThresholds) &&
            languageThresholds.TryGetValue(complexityType, out var typeThresholds))
        {
            return Task.FromResult(typeThresholds);
        }
        
        return Task.FromResult(new Dictionary<string, double>());
    }

    /// <inheritdoc/>
    public Task<bool> SetComplexityThresholdAsync(string language, ComplexityType complexityType, string targetType, double threshold)
    {
        try
        {
            if (!_thresholds.ContainsKey(language))
            {
                _thresholds[language] = new Dictionary<ComplexityType, Dictionary<string, double>>();
            }
            
            if (!_thresholds[language].ContainsKey(complexityType))
            {
                _thresholds[language][complexityType] = new Dictionary<string, double>();
            }
            
            _thresholds[language][complexityType][targetType] = threshold;
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting complexity threshold for {Language}, {ComplexityType}, {TargetType}", 
                language, complexityType, targetType);
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
        int complexity = 1;
        
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
    private double GetThreshold(string language, ComplexityType complexityType, string targetType)
    {
        if (_thresholds.TryGetValue(language, out var languageThresholds) &&
            languageThresholds.TryGetValue(complexityType, out var typeThresholds) &&
            typeThresholds.TryGetValue(targetType, out var threshold))
        {
            return threshold;
        }
        
        // Default thresholds if not configured
        return complexityType switch
        {
            ComplexityType.Cyclomatic => targetType switch
            {
                "Method" => 10,
                "Class" => 20,
                "File" => 50,
                _ => 10
            },
            ComplexityType.Cognitive => targetType switch
            {
                "Method" => 15,
                "Class" => 30,
                "File" => 75,
                _ => 15
            },
            ComplexityType.MaintainabilityIndex => 20,
            _ => 10
        };
    }
}
