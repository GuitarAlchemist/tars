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

namespace TarsEngine.Services;

/// <summary>
/// Analyzer for C# Halstead complexity metrics
/// </summary>
public class CSharpHalsteadAnalyzer
{
    private readonly ILogger<CSharpHalsteadAnalyzer> _logger;
    private readonly Dictionary<string, Dictionary<HalsteadType, Dictionary<string, double>>> _thresholds;

    // C# operators for Halstead complexity calculation
    private static readonly HashSet<SyntaxKind> Operators =
    [
        SyntaxKind.PlusToken,
        SyntaxKind.MinusToken,
        SyntaxKind.AsteriskToken,
        SyntaxKind.SlashToken,
        SyntaxKind.PercentToken,
        SyntaxKind.PlusPlusToken,
        SyntaxKind.MinusMinusToken,

        // Assignment operators
        SyntaxKind.EqualsToken,
        SyntaxKind.PlusEqualsToken,
        SyntaxKind.MinusEqualsToken,
        SyntaxKind.AsteriskEqualsToken,
        SyntaxKind.SlashEqualsToken,
        SyntaxKind.PercentEqualsToken,
        SyntaxKind.AmpersandEqualsToken,
        SyntaxKind.BarEqualsToken,
        SyntaxKind.CaretEqualsToken,
        SyntaxKind.LessThanLessThanEqualsToken,
        SyntaxKind.GreaterThanGreaterThanEqualsToken,

        // Comparison operators
        SyntaxKind.EqualsEqualsToken,
        SyntaxKind.ExclamationEqualsToken,
        SyntaxKind.LessThanToken,
        SyntaxKind.LessThanEqualsToken,
        SyntaxKind.GreaterThanToken,
        SyntaxKind.GreaterThanEqualsToken,

        // Logical operators
        SyntaxKind.AmpersandAmpersandToken,
        SyntaxKind.BarBarToken,
        SyntaxKind.ExclamationToken,

        // Bitwise operators
        SyntaxKind.AmpersandToken,
        SyntaxKind.BarToken,
        SyntaxKind.CaretToken,
        SyntaxKind.TildeToken,
        SyntaxKind.LessThanLessThanToken,
        SyntaxKind.GreaterThanGreaterThanToken,

        // Other operators
        SyntaxKind.QuestionToken,
        SyntaxKind.ColonToken,
        SyntaxKind.DotToken,
        SyntaxKind.QuestionQuestionToken
    ];

    // C# keywords that are considered operators for Halstead complexity
    private static readonly HashSet<SyntaxKind> KeywordOperators =
    [
        SyntaxKind.NewKeyword,
        SyntaxKind.TypeOfKeyword,
        SyntaxKind.CheckedKeyword,
        SyntaxKind.UncheckedKeyword,
        SyntaxKind.DefaultKeyword,
        SyntaxKind.SizeOfKeyword,
        SyntaxKind.IsKeyword,
        SyntaxKind.AsKeyword,
        SyntaxKind.AwaitKeyword,
        SyntaxKind.ThrowKeyword,
        SyntaxKind.NameOfKeyword
    ];

    /// <summary>
    /// Initializes a new instance of the <see cref="CSharpHalsteadAnalyzer"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    public CSharpHalsteadAnalyzer(ILogger<CSharpHalsteadAnalyzer> logger)
    {
        _logger = logger;
        _thresholds = new Dictionary<string, Dictionary<HalsteadType, Dictionary<string, double>>>
        {
            ["C#"] = new Dictionary<HalsteadType, Dictionary<string, double>>
            {
                [HalsteadType.Volume] = new Dictionary<string, double>
                {
                    ["Method"] = 500,
                    ["Class"] = 4000,
                    ["File"] = 10000
                },
                [HalsteadType.Difficulty] = new Dictionary<string, double>
                {
                    ["Method"] = 15,
                    ["Class"] = 30,
                    ["File"] = 50
                },
                [HalsteadType.Effort] = new Dictionary<string, double>
                {
                    ["Method"] = 10000,
                    ["Class"] = 100000,
                    ["File"] = 500000
                }
            }
        };
    }

    /// <summary>
    /// Analyzes Halstead complexity of a C# file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <returns>Halstead complexity metrics for the file</returns>
    public async Task<List<HalsteadMetric>> AnalyzeHalsteadComplexityAsync(string filePath)
    {
        try
        {
            var sourceCode = await File.ReadAllTextAsync(filePath);
            var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
            var root = await syntaxTree.GetRootAsync();

            var metrics = new List<HalsteadMetric>();

            // Analyze methods
            var methodDeclarations = root.DescendantNodes().OfType<MethodDeclarationSyntax>();
            foreach (var method in methodDeclarations)
            {
                var halsteadMetrics = CalculateHalsteadMetrics(method);
                var className = GetClassName(method);
                var methodName = method.Identifier.Text;
                var fullMethodName = $"{className}.{methodName}";

                foreach (var halsteadType in Enum.GetValues<HalsteadType>())
                {
                    var metric = CreateHalsteadMetric(halsteadMetrics, halsteadType, filePath, "C#", fullMethodName, TargetType.Method);
                    metrics.Add(metric);
                }
            }

            // Analyze classes
            var classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
            foreach (var classDecl in classDeclarations)
            {
                var halsteadMetrics = CalculateHalsteadMetrics(classDecl);
                var className = classDecl.Identifier.Text;
                var namespaceName = GetNamespace(classDecl);
                var fullClassName = string.IsNullOrEmpty(namespaceName) ? className : $"{namespaceName}.{className}";

                foreach (var halsteadType in Enum.GetValues<HalsteadType>())
                {
                    var metric = CreateHalsteadMetric(halsteadMetrics, halsteadType, filePath, "C#", fullClassName, TargetType.Class);
                    metrics.Add(metric);
                }
            }

            // Analyze file
            var fileHalsteadMetrics = CalculateHalsteadMetrics(root);
            var fileName = Path.GetFileName(filePath);

            foreach (var halsteadType in Enum.GetValues<HalsteadType>())
            {
                var metric = CreateHalsteadMetric(fileHalsteadMetrics, halsteadType, filePath, "C#", fileName, TargetType.File);
                metrics.Add(metric);
            }

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing Halstead complexity for file {FilePath}", filePath);
            return [];
        }
    }

    /// <summary>
    /// Gets Halstead complexity thresholds for a specific Halstead type
    /// </summary>
    /// <param name="halsteadType">Type of Halstead metric</param>
    /// <returns>Threshold values for the Halstead type</returns>
    public Dictionary<string, double> GetHalsteadThresholds(HalsteadType halsteadType)
    {
        if (_thresholds.TryGetValue("C#", out var languageThresholds) &&
            languageThresholds.TryGetValue(halsteadType, out var typeThresholds))
        {
            return typeThresholds;
        }

        return new Dictionary<string, double>();
    }

    /// <summary>
    /// Sets Halstead complexity threshold for a specific Halstead type and target type
    /// </summary>
    /// <param name="halsteadType">Type of Halstead metric</param>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <param name="threshold">Threshold value</param>
    /// <returns>True if threshold was set successfully</returns>
    public bool SetHalsteadThreshold(HalsteadType halsteadType, string targetType, double threshold)
    {
        try
        {
            if (!_thresholds.ContainsKey("C#"))
            {
                _thresholds["C#"] = new Dictionary<HalsteadType, Dictionary<string, double>>();
            }

            if (!_thresholds["C#"].ContainsKey(halsteadType))
            {
                _thresholds["C#"][halsteadType] = new Dictionary<string, double>();
            }

            _thresholds["C#"][halsteadType][targetType] = threshold;
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting Halstead threshold for {HalsteadType}, {TargetType}",
                halsteadType, targetType);
            return false;
        }
    }

    /// <summary>
    /// Calculates Halstead metrics for a syntax node
    /// </summary>
    /// <param name="node">Syntax node</param>
    /// <returns>Halstead metrics</returns>
    private (int DistinctOperators, int DistinctOperands, int TotalOperators, int TotalOperands) CalculateHalsteadMetrics(SyntaxNode node)
    {
        var operators = new HashSet<string>();
        var operands = new HashSet<string>();
        var operatorCount = 0;
        var operandCount = 0;

        // Process all tokens in the node
        foreach (var token in node.DescendantTokens())
        {
            // Check if token is an operator
            if (Operators.Contains(token.Kind()) || KeywordOperators.Contains(token.Kind()))
            {
                operators.Add(token.Text);
                operatorCount++;
            }
            // Check if token is an identifier or literal (operand)
            else if (token.IsKind(SyntaxKind.IdentifierToken) ||
                     token.IsKind(SyntaxKind.StringLiteralToken) ||
                     token.IsKind(SyntaxKind.NumericLiteralToken) ||
                     token.IsKind(SyntaxKind.CharacterLiteralToken) ||
                     token.IsKind(SyntaxKind.TrueKeyword) ||
                     token.IsKind(SyntaxKind.FalseKeyword) ||
                     token.IsKind(SyntaxKind.NullKeyword))
            {
                operands.Add(token.Text);
                operandCount++;
            }
        }

        return (operators.Count, operands.Count, operatorCount, operandCount);
    }

    /// <summary>
    /// Creates a Halstead metric
    /// </summary>
    /// <param name="halsteadMetrics">Halstead metrics</param>
    /// <param name="halsteadType">Type of Halstead metric</param>
    /// <param name="filePath">File path</param>
    /// <param name="language">Programming language</param>
    /// <param name="target">Target (method, class, etc.)</param>
    /// <param name="targetType">Type of target</param>
    /// <returns>Halstead metric</returns>
    private HalsteadMetric CreateHalsteadMetric(
        (int DistinctOperators, int DistinctOperands, int TotalOperators, int TotalOperands) halsteadMetrics,
        HalsteadType halsteadType,
        string filePath,
        string language,
        string target,
        TargetType targetType)
    {
        var metric = new HalsteadMetric
        {
            Type = halsteadType,
            FilePath = filePath,
            Language = language,
            Target = target,
            TargetType = targetType,
            DistinctOperators = halsteadMetrics.DistinctOperators,
            DistinctOperands = halsteadMetrics.DistinctOperands,
            TotalOperators = halsteadMetrics.TotalOperators,
            TotalOperands = halsteadMetrics.TotalOperands,
            Timestamp = DateTime.UtcNow
        };

        // Set name, description, and value based on Halstead type
        switch (halsteadType)
        {
            case HalsteadType.Vocabulary:
                metric.Name = $"Halstead Vocabulary - {target}";
                metric.Description = $"Halstead vocabulary (n) for {targetType} {target}";
                metric.Value = metric.Vocabulary;
                break;
            case HalsteadType.Length:
                metric.Name = $"Halstead Length - {target}";
                metric.Description = $"Halstead length (N) for {targetType} {target}";
                metric.Value = metric.Length;
                break;
            case HalsteadType.Volume:
                metric.Name = $"Halstead Volume - {target}";
                metric.Description = $"Halstead volume (V) for {targetType} {target}";
                metric.Value = metric.Volume;
                metric.ThresholdValue = GetThreshold(halsteadType, targetType.ToString());
                break;
            case HalsteadType.Difficulty:
                metric.Name = $"Halstead Difficulty - {target}";
                metric.Description = $"Halstead difficulty (D) for {targetType} {target}";
                metric.Value = metric.Difficulty;
                metric.ThresholdValue = GetThreshold(halsteadType, targetType.ToString());
                break;
            case HalsteadType.Effort:
                metric.Name = $"Halstead Effort - {target}";
                metric.Description = $"Halstead effort (E) for {targetType} {target}";
                metric.Value = metric.Effort;
                metric.ThresholdValue = GetThreshold(halsteadType, targetType.ToString());
                break;
            case HalsteadType.TimeRequired:
                metric.Name = $"Halstead Time Required - {target}";
                metric.Description = $"Halstead time required (T) for {targetType} {target}";
                metric.Value = metric.TimeRequired;
                break;
            case HalsteadType.DeliveredBugs:
                metric.Name = $"Halstead Delivered Bugs - {target}";
                metric.Description = $"Halstead delivered bugs (B) for {targetType} {target}";
                metric.Value = metric.DeliveredBugs;
                break;
        }

        return metric;
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
    /// Gets the threshold value for a specific Halstead type and target type
    /// </summary>
    /// <param name="halsteadType">Type of Halstead metric</param>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <returns>Threshold value</returns>
    private double GetThreshold(HalsteadType halsteadType, string targetType)
    {
        if (_thresholds.TryGetValue("C#", out var languageThresholds) &&
            languageThresholds.TryGetValue(halsteadType, out var typeThresholds) &&
            typeThresholds.TryGetValue(targetType, out var threshold))
        {
            return threshold;
        }

        // Default thresholds if not configured
        return halsteadType switch
        {
            HalsteadType.Volume => targetType switch
            {
                "Method" => 500,
                "Class" => 4000,
                "File" => 10000,
                _ => 500
            },
            HalsteadType.Difficulty => targetType switch
            {
                "Method" => 15,
                "Class" => 30,
                "File" => 50,
                _ => 15
            },
            HalsteadType.Effort => targetType switch
            {
                "Method" => 10000,
                "Class" => 100000,
                "File" => 500000,
                _ => 10000
            },
            _ => 0
        };
    }
}
