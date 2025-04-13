using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;

namespace TarsEngine.Services;

/// <summary>
/// Analyzer for C# maintainability index
/// </summary>
public class CSharpMaintainabilityAnalyzer
{
    private readonly ILogger<CSharpMaintainabilityAnalyzer> _logger;
    private readonly CSharpComplexityAnalyzer _complexityAnalyzer;
    private readonly CSharpHalsteadAnalyzer _halsteadAnalyzer;
    private readonly Dictionary<string, Dictionary<string, double>> _thresholds;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="CSharpMaintainabilityAnalyzer"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    /// <param name="complexityAnalyzer">C# complexity analyzer</param>
    /// <param name="halsteadAnalyzer">C# Halstead analyzer</param>
    public CSharpMaintainabilityAnalyzer(
        ILogger<CSharpMaintainabilityAnalyzer> logger,
        CSharpComplexityAnalyzer complexityAnalyzer,
        CSharpHalsteadAnalyzer halsteadAnalyzer)
    {
        _logger = logger;
        _complexityAnalyzer = complexityAnalyzer;
        _halsteadAnalyzer = halsteadAnalyzer;
        _thresholds = new Dictionary<string, Dictionary<string, double>>
        {
            ["C#"] = new()
            {
                ["Method"] = 60,
                ["Class"] = 50,
                ["File"] = 40
            }
        };
    }
    
    /// <summary>
    /// Analyzes maintainability index of a C# file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <returns>Maintainability index metrics for the file</returns>
    public async Task<List<MaintainabilityMetric>> AnalyzeMaintainabilityIndexAsync(string filePath)
    {
        try
        {
            var sourceCode = await File.ReadAllTextAsync(filePath);
            var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
            var root = await syntaxTree.GetRootAsync();
            
            // Get cyclomatic complexity metrics
            var complexityMetrics = await _complexityAnalyzer.AnalyzeCyclomaticComplexityAsync(filePath, "C#");
            
            // Get Halstead volume metrics
            var halsteadMetrics = await _halsteadAnalyzer.AnalyzeHalsteadComplexityAsync(filePath);
            var volumeMetrics = halsteadMetrics.Where(m => m.Type == HalsteadType.Volume).ToList();
            
            var metrics = new List<MaintainabilityMetric>();
            
            // Analyze methods
            var methodDeclarations = root.DescendantNodes().OfType<MethodDeclarationSyntax>();
            foreach (var method in methodDeclarations)
            {
                var className = GetClassName(method);
                var methodName = method.Identifier.Text;
                var fullMethodName = $"{className}.{methodName}";
                
                var methodComplexity = complexityMetrics
                    .FirstOrDefault(m => m.Target == fullMethodName && m.TargetType == TargetType.Method);
                
                var methodVolume = volumeMetrics
                    .FirstOrDefault(m => m.Target == fullMethodName && m.TargetType == TargetType.Method);
                
                if (methodComplexity != null && methodVolume != null)
                {
                    var linesOfCode = CountLinesOfCode(method);
                    var commentPercentage = CalculateCommentPercentage(method);
                    
                    var metric = CreateMaintainabilityMetric(
                        methodVolume.Volume,
                        methodComplexity.Value,
                        linesOfCode,
                        commentPercentage,
                        filePath,
                        "C#",
                        fullMethodName,
                        TargetType.Method);
                    
                    metrics.Add(metric);
                }
            }
            
            // Analyze classes
            var classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
            foreach (var classDecl in classDeclarations)
            {
                var className = classDecl.Identifier.Text;
                var namespaceName = GetNamespace(classDecl);
                var fullClassName = string.IsNullOrEmpty(namespaceName) ? className : $"{namespaceName}.{className}";
                
                var classComplexity = complexityMetrics
                    .FirstOrDefault(m => m.Target == fullClassName && m.TargetType == TargetType.Class);
                
                var classVolume = volumeMetrics
                    .FirstOrDefault(m => m.Target == fullClassName && m.TargetType == TargetType.Class);
                
                if (classComplexity != null && classVolume != null)
                {
                    var linesOfCode = CountLinesOfCode(classDecl);
                    var commentPercentage = CalculateCommentPercentage(classDecl);
                    
                    var metric = CreateMaintainabilityMetric(
                        classVolume.Volume,
                        classComplexity.Value,
                        linesOfCode,
                        commentPercentage,
                        filePath,
                        "C#",
                        fullClassName,
                        TargetType.Class);
                    
                    metrics.Add(metric);
                }
            }
            
            // Analyze file
            var fileName = Path.GetFileName(filePath);
            
            var fileComplexity = complexityMetrics
                .FirstOrDefault(m => m.TargetType == TargetType.File);
            
            var fileVolume = volumeMetrics
                .FirstOrDefault(m => m.TargetType == TargetType.File);
            
            if (fileComplexity != null && fileVolume != null)
            {
                var linesOfCode = CountLinesOfCode(root);
                var commentPercentage = CalculateCommentPercentage(root);
                
                var metric = CreateMaintainabilityMetric(
                    fileVolume.Volume,
                    fileComplexity.Value,
                    linesOfCode,
                    commentPercentage,
                    filePath,
                    "C#",
                    fileName,
                    TargetType.File);
                
                metrics.Add(metric);
            }
            
            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing maintainability index for file {FilePath}", filePath);
            return new List<MaintainabilityMetric>();
        }
    }
    
    /// <summary>
    /// Gets maintainability index thresholds
    /// </summary>
    /// <returns>Threshold values for maintainability index</returns>
    public Dictionary<string, double> GetMaintainabilityThresholds()
    {
        if (_thresholds.TryGetValue("C#", out var thresholds))
        {
            return thresholds;
        }
        
        return new Dictionary<string, double>();
    }
    
    /// <summary>
    /// Sets maintainability index threshold for a specific target type
    /// </summary>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <param name="threshold">Threshold value</param>
    /// <returns>True if threshold was set successfully</returns>
    public bool SetMaintainabilityThreshold(string targetType, double threshold)
    {
        try
        {
            if (!_thresholds.ContainsKey("C#"))
            {
                _thresholds["C#"] = new Dictionary<string, double>();
            }
            
            _thresholds["C#"][targetType] = threshold;
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting maintainability threshold for {TargetType}", targetType);
            return false;
        }
    }
    
    /// <summary>
    /// Creates a maintainability index metric
    /// </summary>
    /// <param name="halsteadVolume">Halstead volume</param>
    /// <param name="cyclomaticComplexity">Cyclomatic complexity</param>
    /// <param name="linesOfCode">Lines of code</param>
    /// <param name="commentPercentage">Comment percentage</param>
    /// <param name="filePath">File path</param>
    /// <param name="language">Programming language</param>
    /// <param name="target">Target (method, class, etc.)</param>
    /// <param name="targetType">Type of target</param>
    /// <returns>Maintainability index metric</returns>
    private MaintainabilityMetric CreateMaintainabilityMetric(
        double halsteadVolume,
        double cyclomaticComplexity,
        int linesOfCode,
        double commentPercentage,
        string filePath,
        string language,
        string target,
        TargetType targetType)
    {
        var metric = new MaintainabilityMetric
        {
            Name = $"Maintainability Index - {target}",
            Description = $"Maintainability index for {targetType} {target}",
            HalsteadVolume = halsteadVolume,
            CyclomaticComplexity = cyclomaticComplexity,
            LinesOfCode = linesOfCode,
            CommentPercentage = commentPercentage,
            FilePath = filePath,
            Language = language,
            Target = target,
            TargetType = targetType,
            Timestamp = DateTime.UtcNow,
            UseMicrosoftFormula = true,
            ThresholdValue = GetThreshold(targetType.ToString())
        };
        
        // Set the value to the calculated maintainability index
        metric.Value = metric.MaintainabilityIndex;
        
        return metric;
    }
    
    /// <summary>
    /// Counts the lines of code in a syntax node
    /// </summary>
    /// <param name="node">Syntax node</param>
    /// <returns>Lines of code</returns>
    private int CountLinesOfCode(SyntaxNode node)
    {
        var text = node.GetText().ToString();
        var lines = text.Split('\n');
        
        // Count non-empty, non-comment lines
        return lines.Count(line => 
            !string.IsNullOrWhiteSpace(line) && 
            !line.TrimStart().StartsWith("//") &&
            !line.TrimStart().StartsWith("/*") &&
            !line.TrimStart().StartsWith("*"));
    }
    
    /// <summary>
    /// Calculates the comment percentage in a syntax node
    /// </summary>
    /// <param name="node">Syntax node</param>
    /// <returns>Comment percentage (0-100)</returns>
    private double CalculateCommentPercentage(SyntaxNode node)
    {
        var text = node.GetText().ToString();
        var lines = text.Split('\n');
        
        var totalLines = lines.Length;
        var commentLines = lines.Count(line => 
            !string.IsNullOrWhiteSpace(line) && 
            (line.TrimStart().StartsWith("//") ||
             line.TrimStart().StartsWith("/*") ||
             line.TrimStart().StartsWith("*")));
        
        return totalLines > 0 ? (double)commentLines / totalLines * 100 : 0;
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
    /// Gets the threshold value for a specific target type
    /// </summary>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <returns>Threshold value</returns>
    private double GetThreshold(string targetType)
    {
        if (_thresholds.TryGetValue("C#", out var thresholds) &&
            thresholds.TryGetValue(targetType, out var threshold))
        {
            return threshold;
        }
        
        // Default thresholds if not configured
        return targetType switch
        {
            "Method" => 60,
            "Class" => 50,
            "File" => 40,
            _ => 50
        };
    }
}
