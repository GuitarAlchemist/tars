using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Analyzer for C# code readability
/// </summary>
public class CSharpReadabilityAnalyzer : IReadabilityAnalyzer
{
    private readonly ILogger<CSharpReadabilityAnalyzer> _logger;
    private readonly Dictionary<string, Dictionary<ReadabilityType, Dictionary<string, double>>> _thresholds;

    // List of common poor identifier names
    private static readonly HashSet<string> PoorIdentifierNames =
    [
        "temp", "tmp", "foo", "bar", "baz", "x", "y", "z", "a", "b", "c", "i", "j", "k",
        "val", "value", "obj", "object", "str", "string", "num", "number", "var", "variable",
        "param", "parameter", "arg", "argument", "item", "element", "data", "result", "ret"
    ];

    // Exceptions for single-letter variables that are commonly used
    private static readonly HashSet<string> SingleLetterExceptions = ["i", "j", "k", "x", "y", "z", "t"];

    /// <summary>
    /// Initializes a new instance of the <see cref="CSharpReadabilityAnalyzer"/> class
    /// </summary>
    /// <param name="logger">Logger</param>
    public CSharpReadabilityAnalyzer(ILogger<CSharpReadabilityAnalyzer> logger)
    {
        _logger = logger;
        _thresholds = new Dictionary<string, Dictionary<ReadabilityType, Dictionary<string, double>>>
        {
            ["C#"] = new()
            {
                [ReadabilityType.IdentifierQuality] = new Dictionary<string, double>
                {
                    ["Method"] = 70,
                    ["Class"] = 70,
                    ["File"] = 70
                },
                [ReadabilityType.CommentQuality] = new Dictionary<string, double>
                {
                    ["Method"] = 60,
                    ["Class"] = 70,
                    ["File"] = 65
                },
                [ReadabilityType.CodeStructure] = new Dictionary<string, double>
                {
                    ["Method"] = 65,
                    ["Class"] = 65,
                    ["File"] = 65
                },
                [ReadabilityType.Overall] = new Dictionary<string, double>
                {
                    ["Method"] = 65,
                    ["Class"] = 65,
                    ["File"] = 65
                }
            }
        };
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeIdentifierQualityAsync(string filePath, string language)
    {
        try
        {
            if (language != "C#")
            {
                _logger.LogWarning("Language {Language} not supported by CSharpReadabilityAnalyzer", language);
                return [];
            }

            var sourceCode = await File.ReadAllTextAsync(filePath);
            var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
            var root = await syntaxTree.GetRootAsync();

            var metrics = new List<ReadabilityMetric>();

            // Analyze methods
            var methodDeclarations = root.DescendantNodes().OfType<MethodDeclarationSyntax>();
            foreach (var method in methodDeclarations)
            {
                var className = GetClassName(method);
                var methodName = method.Identifier.Text;
                var fullMethodName = $"{className}.{methodName}";

                var metric = AnalyzeMethodIdentifierQuality(method, filePath, language, fullMethodName);
                metrics.Add(metric);
            }

            // Analyze classes
            var classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
            foreach (var classDecl in classDeclarations)
            {
                var className = classDecl.Identifier.Text;
                var namespaceName = GetNamespace(classDecl);
                var fullClassName = string.IsNullOrEmpty(namespaceName) ? className : $"{namespaceName}.{className}";

                var metric = AnalyzeClassIdentifierQuality(classDecl, filePath, language, fullClassName);
                metrics.Add(metric);
            }

            // Analyze file
            var fileName = Path.GetFileName(filePath);
            var fileMetric = AnalyzeFileIdentifierQuality(root, filePath, language, fileName);
            metrics.Add(fileMetric);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing identifier quality for file {FilePath}", filePath);
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeCommentQualityAsync(string filePath, string language)
    {
        try
        {
            if (language != "C#")
            {
                _logger.LogWarning("Language {Language} not supported by CSharpReadabilityAnalyzer", language);
                return [];
            }

            var sourceCode = await File.ReadAllTextAsync(filePath);
            var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
            var root = await syntaxTree.GetRootAsync();

            var metrics = new List<ReadabilityMetric>();

            // Analyze methods
            var methodDeclarations = root.DescendantNodes().OfType<MethodDeclarationSyntax>();
            foreach (var method in methodDeclarations)
            {
                var className = GetClassName(method);
                var methodName = method.Identifier.Text;
                var fullMethodName = $"{className}.{methodName}";

                var metric = AnalyzeMethodCommentQuality(method, filePath, language, fullMethodName);
                metrics.Add(metric);
            }

            // Analyze classes
            var classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
            foreach (var classDecl in classDeclarations)
            {
                var className = classDecl.Identifier.Text;
                var namespaceName = GetNamespace(classDecl);
                var fullClassName = string.IsNullOrEmpty(namespaceName) ? className : $"{namespaceName}.{className}";

                var metric = AnalyzeClassCommentQuality(classDecl, filePath, language, fullClassName);
                metrics.Add(metric);
            }

            // Analyze file
            var fileName = Path.GetFileName(filePath);
            var fileMetric = AnalyzeFileCommentQuality(root, filePath, language, fileName);
            metrics.Add(fileMetric);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing comment quality for file {FilePath}", filePath);
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeCodeStructureAsync(string filePath, string language)
    {
        try
        {
            if (language != "C#")
            {
                _logger.LogWarning("Language {Language} not supported by CSharpReadabilityAnalyzer", language);
                return [];
            }

            var sourceCode = await File.ReadAllTextAsync(filePath);
            var syntaxTree = CSharpSyntaxTree.ParseText(sourceCode);
            var root = await syntaxTree.GetRootAsync();

            var metrics = new List<ReadabilityMetric>();

            // Analyze methods
            var methodDeclarations = root.DescendantNodes().OfType<MethodDeclarationSyntax>();
            foreach (var method in methodDeclarations)
            {
                var className = GetClassName(method);
                var methodName = method.Identifier.Text;
                var fullMethodName = $"{className}.{methodName}";

                var metric = AnalyzeMethodCodeStructure(method, filePath, language, fullMethodName);
                metrics.Add(metric);
            }

            // Analyze classes
            var classDeclarations = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
            foreach (var classDecl in classDeclarations)
            {
                var className = classDecl.Identifier.Text;
                var namespaceName = GetNamespace(classDecl);
                var fullClassName = string.IsNullOrEmpty(namespaceName) ? className : $"{namespaceName}.{className}";

                var metric = AnalyzeClassCodeStructure(classDecl, filePath, language, fullClassName);
                metrics.Add(metric);
            }

            // Analyze file
            var fileName = Path.GetFileName(filePath);
            var fileMetric = AnalyzeFileCodeStructure(root, filePath, language, fileName);
            metrics.Add(fileMetric);

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing code structure for file {FilePath}", filePath);
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeOverallReadabilityAsync(string filePath, string language)
    {
        try
        {
            if (language != "C#")
            {
                _logger.LogWarning("Language {Language} not supported by CSharpReadabilityAnalyzer", language);
                return [];
            }

            // Get all metrics first
            var identifierQualityMetrics = await AnalyzeIdentifierQualityAsync(filePath, language);
            var commentQualityMetrics = await AnalyzeCommentQualityAsync(filePath, language);
            var codeStructureMetrics = await AnalyzeCodeStructureAsync(filePath, language);

            var metrics = new List<ReadabilityMetric>();

            // Calculate overall readability for methods
            var methodIdentifierMetrics = identifierQualityMetrics.Where(m => m.TargetType == TargetType.Method).ToList();
            var methodCommentMetrics = commentQualityMetrics.Where(m => m.TargetType == TargetType.Method).ToList();
            var methodStructureMetrics = codeStructureMetrics.Where(m => m.TargetType == TargetType.Method).ToList();

            foreach (var methodIdentifierMetric in methodIdentifierMetrics)
            {
                var methodName = methodIdentifierMetric.Target;
                var methodCommentMetric = methodCommentMetrics.FirstOrDefault(m => m.Target == methodName);
                var methodStructureMetric = methodStructureMetrics.FirstOrDefault(m => m.Target == methodName);

                if (methodCommentMetric != null && methodStructureMetric != null)
                {
                    var overallMetric = new ReadabilityMetric
                    {
                        Name = $"Overall Readability - {methodName}",
                        Description = $"Overall readability for method {methodName}",
                        Type = ReadabilityType.Overall,
                        FilePath = filePath,
                        Language = language,
                        Target = methodName,
                        TargetType = TargetType.Method,
                        AverageIdentifierLength = methodIdentifierMetric.AverageIdentifierLength,
                        PoorlyNamedIdentifierCount = methodIdentifierMetric.PoorlyNamedIdentifierCount,
                        CommentPercentage = methodCommentMetric.CommentPercentage,
                        LinesOfCode = methodStructureMetric.LinesOfCode,
                        AverageLineLength = methodStructureMetric.AverageLineLength,
                        MaxNestingDepth = methodStructureMetric.MaxNestingDepth,
                        AverageNestingDepth = methodStructureMetric.AverageNestingDepth,
                        LongMethodCount = methodStructureMetric.LongMethodCount,
                        LongLineCount = methodStructureMetric.LongLineCount,
                        ComplexExpressionCount = methodStructureMetric.ComplexExpressionCount,
                        MagicNumberCount = methodStructureMetric.MagicNumberCount,
                        Timestamp = DateTime.UtcNow,
                        ThresholdValue = GetThreshold(language, ReadabilityType.Overall, "Method")
                    };

                    // Calculate the value (readability score)
                    overallMetric.Value = overallMetric.ReadabilityScore;

                    metrics.Add(overallMetric);
                }
            }

            // Calculate overall readability for classes
            var classIdentifierMetrics = identifierQualityMetrics.Where(m => m.TargetType == TargetType.Class).ToList();
            var classCommentMetrics = commentQualityMetrics.Where(m => m.TargetType == TargetType.Class).ToList();
            var classStructureMetrics = codeStructureMetrics.Where(m => m.TargetType == TargetType.Class).ToList();

            foreach (var classIdentifierMetric in classIdentifierMetrics)
            {
                var className = classIdentifierMetric.Target;
                var classCommentMetric = classCommentMetrics.FirstOrDefault(m => m.Target == className);
                var classStructureMetric = classStructureMetrics.FirstOrDefault(m => m.Target == className);

                if (classCommentMetric != null && classStructureMetric != null)
                {
                    var overallMetric = new ReadabilityMetric
                    {
                        Name = $"Overall Readability - {className}",
                        Description = $"Overall readability for class {className}",
                        Type = ReadabilityType.Overall,
                        FilePath = filePath,
                        Language = language,
                        Target = className,
                        TargetType = TargetType.Class,
                        AverageIdentifierLength = classIdentifierMetric.AverageIdentifierLength,
                        PoorlyNamedIdentifierCount = classIdentifierMetric.PoorlyNamedIdentifierCount,
                        CommentPercentage = classCommentMetric.CommentPercentage,
                        LinesOfCode = classStructureMetric.LinesOfCode,
                        AverageLineLength = classStructureMetric.AverageLineLength,
                        MaxNestingDepth = classStructureMetric.MaxNestingDepth,
                        AverageNestingDepth = classStructureMetric.AverageNestingDepth,
                        LongMethodCount = classStructureMetric.LongMethodCount,
                        LongLineCount = classStructureMetric.LongLineCount,
                        ComplexExpressionCount = classStructureMetric.ComplexExpressionCount,
                        MagicNumberCount = classStructureMetric.MagicNumberCount,
                        Timestamp = DateTime.UtcNow,
                        ThresholdValue = GetThreshold(language, ReadabilityType.Overall, "Class")
                    };

                    // Calculate the value (readability score)
                    overallMetric.Value = overallMetric.ReadabilityScore;

                    metrics.Add(overallMetric);
                }
            }

            // Calculate overall readability for the file
            var fileIdentifierMetrics = identifierQualityMetrics.Where(m => m.TargetType == TargetType.File).ToList();
            var fileCommentMetrics = commentQualityMetrics.Where(m => m.TargetType == TargetType.File).ToList();
            var fileStructureMetrics = codeStructureMetrics.Where(m => m.TargetType == TargetType.File).ToList();

            if (fileIdentifierMetrics.Any() && fileCommentMetrics.Any() && fileStructureMetrics.Any())
            {
                var fileIdentifierMetric = fileIdentifierMetrics.First();
                var fileCommentMetric = fileCommentMetrics.First();
                var fileStructureMetric = fileStructureMetrics.First();
                var fileName = Path.GetFileName(filePath);

                var overallMetric = new ReadabilityMetric
                {
                    Name = $"Overall Readability - {fileName}",
                    Description = $"Overall readability for file {fileName}",
                    Type = ReadabilityType.Overall,
                    FilePath = filePath,
                    Language = language,
                    Target = fileName,
                    TargetType = TargetType.File,
                    AverageIdentifierLength = fileIdentifierMetric.AverageIdentifierLength,
                    PoorlyNamedIdentifierCount = fileIdentifierMetric.PoorlyNamedIdentifierCount,
                    CommentPercentage = fileCommentMetric.CommentPercentage,
                    LinesOfCode = fileStructureMetric.LinesOfCode,
                    AverageLineLength = fileStructureMetric.AverageLineLength,
                    MaxNestingDepth = fileStructureMetric.MaxNestingDepth,
                    AverageNestingDepth = fileStructureMetric.AverageNestingDepth,
                    LongMethodCount = fileStructureMetric.LongMethodCount,
                    LongLineCount = fileStructureMetric.LongLineCount,
                    ComplexExpressionCount = fileStructureMetric.ComplexExpressionCount,
                    MagicNumberCount = fileStructureMetric.MagicNumberCount,
                    Timestamp = DateTime.UtcNow,
                    ThresholdValue = GetThreshold(language, ReadabilityType.Overall, "File")
                };

                // Calculate the value (readability score)
                overallMetric.Value = overallMetric.ReadabilityScore;

                metrics.Add(overallMetric);
            }

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing overall readability for file {FilePath}", filePath);
            return [];
        }
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeAllReadabilityMetricsAsync(string filePath, string language)
    {
        var metrics = new List<ReadabilityMetric>();

        metrics.AddRange(await AnalyzeIdentifierQualityAsync(filePath, language));
        metrics.AddRange(await AnalyzeCommentQualityAsync(filePath, language));
        metrics.AddRange(await AnalyzeCodeStructureAsync(filePath, language));
        metrics.AddRange(await AnalyzeOverallReadabilityAsync(filePath, language));

        return metrics;
    }

    /// <inheritdoc/>
    public async Task<List<ReadabilityMetric>> AnalyzeProjectReadabilityAsync(string projectPath)
    {
        try
        {
            var metrics = new List<ReadabilityMetric>();

            var csharpFiles = Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories);

            foreach (var file in csharpFiles)
            {
                metrics.AddRange(await AnalyzeAllReadabilityMetricsAsync(file, "C#"));
            }

            // Calculate project-level metrics
            var projectName = Path.GetFileName(projectPath);

            // Calculate average identifier quality
            var identifierQualityMetrics = metrics.Where(m => m.Type == ReadabilityType.IdentifierQuality).ToList();
            if (identifierQualityMetrics.Any())
            {
                var averageIdentifierQuality = identifierQualityMetrics.Average(m => m.Value);
                var projectIdentifierQualityMetric = new ReadabilityMetric
                {
                    Name = $"Identifier Quality - {projectName}",
                    Description = $"Average identifier quality for project {projectName}",
                    Value = averageIdentifierQuality,
                    Type = ReadabilityType.IdentifierQuality,
                    FilePath = projectPath,
                    Language = "C#",
                    Target = projectName,
                    TargetType = TargetType.Project,
                    Timestamp = DateTime.UtcNow,
                    ThresholdValue = GetThreshold("C#", ReadabilityType.IdentifierQuality, "Project")
                };

                metrics.Add(projectIdentifierQualityMetric);
            }

            return metrics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing project readability for {ProjectPath}", projectPath);
            return [];
        }
    }

    /// <inheritdoc/>
    public Task<Dictionary<string, double>> GetReadabilityThresholdsAsync(string language, ReadabilityType readabilityType)
    {
        if (_thresholds.TryGetValue(language, out var languageThresholds) &&
            languageThresholds.TryGetValue(readabilityType, out var typeThresholds))
        {
            return Task.FromResult(typeThresholds);
        }

        return Task.FromResult(new Dictionary<string, double>());
    }

    /// <inheritdoc/>
    public Task<bool> SetReadabilityThresholdAsync(string language, ReadabilityType readabilityType, string targetType, double threshold)
    {
        try
        {
            if (!_thresholds.ContainsKey(language))
            {
                _thresholds[language] = new Dictionary<ReadabilityType, Dictionary<string, double>>();
            }

            if (!_thresholds[language].ContainsKey(readabilityType))
            {
                _thresholds[language][readabilityType] = new Dictionary<string, double>();
            }

            _thresholds[language][readabilityType][targetType] = threshold;
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error setting readability threshold for {Language}, {ReadabilityType}, {TargetType}",
                language, readabilityType, targetType);
            return Task.FromResult(false);
        }
    }

    /// <summary>
    /// Analyzes identifier quality of a method
    /// </summary>
    /// <param name="method">Method declaration syntax</param>
    /// <param name="filePath">File path</param>
    /// <param name="language">Programming language</param>
    /// <param name="fullMethodName">Full method name</param>
    /// <returns>Readability metric</returns>
    private ReadabilityMetric AnalyzeMethodIdentifierQuality(MethodDeclarationSyntax method, string filePath, string language, string fullMethodName)
    {
        // Check method name
        var methodName = method.Identifier.Text;
        var isPascalCase = IsPascalCase(methodName);

        // Check parameters
        var parameters = method.ParameterList.Parameters;
        var parameterNames = parameters.Select(p => p.Identifier.Text).ToList();
        var camelCaseParameters = parameterNames.Count(IsCamelCase);
        var poorNamedParameters = parameterNames.Count(IsLikelyPoorName);

        // Check local variables
        var localVariables = method.DescendantNodes().OfType<VariableDeclarationSyntax>()
            .SelectMany(v => v.Variables)
            .Select(v => v.Identifier.Text)
            .ToList();

        var camelCaseVariables = localVariables.Count(IsCamelCase);
        var poorNamedVariables = localVariables.Count(IsLikelyPoorName);

        // Calculate average identifier length
        var allIdentifiers = new List<string>(parameterNames);
        allIdentifiers.AddRange(localVariables);
        allIdentifiers.Add(methodName);

        var averageIdentifierLength = allIdentifiers.Any()
            ? allIdentifiers.Average(i => i.Length)
            : 0;

        // Count poorly named identifiers
        var poorlyNamedIdentifierCount = poorNamedParameters + poorNamedVariables + (isPascalCase ? 0 : 1);

        // Create metric
        var metric = new ReadabilityMetric
        {
            Name = $"Identifier Quality - {fullMethodName}",
            Description = $"Identifier naming quality for method {fullMethodName}",
            Type = ReadabilityType.IdentifierQuality,
            FilePath = filePath,
            Language = language,
            Target = fullMethodName,
            TargetType = TargetType.Method,
            AverageIdentifierLength = averageIdentifierLength,
            PoorlyNamedIdentifierCount = poorlyNamedIdentifierCount,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold(language, ReadabilityType.IdentifierQuality, "Method")
        };

        // Calculate the value (readability score)
        metric.Value = metric.ReadabilityScore;

        return metric;
    }

    /// <summary>
    /// Analyzes identifier quality of a class
    /// </summary>
    /// <param name="classDecl">Class declaration syntax</param>
    /// <param name="filePath">File path</param>
    /// <param name="language">Programming language</param>
    /// <param name="fullClassName">Full class name</param>
    /// <returns>Readability metric</returns>
    private ReadabilityMetric AnalyzeClassIdentifierQuality(ClassDeclarationSyntax classDecl, string filePath, string language, string fullClassName)
    {
        // Check class name
        var className = classDecl.Identifier.Text;
        var isPascalCase = IsPascalCase(className);

        // Check properties
        var properties = classDecl.DescendantNodes().OfType<PropertyDeclarationSyntax>();
        var propertyNames = properties.Select(p => p.Identifier.Text).ToList();
        var pascalCaseProperties = propertyNames.Count(IsPascalCase);
        var poorNamedProperties = propertyNames.Count(IsLikelyPoorName);

        // Check fields
        var fields = classDecl.DescendantNodes().OfType<FieldDeclarationSyntax>()
            .SelectMany(f => f.Declaration.Variables)
            .Select(v => v.Identifier.Text)
            .ToList();

        var camelCaseFields = fields.Count(IsCamelCase);
        var poorNamedFields = fields.Count(IsLikelyPoorName);

        // Calculate average identifier length
        var allIdentifiers = new List<string>(propertyNames);
        allIdentifiers.AddRange(fields);
        allIdentifiers.Add(className);

        var averageIdentifierLength = allIdentifiers.Any()
            ? allIdentifiers.Average(i => i.Length)
            : 0;

        // Count poorly named identifiers
        var poorlyNamedIdentifierCount = poorNamedProperties + poorNamedFields + (isPascalCase ? 0 : 1);

        // Create metric
        var metric = new ReadabilityMetric
        {
            Name = $"Identifier Quality - {fullClassName}",
            Description = $"Identifier naming quality for class {fullClassName}",
            Type = ReadabilityType.IdentifierQuality,
            FilePath = filePath,
            Language = language,
            Target = fullClassName,
            TargetType = TargetType.Class,
            AverageIdentifierLength = averageIdentifierLength,
            PoorlyNamedIdentifierCount = poorlyNamedIdentifierCount,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold(language, ReadabilityType.IdentifierQuality, "Class")
        };

        // Calculate the value (readability score)
        metric.Value = metric.ReadabilityScore;

        return metric;
    }

    /// <summary>
    /// Analyzes identifier quality of a file
    /// </summary>
    /// <param name="root">Syntax tree root</param>
    /// <param name="filePath">File path</param>
    /// <param name="language">Programming language</param>
    /// <param name="fileName">File name</param>
    /// <returns>Readability metric</returns>
    private ReadabilityMetric AnalyzeFileIdentifierQuality(SyntaxNode root, string filePath, string language, string fileName)
    {
        // Get all identifiers in the file
        var identifiers = new List<string>();

        // Class names
        var classNames = root.DescendantNodes().OfType<ClassDeclarationSyntax>()
            .Select(c => c.Identifier.Text);
        identifiers.AddRange(classNames);

        // Interface names
        var interfaceNames = root.DescendantNodes().OfType<InterfaceDeclarationSyntax>()
            .Select(i => i.Identifier.Text);
        identifiers.AddRange(interfaceNames);

        // Enum names
        var enumNames = root.DescendantNodes().OfType<EnumDeclarationSyntax>()
            .Select(e => e.Identifier.Text);
        identifiers.AddRange(enumNames);

        // Method names
        var methodNames = root.DescendantNodes().OfType<MethodDeclarationSyntax>()
            .Select(m => m.Identifier.Text);
        identifiers.AddRange(methodNames);

        // Property names
        var propertyNames = root.DescendantNodes().OfType<PropertyDeclarationSyntax>()
            .Select(p => p.Identifier.Text);
        identifiers.AddRange(propertyNames);

        // Field names
        var fieldNames = root.DescendantNodes().OfType<FieldDeclarationSyntax>()
            .SelectMany(f => f.Declaration.Variables)
            .Select(v => v.Identifier.Text);
        identifiers.AddRange(fieldNames);

        // Parameter names
        var parameterNames = root.DescendantNodes().OfType<ParameterSyntax>()
            .Select(p => p.Identifier.Text);
        identifiers.AddRange(parameterNames);

        // Local variable names
        var localVariableNames = root.DescendantNodes().OfType<VariableDeclarationSyntax>()
            .SelectMany(v => v.Variables)
            .Select(v => v.Identifier.Text);
        identifiers.AddRange(localVariableNames);

        // Calculate average identifier length
        var averageIdentifierLength = identifiers.Any()
            ? identifiers.Average(i => i.Length)
            : 0;

        // Count poorly named identifiers
        var poorlyNamedIdentifierCount = identifiers.Count(IsLikelyPoorName);

        // Create metric
        var metric = new ReadabilityMetric
        {
            Name = $"Identifier Quality - {fileName}",
            Description = $"Identifier naming quality for file {fileName}",
            Type = ReadabilityType.IdentifierQuality,
            FilePath = filePath,
            Language = language,
            Target = fileName,
            TargetType = TargetType.File,
            AverageIdentifierLength = averageIdentifierLength,
            PoorlyNamedIdentifierCount = poorlyNamedIdentifierCount,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold(language, ReadabilityType.IdentifierQuality, "File")
        };

        // Calculate the value (readability score)
        metric.Value = metric.ReadabilityScore;

        return metric;
    }

    /// <summary>
    /// Checks if a string is in PascalCase
    /// </summary>
    /// <param name="text">Text to check</param>
    /// <returns>True if the text is in PascalCase</returns>
    private bool IsPascalCase(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return false;
        }

        // PascalCase starts with an uppercase letter
        if (!char.IsUpper(text[0]))
        {
            return false;
        }

        // Check for underscores or hyphens (not allowed in PascalCase)
        if (text.Contains('_') || text.Contains('-'))
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Checks if a string is in camelCase
    /// </summary>
    /// <param name="text">Text to check</param>
    /// <returns>True if the text is in camelCase</returns>
    private bool IsCamelCase(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return false;
        }

        // camelCase starts with a lowercase letter
        if (!char.IsLower(text[0]))
        {
            return false;
        }

        // Check for underscores or hyphens (not allowed in camelCase)
        if (text.Contains('_') || text.Contains('-'))
        {
            return false;
        }

        return true;
    }

    /// <summary>
    /// Checks if a string is likely a poor name
    /// </summary>
    /// <param name="text">Text to check</param>
    /// <returns>True if the text is likely a poor name</returns>
    private bool IsLikelyPoorName(string text)
    {
        if (string.IsNullOrEmpty(text))
        {
            return true;
        }

        // Check if it's in the list of poor names
        if (PoorIdentifierNames.Contains(text.ToLowerInvariant()))
        {
            return true;
        }

        // Check if it's a single letter (except for common cases)
        if (text.Length == 1 && !SingleLetterExceptions.Contains(text))
        {
            return true;
        }

        // Check if it's too short (less than 3 characters)
        if (text.Length < 3 && !SingleLetterExceptions.Contains(text))
        {
            return true;
        }

        return false;
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
    /// Analyzes comment quality of a method
    /// </summary>
    /// <param name="method">Method declaration syntax</param>
    /// <param name="filePath">File path</param>
    /// <param name="language">Programming language</param>
    /// <param name="fullMethodName">Full method name</param>
    /// <returns>Readability metric</returns>
    private ReadabilityMetric AnalyzeMethodCommentQuality(MethodDeclarationSyntax method, string filePath, string language, string fullMethodName)
    {
        // Get method text including trivia (comments)
        var methodText = method.ToFullString();
        var methodLines = methodText.Split('\n');
        var totalLines = methodLines.Length;

        // Count comment lines
        var commentLines = methodLines.Count(line =>
            line.TrimStart().StartsWith("//") ||
            line.TrimStart().StartsWith("/*") ||
            line.TrimStart().StartsWith("*"));

        // Calculate comment percentage
        var commentPercentage = totalLines > 0 ? (double)commentLines / totalLines * 100 : 0;

        // Check for XML documentation
        var hasXmlDoc = method.HasLeadingTrivia &&
                       method.GetLeadingTrivia().ToString().Contains("<summary>");

        // Create metric
        var metric = new ReadabilityMetric
        {
            Name = $"Comment Quality - {fullMethodName}",
            Description = $"Comment quality for method {fullMethodName}",
            Type = ReadabilityType.CommentQuality,
            FilePath = filePath,
            Language = language,
            Target = fullMethodName,
            TargetType = TargetType.Method,
            CommentPercentage = commentPercentage,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold(language, ReadabilityType.CommentQuality, "Method")
        };

        // Calculate the value (readability score)
        metric.Value = metric.ReadabilityScore;

        return metric;
    }

    /// <summary>
    /// Analyzes comment quality of a class
    /// </summary>
    /// <param name="classDecl">Class declaration syntax</param>
    /// <param name="filePath">File path</param>
    /// <param name="language">Programming language</param>
    /// <param name="fullClassName">Full class name</param>
    /// <returns>Readability metric</returns>
    private ReadabilityMetric AnalyzeClassCommentQuality(ClassDeclarationSyntax classDecl, string filePath, string language, string fullClassName)
    {
        // Get class text including trivia (comments)
        var classText = classDecl.ToFullString();
        var classLines = classText.Split('\n');
        var totalLines = classLines.Length;

        // Count comment lines
        var commentLines = classLines.Count(line =>
            line.TrimStart().StartsWith("//") ||
            line.TrimStart().StartsWith("/*") ||
            line.TrimStart().StartsWith("*"));

        // Calculate comment percentage
        var commentPercentage = totalLines > 0 ? (double)commentLines / totalLines * 100 : 0;

        // Check for XML documentation
        var hasXmlDoc = classDecl.HasLeadingTrivia &&
                       classDecl.GetLeadingTrivia().ToString().Contains("<summary>");

        // Check if methods have XML documentation
        var methods = classDecl.DescendantNodes().OfType<MethodDeclarationSyntax>().ToList();
        var methodsWithXmlDoc = methods.Count(m =>
            m.HasLeadingTrivia && m.GetLeadingTrivia().ToString().Contains("<summary>"));
        var methodXmlDocPercentage = methods.Any() ? (double)methodsWithXmlDoc / methods.Count * 100 : 0;

        // Create metric
        var metric = new ReadabilityMetric
        {
            Name = $"Comment Quality - {fullClassName}",
            Description = $"Comment quality for class {fullClassName}",
            Type = ReadabilityType.CommentQuality,
            FilePath = filePath,
            Language = language,
            Target = fullClassName,
            TargetType = TargetType.Class,
            CommentPercentage = commentPercentage,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold(language, ReadabilityType.CommentQuality, "Class")
        };

        // Calculate the value (readability score)
        metric.Value = metric.ReadabilityScore;

        return metric;
    }

    /// <summary>
    /// Analyzes comment quality of a file
    /// </summary>
    /// <param name="root">Syntax tree root</param>
    /// <param name="filePath">File path</param>
    /// <param name="language">Programming language</param>
    /// <param name="fileName">File name</param>
    /// <returns>Readability metric</returns>
    private ReadabilityMetric AnalyzeFileCommentQuality(SyntaxNode root, string filePath, string language, string fileName)
    {
        // Get file text including trivia (comments)
        var fileText = root.ToFullString();
        var fileLines = fileText.Split('\n');
        var totalLines = fileLines.Length;

        // Count comment lines
        var commentLines = fileLines.Count(line =>
            line.TrimStart().StartsWith("//") ||
            line.TrimStart().StartsWith("/*") ||
            line.TrimStart().StartsWith("*"));

        // Calculate comment percentage
        var commentPercentage = totalLines > 0 ? (double)commentLines / totalLines * 100 : 0;

        // Check if classes have XML documentation
        var classes = root.DescendantNodes().OfType<ClassDeclarationSyntax>().ToList();
        var classesWithXmlDoc = classes.Count(c =>
            c.HasLeadingTrivia && c.GetLeadingTrivia().ToString().Contains("<summary>"));
        var classXmlDocPercentage = classes.Any() ? (double)classesWithXmlDoc / classes.Count * 100 : 0;

        // Check if methods have XML documentation
        var methods = root.DescendantNodes().OfType<MethodDeclarationSyntax>().ToList();
        var methodsWithXmlDoc = methods.Count(m =>
            m.HasLeadingTrivia && m.GetLeadingTrivia().ToString().Contains("<summary>"));
        var methodXmlDocPercentage = methods.Any() ? (double)methodsWithXmlDoc / methods.Count * 100 : 0;

        // Create metric
        var metric = new ReadabilityMetric
        {
            Name = $"Comment Quality - {fileName}",
            Description = $"Comment quality for file {fileName}",
            Type = ReadabilityType.CommentQuality,
            FilePath = filePath,
            Language = language,
            Target = fileName,
            TargetType = TargetType.File,
            CommentPercentage = commentPercentage,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold(language, ReadabilityType.CommentQuality, "File")
        };

        // Calculate the value (readability score)
        metric.Value = metric.ReadabilityScore;

        return metric;
    }

    /// <summary>
    /// Analyzes code structure of a method
    /// </summary>
    /// <param name="method">Method declaration syntax</param>
    /// <param name="filePath">File path</param>
    /// <param name="language">Programming language</param>
    /// <param name="fullMethodName">Full method name</param>
    /// <returns>Readability metric</returns>
    private ReadabilityMetric AnalyzeMethodCodeStructure(MethodDeclarationSyntax method, string filePath, string language, string fullMethodName)
    {
        // Get method text
        var methodText = method.ToFullString();
        var methodLines = methodText.Split('\n');
        var linesOfCode = methodLines.Length;

        // Check if method is too long
        var isLongMethod = linesOfCode > 30;

        // Calculate average line length
        var averageLineLength = methodLines.Any()
            ? methodLines.Average(line => line.Length)
            : 0;

        // Count long lines (> 100 characters)
        var longLineCount = methodLines.Count(line => line.Length > 100);

        // Calculate nesting depth
        var maxNestingDepth = CalculateMaxNestingDepth(method);
        var averageNestingDepth = CalculateAverageNestingDepth(method);

        // Count complex expressions
        var complexExpressionCount = CountComplexExpressions(method);

        // Count magic numbers
        var magicNumberCount = CountMagicNumbers(method);

        // Create metric
        var metric = new ReadabilityMetric
        {
            Name = $"Code Structure - {fullMethodName}",
            Description = $"Code structure clarity for method {fullMethodName}",
            Type = ReadabilityType.CodeStructure,
            FilePath = filePath,
            Language = language,
            Target = fullMethodName,
            TargetType = TargetType.Method,
            LinesOfCode = linesOfCode,
            AverageLineLength = averageLineLength,
            MaxNestingDepth = maxNestingDepth,
            AverageNestingDepth = averageNestingDepth,
            LongMethodCount = isLongMethod ? 1 : 0,
            LongLineCount = longLineCount,
            ComplexExpressionCount = complexExpressionCount,
            MagicNumberCount = magicNumberCount,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold(language, ReadabilityType.CodeStructure, "Method")
        };

        // Calculate the value (readability score)
        metric.Value = metric.ReadabilityScore;

        return metric;
    }

    /// <summary>
    /// Analyzes code structure of a class
    /// </summary>
    /// <param name="classDecl">Class declaration syntax</param>
    /// <param name="filePath">File path</param>
    /// <param name="language">Programming language</param>
    /// <param name="fullClassName">Full class name</param>
    /// <returns>Readability metric</returns>
    private ReadabilityMetric AnalyzeClassCodeStructure(ClassDeclarationSyntax classDecl, string filePath, string language, string fullClassName)
    {
        // Get class text
        var classText = classDecl.ToFullString();
        var classLines = classText.Split('\n');
        var linesOfCode = classLines.Length;

        // Calculate average line length
        var averageLineLength = classLines.Any()
            ? classLines.Average(line => line.Length)
            : 0;

        // Count long lines (> 100 characters)
        var longLineCount = classLines.Count(line => line.Length > 100);

        // Count long methods
        var methods = classDecl.DescendantNodes().OfType<MethodDeclarationSyntax>().ToList();
        var longMethodCount = methods.Count(m => m.ToFullString().Split('\n').Length > 30);

        // Calculate maximum nesting depth across all methods
        var maxNestingDepth = methods.Any()
            ? methods.Max(CalculateMaxNestingDepth)
            : 0;

        // Calculate average nesting depth across all methods
        var averageNestingDepth = methods.Any()
            ? methods.Average(CalculateAverageNestingDepth)
            : 0;

        // Count complex expressions
        var complexExpressionCount = methods.Sum(CountComplexExpressions);

        // Count magic numbers
        var magicNumberCount = methods.Sum(CountMagicNumbers);

        // Create metric
        var metric = new ReadabilityMetric
        {
            Name = $"Code Structure - {fullClassName}",
            Description = $"Code structure clarity for class {fullClassName}",
            Type = ReadabilityType.CodeStructure,
            FilePath = filePath,
            Language = language,
            Target = fullClassName,
            TargetType = TargetType.Class,
            LinesOfCode = linesOfCode,
            AverageLineLength = averageLineLength,
            MaxNestingDepth = maxNestingDepth,
            AverageNestingDepth = averageNestingDepth,
            LongMethodCount = longMethodCount,
            LongLineCount = longLineCount,
            ComplexExpressionCount = complexExpressionCount,
            MagicNumberCount = magicNumberCount,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold(language, ReadabilityType.CodeStructure, "Class")
        };

        // Calculate the value (readability score)
        metric.Value = metric.ReadabilityScore;

        return metric;
    }

    /// <summary>
    /// Analyzes code structure of a file
    /// </summary>
    /// <param name="root">Syntax tree root</param>
    /// <param name="filePath">File path</param>
    /// <param name="language">Programming language</param>
    /// <param name="fileName">File name</param>
    /// <returns>Readability metric</returns>
    private ReadabilityMetric AnalyzeFileCodeStructure(SyntaxNode root, string filePath, string language, string fileName)
    {
        // Get file text
        var fileText = root.ToFullString();
        var fileLines = fileText.Split('\n');
        var linesOfCode = fileLines.Length;

        // Calculate average line length
        var averageLineLength = fileLines.Any()
            ? fileLines.Average(line => line.Length)
            : 0;

        // Count long lines (> 100 characters)
        var longLineCount = fileLines.Count(line => line.Length > 100);

        // Count long methods
        var methods = root.DescendantNodes().OfType<MethodDeclarationSyntax>().ToList();
        var longMethodCount = methods.Count(m => m.ToFullString().Split('\n').Length > 30);

        // Calculate maximum nesting depth across all methods
        var maxNestingDepth = methods.Any()
            ? methods.Max(CalculateMaxNestingDepth)
            : 0;

        // Calculate average nesting depth across all methods
        var averageNestingDepth = methods.Any()
            ? methods.Average(CalculateAverageNestingDepth)
            : 0;

        // Count complex expressions
        var complexExpressionCount = methods.Sum(CountComplexExpressions);

        // Count magic numbers
        var magicNumberCount = methods.Sum(CountMagicNumbers);

        // Create metric
        var metric = new ReadabilityMetric
        {
            Name = $"Code Structure - {fileName}",
            Description = $"Code structure clarity for file {fileName}",
            Type = ReadabilityType.CodeStructure,
            FilePath = filePath,
            Language = language,
            Target = fileName,
            TargetType = TargetType.File,
            LinesOfCode = linesOfCode,
            AverageLineLength = averageLineLength,
            MaxNestingDepth = maxNestingDepth,
            AverageNestingDepth = averageNestingDepth,
            LongMethodCount = longMethodCount,
            LongLineCount = longLineCount,
            ComplexExpressionCount = complexExpressionCount,
            MagicNumberCount = magicNumberCount,
            Timestamp = DateTime.UtcNow,
            ThresholdValue = GetThreshold(language, ReadabilityType.CodeStructure, "File")
        };

        // Calculate the value (readability score)
        metric.Value = metric.ReadabilityScore;

        return metric;
    }

    /// <summary>
    /// Calculates the maximum nesting depth of a method
    /// </summary>
    /// <param name="method">Method declaration syntax</param>
    /// <returns>Maximum nesting depth</returns>
    private int CalculateMaxNestingDepth(MethodDeclarationSyntax method)
    {
        var maxDepth = 0;
        var currentDepth = 0;

        // Walk the syntax tree and track nesting depth
        foreach (var node in method.DescendantNodes())
        {
            if (node is IfStatementSyntax ||
                node is ForStatementSyntax ||
                node is ForEachStatementSyntax ||
                node is WhileStatementSyntax ||
                node is DoStatementSyntax ||
                node is SwitchStatementSyntax ||
                node is TryStatementSyntax)
            {
                currentDepth++;
                maxDepth = Math.Max(maxDepth, currentDepth);
            }
            else if (node is BlockSyntax && node.Parent is not MethodDeclarationSyntax)
            {
                currentDepth--;
            }
        }

        return maxDepth;
    }

    /// <summary>
    /// Calculates the average nesting depth of a method
    /// </summary>
    /// <param name="method">Method declaration syntax</param>
    /// <returns>Average nesting depth</returns>
    private double CalculateAverageNestingDepth(MethodDeclarationSyntax method)
    {
        var nestingDepths = new List<int>();
        var currentDepth = 0;

        // Walk the syntax tree and track nesting depth
        foreach (var node in method.DescendantNodes())
        {
            if (node is IfStatementSyntax ||
                node is ForStatementSyntax ||
                node is ForEachStatementSyntax ||
                node is WhileStatementSyntax ||
                node is DoStatementSyntax ||
                node is SwitchStatementSyntax ||
                node is TryStatementSyntax)
            {
                currentDepth++;
                nestingDepths.Add(currentDepth);
            }
            else if (node is BlockSyntax && node.Parent is not MethodDeclarationSyntax)
            {
                currentDepth--;
            }
        }

        return nestingDepths.Any() ? nestingDepths.Average() : 0;
    }

    /// <summary>
    /// Counts the number of complex expressions in a method
    /// </summary>
    /// <param name="method">Method declaration syntax</param>
    /// <returns>Number of complex expressions</returns>
    private int CountComplexExpressions(MethodDeclarationSyntax method)
    {
        var complexExpressionCount = 0;

        // Find binary expressions with more than 3 operators
        var binaryExpressions = method.DescendantNodes().OfType<BinaryExpressionSyntax>();
        foreach (var expr in binaryExpressions)
        {
            var operatorCount = CountOperators(expr);
            if (operatorCount > 3)
            {
                complexExpressionCount++;
            }
        }

        // Find conditional expressions (ternary operators)
        var conditionalExpressions = method.DescendantNodes().OfType<ConditionalExpressionSyntax>();
        complexExpressionCount += conditionalExpressions.Count();

        return complexExpressionCount;
    }

    /// <summary>
    /// Counts the number of operators in a binary expression
    /// </summary>
    /// <param name="expr">Binary expression syntax</param>
    /// <returns>Number of operators</returns>
    private int CountOperators(BinaryExpressionSyntax expr)
    {
        var count = 1; // Start with 1 for the current operator

        if (expr.Left is BinaryExpressionSyntax leftBinary)
        {
            count += CountOperators(leftBinary);
        }

        if (expr.Right is BinaryExpressionSyntax rightBinary)
        {
            count += CountOperators(rightBinary);
        }

        return count;
    }

    /// <summary>
    /// Counts the number of magic numbers in a method
    /// </summary>
    /// <param name="method">Method declaration syntax</param>
    /// <returns>Number of magic numbers</returns>
    private int CountMagicNumbers(MethodDeclarationSyntax method)
    {
        var magicNumberCount = 0;

        // Find numeric literals
        var numericLiterals = method.DescendantNodes().OfType<LiteralExpressionSyntax>()
            .Where(l => l.Token.Value is int or double or float or decimal);

        foreach (var literal in numericLiterals)
        {
            var value = literal.Token.Value;

            // Skip common non-magic numbers (0, 1, -1, 2, 10, 100)
            if (value is int intValue && (intValue == 0 || intValue == 1 || intValue == -1 || intValue == 2 || intValue == 10 || intValue == 100))
            {
                continue;
            }

            if (value is double doubleValue && (doubleValue == 0 || doubleValue == 1 || doubleValue == -1 || doubleValue == 2 || doubleValue == 10 || doubleValue == 100))
            {
                continue;
            }

            magicNumberCount++;
        }

        return magicNumberCount;
    }

    /// <summary>
    /// Gets the threshold value for a specific readability type and target type
    /// </summary>
    /// <param name="language">Programming language</param>
    /// <param name="readabilityType">Type of readability</param>
    /// <param name="targetType">Type of target (method, class, etc.)</param>
    /// <returns>Threshold value</returns>
    private double GetThreshold(string language, ReadabilityType readabilityType, string targetType)
    {
        if (_thresholds.TryGetValue(language, out var languageThresholds) &&
            languageThresholds.TryGetValue(readabilityType, out var typeThresholds) &&
            typeThresholds.TryGetValue(targetType, out var threshold))
        {
            return threshold;
        }

        // Default thresholds if not configured
        return readabilityType switch
        {
            ReadabilityType.IdentifierQuality => 70,
            ReadabilityType.CommentQuality => 60,
            ReadabilityType.CodeStructure => 65,
            ReadabilityType.Overall => 65,
            _ => 65
        };
    }
}
