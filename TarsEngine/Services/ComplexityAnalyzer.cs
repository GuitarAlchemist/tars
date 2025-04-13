using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for analyzing code complexity
/// </summary>
public class ComplexityAnalyzer
{
    private readonly ILogger _logger;
    private readonly Dictionary<string, Func<string, CodeStructure, double>> _cyclomaticComplexityCalculators = new();
    private readonly Dictionary<string, Func<string, CodeStructure, double>> _cognitiveComplexityCalculators = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="ComplexityAnalyzer"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public ComplexityAnalyzer(ILogger logger)
    {
        _logger = logger;
        InitializeComplexityCalculators();
    }

    /// <summary>
    /// Calculates the cyclomatic complexity of a method
    /// </summary>
    /// <param name="content">The code content</param>
    /// <param name="structure">The method structure</param>
    /// <returns>The cyclomatic complexity</returns>
    public double CalculateMethodComplexity(string content, CodeStructure structure)
    {
        try
        {
            _logger.LogInformation("Calculating cyclomatic complexity for method {MethodName}", structure.Name);

            if (structure.Type != StructureType.Method)
            {
                _logger.LogWarning("Structure {StructureName} is not a method", structure.Name);
                return 0;
            }

            var language = GetLanguageFromStructure(structure);
            if (_cyclomaticComplexityCalculators.TryGetValue(language, out var calculator))
            {
                return calculator(content, structure);
            }

            // Default calculation if no language-specific calculator is available
            return CalculateDefaultCyclomaticComplexity(content, structure);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating cyclomatic complexity for method {MethodName}", structure.Name);
            return 0;
        }
    }

    /// <summary>
    /// Calculates the cognitive complexity of a method
    /// </summary>
    /// <param name="content">The code content</param>
    /// <param name="structure">The method structure</param>
    /// <returns>The cognitive complexity</returns>
    public double CalculateCognitiveComplexity(string content, CodeStructure structure)
    {
        try
        {
            _logger.LogInformation("Calculating cognitive complexity for method {MethodName}", structure.Name);

            if (structure.Type != StructureType.Method)
            {
                _logger.LogWarning("Structure {StructureName} is not a method", structure.Name);
                return 0;
            }

            var language = GetLanguageFromStructure(structure);
            if (_cognitiveComplexityCalculators.TryGetValue(language, out var calculator))
            {
                return calculator(content, structure);
            }

            // Default calculation if no language-specific calculator is available
            return CalculateDefaultCognitiveComplexity(content, structure);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating cognitive complexity for method {MethodName}", structure.Name);
            return 0;
        }
    }

    /// <summary>
    /// Calculates the complexity of a class
    /// </summary>
    /// <param name="content">The code content</param>
    /// <param name="structure">The class structure</param>
    /// <returns>The class complexity</returns>
    public double CalculateClassComplexity(string content, CodeStructure structure)
    {
        try
        {
            _logger.LogInformation("Calculating complexity for class {ClassName}", structure.Name);

            if (structure.Type != StructureType.Class && structure.Type != StructureType.Interface)
            {
                _logger.LogWarning("Structure {StructureName} is not a class or interface", structure.Name);
                return 0;
            }

            // Extract the class content
            var classContent = ExtractStructureContent(content, structure);

            // Find all methods in the class
            var methodRegex = new Regex(@"(public|private|protected|internal)?\s*(static|virtual|abstract|override|async)?\s*([a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*\(", RegexOptions.Compiled);
            var methodMatches = methodRegex.Matches(classContent);

            double totalComplexity = 0;
            int methodCount = 0;

            foreach (Match match in methodMatches)
            {
                if (match.Groups.Count > 4)
                {
                    var methodName = match.Groups[4].Value;
                    var methodStartLine = structure.Location.StartLine + classContent.Substring(0, match.Index).Count(c => c == '\n');

                    // Create a method structure
                    var methodStructure = new CodeStructure
                    {
                        Type = StructureType.Method,
                        Name = methodName,
                        ParentName = structure.Name,
                        Location = new CodeLocation
                        {
                            StartLine = methodStartLine,
                            Namespace = structure.Location.Namespace,
                            ClassName = structure.Name,
                            MethodName = methodName
                        }
                    };

                    // Calculate method complexity
                    var methodComplexity = CalculateMethodComplexity(content, methodStructure);
                    totalComplexity += methodComplexity;
                    methodCount++;
                }
            }

            // Calculate average method complexity and add a factor for the number of methods
            double classComplexity = methodCount > 0 ? totalComplexity / methodCount : 0;
            classComplexity += Math.Log10(Math.Max(1, methodCount)) * 2; // Factor for number of methods

            return classComplexity;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating complexity for class {ClassName}", structure.Name);
            return 0;
        }
    }

    /// <summary>
    /// Detects complexity issues in the code
    /// </summary>
    /// <param name="content">The code content</param>
    /// <param name="language">The programming language</param>
    /// <param name="structures">The code structures</param>
    /// <returns>The list of complexity issues</returns>
    public List<CodeIssue> DetectComplexityIssues(string content, string language, List<CodeStructure> structures)
    {
        try
        {
            _logger.LogInformation("Detecting complexity issues in {Language} code", language);

            var issues = new List<CodeIssue>();

            // Check method complexity
            foreach (var structure in structures.Where(s => s.Type == StructureType.Method))
            {
                var cyclomaticComplexity = CalculateMethodComplexity(content, structure);
                var cognitiveComplexity = CalculateCognitiveComplexity(content, structure);

                // Check cyclomatic complexity
                if (cyclomaticComplexity > 15)
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Complexity,
                        Severity = cyclomaticComplexity > 30 ? IssueSeverity.Critical : (cyclomaticComplexity > 20 ? IssueSeverity.Major : IssueSeverity.Minor),
                        Title = "High Cyclomatic Complexity",
                        Description = $"Method '{structure.Name}' has a cyclomatic complexity of {cyclomaticComplexity}, which exceeds the recommended maximum of 15.",
                        Location = structure.Location,
                        SuggestedFix = "Consider breaking the method into smaller, more focused methods with less conditional logic.",
                        ImpactScore = Math.Min(1.0, cyclomaticComplexity / 50),
                        FixDifficultyScore = Math.Min(1.0, cyclomaticComplexity / 30),
                        Tags = { "complexity", "maintainability" }
                    });
                }

                // Check cognitive complexity
                if (cognitiveComplexity > 10)
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Complexity,
                        Severity = cognitiveComplexity > 20 ? IssueSeverity.Critical : (cognitiveComplexity > 15 ? IssueSeverity.Major : IssueSeverity.Minor),
                        Title = "High Cognitive Complexity",
                        Description = $"Method '{structure.Name}' has a cognitive complexity of {cognitiveComplexity}, which exceeds the recommended maximum of 10.",
                        Location = structure.Location,
                        SuggestedFix = "Consider simplifying the method by reducing nesting levels and breaking it into smaller methods.",
                        ImpactScore = Math.Min(1.0, cognitiveComplexity / 30),
                        FixDifficultyScore = Math.Min(1.0, cognitiveComplexity / 20),
                        Tags = { "complexity", "maintainability" }
                    });
                }
            }

            // Check class complexity
            foreach (var structure in structures.Where(s => s.Type == StructureType.Class))
            {
                var classComplexity = CalculateClassComplexity(content, structure);

                if (classComplexity > 50)
                {
                    issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Complexity,
                        Severity = classComplexity > 100 ? IssueSeverity.Critical : (classComplexity > 75 ? IssueSeverity.Major : IssueSeverity.Minor),
                        Title = "High Class Complexity",
                        Description = $"Class '{structure.Name}' has a complexity score of {classComplexity}, which indicates it may be too complex.",
                        Location = structure.Location,
                        SuggestedFix = "Consider breaking the class into smaller, more focused classes with single responsibilities.",
                        ImpactScore = Math.Min(1.0, classComplexity / 150),
                        FixDifficultyScore = Math.Min(1.0, classComplexity / 100),
                        Tags = { "complexity", "maintainability" }
                    });
                }
            }

            // Check nesting depth
            var nestingIssues = DetectExcessiveNesting(content, language);
            issues.AddRange(nestingIssues);

            return issues;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting complexity issues in {Language} code", language);
            return new List<CodeIssue>();
        }
    }

    private List<CodeIssue> DetectExcessiveNesting(string content, string language)
    {
        var issues = new List<CodeIssue>();
        var lines = content.Split('\n');
        const int maxNestingLevel = 4;

        try
        {
            // Simple approach to detect nesting levels based on indentation
            var currentNestingLevel = 0;
            var nestingStartLine = -1;
            var maxNestingLevelFound = 0;

            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i].TrimEnd();
                
                // Skip empty lines and comments
                if (string.IsNullOrWhiteSpace(line) || line.TrimStart().StartsWith("//") || line.TrimStart().StartsWith("/*") || line.TrimStart().StartsWith("*"))
                {
                    continue;
                }

                // Count indentation level (simplified approach)
                var indentationLevel = line.Length - line.TrimStart().Length;
                var estimatedNestingLevel = indentationLevel / 4; // Assuming 4 spaces per indentation level
                
                if (estimatedNestingLevel > currentNestingLevel)
                {
                    if (currentNestingLevel == 0)
                    {
                        nestingStartLine = i;
                    }
                    currentNestingLevel = estimatedNestingLevel;
                    maxNestingLevelFound = Math.Max(maxNestingLevelFound, currentNestingLevel);
                }
                else if (estimatedNestingLevel < currentNestingLevel)
                {
                    currentNestingLevel = estimatedNestingLevel;
                    
                    // If we're back to nesting level 0 and we found excessive nesting, report it
                    if (currentNestingLevel == 0 && maxNestingLevelFound > maxNestingLevel && nestingStartLine >= 0)
                    {
                        issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Complexity,
                            Severity = maxNestingLevelFound > maxNestingLevel + 2 ? IssueSeverity.Major : IssueSeverity.Minor,
                            Title = "Excessive Nesting",
                            Description = $"Found code with {maxNestingLevelFound} levels of nesting, which exceeds the recommended maximum of {maxNestingLevel}.",
                            Location = new CodeLocation
                            {
                                StartLine = nestingStartLine,
                                EndLine = i
                            },
                            SuggestedFix = "Consider refactoring the code to reduce nesting by extracting methods or using early returns.",
                            ImpactScore = 0.5,
                            FixDifficultyScore = 0.4,
                            Tags = { "complexity", "nesting" }
                        });
                        
                        maxNestingLevelFound = 0;
                        nestingStartLine = -1;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting excessive nesting");
        }

        return issues;
    }

    private double CalculateDefaultCyclomaticComplexity(string content, CodeStructure structure)
    {
        try
        {
            // Extract the method content
            var methodContent = ExtractStructureContent(content, structure);
            
            // Count decision points (simplified approach)
            double complexity = 1; // Base complexity
            
            // Count if statements
            complexity += CountMatches(methodContent, @"\bif\s*\(");
            
            // Count else if statements
            complexity += CountMatches(methodContent, @"\belse\s+if\s*\(");
            
            // Count switch cases
            complexity += CountMatches(methodContent, @"\bcase\s+[^:]+:");
            
            // Count loops (for, while, do-while, foreach)
            complexity += CountMatches(methodContent, @"\bfor\s*\(");
            complexity += CountMatches(methodContent, @"\bwhile\s*\(");
            complexity += CountMatches(methodContent, @"\bdo\s*\{");
            complexity += CountMatches(methodContent, @"\bforeach\s*\(");
            
            // Count logical operators (&&, ||)
            complexity += CountMatches(methodContent, @"&&|\|\|");
            
            // Count catch blocks
            complexity += CountMatches(methodContent, @"\bcatch\s*\(");
            
            // Count conditional operators (?:)
            complexity += CountMatches(methodContent, @"\?[^;]+:");
            
            return complexity;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating default cyclomatic complexity for method {MethodName}", structure.Name);
            return 1; // Default to base complexity
        }
    }

    private double CalculateDefaultCognitiveComplexity(string content, CodeStructure structure)
    {
        try
        {
            // Extract the method content
            var methodContent = ExtractStructureContent(content, structure);
            
            double complexity = 0;
            
            // Count control flow structures with increasing weights for nesting
            complexity += CountControlFlowStructures(methodContent);
            
            // Add complexity for logical operators
            complexity += CountMatches(methodContent, @"&&|\|\|") * 0.5;
            
            // Add complexity for recursive calls (simplified detection)
            if (methodContent.Contains(structure.Name + "("))
            {
                complexity += 1;
            }
            
            // Add complexity for breaks and continues
            complexity += CountMatches(methodContent, @"\bbreak\b|\bcontinue\b") * 0.5;
            
            // Add complexity for goto statements
            complexity += CountMatches(methodContent, @"\bgoto\b") * 2;
            
            return complexity;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating default cognitive complexity for method {MethodName}", structure.Name);
            return 0;
        }
    }

    private double CountControlFlowStructures(string content)
    {
        double complexity = 0;
        var lines = content.Split('\n');
        var nestingLevel = 0;
        
        foreach (var line in lines)
        {
            var trimmedLine = line.Trim();
            
            // Skip empty lines and comments
            if (string.IsNullOrWhiteSpace(trimmedLine) || trimmedLine.StartsWith("//") || trimmedLine.StartsWith("/*") || trimmedLine.StartsWith("*"))
            {
                continue;
            }
            
            // Check for control flow structures
            if (Regex.IsMatch(trimmedLine, @"\bif\s*\(|\belse\s+if\s*\(|\bfor\s*\(|\bforeach\s*\(|\bwhile\s*\(|\bdo\s*\{|\bcase\s+[^:]+:|\bcatch\s*\("))
            {
                // Add base complexity
                complexity += 1;
                
                // Add nesting complexity
                complexity += nestingLevel;
                
                // Increase nesting level if the line ends with an opening brace
                if (trimmedLine.EndsWith("{"))
                {
                    nestingLevel++;
                }
            }
            else if (trimmedLine.StartsWith("}") && nestingLevel > 0)
            {
                // Decrease nesting level when closing brace is found
                nestingLevel--;
            }
            else if (trimmedLine.EndsWith("{"))
            {
                // Increase nesting level for other blocks
                nestingLevel++;
            }
        }
        
        return complexity;
    }

    private int CountMatches(string content, string pattern)
    {
        return Regex.Matches(content, pattern).Count;
    }

    private string ExtractStructureContent(string content, CodeStructure structure)
    {
        try
        {
            // Get the lines for the structure
            var lines = content.Split('\n');
            var startLine = structure.Location.StartLine;
            var endLine = structure.Location.EndLine;
            
            // If end line is not set, try to calculate it
            if (endLine == 0)
            {
                endLine = CalculateEndLine(content, structure);
            }
            
            // Ensure valid line numbers
            startLine = Math.Max(0, Math.Min(startLine, lines.Length - 1));
            endLine = Math.Max(startLine, Math.Min(endLine, lines.Length - 1));
            
            // Extract the content
            var structureLines = lines.Skip(startLine).Take(endLine - startLine + 1);
            return string.Join("\n", structureLines);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting structure content for {Name}", structure.Name);
            return string.Empty;
        }
    }

    private int CalculateEndLine(string content, CodeStructure structure)
    {
        var lines = content.Split('\n');
        var startLine = structure.Location.StartLine;
        
        // Default to the end of the file
        var endLine = lines.Length - 1;
        
        if (structure.Type == StructureType.Method)
        {
            // Find the method body
            var methodBodyStart = -1;
            var braceCount = 0;
            var inMethod = false;
            
            for (int i = startLine; i < lines.Length; i++)
            {
                var line = lines[i];
                
                if (line.Contains("{"))
                {
                    if (!inMethod)
                    {
                        inMethod = true;
                        methodBodyStart = i;
                    }
                    braceCount++;
                }
                
                if (line.Contains("}"))
                {
                    braceCount--;
                    if (inMethod && braceCount == 0)
                    {
                        endLine = i;
                        break;
                    }
                }
            }
        }
        else if (structure.Type == StructureType.Class || structure.Type == StructureType.Interface)
        {
            // Find the class body
            var braceCount = 0;
            var inClass = false;
            
            for (int i = startLine; i < lines.Length; i++)
            {
                var line = lines[i];
                
                if (line.Contains("{"))
                {
                    if (!inClass)
                    {
                        inClass = true;
                    }
                    braceCount++;
                }
                
                if (line.Contains("}"))
                {
                    braceCount--;
                    if (inClass && braceCount == 0)
                    {
                        endLine = i;
                        break;
                    }
                }
            }
        }
        
        return endLine;
    }

    private string GetLanguageFromStructure(CodeStructure structure)
    {
        // Try to determine language from structure properties
        if (structure.Properties.TryGetValue("Language", out var language))
        {
            return language.ToLowerInvariant();
        }
        
        // Default to C# if not specified
        return "csharp";
    }

    private void InitializeComplexityCalculators()
    {
        // Register C# complexity calculators
        _cyclomaticComplexityCalculators["csharp"] = CalculateCSharpCyclomaticComplexity;
        _cognitiveComplexityCalculators["csharp"] = CalculateCSharpCognitiveComplexity;
        
        // Register F# complexity calculators
        _cyclomaticComplexityCalculators["fsharp"] = CalculateFSharpCyclomaticComplexity;
        _cognitiveComplexityCalculators["fsharp"] = CalculateFSharpCognitiveComplexity;
        
        // Add more language-specific calculators as needed
    }

    private double CalculateCSharpCyclomaticComplexity(string content, CodeStructure structure)
    {
        // Use the default implementation for C#
        return CalculateDefaultCyclomaticComplexity(content, structure);
    }

    private double CalculateCSharpCognitiveComplexity(string content, CodeStructure structure)
    {
        // Use the default implementation for C#
        return CalculateDefaultCognitiveComplexity(content, structure);
    }

    private double CalculateFSharpCyclomaticComplexity(string content, CodeStructure structure)
    {
        try
        {
            // Extract the function content
            var functionContent = ExtractStructureContent(content, structure);
            
            // Count decision points (simplified approach for F#)
            double complexity = 1; // Base complexity
            
            // Count if expressions
            complexity += CountMatches(functionContent, @"\bif\b");
            
            // Count elif expressions
            complexity += CountMatches(functionContent, @"\belif\b");
            
            // Count match cases
            complexity += CountMatches(functionContent, @"\|\s*[^-]+\s*->");
            
            // Count loops (for, while)
            complexity += CountMatches(functionContent, @"\bfor\b");
            complexity += CountMatches(functionContent, @"\bwhile\b");
            
            // Count logical operators (&&, ||)
            complexity += CountMatches(functionContent, @"&&|\|\|");
            
            // Count try/with expressions
            complexity += CountMatches(functionContent, @"\bwith\b");
            
            return complexity;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating F# cyclomatic complexity for function {FunctionName}", structure.Name);
            return 1; // Default to base complexity
        }
    }

    private double CalculateFSharpCognitiveComplexity(string content, CodeStructure structure)
    {
        try
        {
            // Extract the function content
            var functionContent = ExtractStructureContent(content, structure);
            
            double complexity = 0;
            
            // Count control flow structures
            complexity += CountMatches(functionContent, @"\bif\b") * 1;
            complexity += CountMatches(functionContent, @"\belif\b") * 1;
            complexity += CountMatches(functionContent, @"\belse\b") * 0.5;
            complexity += CountMatches(functionContent, @"\|\s*[^-]+\s*->") * 0.5; // Match cases
            complexity += CountMatches(functionContent, @"\bfor\b") * 1;
            complexity += CountMatches(functionContent, @"\bwhile\b") * 1;
            complexity += CountMatches(functionContent, @"\bwith\b") * 1; // Exception handling
            
            // Add complexity for logical operators
            complexity += CountMatches(functionContent, @"&&|\|\|") * 0.5;
            
            // Add complexity for recursive calls (simplified detection)
            if (functionContent.Contains("rec " + structure.Name) || functionContent.Contains(structure.Name + " " + structure.Name))
            {
                complexity += 1;
            }
            
            // Add complexity for nested functions
            complexity += CountMatches(functionContent, @"\blet\s+[a-zA-Z0-9_]+\s*=") * 0.5;
            
            // Add complexity for piping operations (simplified)
            complexity += CountMatches(functionContent, @"\|\>") * 0.2;
            
            return complexity;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating F# cognitive complexity for function {FunctionName}", structure.Name);
            return 0;
        }
    }
}
