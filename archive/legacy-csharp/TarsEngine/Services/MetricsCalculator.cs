using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Calculates code metrics for C# code
/// </summary>
public class MetricsCalculator(ILogger<MetricsCalculator> logger) : IMetricsCalculator
{
    private readonly ILogger<MetricsCalculator> _logger = logger;

    /// <inheritdoc/>
    public string Language => "csharp";

    /// <inheritdoc/>
    public List<CodeMetric> CalculateMetrics(string content, List<CodeStructure> structures, bool analyzeComplexity, bool analyzeMaintainability)
    {
        var metrics = new List<CodeMetric>();

        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return metrics;
            }

            // Add basic metrics
            metrics.Add(new CodeMetric
            {
                Type = MetricType.Size,
                Value = content.Count(c => c == '\n') + 1,
                Name = "Lines of Code"
            });

            metrics.Add(new CodeMetric
            {
                Type = MetricType.Size,
                Value = content.Length,
                Name = "Character Count"
            });

            // Add structure counts
            metrics.Add(new CodeMetric
            {
                Type = MetricType.Size,
                Value = structures.Count(s => s.Type == StructureType.Class),
                Name = "Class Count"
            });

            metrics.Add(new CodeMetric
            {
                Type = MetricType.Size,
                Value = structures.Count(s => s.Type == StructureType.Method),
                Name = "Method Count"
            });

            metrics.Add(new CodeMetric
            {
                Type = MetricType.Size,
                Value = structures.Count(s => s.Type == StructureType.Interface),
                Name = "Interface Count"
            });

            // Calculate Halstead metrics if complexity analysis is enabled
            if (analyzeComplexity)
            {
                // Count unique operators
                var operatorRegex = new Regex(@"[+\-*/=<>!&|^~%]|==|!=|<=|>=|&&|\|\||<<|>>|\+\+|--|->|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=", RegexOptions.Compiled);
                var operatorMatches = operatorRegex.Matches(content);
                var uniqueOperators = new HashSet<string>();
                foreach (Match match in operatorMatches)
                {
                    uniqueOperators.Add(match.Value);
                }

                // Count unique operands (identifiers and literals)
                var operandRegex = new Regex(@"\b[a-zA-Z_][a-zA-Z0-9_]*\b|""[^""]*""|'[^']*'|\d+(\.\d+)?", RegexOptions.Compiled);
                var operandMatches = operandRegex.Matches(content);
                var uniqueOperands = new HashSet<string>();
                foreach (Match match in operandMatches)
                {
                    uniqueOperands.Add(match.Value);
                }

                var n1 = uniqueOperators.Count;
                var n2 = uniqueOperands.Count;
                var N1 = operatorMatches.Count;
                var N2 = operandMatches.Count;

                // Calculate Halstead metrics
                var programVocabulary = n1 + n2;
                var programLength = N1 + N2;
                var calculatedProgramLength = n1 * Math.Log2(n1) + n2 * Math.Log2(n2);
                var volume = programLength * Math.Log2(programVocabulary);
                var difficulty = (n1 / 2.0) * (N2 / (double)n2);
                var effort = difficulty * volume;
                var timeToImplement = effort / 18.0; // Time in seconds
                var deliveredBugs = Math.Pow(effort, 2.0/3.0) / 3000.0;

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = programVocabulary,
                    Name = "Halstead Vocabulary"
                });

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = programLength,
                    Name = "Halstead Length"
                });

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = volume,
                    Name = "Halstead Volume"
                });

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = difficulty,
                    Name = "Halstead Difficulty"
                });

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = effort,
                    Name = "Halstead Effort"
                });

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = timeToImplement,
                    Name = "Halstead Time (seconds)"
                });

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Complexity,
                    Value = deliveredBugs,
                    Name = "Halstead Bugs"
                });
            }

            // Calculate cyclomatic complexity for each method
            if (analyzeComplexity)
            {
                var methodStructures = structures.Where(s => s.Type == StructureType.Method).ToList();
                foreach (var method in methodStructures)
                {
                    // Get method content
                    var methodContent = GetMethodContent(content, method);
                    if (string.IsNullOrWhiteSpace(methodContent))
                    {
                        continue;
                    }

                    var complexity = CalculateCyclomaticComplexity(methodContent);

                    metrics.Add(new CodeMetric
                    {
                        Type = MetricType.Complexity,
                        Value = complexity,
                        Name = $"Cyclomatic Complexity: {method.Name}"
                    });
                }

                // Calculate average cyclomatic complexity
                var complexityMetrics = metrics.Where(m => m.Name.StartsWith("Cyclomatic Complexity:")).ToList();
                if (complexityMetrics.Count > 0)
                {
                    var avgComplexity = complexityMetrics.Average(m => m.Value);
                    metrics.Add(new CodeMetric
                    {
                        Type = MetricType.Complexity,
                        Value = avgComplexity,
                        Name = "Average Cyclomatic Complexity"
                    });
                }
            }

            // Calculate maintainability index if enabled
            if (analyzeMaintainability && analyzeComplexity)
            {
                // Get Halstead volume
                var halsteadVolume = metrics.FirstOrDefault(m => m.Name == "Halstead Volume")?.Value ?? 0;

                // Get average cyclomatic complexity
                var avgComplexity = metrics.FirstOrDefault(m => m.Name == "Average Cyclomatic Complexity")?.Value ?? 0;

                // Get lines of code
                var linesOfCode = metrics.FirstOrDefault(m => m.Name == "Lines of Code")?.Value ?? 0;

                // Calculate maintainability index
                var maintainabilityIndex = CalculateMaintainabilityIndex(halsteadVolume, avgComplexity, (int)linesOfCode);

                metrics.Add(new CodeMetric
                {
                    Type = MetricType.Maintainability,
                    Value = maintainabilityIndex,
                    Name = "Maintainability Index"
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating metrics for C# code");
        }

        return metrics;
    }

    /// <inheritdoc/>
    public int CalculateCyclomaticComplexity(string methodContent)
    {
        if (string.IsNullOrWhiteSpace(methodContent))
        {
            return 1;
        }

        // Start with 1 (base complexity)
        var complexity = 1;

        // Count decision points
        complexity += Regex.Matches(methodContent, @"\bif\b").Count;
        complexity += Regex.Matches(methodContent, @"\belse\s+if\b").Count;
        complexity += Regex.Matches(methodContent, @"\bwhile\b").Count;
        complexity += Regex.Matches(methodContent, @"\bfor\b").Count;
        complexity += Regex.Matches(methodContent, @"\bforeach\b").Count;
        complexity += Regex.Matches(methodContent, @"\bcase\b").Count;
        complexity += Regex.Matches(methodContent, @"\bcatch\b").Count;
        complexity += Regex.Matches(methodContent, @"\b\|\|\b").Count;
        complexity += Regex.Matches(methodContent, @"\b&&\b").Count;
        complexity += Regex.Matches(methodContent, @"\?\s*[^:]+\s*:").Count; // Ternary operators

        return complexity;
    }

    /// <inheritdoc/>
    public double CalculateMaintainabilityIndex(double halsteadVolume, double cyclomaticComplexity, int linesOfCode)
    {
        // MI = 171 - 5.2 * ln(HV) - 0.23 * CC - 16.2 * ln(LOC)
        var maintainabilityIndex = 171 - 5.2 * Math.Log(Math.Max(1, halsteadVolume)) - 0.23 * cyclomaticComplexity - 16.2 * Math.Log(Math.Max(1, linesOfCode));

        // Normalize to 0-100 scale
        maintainabilityIndex = Math.Max(0, Math.Min(100, maintainabilityIndex));

        return maintainabilityIndex;
    }

    /// <inheritdoc/>
    public Dictionary<MetricType, string> GetAvailableMetricTypes()
    {
        return new Dictionary<MetricType, string>
        {
            { MetricType.Size, "Lines of C# code" },
            { MetricType.Size, "Character count in C# code" },
            { MetricType.Size, "Number of classes in C# code" },
            { MetricType.Size, "Number of methods in C# code" },
            { MetricType.Size, "Number of interfaces in C# code" },
            { MetricType.Complexity, "Cyclomatic complexity of C# methods" },
            { MetricType.Complexity, "Average cyclomatic complexity of C# methods" },
            { MetricType.Complexity, "Halstead vocabulary of C# code" },
            { MetricType.Complexity, "Halstead length of C# code" },
            { MetricType.Complexity, "Halstead volume of C# code" },
            { MetricType.Complexity, "Halstead difficulty of C# code" },
            { MetricType.Complexity, "Halstead effort of C# code" },
            { MetricType.Complexity, "Halstead time to implement C# code" },
            { MetricType.Complexity, "Halstead estimated bugs in C# code" },
            { MetricType.Maintainability, "Maintainability index of C# code" }
        };
    }

    /// <summary>
    /// Gets the content of a method from the full code content
    /// </summary>
    /// <param name="content">The full code content</param>
    /// <param name="method">The method structure</param>
    /// <returns>The method content</returns>
    private string GetMethodContent(string content, CodeStructure method)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(content) || method == null)
            {
                return string.Empty;
            }

            // Find the method declaration
            var methodRegex = new Regex($@"(public|private|protected|internal)?\s*(static|virtual|abstract|override|async)?\s*[a-zA-Z0-9_<>]+\s+{Regex.Escape(method.Name)}\s*\(", RegexOptions.Compiled);
            var match = methodRegex.Match(content);
            if (!match.Success)
            {
                return string.Empty;
            }

            // Find the opening brace
            var openBracePos = content.IndexOf('{', match.Index);
            if (openBracePos == -1)
            {
                return string.Empty;
            }

            // Find the matching closing brace
            var closeBracePos = FindMatchingBrace(content, openBracePos);
            if (closeBracePos == -1)
            {
                return string.Empty;
            }

            // Extract the method content
            return content.Substring(match.Index, closeBracePos - match.Index + 1);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting method content");
            return string.Empty;
        }
    }

    /// <summary>
    /// Finds the position of the matching closing brace
    /// </summary>
    /// <param name="content">The content to search in</param>
    /// <param name="openBracePos">The position of the opening brace</param>
    /// <returns>The position of the matching closing brace, or -1 if not found</returns>
    private static int FindMatchingBrace(string content, int openBracePos)
    {
        var braceCount = 1;
        for (var i = openBracePos + 1; i < content.Length; i++)
        {
            if (content[i] == '{')
            {
                braceCount++;
            }
            else if (content[i] == '}')
            {
                braceCount--;
                if (braceCount == 0)
                {
                    return i;
                }
            }
        }
        return -1;
    }
}