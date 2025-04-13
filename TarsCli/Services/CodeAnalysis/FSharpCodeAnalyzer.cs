using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services.CodeAnalysis
{
    /// <summary>
    /// Analyzer for F# code
    /// </summary>
    public class FSharpCodeAnalyzer : ICodeAnalyzer
    {
        private readonly ILogger<FSharpCodeAnalyzer> _logger;
        private readonly SecurityVulnerabilityAnalyzer _securityAnalyzer;

        /// <summary>
        /// Initializes a new instance of the FSharpCodeAnalyzer class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="securityAnalyzer">Security vulnerability analyzer</param>
        public FSharpCodeAnalyzer(
            ILogger<FSharpCodeAnalyzer> logger,
            SecurityVulnerabilityAnalyzer securityAnalyzer)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _securityAnalyzer = securityAnalyzer ?? throw new ArgumentNullException(nameof(securityAnalyzer));
        }

        /// <inheritdoc/>
        public async Task<CodeAnalysisResult> AnalyzeFileAsync(string filePath, string fileContent)
        {
            _logger.LogInformation($"Analyzing F# file: {filePath}");

            var result = new CodeAnalysisResult
            {
                FilePath = filePath,
                NeedsImprovement = false
            };

            try
            {
                // Split the file content into lines for analysis
                var lines = fileContent.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

                // Calculate basic metrics
                result.Metrics["LineCount"] = lines.Length;
                result.Metrics["EmptyLineCount"] = lines.Count(line => string.IsNullOrWhiteSpace(line));
                result.Metrics["CommentLineCount"] = lines.Count(line => line.Trim().StartsWith("//") || line.Trim().StartsWith("(*") || line.Trim().StartsWith("*)"));
                result.Metrics["CodeLineCount"] = result.Metrics["LineCount"] - result.Metrics["EmptyLineCount"] - result.Metrics["CommentLineCount"];

                // Check for missing XML documentation on public members
                await CheckMissingDocumentationAsync(lines, result);

                // Check for unused bindings
                await CheckUnusedBindingsAsync(lines, result);

                // Check for long functions
                await CheckLongFunctionsAsync(lines, result);

                // Check for mutable variables
                await CheckMutableVariablesAsync(lines, result);

                // Check for imperative code
                await CheckImperativeCodeAsync(lines, result);

                // Check for security vulnerabilities
                var securityIssues = await _securityAnalyzer.AnalyzeAsync(filePath, fileContent, "fs");
                result.Issues.AddRange(securityIssues);

                // Set needs improvement flag if any issues were found
                result.NeedsImprovement = result.Issues.Count > 0;

                _logger.LogInformation($"Analysis completed for {filePath}. Found {result.Issues.Count} issues.");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error analyzing file {filePath}");
                result.Issues.Add(new CodeIssue
                {
                    Type = CodeIssueType.Functional,
                    Severity = IssueSeverity.Error,
                    Description = $"Error analyzing file: {ex.Message}",
                    LineNumber = 1,
                    ColumnNumber = 1
                });
            }

            return result;
        }

        /// <inheritdoc/>
        public IEnumerable<string> GetSupportedFileExtensions()
        {
            return new[] { ".fs", ".fsx" };
        }

        /// <summary>
        /// Checks for missing XML documentation on public members
        /// </summary>
        /// <param name="lines">Lines of code</param>
        /// <param name="result">Analysis result</param>
        private async Task CheckMissingDocumentationAsync(string[] lines, CodeAnalysisResult result)
        {
            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i].Trim();

                // Check for public functions or types without XML documentation
                if ((line.StartsWith("let ") || line.StartsWith("type ")) && !line.Contains("private"))
                {
                    // Check if the previous line has XML documentation
                    bool hasDocumentation = false;
                    for (int j = i - 1; j >= 0 && j >= i - 5; j--)
                    {
                        if (lines[j].Trim().StartsWith("///"))
                        {
                            hasDocumentation = true;
                            break;
                        }
                        if (!string.IsNullOrWhiteSpace(lines[j]) && !lines[j].Trim().StartsWith("[<"))
                        {
                            break;
                        }
                    }

                    if (!hasDocumentation)
                    {
                        result.Issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Documentation,
                            Severity = IssueSeverity.Warning,
                            Description = "Missing XML documentation for public member",
                            LineNumber = i + 1,
                            ColumnNumber = 1,
                            CodeSegment = lines[i],
                            SuggestedFix = $"/// <summary>\n/// Description of this member\n/// </summary>\n{lines[i]}"
                        });
                    }
                }
            }

            await Task.CompletedTask;
        }

        /// <summary>
        /// Checks for unused bindings
        /// </summary>
        /// <param name="lines">Lines of code</param>
        /// <param name="result">Analysis result</param>
        private async Task CheckUnusedBindingsAsync(string[] lines, CodeAnalysisResult result)
        {
            // Simple regex to find let bindings
            var bindingRegex = new Regex(@"let\s+(\w+)\s*=");

            // Find all bindings
            var bindings = new Dictionary<string, int>();
            for (int i = 0; i < lines.Length; i++)
            {
                var matches = bindingRegex.Matches(lines[i]);
                foreach (Match match in matches)
                {
                    if (match.Groups.Count > 1)
                    {
                        var bindingName = match.Groups[1].Value;
                        bindings[bindingName] = i;
                    }
                }
            }

            // Check if each binding is used elsewhere in the code
            foreach (var binding in bindings)
            {
                var bindingName = binding.Key;
                var lineNumber = binding.Value;

                // Skip if it's a public function (likely exported)
                if (lines[lineNumber].Trim().StartsWith("let ") && !lines[lineNumber].Contains("private"))
                {
                    continue;
                }

                bool isUsed = false;
                for (int i = 0; i < lines.Length; i++)
                {
                    if (i == lineNumber)
                    {
                        continue; // Skip the declaration line
                    }

                    // Simple check for binding usage (this is a simplification)
                    if (lines[i].Contains(bindingName))
                    {
                        isUsed = true;
                        break;
                    }
                }

                if (!isUsed)
                {
                    result.Issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Maintainability,
                        Severity = IssueSeverity.Warning,
                        Description = $"Unused binding: {bindingName}",
                        LineNumber = lineNumber + 1,
                        ColumnNumber = lines[lineNumber].IndexOf(bindingName) + 1,
                        CodeSegment = lines[lineNumber],
                        SuggestedFix = $"// Remove unused binding: {bindingName}"
                    });
                }
            }

            await Task.CompletedTask;
        }

        /// <summary>
        /// Checks for long functions
        /// </summary>
        /// <param name="lines">Lines of code</param>
        /// <param name="result">Analysis result</param>
        private async Task CheckLongFunctionsAsync(string[] lines, CodeAnalysisResult result)
        {
            const int MaxFunctionLength = 30; // Maximum acceptable function length

            int functionStartLine = -1;
            string functionName = string.Empty;

            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i].Trim();

                // Check for function declaration
                if (functionStartLine == -1 && line.StartsWith("let ") && !line.Contains("="))
                {
                    functionStartLine = i;
                    var match = Regex.Match(line, @"let\s+(\w+)");
                    if (match.Success)
                    {
                        functionName = match.Groups[1].Value;
                    }
                }
                else if (functionStartLine == -1 && line.StartsWith("let ") && line.Contains("="))
                {
                    functionStartLine = i;
                    var match = Regex.Match(line, @"let\s+(\w+)");
                    if (match.Success)
                    {
                        functionName = match.Groups[1].Value;
                    }
                }

                // Function end found (next function or module)
                if (functionStartLine != -1 && (i > functionStartLine) &&
                    (line.StartsWith("let ") || line.StartsWith("type ") || line.StartsWith("module ") || i == lines.Length - 1))
                {
                    int functionLength = i - functionStartLine;
                    if (i == lines.Length - 1)
                    {
                        functionLength++; // Include the last line
                    }

                    if (functionLength > MaxFunctionLength)
                    {
                        result.Issues.Add(new CodeIssue
                        {
                            Type = CodeIssueType.Complexity,
                            Severity = IssueSeverity.Warning,
                            Description = $"Function '{functionName}' is too long ({functionLength} lines)",
                            LineNumber = functionStartLine + 1,
                            ColumnNumber = 1,
                            CodeSegment = lines[functionStartLine],
                            SuggestedFix = $"// Consider refactoring function '{functionName}' into smaller functions"
                        });
                    }

                    functionStartLine = -1;
                    functionName = string.Empty;

                    // If this line is a new function, process it in the next iteration
                    if (line.StartsWith("let "))
                    {
                        i--;
                    }
                }
            }

            await Task.CompletedTask;
        }

        /// <summary>
        /// Checks for mutable variables
        /// </summary>
        /// <param name="lines">Lines of code</param>
        /// <param name="result">Analysis result</param>
        private async Task CheckMutableVariablesAsync(string[] lines, CodeAnalysisResult result)
        {
            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i].Trim();

                // Check for mutable variables
                if (line.Contains("mutable "))
                {
                    result.Issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Design,
                        Severity = IssueSeverity.Warning,
                        Description = "Mutable variable found",
                        LineNumber = i + 1,
                        ColumnNumber = line.IndexOf("mutable ") + 1,
                        CodeSegment = lines[i],
                        SuggestedFix = "// Consider using immutable values and functional transformations instead of mutable variables"
                    });
                }

                // Check for variable mutation
                if (line.Contains("<-"))
                {
                    result.Issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Design,
                        Severity = IssueSeverity.Info,
                        Description = "Variable mutation found",
                        LineNumber = i + 1,
                        ColumnNumber = line.IndexOf("<-") + 1,
                        CodeSegment = lines[i],
                        SuggestedFix = "// Consider using immutable values and functional transformations instead of mutation"
                    });
                }
            }

            await Task.CompletedTask;
        }

        /// <summary>
        /// Checks for imperative code
        /// </summary>
        /// <param name="lines">Lines of code</param>
        /// <param name="result">Analysis result</param>
        private async Task CheckImperativeCodeAsync(string[] lines, CodeAnalysisResult result)
        {
            for (int i = 0; i < lines.Length; i++)
            {
                var line = lines[i].Trim();

                // Check for for loops
                if (line.StartsWith("for ") && line.Contains(" do"))
                {
                    result.Issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Design,
                        Severity = IssueSeverity.Info,
                        Description = "Imperative for loop found",
                        LineNumber = i + 1,
                        ColumnNumber = 1,
                        CodeSegment = lines[i],
                        SuggestedFix = "// Consider using functional constructs like List.map, List.iter, or List.fold instead of for loops"
                    });
                }

                // Check for while loops
                if (line.StartsWith("while ") && line.Contains(" do"))
                {
                    result.Issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Design,
                        Severity = IssueSeverity.Info,
                        Description = "Imperative while loop found",
                        LineNumber = i + 1,
                        ColumnNumber = 1,
                        CodeSegment = lines[i],
                        SuggestedFix = "// Consider using recursive functions instead of while loops"
                    });
                }

                // Check for mutable collections
                if (line.Contains("ResizeArray") || line.Contains("Dictionary") || line.Contains("HashSet"))
                {
                    result.Issues.Add(new CodeIssue
                    {
                        Type = CodeIssueType.Design,
                        Severity = IssueSeverity.Info,
                        Description = "Mutable collection found",
                        LineNumber = i + 1,
                        ColumnNumber = 1,
                        CodeSegment = lines[i],
                        SuggestedFix = "// Consider using immutable collections like List, Map, or Set instead of mutable collections"
                    });
                }
            }

            await Task.CompletedTask;
        }
    }
}
