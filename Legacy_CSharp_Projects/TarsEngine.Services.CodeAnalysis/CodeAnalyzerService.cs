using Microsoft.Extensions.Logging;
using TarsEngine.Services.Abstractions.CodeAnalysis;
using TarsEngine.Services.Abstractions.Models.CodeAnalysis;
using TarsEngine.Services.Core.Base;

namespace TarsEngine.Services.CodeAnalysis
{
    /// <summary>
    /// Implementation of the ICodeAnalyzerService interface.
    /// </summary>
    public class CodeAnalyzerService : ServiceBase, ICodeAnalyzerService
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CodeAnalyzerService"/> class.
        /// </summary>
        /// <param name="logger">The logger instance.</param>
        public CodeAnalyzerService(ILogger<CodeAnalyzerService> logger)
            : base(logger)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Code Analyzer Service";

        /// <inheritdoc/>
        public async Task<CodeAnalysisResult> AnalyzeCodeAsync(string code, string language)
        {
            Logger.LogInformation("Analyzing code snippet in language: {Language}", language);
            
            // Simulate code analysis
            await Task.Delay(100);
            
            var result = new CodeAnalysisResult
            {
                Language = language,
                AnalysisTimestamp = DateTime.UtcNow
            };
            
            // Add some sample metrics
            result.Metrics.Add(new CodeMetric
            {
                Name = "Lines of Code",
                Value = code.Split('\n').Length,
                Category = "Size"
            });
            
            result.Metrics.Add(new CodeMetric
            {
                Name = "Cyclomatic Complexity",
                Value = CalculateComplexity(code, language),
                Category = "Complexity"
            });
            
            // Add some sample issues
            if (code.Contains("TODO"))
            {
                result.Issues.Add(new CodeIssue
                {
                    Title = "TODO Comment",
                    Description = "Code contains TODO comments that should be addressed",
                    Severity = IssueSeverity.Minor,
                    LineNumber = GetLineNumber(code, "TODO")
                });
            }
            
            if (code.Contains("catch (Exception"))
            {
                result.Issues.Add(new CodeIssue
                {
                    Title = "Generic Exception Catch",
                    Description = "Catching generic exceptions is not recommended",
                    Severity = IssueSeverity.Moderate,
                    LineNumber = GetLineNumber(code, "catch (Exception"),
                    SuggestedFix = "Catch specific exception types instead"
                });
            }
            
            return result;
        }

        /// <inheritdoc/>
        public async Task<CodeAnalysisResult> AnalyzeFileAsync(string filePath)
        {
            Logger.LogInformation("Analyzing file: {FilePath}", filePath);
            
            if (!File.Exists(filePath))
            {
                Logger.LogWarning("File not found: {FilePath}", filePath);
                throw new FileNotFoundException("File not found", filePath);
            }
            
            string code = await File.ReadAllTextAsync(filePath);
            string language = GetLanguageFromExtension(Path.GetExtension(filePath));
            
            var result = await AnalyzeCodeAsync(code, language);
            result.FilePath = filePath;
            
            return result;
        }
        
        private int GetLineNumber(string code, string searchText)
        {
            var lines = code.Split('\n');
            for (int i = 0; i < lines.Length; i++)
            {
                if (lines[i].Contains(searchText))
                {
                    return i + 1;
                }
            }
            
            return 0;
        }
        
        private double CalculateComplexity(string code, string language)
        {
            // This is a very simplified complexity calculation
            // In a real implementation, this would use a proper parser
            
            double complexity = 1;
            
            switch (language.ToLowerInvariant())
            {
                case "csharp":
                case "c#":
                    complexity += CountOccurrences(code, "if ") * 1;
                    complexity += CountOccurrences(code, "else ") * 1;
                    complexity += CountOccurrences(code, "for ") * 1;
                    complexity += CountOccurrences(code, "foreach ") * 1;
                    complexity += CountOccurrences(code, "while ") * 1;
                    complexity += CountOccurrences(code, "switch ") * 1;
                    complexity += CountOccurrences(code, "case ") * 0.5;
                    complexity += CountOccurrences(code, "catch ") * 1;
                    break;
                    
                case "javascript":
                case "js":
                    complexity += CountOccurrences(code, "if ") * 1;
                    complexity += CountOccurrences(code, "else ") * 1;
                    complexity += CountOccurrences(code, "for ") * 1;
                    complexity += CountOccurrences(code, "while ") * 1;
                    complexity += CountOccurrences(code, "switch ") * 1;
                    complexity += CountOccurrences(code, "case ") * 0.5;
                    complexity += CountOccurrences(code, "catch ") * 1;
                    complexity += CountOccurrences(code, "? ") * 1;
                    break;
                    
                default:
                    complexity += CountOccurrences(code, "if ") * 1;
                    complexity += CountOccurrences(code, "else ") * 1;
                    complexity += CountOccurrences(code, "for ") * 1;
                    complexity += CountOccurrences(code, "while ") * 1;
                    break;
            }
            
            return complexity;
        }
        
        private int CountOccurrences(string text, string pattern)
        {
            int count = 0;
            int index = 0;
            
            while ((index = text.IndexOf(pattern, index, StringComparison.OrdinalIgnoreCase)) != -1)
            {
                count++;
                index += pattern.Length;
            }
            
            return count;
        }
        
        private string GetLanguageFromExtension(string extension)
        {
            return extension.ToLowerInvariant() switch
            {
                ".cs" => "csharp",
                ".js" => "javascript",
                ".ts" => "typescript",
                ".py" => "python",
                ".java" => "java",
                ".cpp" => "cpp",
                ".c" => "c",
                ".go" => "go",
                ".rb" => "ruby",
                ".php" => "php",
                ".swift" => "swift",
                ".kt" => "kotlin",
                ".rs" => "rust",
                _ => "unknown"
            };
        }
    }
}
