using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services
{
    /// <summary>
    /// Service for analyzing code and extracting information about its structure
    /// </summary>
    public class CodeAnalysisService : ICodeAnalysisService
    {
        private readonly ILogger<CodeAnalysisService> _logger;

        public CodeAnalysisService(ILogger<CodeAnalysisService> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Analyzes a file and extracts information about its structure
        /// </summary>
        /// <param name="filePath">Path to the file to analyze</param>
        /// <returns>A CodeAnalysisResult containing information about the file</returns>
        public virtual async Task<CodeAnalysisResult> AnalyzeFileAsync(string filePath)
        {
            try
            {
                _logger.LogInformation($"Analyzing file: {filePath}");

                if (!File.Exists(filePath))
                {
                    _logger.LogError($"File not found: {filePath}");
                    return new CodeAnalysisResult
                    {
                        Success = false,
                        ErrorMessage = $"File not found: {filePath}"
                    };
                }

                string content = await File.ReadAllTextAsync(filePath);
                string extension = Path.GetExtension(filePath).ToLowerInvariant();

                var result = new CodeAnalysisResult
                {
                    FilePath = filePath,
                    Language = DetermineLanguage(extension),
                    Success = true
                };

                // Analyze the code based on the language
                switch (result.Language)
                {
                    case ProgrammingLanguage.CSharp:
                        AnalyzeCSharpCode(content, result);
                        break;
                    case ProgrammingLanguage.FSharp:
                        AnalyzeFSharpCode(content, result);
                        break;
                    case ProgrammingLanguage.JavaScript:
                    case ProgrammingLanguage.TypeScript:
                        AnalyzeJavaScriptCode(content, result);
                        break;
                    case ProgrammingLanguage.Python:
                        AnalyzePythonCode(content, result);
                        break;
                    default:
                        // Generic analysis for other languages
                        AnalyzeGenericCode(content, result);
                        break;
                }

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error analyzing file {filePath}");
                return new CodeAnalysisResult
                {
                    Success = false,
                    ErrorMessage = $"Error analyzing file: {ex.Message}"
                };
            }
        }

        /// <summary>
        /// Analyzes a directory and extracts information about its structure
        /// </summary>
        /// <param name="directoryPath">Path to the directory to analyze</param>
        /// <param name="recursive">Whether to analyze subdirectories</param>
        /// <returns>A list of CodeAnalysisResult containing information about each file</returns>
        public virtual async Task<List<CodeAnalysisResult>> AnalyzeDirectoryAsync(string directoryPath, bool recursive = true)
        {
            try
            {
                _logger.LogInformation($"Analyzing directory: {directoryPath}");

                if (!Directory.Exists(directoryPath))
                {
                    _logger.LogError($"Directory not found: {directoryPath}");
                    return new List<CodeAnalysisResult>();
                }

                var results = new List<CodeAnalysisResult>();
                var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

                // Get all code files in the directory
                var codeExtensions = new[] { ".cs", ".fs", ".js", ".ts", ".py", ".java", ".cpp", ".h", ".hpp" };
                var files = Directory.GetFiles(directoryPath, "*.*", searchOption)
                    .Where(f => codeExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                    .ToList();

                foreach (var file in files)
                {
                    var result = await AnalyzeFileAsync(file);
                    results.Add(result);
                }

                return results;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error analyzing directory {directoryPath}");
                return new List<CodeAnalysisResult>();
            }
        }

        /// <summary>
        /// Determines the programming language based on the file extension
        /// </summary>
        private ProgrammingLanguage DetermineLanguage(string extension)
        {
            return extension switch
            {
                ".cs" => ProgrammingLanguage.CSharp,
                ".fs" => ProgrammingLanguage.FSharp,
                ".js" => ProgrammingLanguage.JavaScript,
                ".ts" => ProgrammingLanguage.TypeScript,
                ".py" => ProgrammingLanguage.Python,
                ".java" => ProgrammingLanguage.Java,
                ".cpp" or ".h" or ".hpp" => ProgrammingLanguage.Cpp,
                _ => ProgrammingLanguage.Unknown
            };
        }

        /// <summary>
        /// Analyzes C# code and extracts information about its structure
        /// </summary>
        private void AnalyzeCSharpCode(string content, CodeAnalysisResult result)
        {
            // Extract namespaces
            var namespaceRegex = new Regex(@"namespace\s+([^\s{]+)");
            var namespaceMatches = namespaceRegex.Matches(content);
            result.Namespaces = namespaceMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract classes
            var classRegex = new Regex(@"(?:public|private|protected|internal)?\s*(?:static|abstract|sealed)?\s*class\s+([^\s:<]+)");
            var classMatches = classRegex.Matches(content);
            result.Classes = classMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract interfaces
            var interfaceRegex = new Regex(@"(?:public|private|protected|internal)?\s*interface\s+([^\s:<]+)");
            var interfaceMatches = interfaceRegex.Matches(content);
            result.Interfaces = interfaceMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract methods
            var methodRegex = new Regex(@"(?:public|private|protected|internal)?\s*(?:static|virtual|abstract|override)?\s*(?:[a-zA-Z0-9_<>]+)\s+([a-zA-Z0-9_]+)\s*\(");
            var methodMatches = methodRegex.Matches(content);
            result.Methods = methodMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract using statements
            var usingRegex = new Regex(@"using\s+([^;]+);");
            var usingMatches = usingRegex.Matches(content);
            result.Imports = usingMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract dependencies (types used in the code)
            var dependencies = new HashSet<string>();
            foreach (var className in result.Classes)
            {
                var usageRegex = new Regex($@"\b{className}\b");
                var usageMatches = usageRegex.Matches(content);
                if (usageMatches.Count > 0)
                {
                    dependencies.Add(className);
                }
            }
            result.Dependencies = dependencies.ToList();
        }

        /// <summary>
        /// Analyzes F# code and extracts information about its structure
        /// </summary>
        private void AnalyzeFSharpCode(string content, CodeAnalysisResult result)
        {
            // Extract namespaces
            var namespaceRegex = new Regex(@"namespace\s+([^\s]+)");
            var namespaceMatches = namespaceRegex.Matches(content);
            result.Namespaces = namespaceMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract modules
            var moduleRegex = new Regex(@"module\s+([^\s=]+)");
            var moduleMatches = moduleRegex.Matches(content);
            result.Modules = moduleMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract types
            var typeRegex = new Regex(@"type\s+([^\s<=(]+)");
            var typeMatches = typeRegex.Matches(content);
            result.Types = typeMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract functions
            var functionRegex = new Regex(@"let\s+(?:rec\s+)?([a-zA-Z0-9_]+)\s*(?:<[^>]*>)?\s*(?:\([^)]*\))?\s*=");
            var functionMatches = functionRegex.Matches(content);
            result.Functions = functionMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract open statements
            var openRegex = new Regex(@"open\s+([^\s]+)");
            var openMatches = openRegex.Matches(content);
            result.Imports = openMatches.Select(m => m.Groups[1].Value).ToList();
        }

        /// <summary>
        /// Analyzes JavaScript/TypeScript code and extracts information about its structure
        /// </summary>
        private void AnalyzeJavaScriptCode(string content, CodeAnalysisResult result)
        {
            // Extract classes
            var classRegex = new Regex(@"class\s+([a-zA-Z0-9_]+)");
            var classMatches = classRegex.Matches(content);
            result.Classes = classMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract functions
            var functionRegex = new Regex(@"function\s+([a-zA-Z0-9_]+)\s*\(");
            var functionMatches = functionRegex.Matches(content);
            result.Functions = functionMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract arrow functions with names
            var arrowFunctionRegex = new Regex(@"(?:const|let|var)\s+([a-zA-Z0-9_]+)\s*=\s*(?:\([^)]*\)|[a-zA-Z0-9_]+)\s*=>");
            var arrowFunctionMatches = arrowFunctionRegex.Matches(content);
            result.Functions.AddRange(arrowFunctionMatches.Select(m => m.Groups[1].Value));

            // Extract imports
            var importRegex = new Regex(@"import\s+(?:{[^}]*}|[^;]*)\s+from\s+['""]([^'""]+)['""];");
            var importMatches = importRegex.Matches(content);
            result.Imports = importMatches.Select(m => m.Groups[1].Value).ToList();
        }

        /// <summary>
        /// Analyzes Python code and extracts information about its structure
        /// </summary>
        private void AnalyzePythonCode(string content, CodeAnalysisResult result)
        {
            // Extract classes
            var classRegex = new Regex(@"class\s+([a-zA-Z0-9_]+)");
            var classMatches = classRegex.Matches(content);
            result.Classes = classMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract functions
            var functionRegex = new Regex(@"def\s+([a-zA-Z0-9_]+)\s*\(");
            var functionMatches = functionRegex.Matches(content);
            result.Functions = functionMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract imports
            var importRegex = new Regex(@"(?:import|from)\s+([a-zA-Z0-9_.]+)");
            var importMatches = importRegex.Matches(content);
            result.Imports = importMatches.Select(m => m.Groups[1].Value).ToList();
        }

        /// <summary>
        /// Generic code analysis for languages without specific analyzers
        /// </summary>
        private void AnalyzeGenericCode(string content, CodeAnalysisResult result)
        {
            // Extract function-like patterns
            var functionRegex = new Regex(@"(?:function|def|void|int|string|bool|float|double|var|let)\s+([a-zA-Z0-9_]+)\s*\(");
            var functionMatches = functionRegex.Matches(content);
            result.Functions = functionMatches.Select(m => m.Groups[1].Value).ToList();

            // Extract class-like patterns
            var classRegex = new Regex(@"(?:class|struct|interface|enum)\s+([a-zA-Z0-9_]+)");
            var classMatches = classRegex.Matches(content);
            result.Classes = classMatches.Select(m => m.Groups[1].Value).ToList();
        }
    }

    /// <summary>
    /// Represents the result of a code analysis
    /// </summary>
    public class CodeAnalysisResult
    {
        public string FilePath { get; set; }
        public ProgrammingLanguage Language { get; set; }
        public bool Success { get; set; }
        public string ErrorMessage { get; set; }

        public List<string> Namespaces { get; set; } = new List<string>();
        public List<string> Classes { get; set; } = new List<string>();
        public List<string> Interfaces { get; set; } = new List<string>();
        public List<string> Methods { get; set; } = new List<string>();
        public List<string> Functions { get; set; } = new List<string>();
        public List<string> Types { get; set; } = new List<string>();
        public List<string> Modules { get; set; } = new List<string>();
        public List<string> Imports { get; set; } = new List<string>();
        public List<string> Dependencies { get; set; } = new List<string>();
    }

    /// <summary>
    /// Represents a programming language
    /// </summary>
    public enum ProgrammingLanguage
    {
        Unknown,
        CSharp,
        FSharp,
        JavaScript,
        TypeScript,
        Python,
        Java,
        Cpp
    }
}
