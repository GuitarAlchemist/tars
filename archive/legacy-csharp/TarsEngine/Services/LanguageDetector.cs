using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Services;

/// <summary>
/// Detects programming languages from file content or file extensions
/// </summary>
public class LanguageDetector(ILogger<LanguageDetector> logger)
{
    private readonly ILogger<LanguageDetector> _logger = logger;
        
    private static readonly Dictionary<string, ProgrammingLanguage> _extensionMap = new()
    {
        { ".cs", ProgrammingLanguage.CSharp },
        { ".fs", ProgrammingLanguage.FSharp },
        { ".fsx", ProgrammingLanguage.FSharp },
        { ".js", ProgrammingLanguage.JavaScript },
        { ".ts", ProgrammingLanguage.TypeScript },
        { ".py", ProgrammingLanguage.Python },
        { ".java", ProgrammingLanguage.Java },
        { ".cpp", ProgrammingLanguage.Cpp },
        { ".h", ProgrammingLanguage.Cpp },
        { ".hpp", ProgrammingLanguage.Cpp },
        { ".c", ProgrammingLanguage.Cpp }
    };
        
    private static readonly Dictionary<ProgrammingLanguage, List<string>> _languagePatterns = new()
    {
        { 
            ProgrammingLanguage.CSharp,
            [
                @"using\s+[a-zA-Z0-9_.]+;",
                @"namespace\s+[a-zA-Z0-9_.]+",
                @"(public|private|protected|internal)\s+(class|interface|struct|enum)",
                @"(public|private|protected|internal)\s+(static|virtual|abstract|override)?\s+[a-zA-Z0-9_<>]+\s+[a-zA-Z0-9_]+\s*\("
            ]
        },
        { 
            ProgrammingLanguage.FSharp,
            [
                @"open\s+[a-zA-Z0-9_.]+",
                @"module\s+[a-zA-Z0-9_.]+",
                @"type\s+[a-zA-Z0-9_]+(?:<[^>]+>)?(?:\s*=|\s*\(|\s*:)",
                @"let\s+(rec\s+)?[a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)*\s*=",
                @"match\s+[a-zA-Z0-9_]+\s+with"
            ]
        },
        { 
            ProgrammingLanguage.JavaScript,
            [
                @"const\s+[a-zA-Z0-9_$]+\s*=",
                @"let\s+[a-zA-Z0-9_$]+\s*=",
                @"var\s+[a-zA-Z0-9_$]+\s*=",
                @"function\s+[a-zA-Z0-9_$]+\s*\(",
                @"import\s+{[^}]+}\s+from",
                @"export\s+(default\s+)?(function|class|const|let|var)",
                @"document\.getElementById",
                @"window\."
            ]
        },
        { 
            ProgrammingLanguage.TypeScript,
            [
                @"interface\s+[a-zA-Z0-9_$]+\s*{",
                @"class\s+[a-zA-Z0-9_$]+(?:<[^>]+>)?(?:\s+implements\s+[a-zA-Z0-9_$]+)?",
                @"(const|let|var)\s+[a-zA-Z0-9_$]+\s*:\s*[a-zA-Z0-9_$<>]+",
                @"function\s+[a-zA-Z0-9_$]+\s*\([^)]*\)\s*:\s*[a-zA-Z0-9_$<>]+",
                @"import\s+{[^}]+}\s+from",
                @"export\s+(default\s+)?(function|class|interface|type|const|let|var)"
            ]
        },
        { 
            ProgrammingLanguage.Python,
            [
                @"import\s+[a-zA-Z0-9_.]+",
                @"from\s+[a-zA-Z0-9_.]+\s+import",
                @"def\s+[a-zA-Z0-9_]+\s*\(",
                @"class\s+[a-zA-Z0-9_]+(?:\([^)]*\))?:",
                @"if\s+__name__\s*==\s*[""']__main__[""']"
            ]
        },
        { 
            ProgrammingLanguage.Java,
            [
                @"package\s+[a-zA-Z0-9_.]+;",
                @"import\s+[a-zA-Z0-9_.]+;",
                @"(public|private|protected)\s+(static\s+)?(class|interface|enum)",
                @"(public|private|protected)\s+(static\s+)?(final\s+)?[a-zA-Z0-9_<>]+\s+[a-zA-Z0-9_]+\s*\(",
                @"System\.out\.println"
            ]
        },
        { 
            ProgrammingLanguage.Cpp,
            [
                @"#include\s+[<""][a-zA-Z0-9_.]+[>""]",
                @"namespace\s+[a-zA-Z0-9_]+\s*{",
                @"(public|private|protected):",
                @"class\s+[a-zA-Z0-9_]+(?:\s*:\s*(?:public|private|protected)\s+[a-zA-Z0-9_]+)?",
                @"std::",
                @"int\s+main\s*\(\s*(?:int\s+argc,\s*char\s*\*\s*argv\[\s*\]|void)\s*\)"
            ]
        }
    };

    /// <summary>
    /// Detects the programming language from file content
    /// </summary>
    /// <param name="content">The file content</param>
    /// <param name="filePath">Optional file path to help with detection</param>
    /// <returns>The detected programming language</returns>
    public ProgrammingLanguage DetectLanguage(string content, string? filePath = null)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return ProgrammingLanguage.Unknown;
            }

            // First try to detect by file extension if available
            if (!string.IsNullOrWhiteSpace(filePath))
            {
                var extension = Path.GetExtension(filePath).ToLowerInvariant();
                if (_extensionMap.TryGetValue(extension, out var language))
                {
                    return language;
                }
            }

            // Then try to detect by content patterns
            var scores = new Dictionary<ProgrammingLanguage, int>();
            foreach (var language in _languagePatterns.Keys)
            {
                scores[language] = 0;
                foreach (var pattern in _languagePatterns[language])
                {
                    var matches = Regex.Matches(content, pattern);
                    scores[language] += matches.Count;
                }
            }

            // Return the language with the highest score
            var bestMatch = scores.OrderByDescending(s => s.Value).FirstOrDefault();
            return bestMatch.Value > 0 ? bestMatch.Key : ProgrammingLanguage.Unknown;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting programming language");
            return ProgrammingLanguage.Unknown;
        }
    }

    /// <summary>
    /// Gets the language name from the programming language enum
    /// </summary>
    /// <param name="language">The programming language</param>
    /// <returns>The language name</returns>
    public string GetLanguageName(ProgrammingLanguage language)
    {
        return language switch
        {
            ProgrammingLanguage.CSharp => "csharp",
            ProgrammingLanguage.FSharp => "fsharp",
            ProgrammingLanguage.JavaScript => "javascript",
            ProgrammingLanguage.TypeScript => "typescript",
            ProgrammingLanguage.Python => "python",
            ProgrammingLanguage.Java => "java",
            ProgrammingLanguage.Cpp => "cpp",
            _ => "unknown"
        };
    }

    /// <summary>
    /// Gets the programming language enum from the language name
    /// </summary>
    /// <param name="languageName">The language name</param>
    /// <returns>The programming language enum</returns>
    public ProgrammingLanguage GetLanguageFromName(string languageName)
    {
        return languageName.ToLowerInvariant() switch
        {
            "csharp" or "c#" => ProgrammingLanguage.CSharp,
            "fsharp" or "f#" => ProgrammingLanguage.FSharp,
            "javascript" or "js" => ProgrammingLanguage.JavaScript,
            "typescript" or "ts" => ProgrammingLanguage.TypeScript,
            "python" or "py" => ProgrammingLanguage.Python,
            "java" => ProgrammingLanguage.Java,
            "cpp" or "c++" => ProgrammingLanguage.Cpp,
            _ => ProgrammingLanguage.Unknown
        };
    }

    /// <summary>
    /// Gets the file extensions for a programming language
    /// </summary>
    /// <param name="language">The programming language</param>
    /// <returns>The file extensions</returns>
    public List<string> GetFileExtensions(ProgrammingLanguage language)
    {
        return _extensionMap
            .Where(kv => kv.Value == language)
            .Select(kv => kv.Key)
            .ToList();
    }

    /// <summary>
    /// Gets all supported programming languages
    /// </summary>
    /// <returns>The supported programming languages</returns>
    public List<ProgrammingLanguage> GetSupportedLanguages()
    {
        return _languagePatterns.Keys.ToList();
    }
}