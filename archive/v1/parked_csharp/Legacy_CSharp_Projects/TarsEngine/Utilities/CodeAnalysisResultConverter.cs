using TarsEngine.Models;
using TarsEngine.Models.Unified;
using ModelCodeAnalysisResult = TarsEngine.Models.CodeAnalysisResult;
using ServiceCodeAnalysisResult = TarsEngine.Services.CodeAnalysisResult;
using ServiceProgrammingLanguage = TarsEngine.Services.ProgrammingLanguage;

namespace TarsEngine.Utilities;

/// <summary>
/// Provides conversion methods between different CodeAnalysisResult classes
/// </summary>
public static class CodeAnalysisResultConverter
{
    /// <summary>
    /// Converts from Models.CodeAnalysisResult to CodeAnalysisResultUnified
    /// </summary>
    public static CodeAnalysisResultUnified ToUnified(ModelCodeAnalysisResult result)
    {
        if (result == null)
        {
            return new CodeAnalysisResultUnified();
        }

        var unified = new CodeAnalysisResultUnified
        {
            Path = result.Path ?? string.Empty,
            FilePath = result.Path ?? string.Empty,
            Language = result.Language ?? string.Empty,
            IsSuccessful = result.IsSuccessful,
            Success = result.IsSuccessful,
            ErrorMessage = result.Errors != null && result.Errors.Any()
                ? string.Join(Environment.NewLine, result.Errors)
                : string.Empty,
            Errors = result.Errors ?? [],
            Issues = result.Issues ?? [],
            Metrics = result.Metrics ?? [],
            Structures = result.Structures ?? [],
            AnalyzedAt = result.AnalyzedAt != default ? result.AnalyzedAt : DateTime.UtcNow
        };

        // These properties might not exist in all versions of CodeAnalysisResult
        try
        {
            // Use reflection to check if these properties exist
            var type = typeof(ModelCodeAnalysisResult);

            if (type.GetProperty("Namespaces") != null)
            {
                unified.Namespaces = result.GetType().GetProperty("Namespaces")?.GetValue(result) as List<string> ??
                                     [];
            }

            if (type.GetProperty("Classes") != null)
            {
                unified.Classes = result.GetType().GetProperty("Classes")?.GetValue(result) as List<string> ?? [];
            }

            if (type.GetProperty("Interfaces") != null)
            {
                unified.Interfaces = result.GetType().GetProperty("Interfaces")?.GetValue(result) as List<string> ??
                                     [];
            }

            if (type.GetProperty("Methods") != null)
            {
                unified.Methods = result.GetType().GetProperty("Methods")?.GetValue(result) as List<string> ?? [];
            }

            if (type.GetProperty("Properties") != null)
            {
                unified.Properties = result.GetType().GetProperty("Properties")?.GetValue(result) as List<string> ??
                                     [];
            }
        }
        catch
        {
            // Ignore any reflection errors
        }

        return unified;
    }

    /// <summary>
    /// Converts from Services.CodeAnalysisResult to CodeAnalysisResultUnified
    /// </summary>
    public static CodeAnalysisResultUnified ToUnified(ServiceCodeAnalysisResult result)
    {
        if (result == null)
        {
            return new CodeAnalysisResultUnified();
        }

        var unified = new CodeAnalysisResultUnified
        {
            FilePath = result.FilePath ?? string.Empty,
            Path = result.FilePath ?? string.Empty,
            Language = ConvertLanguageToString(result.Language),
            IsSuccessful = result.Success,
            Success = result.Success,
            ErrorMessage = result.ErrorMessage ?? string.Empty,
            AnalyzedAt = result.AnalyzedAt != default ? result.AnalyzedAt : DateTime.UtcNow
        };

        // Copy collections if they exist
        if (result.Issues != null)
        {
            unified.Issues = new List<CodeIssue>(result.Issues);
        }

        if (result.Metrics != null)
        {
            unified.Metrics = new List<CodeMetric>(result.Metrics);
        }

        if (result.Structures != null)
        {
            unified.Structures = new List<CodeStructure>(result.Structures);
        }

        if (result.Errors != null)
        {
            unified.Errors = new List<string>(result.Errors);
        }

        // These properties might not exist in all versions of CodeAnalysisResult
        try
        {
            // Use reflection to check if these properties exist
            var type = typeof(ServiceCodeAnalysisResult);

            if (type.GetProperty("Namespaces") != null)
            {
                unified.Namespaces = result.GetType().GetProperty("Namespaces")?.GetValue(result) as List<string> ??
                                     [];
            }

            if (type.GetProperty("Classes") != null)
            {
                unified.Classes = result.GetType().GetProperty("Classes")?.GetValue(result) as List<string> ?? [];
            }

            if (type.GetProperty("Interfaces") != null)
            {
                unified.Interfaces = result.GetType().GetProperty("Interfaces")?.GetValue(result) as List<string> ??
                                     [];
            }

            if (type.GetProperty("Methods") != null)
            {
                unified.Methods = result.GetType().GetProperty("Methods")?.GetValue(result) as List<string> ?? [];
            }

            if (type.GetProperty("Properties") != null)
            {
                unified.Properties = result.GetType().GetProperty("Properties")?.GetValue(result) as List<string> ??
                                     [];
            }

            if (type.GetProperty("Functions") != null)
            {
                unified.Methods = result.GetType().GetProperty("Functions")?.GetValue(result) as List<string> ?? unified.Methods;
            }
        }
        catch
        {
            // Ignore any reflection errors
        }

        return unified;
    }

    /// <summary>
    /// Converts from CodeAnalysisResultUnified to Models.CodeAnalysisResult
    /// </summary>
    public static ModelCodeAnalysisResult ToModelResult(CodeAnalysisResultUnified result)
    {
        if (result == null)
        {
            return new ModelCodeAnalysisResult
            {
                IsSuccessful = false,
                Errors = ["Null unified result"]
            };
        }

        var modelResult = new ModelCodeAnalysisResult
        {
            Path = result.FilePath ?? result.Path ?? string.Empty,
            Language = result.Language ?? string.Empty,
            IsSuccessful = result.IsSuccessful || result.Success,
            Errors = result.Errors ?? [],
            Issues = result.Issues ?? [],
            Metrics = result.Metrics ?? [],
            Structures = result.Structures ?? [],
            AnalyzedAt = result.AnalyzedAt
        };

        // Check if the ModelCodeAnalysisResult has these properties before setting them
        var type = typeof(ModelCodeAnalysisResult);

        if (type.GetProperty("Namespaces") != null && result.Namespaces != null)
        {
            type.GetProperty("Namespaces")?.SetValue(modelResult, result.Namespaces);
        }

        if (type.GetProperty("Classes") != null && result.Classes != null)
        {
            type.GetProperty("Classes")?.SetValue(modelResult, result.Classes);
        }

        if (type.GetProperty("Interfaces") != null && result.Interfaces != null)
        {
            type.GetProperty("Interfaces")?.SetValue(modelResult, result.Interfaces);
        }

        if (type.GetProperty("Methods") != null && result.Methods != null)
        {
            type.GetProperty("Methods")?.SetValue(modelResult, result.Methods);
        }

        if (type.GetProperty("Properties") != null && result.Properties != null)
        {
            type.GetProperty("Properties")?.SetValue(modelResult, result.Properties);
        }

        return modelResult;
    }

    /// <summary>
    /// Converts from CodeAnalysisResultUnified to Services.CodeAnalysisResult
    /// </summary>
    public static ServiceCodeAnalysisResult ToServiceResult(CodeAnalysisResultUnified result)
    {
        if (result == null)
        {
            return new ServiceCodeAnalysisResult
            {
                FilePath = "unknown",
                Success = false,
                ErrorMessage = "Null unified result"
            };
        }

        var serviceResult = new ServiceCodeAnalysisResult
        {
            FilePath = result.FilePath ?? result.Path ?? string.Empty,
            Language = ConvertLanguageFromString(result.Language),
            Success = result.IsSuccessful || result.Success,
            ErrorMessage = result.ErrorMessage ?? (result.Errors != null && result.Errors.Any()
                ? string.Join(Environment.NewLine, result.Errors)
                : string.Empty),
            AnalyzedAt = result.AnalyzedAt
        };

        // Copy collections if they exist
        if (result.Issues != null)
        {
            serviceResult.Issues = new List<CodeIssue>(result.Issues);
        }

        if (result.Metrics != null)
        {
            serviceResult.Metrics = new List<CodeMetric>(result.Metrics);
        }

        if (result.Structures != null)
        {
            serviceResult.Structures = new List<CodeStructure>(result.Structures);
        }

        if (result.Errors != null)
        {
            serviceResult.Errors = new List<string>(result.Errors);
        }

        // Check if the ServiceCodeAnalysisResult has these properties before setting them
        var type = typeof(ServiceCodeAnalysisResult);

        if (type.GetProperty("Namespaces") != null && result.Namespaces != null)
        {
            type.GetProperty("Namespaces")?.SetValue(serviceResult, result.Namespaces);
        }

        if (type.GetProperty("Classes") != null && result.Classes != null)
        {
            type.GetProperty("Classes")?.SetValue(serviceResult, result.Classes);
        }

        if (type.GetProperty("Interfaces") != null && result.Interfaces != null)
        {
            type.GetProperty("Interfaces")?.SetValue(serviceResult, result.Interfaces);
        }

        if (type.GetProperty("Methods") != null && result.Methods != null)
        {
            type.GetProperty("Methods")?.SetValue(serviceResult, result.Methods);
        }

        if (type.GetProperty("Properties") != null && result.Properties != null)
        {
            type.GetProperty("Properties")?.SetValue(serviceResult, result.Properties);
        }

        if (type.GetProperty("Functions") != null && result.Methods != null)
        {
            type.GetProperty("Functions")?.SetValue(serviceResult, result.Methods);
        }

        return serviceResult;
    }

    /// <summary>
    /// Converts a ServiceProgrammingLanguage enum to a language string
    /// </summary>
    /// <param name="language">The ServiceProgrammingLanguage enum value</param>
    /// <returns>The corresponding language string</returns>
    private static string ConvertLanguageToString(ServiceProgrammingLanguage language)
    {
        return language switch
        {
            ServiceProgrammingLanguage.CSharp => "csharp",
            ServiceProgrammingLanguage.FSharp => "fsharp",
            ServiceProgrammingLanguage.JavaScript => "javascript",
            ServiceProgrammingLanguage.TypeScript => "typescript",
            ServiceProgrammingLanguage.Python => "python",
            ServiceProgrammingLanguage.Java => "java",
            ServiceProgrammingLanguage.Cpp => "cpp",
            _ => "unknown"
        };
    }

    /// <summary>
    /// Converts a language string to a ServiceProgrammingLanguage enum
    /// </summary>
    /// <param name="language">The language string</param>
    /// <returns>The corresponding ServiceProgrammingLanguage enum value</returns>
    private static ServiceProgrammingLanguage ConvertLanguageFromString(string language)
    {
        return language?.ToLowerInvariant() switch
        {
            "csharp" => ServiceProgrammingLanguage.CSharp,
            "fsharp" => ServiceProgrammingLanguage.FSharp,
            "javascript" => ServiceProgrammingLanguage.JavaScript,
            "typescript" => ServiceProgrammingLanguage.TypeScript,
            "python" => ServiceProgrammingLanguage.Python,
            "java" => ServiceProgrammingLanguage.Java,
            "cpp" => ServiceProgrammingLanguage.Cpp,
            _ => ServiceProgrammingLanguage.Unknown
        };
    }
}