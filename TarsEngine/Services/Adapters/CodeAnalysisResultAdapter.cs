using System;
using System.Linq;
using TarsEngine.Models;
using ModelCodeAnalysisResult = TarsEngine.Models.CodeAnalysisResult;
using ServiceCodeAnalysisResult = TarsEngine.Services.CodeAnalysisResult;
using ServicesProgrammingLanguage = TarsEngine.Services.Models.ProgrammingLanguage;
using ServiceProgrammingLanguage = TarsEngine.Services.ProgrammingLanguage;

namespace TarsEngine.Services.Adapters;

/// <summary>
/// Adapter for converting between different CodeAnalysisResult types
/// </summary>
public static class CodeAnalysisResultAdapter
{
    /// <summary>
    /// Converts a Models.CodeAnalysisResult to a Services.CodeAnalysisResult
    /// </summary>
    /// <param name="modelResult">The Models.CodeAnalysisResult to convert</param>
    /// <returns>The converted Services.CodeAnalysisResult</returns>
    public static ServiceCodeAnalysisResult ToServiceResult(this ModelCodeAnalysisResult modelResult)
    {
        if (modelResult == null)
        {
            return new ServiceCodeAnalysisResult
            {
                Success = false,
                ErrorMessage = "Source result was null"
            };
        }

        return new ServiceCodeAnalysisResult
        {
            FilePath = modelResult.FilePath,
            Language = (ServiceProgrammingLanguage)ConvertLanguageToServiceEnum(modelResult.Language),
            Success = modelResult.Success,
            ErrorMessage = modelResult.Errors.Any() ? string.Join(Environment.NewLine, modelResult.Errors) : string.Empty
        };
    }

    /// <summary>
    /// Converts a Services.CodeAnalysisResult to a Models.CodeAnalysisResult
    /// </summary>
    /// <param name="serviceResult">The Services.CodeAnalysisResult to convert</param>
    /// <returns>The converted Models.CodeAnalysisResult</returns>
    public static ModelCodeAnalysisResult ToModelResult(this ServiceCodeAnalysisResult serviceResult)
    {
        if (serviceResult == null)
        {
            return new ModelCodeAnalysisResult
            {
                Success = false,
                Errors = { "Source result was null" }
            };
        }

        var result = new ModelCodeAnalysisResult
        {
            FilePath = serviceResult.FilePath,
            Path = serviceResult.FilePath,
            Language = ConvertLanguageToString((ServicesProgrammingLanguage)serviceResult.Language),
            Success = serviceResult.Success,
            IsSuccessful = serviceResult.Success
        };

        if (!string.IsNullOrEmpty(serviceResult.ErrorMessage))
        {
            result.Errors.Add(serviceResult.ErrorMessage);
        }

        return result;
    }

    /// <summary>
    /// Converts a language string to a ServicesProgrammingLanguage enum
    /// </summary>
    /// <param name="language">The language string</param>
    /// <returns>The corresponding ServicesProgrammingLanguage enum value</returns>
    private static ServicesProgrammingLanguage ConvertLanguageToServiceEnum(string language)
    {
        return language.ToLowerInvariant() switch
        {
            "csharp" => ServicesProgrammingLanguage.CSharp,
            "fsharp" => ServicesProgrammingLanguage.FSharp,
            "javascript" => ServicesProgrammingLanguage.JavaScript,
            "typescript" => ServicesProgrammingLanguage.TypeScript,
            "python" => ServicesProgrammingLanguage.Python,
            "java" => ServicesProgrammingLanguage.Java,
            "cpp" => ServicesProgrammingLanguage.Cpp,
            _ => ServicesProgrammingLanguage.Unknown
        };
    }

    /// <summary>
    /// Converts a ServicesProgrammingLanguage enum to a language string
    /// </summary>
    /// <param name="language">The ServicesProgrammingLanguage enum value</param>
    /// <returns>The corresponding language string</returns>
    private static string ConvertLanguageToString(ServicesProgrammingLanguage language)
    {
        return language switch
        {
            ServicesProgrammingLanguage.CSharp => "csharp",
            ServicesProgrammingLanguage.FSharp => "fsharp",
            ServicesProgrammingLanguage.JavaScript => "javascript",
            ServicesProgrammingLanguage.TypeScript => "typescript",
            ServicesProgrammingLanguage.Python => "python",
            ServicesProgrammingLanguage.Java => "java",
            ServicesProgrammingLanguage.Cpp => "cpp",
            _ => "unknown"
        };
    }
}
