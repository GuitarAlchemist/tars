using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using TarsEngine.Models;
using TarsEngine.Models.Unified;
using TarsEngine.Utilities;
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
                FilePath = "unknown",
                Success = false,
                ErrorMessage = "Source result was null"
            };
        }

        // Get the Path property value using reflection if FilePath doesn't exist
        string filePath = string.Empty;
        var pathProperty = modelResult.GetType().GetProperty("Path");
        if (pathProperty != null)
        {
            filePath = pathProperty.GetValue(modelResult) as string ?? string.Empty;
        }

        // Get the FilePath property value using reflection if it exists
        var filePathProperty = modelResult.GetType().GetProperty("FilePath");
        if (filePathProperty != null)
        {
            var filePathValue = filePathProperty.GetValue(modelResult) as string;
            if (!string.IsNullOrEmpty(filePathValue))
            {
                filePath = filePathValue;
            }
        }

        // Get the IsSuccessful property value using reflection
        bool isSuccessful = true;
        var isSuccessfulProperty = modelResult.GetType().GetProperty("IsSuccessful");
        if (isSuccessfulProperty != null)
        {
            var isSuccessfulValue = isSuccessfulProperty.GetValue(modelResult);
            if (isSuccessfulValue != null)
            {
                isSuccessful = (bool)isSuccessfulValue;
            }
        }

        // Get the Success property value using reflection if it exists
        var successProperty = modelResult.GetType().GetProperty("Success");
        if (successProperty != null)
        {
            var successValue = successProperty.GetValue(modelResult);
            if (successValue != null)
            {
                isSuccessful = (bool)successValue;
            }
        }

        // Get the error message from Errors collection or ErrorMessage property
        string errorMessage = string.Empty;
        if (modelResult.Errors != null && modelResult.Errors.Any())
        {
            errorMessage = string.Join(Environment.NewLine, modelResult.Errors);
        }
        else
        {
            var errorMessageProperty = modelResult.GetType().GetProperty("ErrorMessage");
            if (errorMessageProperty != null)
            {
                errorMessage = errorMessageProperty.GetValue(modelResult) as string ?? string.Empty;
            }
        }

        // Create the result
        var result = new ServiceCodeAnalysisResult
        {
            FilePath = filePath,
            Language = (ServiceProgrammingLanguage)ConvertLanguageToServiceEnum(modelResult.Language ?? "csharp"),
            Success = isSuccessful,
            ErrorMessage = errorMessage,
            AnalyzedAt = modelResult.AnalyzedAt
        };

        // Add extension properties
        result.Issues = modelResult.Issues?.ToList() ?? new List<CodeIssue>();
        result.Metrics = modelResult.Metrics?.ToList() ?? new List<CodeMetric>();
        result.Structures = modelResult.Structures?.ToList() ?? new List<CodeStructure>();

        // Copy Errors collection if it exists
        if (modelResult.Errors != null)
        {
            result.Errors = new List<string>(modelResult.Errors);
        }

        return result;
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
                IsSuccessful = false,
                Errors = { "Source result was null" }
            };
        }

        // Get the FilePath property value using reflection
        string filePath = string.Empty;
        var filePathProperty = serviceResult.GetType().GetProperty("FilePath");
        if (filePathProperty != null)
        {
            filePath = filePathProperty.GetValue(serviceResult) as string ?? string.Empty;
        }

        // Get the Success property value using reflection
        bool isSuccessful = true;
        var successProperty = serviceResult.GetType().GetProperty("Success");
        if (successProperty != null)
        {
            var successValue = successProperty.GetValue(serviceResult);
            if (successValue != null)
            {
                isSuccessful = (bool)successValue;
            }
        }

        // Get the IsSuccessful property value using reflection if it exists
        var isSuccessfulProperty = serviceResult.GetType().GetProperty("IsSuccessful");
        if (isSuccessfulProperty != null)
        {
            var isSuccessfulValue = isSuccessfulProperty.GetValue(serviceResult);
            if (isSuccessfulValue != null)
            {
                isSuccessful = (bool)isSuccessfulValue;
            }
        }

        // Get the error message
        string errorMessage = string.Empty;
        var errorMessageProperty = serviceResult.GetType().GetProperty("ErrorMessage");
        if (errorMessageProperty != null)
        {
            errorMessage = errorMessageProperty.GetValue(serviceResult) as string ?? string.Empty;
        }

        // Create the result
        var result = new ModelCodeAnalysisResult
        {
            Path = filePath,
            Language = ConvertLanguageToString((ServicesProgrammingLanguage)serviceResult.Language),
            IsSuccessful = isSuccessful,
            AnalyzedAt = serviceResult.AnalyzedAt != default ? serviceResult.AnalyzedAt : DateTime.UtcNow
        };

        // Set FilePath property if it exists
        var resultFilePathProperty = result.GetType().GetProperty("FilePath");
        if (resultFilePathProperty != null)
        {
            resultFilePathProperty.SetValue(result, filePath);
        }

        // Copy collections
        if (serviceResult.Issues != null)
        {
            result.Issues = new List<CodeIssue>(serviceResult.Issues);
        }

        if (serviceResult.Metrics != null)
        {
            result.Metrics = new List<CodeMetric>(serviceResult.Metrics);
        }

        if (serviceResult.Structures != null)
        {
            result.Structures = new List<CodeStructure>(serviceResult.Structures);
        }

        // Add error message to Errors collection
        if (!string.IsNullOrEmpty(errorMessage))
        {
            result.Errors.Add(errorMessage);
        }

        // Copy Errors collection if it exists
        if (serviceResult.Errors != null && serviceResult.Errors.Any())
        {
            foreach (var error in serviceResult.Errors)
            {
                if (!result.Errors.Contains(error))
                {
                    result.Errors.Add(error);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Converts a Models.CodeAnalysisResult to a CodeAnalysisResultUnified
    /// </summary>
    /// <param name="modelResult">The Models.CodeAnalysisResult to convert</param>
    /// <returns>The converted CodeAnalysisResultUnified</returns>
    public static CodeAnalysisResultUnified ToUnified(this ModelCodeAnalysisResult modelResult)
    {
        return CodeAnalysisResultConverter.ToUnified(modelResult);
    }

    /// <summary>
    /// Converts a Services.CodeAnalysisResult to a CodeAnalysisResultUnified
    /// </summary>
    /// <param name="serviceResult">The Services.CodeAnalysisResult to convert</param>
    /// <returns>The converted CodeAnalysisResultUnified</returns>
    public static CodeAnalysisResultUnified ToUnified(this ServiceCodeAnalysisResult serviceResult)
    {
        return CodeAnalysisResultConverter.ToUnified(serviceResult);
    }

    /// <summary>
    /// Converts a CodeAnalysisResultUnified to a Models.CodeAnalysisResult
    /// </summary>
    /// <param name="unifiedResult">The CodeAnalysisResultUnified to convert</param>
    /// <returns>The converted Models.CodeAnalysisResult</returns>
    public static ModelCodeAnalysisResult ToModelResult(this CodeAnalysisResultUnified unifiedResult)
    {
        return CodeAnalysisResultConverter.ToModelResult(unifiedResult);
    }

    /// <summary>
    /// Converts a CodeAnalysisResultUnified to a Services.CodeAnalysisResult
    /// </summary>
    /// <param name="unifiedResult">The CodeAnalysisResultUnified to convert</param>
    /// <returns>The converted Services.CodeAnalysisResult</returns>
    public static ServiceCodeAnalysisResult ToServiceResult(this CodeAnalysisResultUnified unifiedResult)
    {
        return CodeAnalysisResultConverter.ToServiceResult(unifiedResult);
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
