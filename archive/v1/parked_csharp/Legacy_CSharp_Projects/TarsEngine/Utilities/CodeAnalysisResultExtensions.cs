using TarsEngine.Models;
using TarsEngine.Services;
using ModelResult = TarsEngine.Models.CodeAnalysisResult;
using ServiceResult = TarsEngine.Services.CodeAnalysisResult;

namespace TarsEngine.Utilities;

/// <summary>
/// Extension methods for CodeAnalysisResult to provide compatibility across different versions
/// </summary>
public static class CodeAnalysisResultExtensions
{
    /// <summary>
    /// Gets the file path from the Model CodeAnalysisResult
    /// </summary>
    public static string GetFilePath(this ModelResult result)
    {
        // Check if the Path property exists using reflection
        var pathProperty = result.GetType().GetProperty("Path");
        if (pathProperty != null)
        {
            var path = pathProperty.GetValue(result) as string;
            return path ?? string.Empty;
        }

        // Check if the FilePath property exists using reflection
        var filePathProperty = result.GetType().GetProperty("FilePath");
        if (filePathProperty != null)
        {
            var filePath = filePathProperty.GetValue(result) as string;
            return filePath ?? string.Empty;
        }

        return string.Empty;
    }

    /// <summary>
    /// Gets the file path from the Service CodeAnalysisResult
    /// </summary>
    public static string GetFilePath(this ServiceResult result)
    {
        // Check if the Path property exists using reflection
        var pathProperty = result.GetType().GetProperty("Path");
        if (pathProperty != null)
        {
            var path = pathProperty.GetValue(result) as string;
            return path ?? string.Empty;
        }

        // Check if the FilePath property exists using reflection
        var filePathProperty = result.GetType().GetProperty("FilePath");
        if (filePathProperty != null)
        {
            var filePath = filePathProperty.GetValue(result) as string;
            return filePath ?? string.Empty;
        }

        return string.Empty;
    }

    /// <summary>
    /// Gets whether the analysis was successful for Model CodeAnalysisResult
    /// </summary>
    public static bool IsSuccessful(this ModelResult result)
    {
        // Check if the Success property exists using reflection
        var successProperty = result.GetType().GetProperty("Success");
        if (successProperty != null)
        {
            var success = successProperty.GetValue(result);
            return success != null && (bool)success;
        }

        // Check if the IsSuccess property exists using reflection
        var isSuccessProperty = result.GetType().GetProperty("IsSuccess");
        if (isSuccessProperty != null)
        {
            var isSuccess = isSuccessProperty.GetValue(result);
            return isSuccess != null && (bool)isSuccess;
        }

        return true; // Default to true if no property is found
    }

    /// <summary>
    /// Gets whether the analysis was successful for Service CodeAnalysisResult
    /// </summary>
    public static bool IsSuccessful(this ServiceResult result)
    {
        // Check if the Success property exists using reflection
        var successProperty = result.GetType().GetProperty("Success");
        if (successProperty != null)
        {
            var success = successProperty.GetValue(result);
            return success != null && (bool)success;
        }

        // Check if the IsSuccess property exists using reflection
        var isSuccessProperty = result.GetType().GetProperty("IsSuccess");
        if (isSuccessProperty != null)
        {
            var isSuccess = isSuccessProperty.GetValue(result);
            return isSuccess != null && (bool)isSuccess;
        }

        return true; // Default to true if no property is found
    }

    /// <summary>
    /// Gets the error message from the Model CodeAnalysisResult
    /// </summary>
    public static string GetErrorMessage(this ModelResult result)
    {
        // Check if the ErrorMessage property exists using reflection
        var errorMessageProperty = result.GetType().GetProperty("ErrorMessage");
        if (errorMessageProperty != null)
        {
            var errorMessage = errorMessageProperty.GetValue(result) as string;
            return errorMessage ?? string.Empty;
        }

        // Check if the Error property exists using reflection
        var errorProperty = result.GetType().GetProperty("Error");
        if (errorProperty != null)
        {
            var error = errorProperty.GetValue(result) as string;
            return error ?? string.Empty;
        }

        return string.Empty;
    }

    /// <summary>
    /// Gets the error message from the Service CodeAnalysisResult
    /// </summary>
    public static string GetErrorMessage(this ServiceResult result)
    {
        // Check if the ErrorMessage property exists using reflection
        var errorMessageProperty = result.GetType().GetProperty("ErrorMessage");
        if (errorMessageProperty != null)
        {
            var errorMessage = errorMessageProperty.GetValue(result) as string;
            return errorMessage ?? string.Empty;
        }

        // Check if the Error property exists using reflection
        var errorProperty = result.GetType().GetProperty("Error");
        if (errorProperty != null)
        {
            var error = errorProperty.GetValue(result) as string;
            return error ?? string.Empty;
        }

        return string.Empty;
    }

    /// <summary>
    /// Gets the list of issues from the Model CodeAnalysisResult
    /// </summary>
    public static List<CodeIssue> GetIssues(this ModelResult result)
    {
        // Check if the Issues property exists using reflection
        var issuesProperty = result.GetType().GetProperty("Issues");
        if (issuesProperty != null)
        {
            var issues = issuesProperty.GetValue(result) as List<CodeIssue>;
            return issues ?? new List<CodeIssue>();
        }

        return new List<CodeIssue>();
    }

    /// <summary>
    /// Gets the list of issues from the Service CodeAnalysisResult
    /// </summary>
    public static List<CodeIssue> GetIssues(this ServiceResult result)
    {
        // Check if the Issues property exists using reflection
        var issuesProperty = result.GetType().GetProperty("Issues");
        if (issuesProperty != null)
        {
            var issues = issuesProperty.GetValue(result) as List<CodeIssue>;
            return issues ?? new List<CodeIssue>();
        }

        return new List<CodeIssue>();
    }

    /// <summary>
    /// Gets the list of metrics from the Model CodeAnalysisResult
    /// </summary>
    public static List<CodeMetric> GetMetrics(this ModelResult result)
    {
        // Check if the Metrics property exists using reflection
        var metricsProperty = result.GetType().GetProperty("Metrics");
        if (metricsProperty != null)
        {
            var metrics = metricsProperty.GetValue(result) as List<CodeMetric>;
            return metrics ?? new List<CodeMetric>();
        }

        return new List<CodeMetric>();
    }

    /// <summary>
    /// Gets the list of metrics from the Service CodeAnalysisResult
    /// </summary>
    public static List<CodeMetric> GetMetrics(this ServiceResult result)
    {
        // Check if the Metrics property exists using reflection
        var metricsProperty = result.GetType().GetProperty("Metrics");
        if (metricsProperty != null)
        {
            var metrics = metricsProperty.GetValue(result) as List<CodeMetric>;
            return metrics ?? new List<CodeMetric>();
        }

        return new List<CodeMetric>();
    }

    /// <summary>
    /// Gets the list of structures from the Model CodeAnalysisResult
    /// </summary>
    public static List<CodeStructure> GetStructures(this ModelResult result)
    {
        // Provide a default implementation that can be overridden
        return result.Structures ?? new List<CodeStructure>();
    }

    /// <summary>
    /// Gets the list of structures from the Service CodeAnalysisResult
    /// </summary>
    public static List<CodeStructure> GetStructures(this ServiceResult result)
    {
        // Provide a default implementation that can be overridden
        return result.Structures ?? new List<CodeStructure>();
    }

    /// <summary>
    /// Gets the list of errors from the Model CodeAnalysisResult
    /// </summary>
    public static List<string> GetErrors(this ModelResult result)
    {
        // Provide a default implementation that can be overridden
        return result.Errors ?? new List<string>();
    }

    /// <summary>
    /// Gets the list of errors from the Service CodeAnalysisResult
    /// </summary>
    public static List<string> GetErrors(this ServiceResult result)
    {
        // Provide a default implementation that can be overridden
        return result.Errors ?? new List<string>();
    }

    /// <summary>
    /// Gets the programming language from the Model CodeAnalysisResult
    /// </summary>
    public static ProgrammingLanguage GetLanguage(this ModelResult result)
    {
        // Check if the Language property exists using reflection
        var languageProperty = result.GetType().GetProperty("Language");
        if (languageProperty != null)
        {
            var language = languageProperty.GetValue(result);
            if (language != null)
            {
                // Handle the case where Language is a string
                if (language is string languageStr)
                {
                    // Try to parse the string to ProgrammingLanguage
                    if (Enum.TryParse<ProgrammingLanguage>(languageStr, out var parsedLanguage))
                    {
                        return parsedLanguage;
                    }
                }
                // Handle the case where Language is already a ProgrammingLanguage
                else if (language is ProgrammingLanguage programmingLanguage)
                {
                    return programmingLanguage;
                }
            }
        }

        return ProgrammingLanguage.CSharp; // Default to C# if no property is found
    }

    /// <summary>
    /// Gets the programming language from the Service CodeAnalysisResult
    /// </summary>
    public static ProgrammingLanguage GetLanguage(this ServiceResult result)
    {
        // Check if the Language property exists using reflection
        var languageProperty = result.GetType().GetProperty("Language");
        if (languageProperty != null)
        {
            var language = languageProperty.GetValue(result);
            if (language != null)
            {
                // Handle the case where Language is a string
                if (language is string languageStr)
                {
                    // Try to parse the string to ProgrammingLanguage
                    if (Enum.TryParse<ProgrammingLanguage>(languageStr, out var parsedLanguage))
                    {
                        return parsedLanguage;
                    }
                }
                // Handle the case where Language is already a ProgrammingLanguage
                else if (language is ProgrammingLanguage programmingLanguage)
                {
                    return programmingLanguage;
                }
            }
        }

        return ProgrammingLanguage.CSharp; // Default to C# if no property is found
    }

    /// <summary>
    /// Gets the analyzed timestamp from the Model CodeAnalysisResult
    /// </summary>
    public static DateTime GetAnalyzedAt(this ModelResult result)
    {
        // Provide a default implementation that can be overridden
        return result.AnalyzedAt;
    }

    /// <summary>
    /// Gets the analyzed timestamp from the Service CodeAnalysisResult
    /// </summary>
    public static DateTime GetAnalyzedAt(this ServiceResult result)
    {
        // Provide a default implementation that can be overridden
        return result.AnalyzedAt;
    }
}