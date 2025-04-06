using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for filling metascript templates with parameter values
/// </summary>
public class TemplateFiller
{
    private readonly ILogger _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="TemplateFiller"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public TemplateFiller(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Fills a template with parameter values
    /// </summary>
    /// <param name="template">The template to fill</param>
    /// <param name="parameters">The parameter values</param>
    /// <returns>The filled template</returns>
    public string FillTemplate(MetascriptTemplate template, Dictionary<string, string> parameters)
    {
        try
        {
            _logger.LogInformation("Filling template: {TemplateName} ({TemplateId})", template.Name, template.Id);

            var filledTemplate = template.Code;

            // Replace parameter placeholders
            foreach (var parameter in template.Parameters)
            {
                if (parameters.TryGetValue(parameter.Name, out var value))
                {
                    // Replace ${paramName} and $paramName
                    filledTemplate = filledTemplate.Replace($"${{{parameter.Name}}}", value);
                    filledTemplate = filledTemplate.Replace($"${parameter.Name}", value);
                }
                else if (parameter.IsRequired)
                {
                    _logger.LogWarning("Required parameter {ParameterName} not provided", parameter.Name);
                    throw new ArgumentException($"Required parameter {parameter.Name} not provided");
                }
                else if (parameter.DefaultValue != null)
                {
                    // Use default value
                    filledTemplate = filledTemplate.Replace($"${{{parameter.Name}}}", parameter.DefaultValue);
                    filledTemplate = filledTemplate.Replace($"${parameter.Name}", parameter.DefaultValue);
                }
            }

            return filledTemplate;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error filling template: {TemplateName} ({TemplateId})", template.Name, template.Id);
            throw;
        }
    }

    /// <summary>
    /// Extracts parameter values from a pattern match
    /// </summary>
    /// <param name="patternMatch">The pattern match</param>
    /// <param name="template">The template</param>
    /// <returns>The extracted parameter values</returns>
    public Dictionary<string, string> ExtractParametersFromPatternMatch(PatternMatch patternMatch, MetascriptTemplate template)
    {
        try
        {
            _logger.LogInformation("Extracting parameters from pattern match for template: {TemplateName} ({TemplateId})", template.Name, template.Id);

            var parameters = new Dictionary<string, string>();

            // Extract parameters from pattern match
            foreach (var parameter in template.Parameters)
            {
                if (parameter.Source == MetascriptParameterSource.PatternMatch)
                {
                    if (string.IsNullOrEmpty(parameter.SourcePath))
                    {
                        _logger.LogWarning("Parameter {ParameterName} has PatternMatch source but no SourcePath", parameter.Name);
                        continue;
                    }

                    // Extract parameter value from pattern match
                    var value = ExtractValueFromPatternMatch(patternMatch, parameter.SourcePath);
                    if (value != null)
                    {
                        parameters[parameter.Name] = value;
                    }
                    else if (parameter.IsRequired)
                    {
                        _logger.LogWarning("Required parameter {ParameterName} could not be extracted from pattern match", parameter.Name);
                    }
                    else if (parameter.DefaultValue != null)
                    {
                        parameters[parameter.Name] = parameter.DefaultValue;
                    }
                }
                else if (parameter.Source == MetascriptParameterSource.System)
                {
                    // Set system parameters
                    parameters[parameter.Name] = GetSystemParameterValue(parameter.Name, patternMatch);
                }
                else if (parameter.DefaultValue != null)
                {
                    parameters[parameter.Name] = parameter.DefaultValue;
                }
            }

            return parameters;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting parameters from pattern match for template: {TemplateName} ({TemplateId})", template.Name, template.Id);
            return new Dictionary<string, string>();
        }
    }

    /// <summary>
    /// Validates parameter values
    /// </summary>
    /// <param name="parameters">The parameter values</param>
    /// <param name="template">The template</param>
    /// <returns>The validation result</returns>
    public (bool IsValid, List<string> Errors) ValidateParameters(Dictionary<string, string> parameters, MetascriptTemplate template)
    {
        try
        {
            _logger.LogInformation("Validating parameters for template: {TemplateName} ({TemplateId})", template.Name, template.Id);

            var isValid = true;
            var errors = new List<string>();

            // Check that all required parameters are provided
            foreach (var parameter in template.Parameters.Where(p => p.IsRequired))
            {
                if (!parameters.ContainsKey(parameter.Name))
                {
                    isValid = false;
                    errors.Add($"Required parameter {parameter.Name} not provided");
                }
            }

            // Validate parameter values
            foreach (var parameter in template.Parameters)
            {
                if (parameters.TryGetValue(parameter.Name, out var value))
                {
                    // Validate parameter value
                    var (paramValid, paramErrors) = ValidateParameterValue(parameter, value);
                    if (!paramValid)
                    {
                        isValid = false;
                        errors.AddRange(paramErrors);
                    }
                }
            }

            return (isValid, errors);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating parameters for template: {TemplateName} ({TemplateId})", template.Name, template.Id);
            return (false, new List<string> { ex.Message });
        }
    }

    /// <summary>
    /// Validates a parameter value
    /// </summary>
    /// <param name="parameter">The parameter</param>
    /// <param name="value">The parameter value</param>
    /// <returns>The validation result</returns>
    private (bool IsValid, List<string> Errors) ValidateParameterValue(MetascriptParameter parameter, string value)
    {
        var isValid = true;
        var errors = new List<string>();

        try
        {
            // Validate parameter value based on type
            switch (parameter.Type)
            {
                case MetascriptParameterType.Integer:
                    if (!int.TryParse(value, out var intValue))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be an integer");
                    }
                    else
                    {
                        if (parameter.MinValue.HasValue && intValue < parameter.MinValue)
                        {
                            isValid = false;
                            errors.Add($"Parameter {parameter.Name} must be at least {parameter.MinValue}");
                        }
                        if (parameter.MaxValue.HasValue && intValue > parameter.MaxValue)
                        {
                            isValid = false;
                            errors.Add($"Parameter {parameter.Name} must be at most {parameter.MaxValue}");
                        }
                    }
                    break;

                case MetascriptParameterType.Float:
                    if (!double.TryParse(value, out var doubleValue))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be a number");
                    }
                    else
                    {
                        if (parameter.MinValue.HasValue && doubleValue < parameter.MinValue)
                        {
                            isValid = false;
                            errors.Add($"Parameter {parameter.Name} must be at least {parameter.MinValue}");
                        }
                        if (parameter.MaxValue.HasValue && doubleValue > parameter.MaxValue)
                        {
                            isValid = false;
                            errors.Add($"Parameter {parameter.Name} must be at most {parameter.MaxValue}");
                        }
                    }
                    break;

                case MetascriptParameterType.Boolean:
                    if (!bool.TryParse(value, out _))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be a boolean (true or false)");
                    }
                    break;

                case MetascriptParameterType.String:
                    if (parameter.MinLength.HasValue && value.Length < parameter.MinLength)
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be at least {parameter.MinLength} characters long");
                    }
                    if (parameter.MaxLength.HasValue && value.Length > parameter.MaxLength)
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be at most {parameter.MaxLength} characters long");
                    }
                    if (!string.IsNullOrEmpty(parameter.Pattern))
                    {
                        var regex = new Regex(parameter.Pattern);
                        if (!regex.IsMatch(value))
                        {
                            isValid = false;
                            errors.Add($"Parameter {parameter.Name} must match pattern {parameter.Pattern}");
                        }
                    }
                    break;

                case MetascriptParameterType.Enum:
                    if (parameter.AllowedValues != null && !parameter.AllowedValues.Contains(value))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be one of: {string.Join(", ", parameter.AllowedValues)}");
                    }
                    break;

                case MetascriptParameterType.Date:
                    if (!DateTime.TryParse(value, out _))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be a valid date");
                    }
                    break;

                case MetascriptParameterType.Time:
                    if (!TimeSpan.TryParse(value, out _))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be a valid time");
                    }
                    break;

                case MetascriptParameterType.DateTime:
                    if (!DateTime.TryParse(value, out _))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be a valid date and time");
                    }
                    break;

                case MetascriptParameterType.TimeSpan:
                    if (!TimeSpan.TryParse(value, out _))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be a valid time span");
                    }
                    break;

                case MetascriptParameterType.Regex:
                    try
                    {
                        _ = new Regex(value);
                    }
                    catch
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be a valid regular expression");
                    }
                    break;

                case MetascriptParameterType.FilePath:
                    if (!System.IO.Path.IsPathRooted(value) && !value.Contains('/') && !value.Contains('\\'))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be a valid file path");
                    }
                    break;

                case MetascriptParameterType.DirectoryPath:
                    if (!System.IO.Path.IsPathRooted(value) && !value.Contains('/') && !value.Contains('\\'))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be a valid directory path");
                    }
                    break;

                case MetascriptParameterType.Url:
                    if (!Uri.TryCreate(value, UriKind.Absolute, out _))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be a valid URL");
                    }
                    break;

                case MetascriptParameterType.Email:
                    var emailRegex = new Regex(@"^[^@\s]+@[^@\s]+\.[^@\s]+$");
                    if (!emailRegex.IsMatch(value))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be a valid email address");
                    }
                    break;

                case MetascriptParameterType.Phone:
                    var phoneRegex = new Regex(@"^\+?[0-9\s\-\(\)]+$");
                    if (!phoneRegex.IsMatch(value))
                    {
                        isValid = false;
                        errors.Add($"Parameter {parameter.Name} must be a valid phone number");
                    }
                    break;
            }
        }
        catch (Exception ex)
        {
            isValid = false;
            errors.Add($"Error validating parameter {parameter.Name}: {ex.Message}");
        }

        return (isValid, errors);
    }

    /// <summary>
    /// Extracts a value from a pattern match
    /// </summary>
    /// <param name="patternMatch">The pattern match</param>
    /// <param name="sourcePath">The source path</param>
    /// <returns>The extracted value</returns>
    private string? ExtractValueFromPatternMatch(PatternMatch patternMatch, string sourcePath)
    {
        try
        {
            // Handle special source paths
            switch (sourcePath)
            {
                case "MatchedText":
                    return patternMatch.MatchedText;
                case "FilePath":
                    return patternMatch.FilePath;
                case "Language":
                    return patternMatch.Language;
                case "PatternId":
                    return patternMatch.PatternId;
                case "PatternName":
                    return patternMatch.PatternName;
                case "SuggestedReplacement":
                    return patternMatch.SuggestedReplacement;
            }

            // Handle regex capture groups
            if (sourcePath.StartsWith("Capture:"))
            {
                var captureGroup = sourcePath.Substring("Capture:".Length);
                var regex = new Regex(patternMatch.PatternId);
                var match = regex.Match(patternMatch.MatchedText);

                if (match.Success)
                {
                    if (int.TryParse(captureGroup, out var groupIndex))
                    {
                        if (groupIndex >= 0 && groupIndex < match.Groups.Count)
                        {
                            return match.Groups[groupIndex].Value;
                        }
                    }
                    else
                    {
                        if (match.Groups[captureGroup] != null)
                        {
                            return match.Groups[captureGroup].Value;
                        }
                    }
                }
            }

            // Handle metadata
            if (sourcePath.StartsWith("Metadata:"))
            {
                var metadataKey = sourcePath.Substring("Metadata:".Length);
                if (patternMatch.Metadata.TryGetValue(metadataKey, out var metadataValue))
                {
                    return metadataValue;
                }
            }

            // Handle location
            if (sourcePath.StartsWith("Location:"))
            {
                var locationProperty = sourcePath.Substring("Location:".Length);
                switch (locationProperty)
                {
                    case "StartLine":
                        return patternMatch.Location.StartLine.ToString();
                    case "EndLine":
                        return patternMatch.Location.EndLine.ToString();
                    case "StartColumn":
                        return patternMatch.Location.StartColumn.ToString();
                    case "EndColumn":
                        return patternMatch.Location.EndColumn.ToString();
                    case "Namespace":
                        return patternMatch.Location.Namespace;
                    case "ClassName":
                        return patternMatch.Location.ClassName;
                    case "MethodName":
                        return patternMatch.Location.MethodName;
                }
            }

            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting value from pattern match with source path: {SourcePath}", sourcePath);
            return null;
        }
    }

    /// <summary>
    /// Gets a system parameter value
    /// </summary>
    /// <param name="parameterName">The parameter name</param>
    /// <param name="patternMatch">The pattern match</param>
    /// <returns>The system parameter value</returns>
    private string GetSystemParameterValue(string parameterName, PatternMatch patternMatch)
    {
        try
        {
            switch (parameterName)
            {
                case "CurrentDate":
                    return DateTime.Now.ToString("yyyy-MM-dd");
                case "CurrentTime":
                    return DateTime.Now.ToString("HH:mm:ss");
                case "CurrentDateTime":
                    return DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
                case "CurrentYear":
                    return DateTime.Now.Year.ToString();
                case "CurrentMonth":
                    return DateTime.Now.Month.ToString();
                case "CurrentDay":
                    return DateTime.Now.Day.ToString();
                case "CurrentHour":
                    return DateTime.Now.Hour.ToString();
                case "CurrentMinute":
                    return DateTime.Now.Minute.ToString();
                case "CurrentSecond":
                    return DateTime.Now.Second.ToString();
                case "RandomGuid":
                    return Guid.NewGuid().ToString();
                case "RandomNumber":
                    return new Random().Next(1000).ToString();
                case "FileName":
                    return System.IO.Path.GetFileName(patternMatch.FilePath);
                case "FileNameWithoutExtension":
                    return System.IO.Path.GetFileNameWithoutExtension(patternMatch.FilePath);
                case "FileExtension":
                    return System.IO.Path.GetExtension(patternMatch.FilePath);
                case "DirectoryName":
                    return System.IO.Path.GetDirectoryName(patternMatch.FilePath) ?? string.Empty;
                default:
                    return string.Empty;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting system parameter value: {ParameterName}", parameterName);
            return string.Empty;
        }
    }
}
