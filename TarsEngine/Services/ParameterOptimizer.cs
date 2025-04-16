using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for optimizing metascript parameters
/// </summary>
public class ParameterOptimizer
{
    private readonly ILogger _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="ParameterOptimizer"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public ParameterOptimizer(ILogger logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Optimizes parameter values for a template
    /// </summary>
    /// <param name="parameters">The parameter values</param>
    /// <param name="template">The template</param>
    /// <param name="context">The optimization context</param>
    /// <returns>The optimized parameter values</returns>
    public Dictionary<string, string> OptimizeParameters(Dictionary<string, string> parameters, MetascriptTemplate template, Dictionary<string, object>? context = null)
    {
        try
        {
            _logger.LogInformation("Optimizing parameters for template: {TemplateName} ({TemplateId})", template.Name, template.Id);

            var optimizedParameters = new Dictionary<string, string>(parameters);
            context ??= new Dictionary<string, object>();

            // Optimize parameters based on their type and interdependencies
            foreach (var parameter in template.Parameters)
            {
                if (optimizedParameters.TryGetValue(parameter.Name, out var value))
                {
                    var optimizedValue = OptimizeParameterValue(parameter, value, optimizedParameters, template, context);
                    optimizedParameters[parameter.Name] = optimizedValue;
                }
            }

            return optimizedParameters;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error optimizing parameters for template: {TemplateName} ({TemplateId})", template.Name, template.Id);
            return parameters;
        }
    }

    /// <summary>
    /// Optimizes a parameter value
    /// </summary>
    /// <param name="parameter">The parameter</param>
    /// <param name="value">The parameter value</param>
    /// <param name="parameters">All parameter values</param>
    /// <param name="template">The template</param>
    /// <param name="context">The optimization context</param>
    /// <returns>The optimized parameter value</returns>
    private string OptimizeParameterValue(MetascriptParameter parameter, string value, Dictionary<string, string> parameters, MetascriptTemplate template, Dictionary<string, object> context)
    {
        try
        {
            // Optimize parameter value based on type
            switch (parameter.Type)
            {
                case MetascriptParameterType.Integer:
                    return OptimizeIntegerParameter(parameter, value, parameters, context);

                case MetascriptParameterType.Float:
                    return OptimizeFloatParameter(parameter, value, parameters, context);

                case MetascriptParameterType.String:
                    return OptimizeStringParameter(parameter, value, parameters, context);

                case MetascriptParameterType.Code:
                    return OptimizeCodeParameter(parameter, value, parameters, context);

                case MetascriptParameterType.FilePath:
                    return OptimizeFilePathParameter(parameter, value, parameters, context);

                case MetascriptParameterType.DirectoryPath:
                    return OptimizeDirectoryPathParameter(parameter, value, parameters, context);

                default:
                    return value;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error optimizing parameter value: {ParameterName}", parameter.Name);
            return value;
        }
    }

    /// <summary>
    /// Optimizes an integer parameter value
    /// </summary>
    /// <param name="parameter">The parameter</param>
    /// <param name="value">The parameter value</param>
    /// <param name="parameters">All parameter values</param>
    /// <param name="context">The optimization context</param>
    /// <returns>The optimized parameter value</returns>
    private string OptimizeIntegerParameter(MetascriptParameter parameter, string value, Dictionary<string, string> parameters, Dictionary<string, object> context)
    {
        if (!int.TryParse(value, out var intValue))
        {
            return value;
        }

        // Apply constraints
        if (parameter.MinValue.HasValue)
        {
            intValue = Math.Max(intValue, (int)parameter.MinValue.Value);
        }

        if (parameter.MaxValue.HasValue)
        {
            intValue = Math.Min(intValue, (int)parameter.MaxValue.Value);
        }

        // Apply optimizations based on context
        if (context.TryGetValue("OptimizeForPerformance", out var optimizeForPerformance) && (bool)optimizeForPerformance)
        {
            // Optimize for performance (e.g., increase buffer sizes, thread counts, etc.)
            if (parameter.Name.Contains("BufferSize") || parameter.Name.Contains("CacheSize"))
            {
                intValue = Math.Max(intValue, 4096);
            }
            else if (parameter.Name.Contains("ThreadCount") || parameter.Name.Contains("ParallelCount"))
            {
                intValue = Math.Max(intValue, Environment.ProcessorCount);
            }
            else if (parameter.Name.Contains("Timeout"))
            {
                intValue = Math.Max(intValue, 30000);
            }
        }
        else if (context.TryGetValue("OptimizeForMemory", out var optimizeForMemory) && (bool)optimizeForMemory)
        {
            // Optimize for memory usage (e.g., decrease buffer sizes, cache sizes, etc.)
            if (parameter.Name.Contains("BufferSize") || parameter.Name.Contains("CacheSize"))
            {
                intValue = Math.Min(intValue, 1024);
            }
            else if (parameter.Name.Contains("ThreadCount") || parameter.Name.Contains("ParallelCount"))
            {
                intValue = Math.Min(intValue, Math.Max(1, Environment.ProcessorCount / 2));
            }
        }

        return intValue.ToString();
    }

    /// <summary>
    /// Optimizes a float parameter value
    /// </summary>
    /// <param name="parameter">The parameter</param>
    /// <param name="value">The parameter value</param>
    /// <param name="parameters">All parameter values</param>
    /// <param name="context">The optimization context</param>
    /// <returns>The optimized parameter value</returns>
    private string OptimizeFloatParameter(MetascriptParameter parameter, string value, Dictionary<string, string> parameters, Dictionary<string, object> context)
    {
        if (!double.TryParse(value, out var doubleValue))
        {
            return value;
        }

        // Apply constraints
        if (parameter.MinValue.HasValue)
        {
            doubleValue = Math.Max(doubleValue, parameter.MinValue.Value);
        }

        if (parameter.MaxValue.HasValue)
        {
            doubleValue = Math.Min(doubleValue, parameter.MaxValue.Value);
        }

        // Apply optimizations based on context
        if (context.TryGetValue("OptimizeForPrecision", out var optimizeForPrecision) && (bool)optimizeForPrecision)
        {
            // Optimize for precision (e.g., increase decimal places, etc.)
            if (parameter.Name.Contains("Tolerance") || parameter.Name.Contains("Epsilon"))
            {
                doubleValue = Math.Min(doubleValue, 1e-10);
            }
        }

        return doubleValue.ToString();
    }

    /// <summary>
    /// Optimizes a string parameter value
    /// </summary>
    /// <param name="parameter">The parameter</param>
    /// <param name="value">The parameter value</param>
    /// <param name="parameters">All parameter values</param>
    /// <param name="context">The optimization context</param>
    /// <returns>The optimized parameter value</returns>
    private string OptimizeStringParameter(MetascriptParameter parameter, string value, Dictionary<string, string> parameters, Dictionary<string, object> context)
    {
        // Apply constraints
        if (parameter.MinLength.HasValue && value.Length < parameter.MinLength)
        {
            value = value.PadRight((int)parameter.MinLength.Value);
        }

        if (parameter.MaxLength.HasValue && value.Length > parameter.MaxLength)
        {
            value = value.Substring(0, (int)parameter.MaxLength.Value);
        }

        // Apply optimizations based on context
        if (context.TryGetValue("OptimizeForReadability", out var optimizeForReadability) && (bool)optimizeForReadability)
        {
            // Optimize for readability (e.g., capitalize first letter, etc.)
            if (parameter.Name.Contains("Name") || parameter.Name.Contains("Title"))
            {
                if (!string.IsNullOrEmpty(value))
                {
                    value = char.ToUpper(value[0]) + value.Substring(1);
                }
            }
        }

        return value;
    }

    /// <summary>
    /// Optimizes a code parameter value
    /// </summary>
    /// <param name="parameter">The parameter</param>
    /// <param name="value">The parameter value</param>
    /// <param name="parameters">All parameter values</param>
    /// <param name="context">The optimization context</param>
    /// <returns>The optimized parameter value</returns>
    private string OptimizeCodeParameter(MetascriptParameter parameter, string value, Dictionary<string, string> parameters, Dictionary<string, object> context)
    {
        // Apply optimizations based on context
        if (context.TryGetValue("OptimizeForReadability", out var optimizeForReadability) && (bool)optimizeForReadability)
        {
            // Optimize for readability (e.g., add indentation, etc.)
            var lines = value.Split('\n');
            var indentedLines = new List<string>();
            var indentLevel = 0;

            foreach (var line in lines)
            {
                var trimmedLine = line.Trim();
                if (trimmedLine.EndsWith('{'))
                {
                    indentedLines.Add(new string(' ', indentLevel * 4) + trimmedLine);
                    indentLevel++;
                }
                else if (trimmedLine.StartsWith('}'))
                {
                    indentLevel = Math.Max(0, indentLevel - 1);
                    indentedLines.Add(new string(' ', indentLevel * 4) + trimmedLine);
                }
                else
                {
                    indentedLines.Add(new string(' ', indentLevel * 4) + trimmedLine);
                }
            }

            value = string.Join('\n', indentedLines);
        }

        return value;
    }

    /// <summary>
    /// Optimizes a file path parameter value
    /// </summary>
    /// <param name="parameter">The parameter</param>
    /// <param name="value">The parameter value</param>
    /// <param name="parameters">All parameter values</param>
    /// <param name="context">The optimization context</param>
    /// <returns>The optimized parameter value</returns>
    private string OptimizeFilePathParameter(MetascriptParameter parameter, string value, Dictionary<string, string> parameters, Dictionary<string, object> context)
    {
        // Normalize file path
        value = value.Replace('\\', '/');

        // Ensure file extension
        if (context.TryGetValue("FileExtension", out var fileExtension) && !string.IsNullOrEmpty((string)fileExtension))
        {
            var extension = (string)fileExtension;
            if (!extension.StartsWith('.'))
            {
                extension = '.' + extension;
            }

            if (!value.EndsWith(extension, StringComparison.OrdinalIgnoreCase))
            {
                value = Path.ChangeExtension(value, extension.TrimStart('.'));
            }
        }

        return value;
    }

    /// <summary>
    /// Optimizes a directory path parameter value
    /// </summary>
    /// <param name="parameter">The parameter</param>
    /// <param name="value">The parameter value</param>
    /// <param name="parameters">All parameter values</param>
    /// <param name="context">The optimization context</param>
    /// <returns>The optimized parameter value</returns>
    private string OptimizeDirectoryPathParameter(MetascriptParameter parameter, string value, Dictionary<string, string> parameters, Dictionary<string, object> context)
    {
        // Normalize directory path
        value = value.Replace('\\', '/');

        // Ensure trailing slash
        if (!value.EndsWith('/'))
        {
            value += '/';
        }

        return value;
    }
}
