using System.Collections.Generic;

namespace TarsEngine.Models;

/// <summary>
/// Represents a parameter for a metascript template
/// </summary>
public class MetascriptParameter
{
    /// <summary>
    /// Gets or sets the name of the parameter
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the parameter
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the type of the parameter
    /// </summary>
    public MetascriptParameterType Type { get; set; } = MetascriptParameterType.String;

    /// <summary>
    /// Gets or sets whether the parameter is required
    /// </summary>
    public bool IsRequired { get; set; } = true;

    /// <summary>
    /// Gets or sets the default value of the parameter
    /// </summary>
    public string? DefaultValue { get; set; }

    /// <summary>
    /// Gets or sets the minimum value for numeric parameters
    /// </summary>
    public double? MinValue { get; set; }

    /// <summary>
    /// Gets or sets the maximum value for numeric parameters
    /// </summary>
    public double? MaxValue { get; set; }

    /// <summary>
    /// Gets or sets the minimum length for string parameters
    /// </summary>
    public int? MinLength { get; set; }

    /// <summary>
    /// Gets or sets the maximum length for string parameters
    /// </summary>
    public int? MaxLength { get; set; }

    /// <summary>
    /// Gets or sets the pattern for string parameters
    /// </summary>
    public string? Pattern { get; set; }

    /// <summary>
    /// Gets or sets the allowed values for enum parameters
    /// </summary>
    public List<string>? AllowedValues { get; set; }

    /// <summary>
    /// Gets or sets the source of the parameter value
    /// </summary>
    public MetascriptParameterSource Source { get; set; } = MetascriptParameterSource.Manual;

    /// <summary>
    /// Gets or sets the source path for the parameter value
    /// </summary>
    public string? SourcePath { get; set; }

    /// <summary>
    /// Gets or sets additional metadata about the parameter
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
}
