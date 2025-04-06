using System;
using System.Collections.Generic;

namespace TarsEngine.Models;

/// <summary>
/// Represents a metascript template
/// </summary>
public class MetascriptTemplate
{
    /// <summary>
    /// Gets or sets the ID of the template
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the name of the template
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the template
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the template code
    /// </summary>
    public string Code { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the programming language of the template
    /// </summary>
    public string Language { get; set; } = "meta";

    /// <summary>
    /// Gets or sets the version of the template
    /// </summary>
    public string Version { get; set; } = "1.0.0";

    /// <summary>
    /// Gets or sets the timestamp when the template was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the timestamp when the template was last updated
    /// </summary>
    public DateTime? UpdatedAt { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the template was last used
    /// </summary>
    public DateTime? LastUsedAt { get; set; }

    /// <summary>
    /// Gets or sets the number of times the template has been used
    /// </summary>
    public int UsageCount { get; set; }

    /// <summary>
    /// Gets or sets the parameters required by the template
    /// </summary>
    public List<MetascriptParameter> Parameters { get; set; } = new List<MetascriptParameter>();

    /// <summary>
    /// Gets or sets the pattern IDs that can use this template
    /// </summary>
    public List<string> ApplicablePatterns { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the list of tags associated with the template
    /// </summary>
    public List<string> Tags { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets additional metadata about the template
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();

    /// <summary>
    /// Gets or sets the list of example metascripts generated from this template
    /// </summary>
    public List<string> Examples { get; set; } = new List<string>();
}
