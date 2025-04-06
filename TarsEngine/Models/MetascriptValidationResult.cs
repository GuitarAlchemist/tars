using System;
using System.Collections.Generic;

namespace TarsEngine.Models;

/// <summary>
/// Represents the result of validating a metascript
/// </summary>
public class MetascriptValidationResult
{
    /// <summary>
    /// Gets or sets the ID of the metascript
    /// </summary>
    public string MetascriptId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether the validation was successful
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// Gets or sets the validation status
    /// </summary>
    public MetascriptValidationStatus Status { get; set; } = MetascriptValidationStatus.NotValidated;

    /// <summary>
    /// Gets or sets the validation messages
    /// </summary>
    public List<string> Messages { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the validation errors
    /// </summary>
    public List<string> Errors { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the validation warnings
    /// </summary>
    public List<string> Warnings { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the timestamp when the validation was performed
    /// </summary>
    public DateTime ValidatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the validation time in milliseconds
    /// </summary>
    public long ValidationTimeMs { get; set; }

    /// <summary>
    /// Gets or sets additional metadata about the validation
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
}
