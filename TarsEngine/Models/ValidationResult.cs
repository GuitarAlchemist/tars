using System;
using System.Collections.Generic;

namespace TarsEngine.Models;

/// <summary>
/// Represents the result of a validation rule
/// </summary>
public class ValidationResult
{
    /// <summary>
    /// Gets or sets the name of the validation rule
    /// </summary>
    public string RuleName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether the validation rule passed
    /// </summary>
    public bool IsPassed { get; set; }

    /// <summary>
    /// Gets or sets the severity of the validation rule
    /// </summary>
    public ValidationRuleSeverity Severity { get; set; } = ValidationRuleSeverity.Error;

    /// <summary>
    /// Gets or sets the message of the validation result
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the target of the validation rule
    /// </summary>
    public string Target { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp when the validation rule was executed
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the details of the validation result
    /// </summary>
    public string Details { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the exception that caused the validation rule to fail
    /// </summary>
    public Exception? Exception { get; set; }

    /// <summary>
    /// Gets or sets additional metadata about the validation result
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
}
