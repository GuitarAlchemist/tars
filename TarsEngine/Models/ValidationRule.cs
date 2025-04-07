using System.Collections.Generic;

namespace TarsEngine.Models;

/// <summary>
/// Represents a validation rule for an execution step
/// </summary>
public class ValidationRule
{
    /// <summary>
    /// Gets or sets the name of the validation rule
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the validation rule
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the type of the validation rule
    /// </summary>
    public ValidationRuleType Type { get; set; } = ValidationRuleType.Other;

    /// <summary>
    /// Gets or sets the severity of the validation rule
    /// </summary>
    public ValidationRuleSeverity Severity { get; set; } = ValidationRuleSeverity.Error;

    /// <summary>
    /// Gets or sets the target of the validation rule
    /// </summary>
    public string Target { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the condition of the validation rule
    /// </summary>
    public string Condition { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the parameters of the validation rule
    /// </summary>
    public Dictionary<string, string> Parameters { get; set; } = new Dictionary<string, string>();

    /// <summary>
    /// Gets or sets the error message of the validation rule
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether the validation rule is required
    /// </summary>
    public bool IsRequired { get; set; } = true;

    /// <summary>
    /// Gets or sets additional metadata about the validation rule
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
}
