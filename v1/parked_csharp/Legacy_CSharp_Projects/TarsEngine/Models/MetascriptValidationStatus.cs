namespace TarsEngine.Models;

/// <summary>
/// Represents the validation status of a metascript
/// </summary>
public enum MetascriptValidationStatus
{
    /// <summary>
    /// Metascript has not been validated
    /// </summary>
    NotValidated,

    /// <summary>
    /// Metascript validation is in progress
    /// </summary>
    Validating,

    /// <summary>
    /// Metascript validation completed successfully
    /// </summary>
    Valid,

    /// <summary>
    /// Metascript validation failed
    /// </summary>
    Invalid,

    /// <summary>
    /// Metascript validation has warnings
    /// </summary>
    Warning,

    /// <summary>
    /// Metascript validation status is unknown
    /// </summary>
    Unknown
}
