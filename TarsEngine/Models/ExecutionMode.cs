namespace TarsEngine.Models;

/// <summary>
/// Represents the mode of execution
/// </summary>
public enum ExecutionMode
{
    /// <summary>
    /// Dry run mode (no changes are made)
    /// </summary>
    DryRun,

    /// <summary>
    /// Real mode (changes are made)
    /// </summary>
    Real,

    /// <summary>
    /// Interactive mode (user confirmation is required)
    /// </summary>
    Interactive
}
