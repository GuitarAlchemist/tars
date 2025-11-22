namespace TarsEngine.Models;

/// <summary>
/// Represents the environment for execution
/// </summary>
public enum ExecutionEnvironment
{
    /// <summary>
    /// Sandbox environment (isolated)
    /// </summary>
    Sandbox,

    /// <summary>
    /// Development environment
    /// </summary>
    Development,

    /// <summary>
    /// Testing environment
    /// </summary>
    Testing,

    /// <summary>
    /// Staging environment
    /// </summary>
    Staging,

    /// <summary>
    /// Production environment
    /// </summary>
    Production
}
