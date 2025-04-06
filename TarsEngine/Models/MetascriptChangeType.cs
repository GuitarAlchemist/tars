namespace TarsEngine.Models;

/// <summary>
/// Represents the type of change made by a metascript
/// </summary>
public enum MetascriptChangeType
{
    /// <summary>
    /// File creation
    /// </summary>
    Creation,

    /// <summary>
    /// File modification
    /// </summary>
    Modification,

    /// <summary>
    /// File deletion
    /// </summary>
    Deletion,

    /// <summary>
    /// File renaming
    /// </summary>
    Rename,

    /// <summary>
    /// File moving
    /// </summary>
    Move,

    /// <summary>
    /// File copying
    /// </summary>
    Copy,

    /// <summary>
    /// Other change type
    /// </summary>
    Other
}
