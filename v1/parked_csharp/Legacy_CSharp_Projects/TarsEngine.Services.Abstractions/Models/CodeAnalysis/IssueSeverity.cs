namespace TarsEngine.Services.Abstractions.Models.CodeAnalysis
{
    /// <summary>
    /// Represents the severity level of a code issue.
    /// </summary>
    public enum IssueSeverity
    {
        /// <summary>
        /// Informational issue that does not affect code quality.
        /// </summary>
        Info = 0,

        /// <summary>
        /// Minor issue that might affect code quality.
        /// </summary>
        Minor = 1,

        /// <summary>
        /// Moderate issue that affects code quality.
        /// </summary>
        Moderate = 2,

        /// <summary>
        /// Major issue that significantly affects code quality.
        /// </summary>
        Major = 3,

        /// <summary>
        /// Critical issue that severely affects code quality or functionality.
        /// </summary>
        Critical = 4
    }
}
