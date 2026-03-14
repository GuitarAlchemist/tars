namespace TarsEngine.Services.Abstractions.Models.Metascript
{
    /// <summary>
    /// Represents the status of a Metascript validation.
    /// </summary>
    public enum MetascriptValidationStatus
    {
        /// <summary>
        /// The validation status is unknown.
        /// </summary>
        Unknown = 0,

        /// <summary>
        /// The validation is in progress.
        /// </summary>
        InProgress = 1,

        /// <summary>
        /// The validation completed successfully.
        /// </summary>
        Completed = 2,

        /// <summary>
        /// The validation failed.
        /// </summary>
        Failed = 3,

        /// <summary>
        /// The validation found errors.
        /// </summary>
        Error = 4,

        /// <summary>
        /// The validation found warnings.
        /// </summary>
        Warning = 5,

        /// <summary>
        /// The validation found both errors and warnings.
        /// </summary>
        ErrorAndWarning = 6
    }
}
