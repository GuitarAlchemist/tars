namespace TarsEngine.Services.Abstractions.Models.Metascript
{
    /// <summary>
    /// Represents the result of validating a Metascript.
    /// </summary>
    public class MetascriptValidationResult
    {
        /// <summary>
        /// Gets or sets a value indicating whether the validation was successful.
        /// </summary>
        public bool IsValid { get; set; }

        /// <summary>
        /// Gets or sets the list of validation errors.
        /// </summary>
        public List<string> Errors { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets the list of validation warnings.
        /// </summary>
        public List<string> Warnings { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets the status of the validation.
        /// </summary>
        public MetascriptValidationStatus Status { get; set; } = MetascriptValidationStatus.Unknown;

        /// <summary>
        /// Gets or sets the timestamp when the validation was performed.
        /// </summary>
        public DateTime ValidationTimestamp { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets a value indicating whether there are any errors.
        /// </summary>
        public bool HasErrors => Errors.Count > 0;

        /// <summary>
        /// Gets a value indicating whether there are any warnings.
        /// </summary>
        public bool HasWarnings => Warnings.Count > 0;
    }
}
