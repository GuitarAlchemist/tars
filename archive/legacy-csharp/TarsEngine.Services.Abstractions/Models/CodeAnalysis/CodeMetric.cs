namespace TarsEngine.Services.Abstractions.Models.CodeAnalysis
{
    /// <summary>
    /// Represents a metric calculated during code analysis.
    /// </summary>
    public class CodeMetric
    {
        /// <summary>
        /// Gets or sets the name of the metric.
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the value of the metric.
        /// </summary>
        public double Value { get; set; }

        /// <summary>
        /// Gets or sets the description of the metric.
        /// </summary>
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the category of the metric.
        /// </summary>
        public string Category { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the minimum acceptable value for the metric.
        /// </summary>
        public double? MinAcceptableValue { get; set; }

        /// <summary>
        /// Gets or sets the maximum acceptable value for the metric.
        /// </summary>
        public double? MaxAcceptableValue { get; set; }

        /// <summary>
        /// Gets a value indicating whether the metric is within acceptable range.
        /// </summary>
        public bool IsWithinAcceptableRange
        {
            get
            {
                bool isAboveMin = !MinAcceptableValue.HasValue || Value >= MinAcceptableValue.Value;
                bool isBelowMax = !MaxAcceptableValue.HasValue || Value <= MaxAcceptableValue.Value;
                return isAboveMin && isBelowMax;
            }
        }
    }
}
