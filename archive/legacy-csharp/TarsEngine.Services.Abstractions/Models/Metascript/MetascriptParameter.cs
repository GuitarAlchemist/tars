namespace TarsEngine.Services.Abstractions.Models.Metascript
{
    /// <summary>
    /// Represents a parameter for a Metascript template.
    /// </summary>
    public class MetascriptParameter
    {
        /// <summary>
        /// Gets or sets the name of the parameter.
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the description of the parameter.
        /// </summary>
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the type of the parameter.
        /// </summary>
        public string Type { get; set; } = "string";

        /// <summary>
        /// Gets or sets a value indicating whether the parameter is required.
        /// </summary>
        public bool IsRequired { get; set; } = true;

        /// <summary>
        /// Gets or sets the default value of the parameter.
        /// </summary>
        public object? DefaultValue { get; set; }
    }
}
