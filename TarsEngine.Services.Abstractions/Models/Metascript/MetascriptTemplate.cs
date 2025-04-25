using TarsEngine.Services.Abstractions.Models;

namespace TarsEngine.Services.Abstractions.Models.Metascript
{
    /// <summary>
    /// Represents a Metascript template.
    /// </summary>
    public class MetascriptTemplate : IEntity
    {
        /// <summary>
        /// Gets or sets the unique identifier for the template.
        /// </summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Gets or sets the name of the template.
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the description of the template.
        /// </summary>
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the content of the template.
        /// </summary>
        public string Content { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the parameters required by the template.
        /// </summary>
        public List<MetascriptParameter> Parameters { get; set; } = new List<MetascriptParameter>();

        /// <summary>
        /// Gets or sets the creation time of the template.
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets the last update time of the template.
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets the author of the template.
        /// </summary>
        public string Author { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the version of the template.
        /// </summary>
        public string Version { get; set; } = "1.0.0";

        /// <summary>
        /// Gets or sets the tags associated with the template.
        /// </summary>
        public List<string> Tags { get; set; } = new List<string>();
    }
}
