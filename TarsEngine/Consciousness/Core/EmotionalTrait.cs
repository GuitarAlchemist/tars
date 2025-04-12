using System;

namespace TarsEngine.Consciousness.Core
{
    /// <summary>
    /// Represents an emotional trait that influences emotional responses
    /// </summary>
    public class EmotionalTrait
    {
        /// <summary>
        /// Gets or sets the unique identifier for the trait
        /// </summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the name of the emotional trait
        /// </summary>
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the category of the emotional trait
        /// </summary>
        public string Category { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the intensity of the emotional trait (0.0 to 1.0)
        /// </summary>
        public double Intensity { get; set; }
        
        /// <summary>
        /// Gets or sets the description of the emotional trait
        /// </summary>
        public string Description { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the timestamp when the trait was created or last modified
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }
}
