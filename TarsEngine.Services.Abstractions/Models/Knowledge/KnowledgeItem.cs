namespace TarsEngine.Services.Abstractions.Models.Knowledge
{
    /// <summary>
    /// Represents a knowledge item stored in the repository.
    /// </summary>
    public class KnowledgeItem : IEntity
    {
        /// <summary>
        /// Gets or sets the unique identifier for the knowledge item.
        /// </summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Gets or sets the title of the knowledge item.
        /// </summary>
        public string Title { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the content of the knowledge item.
        /// </summary>
        public string Content { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the type of the knowledge item.
        /// </summary>
        public KnowledgeType Type { get; set; } = KnowledgeType.General;

        /// <summary>
        /// Gets or sets the tags associated with the knowledge item.
        /// </summary>
        public List<string> Tags { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets the creation time of the knowledge item.
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets the last update time of the knowledge item.
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets the source of the knowledge item.
        /// </summary>
        public string Source { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the confidence score of the knowledge item.
        /// </summary>
        public double Confidence { get; set; } = 1.0;

        /// <summary>
        /// Gets or sets the relevance score of the knowledge item.
        /// </summary>
        public double Relevance { get; set; } = 1.0;
    }
}
