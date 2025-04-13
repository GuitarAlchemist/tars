namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Service for transferring knowledge between TARS and external systems like Augment Code
/// </summary>
public interface IKnowledgeTransferService
{
    /// <summary>
    /// Extracts knowledge from a conversation or text
    /// </summary>
    /// <param name="text">The text to extract knowledge from</param>
    /// <param name="context">Optional context information</param>
    /// <returns>A collection of knowledge items</returns>
    Task<IEnumerable<KnowledgeItem>> ExtractKnowledgeAsync(string text, Dictionary<string, string>? context = null);

    /// <summary>
    /// Shares knowledge with an external system
    /// </summary>
    /// <param name="knowledgeItems">The knowledge items to share</param>
    /// <param name="target">The target system to share with</param>
    /// <param name="options">Optional sharing options</param>
    /// <returns>A result indicating success or failure</returns>
    Task<KnowledgeTransferResult> ShareKnowledgeAsync(IEnumerable<KnowledgeItem> knowledgeItems, string target, Dictionary<string, string>? options = null);

    /// <summary>
    /// Retrieves knowledge from an external system
    /// </summary>
    /// <param name="query">The query to retrieve knowledge</param>
    /// <param name="source">The source system to retrieve from</param>
    /// <param name="options">Optional retrieval options</param>
    /// <returns>A collection of knowledge items</returns>
    Task<IEnumerable<KnowledgeItem>> RetrieveKnowledgeAsync(string query, string source, Dictionary<string, string>? options = null);

    /// <summary>
    /// Organizes knowledge items into a structured format
    /// </summary>
    /// <param name="knowledgeItems">The knowledge items to organize</param>
    /// <param name="format">The format to organize into</param>
    /// <returns>The organized knowledge</returns>
    Task<OrganizedKnowledge> OrganizeKnowledgeAsync(IEnumerable<KnowledgeItem> knowledgeItems, string format);

    /// <summary>
    /// Applies knowledge to improve code or documentation
    /// </summary>
    /// <param name="knowledgeItems">The knowledge items to apply</param>
    /// <param name="target">The target to apply knowledge to</param>
    /// <param name="options">Optional application options</param>
    /// <returns>A result indicating success or failure</returns>
    Task<KnowledgeApplicationResult> ApplyKnowledgeAsync(IEnumerable<KnowledgeItem> knowledgeItems, string target, Dictionary<string, string>? options = null);
}

/// <summary>
/// Represents a knowledge item
/// </summary>
public class KnowledgeItem
{
    /// <summary>
    /// Gets or sets the unique identifier for the knowledge item
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the type of knowledge
    /// </summary>
    public KnowledgeType Type { get; set; }

    /// <summary>
    /// Gets or sets the content of the knowledge item
    /// </summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the source of the knowledge
    /// </summary>
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence level (0-1)
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the knowledge was acquired
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the metadata associated with the knowledge item
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();

    /// <summary>
    /// Gets or sets the related knowledge items
    /// </summary>
    public List<string> RelatedItems { get; set; } = new();
}

/// <summary>
/// Represents the type of knowledge
/// </summary>
public enum KnowledgeType
{
    /// <summary>
    /// Factual knowledge
    /// </summary>
    Fact,

    /// <summary>
    /// Conceptual knowledge
    /// </summary>
    Concept,

    /// <summary>
    /// Procedural knowledge
    /// </summary>
    Procedure,

    /// <summary>
    /// Code-related knowledge
    /// </summary>
    Code,

    /// <summary>
    /// Architecture-related knowledge
    /// </summary>
    Architecture,

    /// <summary>
    /// Design pattern knowledge
    /// </summary>
    DesignPattern,

    /// <summary>
    /// Best practice knowledge
    /// </summary>
    BestPractice,

    /// <summary>
    /// User preference knowledge
    /// </summary>
    UserPreference,

    /// <summary>
    /// Project-specific knowledge
    /// </summary>
    ProjectSpecific,

    /// <summary>
    /// Other types of knowledge
    /// </summary>
    Other
}

/// <summary>
/// Represents the result of a knowledge transfer operation
/// </summary>
public class KnowledgeTransferResult
{
    /// <summary>
    /// Gets or sets whether the transfer was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets the message associated with the result
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the number of items transferred
    /// </summary>
    public int ItemsTransferred { get; set; }

    /// <summary>
    /// Gets or sets the errors that occurred during transfer
    /// </summary>
    public List<string> Errors { get; set; } = new();

    /// <summary>
    /// Gets or sets the warnings that occurred during transfer
    /// </summary>
    public List<string> Warnings { get; set; } = new();
}

/// <summary>
/// Represents organized knowledge
/// </summary>
public class OrganizedKnowledge
{
    /// <summary>
    /// Gets or sets the format of the organized knowledge
    /// </summary>
    public string Format { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the content of the organized knowledge
    /// </summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the structure of the organized knowledge
    /// </summary>
    public Dictionary<string, object> Structure { get; set; } = new();

    /// <summary>
    /// Gets or sets the metadata associated with the organized knowledge
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}

/// <summary>
/// Represents the result of applying knowledge
/// </summary>
public class KnowledgeApplicationResult
{
    /// <summary>
    /// Gets or sets whether the application was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Gets or sets the message associated with the result
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the changes made during application
    /// </summary>
    public List<string> Changes { get; set; } = new();

    /// <summary>
    /// Gets or sets the errors that occurred during application
    /// </summary>
    public List<string> Errors { get; set; } = new();

    /// <summary>
    /// Gets or sets the warnings that occurred during application
    /// </summary>
    public List<string> Warnings { get; set; } = new();
}
