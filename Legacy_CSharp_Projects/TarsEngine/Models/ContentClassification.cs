namespace TarsEngine.Models;

/// <summary>
/// Represents the classification of content
/// </summary>
public class ContentClassification
{
    /// <summary>
    /// Gets or sets the unique identifier for the classification
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the content being classified
    /// </summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the primary category of the content
    /// </summary>
    public ContentCategory PrimaryCategory { get; set; }

    /// <summary>
    /// Gets or sets the secondary categories of the content
    /// </summary>
    public List<ContentCategory> SecondaryCategories { get; set; } = new();

    /// <summary>
    /// Gets or sets the tags associated with the content
    /// </summary>
    public List<string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets the relevance score (0-1) of the content
    /// </summary>
    public double RelevanceScore { get; set; }

    /// <summary>
    /// Gets or sets the confidence score (0-1) of the classification
    /// </summary>
    public double ConfidenceScore { get; set; }

    /// <summary>
    /// Gets or sets the quality score (0-1) of the content
    /// </summary>
    public double QualityScore { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the classification was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the source of the classification (e.g., rule-based, ML model)
    /// </summary>
    public string ClassificationSource { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the metadata associated with the classification
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();

    /// <summary>
    /// Gets or sets the related content IDs
    /// </summary>
    public List<string> RelatedContentIds { get; set; } = new();
}

/// <summary>
/// Represents the category of content
/// </summary>
public enum ContentCategory
{
    /// <summary>
    /// Unknown category
    /// </summary>
    Unknown,

    /// <summary>
    /// Conceptual explanation
    /// </summary>
    Concept,

    /// <summary>
    /// Code example
    /// </summary>
    CodeExample,

    /// <summary>
    /// Algorithm description
    /// </summary>
    Algorithm,

    /// <summary>
    /// Design pattern
    /// </summary>
    DesignPattern,

    /// <summary>
    /// Architecture description
    /// </summary>
    Architecture,

    /// <summary>
    /// Best practice
    /// </summary>
    BestPractice,

    /// <summary>
    /// Tutorial or guide
    /// </summary>
    Tutorial,

    /// <summary>
    /// API documentation
    /// </summary>
    ApiDoc,

    /// <summary>
    /// Question or inquiry
    /// </summary>
    Question,

    /// <summary>
    /// Answer or response
    /// </summary>
    Answer,

    /// <summary>
    /// Insight or reflection
    /// </summary>
    Insight,

    /// <summary>
    /// Problem description
    /// </summary>
    Problem,

    /// <summary>
    /// Solution description
    /// </summary>
    Solution,

    /// <summary>
    /// Testing approach
    /// </summary>
    Testing,

    /// <summary>
    /// Performance optimization
    /// </summary>
    Performance,

    /// <summary>
    /// Security consideration
    /// </summary>
    Security,

    /// <summary>
    /// User interface design
    /// </summary>
    UserInterface,

    /// <summary>
    /// Data structure
    /// </summary>
    DataStructure,

    /// <summary>
    /// Database design
    /// </summary>
    Database,

    /// <summary>
    /// Configuration information
    /// </summary>
    Configuration,

    /// <summary>
    /// Deployment information
    /// </summary>
    Deployment,

    /// <summary>
    /// Error handling
    /// </summary>
    ErrorHandling,

    /// <summary>
    /// Debugging information
    /// </summary>
    Debugging,

    /// <summary>
    /// Logging information
    /// </summary>
    Logging,

    /// <summary>
    /// Monitoring information
    /// </summary>
    Monitoring,

    /// <summary>
    /// General information
    /// </summary>
    General
}

/// <summary>
/// Represents a classification rule
/// </summary>
public class ClassificationRule
{
    /// <summary>
    /// Gets or sets the unique identifier for the rule
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the name of the rule
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description of the rule
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the category to assign if the rule matches
    /// </summary>
    public ContentCategory Category { get; set; }

    /// <summary>
    /// Gets or sets the keywords that trigger this rule
    /// </summary>
    public List<string> Keywords { get; set; } = new();

    /// <summary>
    /// Gets or sets the regular expression patterns that trigger this rule
    /// </summary>
    public List<string> Patterns { get; set; } = new();

    /// <summary>
    /// Gets or sets the minimum confidence score for this rule to apply
    /// </summary>
    public double MinConfidence { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the weight of this rule (0-1)
    /// </summary>
    public double Weight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether this rule is enabled
    /// </summary>
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the tags to apply if the rule matches
    /// </summary>
    public List<string> Tags { get; set; } = new();
}

/// <summary>
/// Represents a batch of content classifications
/// </summary>
public class ContentClassificationBatch
{
    /// <summary>
    /// Gets or sets the unique identifier for the batch
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the classifications in the batch
    /// </summary>
    public List<ContentClassification> Classifications { get; set; } = new();

    /// <summary>
    /// Gets or sets the timestamp when the batch was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the source of the batch
    /// </summary>
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the metadata associated with the batch
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}
