using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace TarsEngine.Models;

/// <summary>
/// Represents a knowledge item extracted from content
/// </summary>
public class KnowledgeItem
{
    /// <summary>
    /// Gets or sets the unique identifier for the knowledge item
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the type of knowledge
    /// </summary>
    public KnowledgeType Type { get; set; }

    /// <summary>
    /// Gets or sets the content of the knowledge item
    /// </summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the source of the knowledge item
    /// </summary>
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the context of the knowledge item
    /// </summary>
    public string Context { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the confidence score (0-1) of the knowledge item
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// Gets or sets the relevance score (0-1) of the knowledge item
    /// </summary>
    public double Relevance { get; set; }

    /// <summary>
    /// Gets or sets the timestamp when the knowledge item was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the timestamp when the knowledge item was last updated
    /// </summary>
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the tags associated with the knowledge item
    /// </summary>
    public List<string> Tags { get; set; } = new();

    /// <summary>
    /// Gets or sets the metadata associated with the knowledge item
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();

    /// <summary>
    /// Gets or sets the related knowledge item IDs
    /// </summary>
    public List<string> RelatedIds { get; set; } = new();

    /// <summary>
    /// Gets or sets the validation status of the knowledge item
    /// </summary>
    public ValidationStatus ValidationStatus { get; set; } = ValidationStatus.Unvalidated;

    /// <summary>
    /// Gets or sets the validation notes for the knowledge item
    /// </summary>
    public string ValidationNotes { get; set; } = string.Empty;
}

/// <summary>
/// Represents the type of knowledge
/// </summary>
public enum KnowledgeType
{
    /// <summary>
    /// Unknown knowledge type
    /// </summary>
    Unknown,

    /// <summary>
    /// Concept or definition
    /// </summary>
    Concept,

    /// <summary>
    /// Code pattern or idiom
    /// </summary>
    CodePattern,

    /// <summary>
    /// Algorithm or technique
    /// </summary>
    Algorithm,

    /// <summary>
    /// Design pattern
    /// </summary>
    DesignPattern,

    /// <summary>
    /// Best practice
    /// </summary>
    BestPractice,

    /// <summary>
    /// API usage example
    /// </summary>
    ApiUsage,

    /// <summary>
    /// Error pattern or anti-pattern
    /// </summary>
    ErrorPattern,

    /// <summary>
    /// Performance optimization
    /// </summary>
    Performance,

    /// <summary>
    /// Security consideration
    /// </summary>
    Security,

    /// <summary>
    /// Testing approach
    /// </summary>
    Testing,

    /// <summary>
    /// Insight or reflection
    /// </summary>
    Insight,

    /// <summary>
    /// Question or inquiry
    /// </summary>
    Question,

    /// <summary>
    /// Answer or response
    /// </summary>
    Answer,

    /// <summary>
    /// Tool or library
    /// </summary>
    Tool,

    /// <summary>
    /// Resource or reference
    /// </summary>
    Resource
}

/// <summary>
/// Represents the validation status of a knowledge item
/// </summary>
public enum ValidationStatus
{
    /// <summary>
    /// Not yet validated
    /// </summary>
    Unvalidated,

    /// <summary>
    /// Validated and confirmed
    /// </summary>
    Validated,

    /// <summary>
    /// Rejected as invalid
    /// </summary>
    Rejected,

    /// <summary>
    /// Needs further review
    /// </summary>
    NeedsReview,

    /// <summary>
    /// Partially validated
    /// </summary>
    PartiallyValidated
}

/// <summary>
/// Represents a knowledge extraction result
/// </summary>
public class KnowledgeExtractionResult
{
    /// <summary>
    /// Gets or sets the extracted knowledge items
    /// </summary>
    public List<KnowledgeItem> Items { get; set; } = new();

    /// <summary>
    /// Gets or sets the source of the extraction
    /// </summary>
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp when the extraction was performed
    /// </summary>
    public DateTime ExtractedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the errors that occurred during extraction
    /// </summary>
    public List<string> Errors { get; set; } = new();

    /// <summary>
    /// Gets or sets the warnings that occurred during extraction
    /// </summary>
    public List<string> Warnings { get; set; } = new();

    /// <summary>
    /// Gets or sets the metadata associated with the extraction
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
}

/// <summary>
/// Represents a knowledge validation rule
/// </summary>
public class KnowledgeValidationRule
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
    /// Gets or sets the knowledge types this rule applies to
    /// </summary>
    public List<KnowledgeType> ApplicableTypes { get; set; } = new();

    /// <summary>
    /// Gets or sets the validation criteria
    /// </summary>
    public string ValidationCriteria { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the error message to display when validation fails
    /// </summary>
    public string ErrorMessage { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the severity of the rule
    /// </summary>
    public ValidationSeverity Severity { get; set; } = ValidationSeverity.Warning;

    /// <summary>
    /// Gets or sets whether this rule is enabled
    /// </summary>
    public bool IsEnabled { get; set; } = true;
}

/// <summary>
/// Represents the severity of a validation rule
/// </summary>
public enum ValidationSeverity
{
    /// <summary>
    /// Information only
    /// </summary>
    Info,

    /// <summary>
    /// Warning
    /// </summary>
    Warning,

    /// <summary>
    /// Error
    /// </summary>
    Error,

    /// <summary>
    /// Critical error
    /// </summary>
    Critical
}

/// <summary>
/// Represents a knowledge validation result
/// </summary>
public class KnowledgeValidationResult
{
    /// <summary>
    /// Gets or sets whether the validation passed
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// Gets or sets the validation issues
    /// </summary>
    public List<ValidationIssue> Issues { get; set; } = new();

    /// <summary>
    /// Gets or sets the knowledge item being validated
    /// </summary>
    public KnowledgeItem Item { get; set; } = new();

    /// <summary>
    /// Gets or sets the timestamp when the validation was performed
    /// </summary>
    public DateTime ValidatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Represents a validation issue
/// </summary>
public class ValidationIssue
{
    /// <summary>
    /// Gets or sets the rule that triggered the issue
    /// </summary>
    public string RuleId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the rule name
    /// </summary>
    public string RuleName { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the message describing the issue
    /// </summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the severity of the issue
    /// </summary>
    public ValidationSeverity Severity { get; set; }
}

/// <summary>
/// Represents a knowledge relationship
/// </summary>
public class KnowledgeRelationship
{
    /// <summary>
    /// Gets or sets the unique identifier for the relationship
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the source knowledge item ID
    /// </summary>
    public string SourceId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the target knowledge item ID
    /// </summary>
    public string TargetId { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the type of relationship
    /// </summary>
    public RelationshipType Type { get; set; }

    /// <summary>
    /// Gets or sets the strength of the relationship (0-1)
    /// </summary>
    public double Strength { get; set; }

    /// <summary>
    /// Gets or sets the description of the relationship
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp when the relationship was created
    /// </summary>
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Represents the type of relationship between knowledge items
/// </summary>
public enum RelationshipType
{
    /// <summary>
    /// Unknown relationship type
    /// </summary>
    Unknown,

    /// <summary>
    /// Related to
    /// </summary>
    RelatedTo,

    /// <summary>
    /// Depends on
    /// </summary>
    DependsOn,

    /// <summary>
    /// Is a type of
    /// </summary>
    IsA,

    /// <summary>
    /// Is part of
    /// </summary>
    IsPartOf,

    /// <summary>
    /// Is similar to
    /// </summary>
    IsSimilarTo,

    /// <summary>
    /// Is opposite of
    /// </summary>
    IsOppositeOf,

    /// <summary>
    /// Follows
    /// </summary>
    Follows,

    /// <summary>
    /// Precedes
    /// </summary>
    Precedes,

    /// <summary>
    /// Implements
    /// </summary>
    Implements,

    /// <summary>
    /// Is implemented by
    /// </summary>
    IsImplementedBy,

    /// <summary>
    /// Is alternative to
    /// </summary>
    IsAlternativeTo,

    /// <summary>
    /// Answers
    /// </summary>
    Answers,

    /// <summary>
    /// Questions
    /// </summary>
    Questions
}
