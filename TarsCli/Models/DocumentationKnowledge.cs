using System.Text.Json.Serialization;

namespace TarsCli.Models;

/// <summary>
/// Represents knowledge extracted from documentation
/// </summary>
public class DocumentationKnowledge
{
    /// <summary>
    /// The title of the document
    /// </summary>
    public string Title { get; set; }

    /// <summary>
    /// A concise summary of the document
    /// </summary>
    public string Summary { get; set; }

    /// <summary>
    /// Key concepts and their definitions
    /// </summary>
    [JsonPropertyName("key_concepts")]
    public List<Concept> KeyConcepts { get; set; } = new List<Concept>();

    /// <summary>
    /// Important insights and conclusions
    /// </summary>
    public List<Insight> Insights { get; set; } = new List<Insight>();

    /// <summary>
    /// Technical details and specifications
    /// </summary>
    [JsonPropertyName("technical_details")]
    public List<TechnicalDetail> TechnicalDetails { get; set; } = new List<TechnicalDetail>();

    /// <summary>
    /// Design decisions and rationales
    /// </summary>
    [JsonPropertyName("design_decisions")]
    public List<DesignDecision> DesignDecisions { get; set; } = new List<DesignDecision>();

    /// <summary>
    /// Relationships between concepts
    /// </summary>
    public List<Relationship> Relationships { get; set; } = new List<Relationship>();

    /// <summary>
    /// The source file from which the knowledge was extracted
    /// </summary>
    public string SourceFile { get; set; }

    /// <summary>
    /// The date and time when the knowledge was extracted
    /// </summary>
    public DateTime ExtractionDate { get; set; }
}

/// <summary>
/// Represents a concept and its definition
/// </summary>
public class Concept
{
    /// <summary>
    /// The name of the concept
    /// </summary>
    public string Name { get; set; }

    /// <summary>
    /// The definition of the concept
    /// </summary>
    public string Definition { get; set; }

    /// <summary>
    /// Related concepts
    /// </summary>
    [JsonPropertyName("related_concepts")]
    public List<string> RelatedConcepts { get; set; } = new List<string>();
}

/// <summary>
/// Represents an insight or conclusion
/// </summary>
public class Insight
{
    /// <summary>
    /// Description of the insight
    /// </summary>
    public string Description { get; set; }

    /// <summary>
    /// Why this insight is important
    /// </summary>
    public string Importance { get; set; }

    /// <summary>
    /// Applications of this insight
    /// </summary>
    public List<string> Applications { get; set; } = new List<string>();
}

/// <summary>
/// Represents technical details and specifications
/// </summary>
public class TechnicalDetail
{
    /// <summary>
    /// The topic of the technical detail
    /// </summary>
    public string Topic { get; set; }

    /// <summary>
    /// The technical details
    /// </summary>
    public string Details { get; set; }

    /// <summary>
    /// Code examples related to the technical detail
    /// </summary>
    [JsonPropertyName("code_examples")]
    public List<string> CodeExamples { get; set; } = new List<string>();
}

/// <summary>
/// Represents a design decision and its rationale
/// </summary>
public class DesignDecision
{
    /// <summary>
    /// The decision made
    /// </summary>
    public string Decision { get; set; }

    /// <summary>
    /// Why this decision was made
    /// </summary>
    public string Rationale { get; set; }

    /// <summary>
    /// Alternative options that were considered
    /// </summary>
    public List<string> Alternatives { get; set; } = new List<string>();
}

/// <summary>
/// Represents a relationship between two concepts
/// </summary>
public class Relationship
{
    /// <summary>
    /// The source concept
    /// </summary>
    public string From { get; set; }

    /// <summary>
    /// The target concept
    /// </summary>
    public string To { get; set; }

    /// <summary>
    /// How the source concept relates to the target concept
    /// </summary>
    public string RelationshipType { get; set; }
}
