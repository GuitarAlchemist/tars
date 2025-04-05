using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace TarsCli.Models;

/// <summary>
/// Represents a structured message for collaboration between TARS and Augment
/// </summary>
public class CollaborationMessage
{
    /// <summary>
    /// Unique identifier for the message
    /// </summary>
    [JsonPropertyName("id")]
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Timestamp when the message was created
    /// </summary>
    [JsonPropertyName("timestamp")]
    public DateTime Timestamp { get; set; } = DateTime.Now;

    /// <summary>
    /// The sender of the message (TARS or Augment)
    /// </summary>
    [JsonPropertyName("sender")]
    public string Sender { get; set; } = "TARS";

    /// <summary>
    /// The recipient of the message (TARS or Augment)
    /// </summary>
    [JsonPropertyName("recipient")]
    public string Recipient { get; set; } = "Augment";

    /// <summary>
    /// The type of message
    /// </summary>
    [JsonPropertyName("type")]
    public string Type { get; set; } = "knowledge";

    /// <summary>
    /// The operation to perform
    /// </summary>
    [JsonPropertyName("operation")]
    public string Operation { get; set; }

    /// <summary>
    /// The status of the message (request, in_progress, completed, error)
    /// </summary>
    [JsonPropertyName("status")]
    public string Status { get; set; } = "request";

    /// <summary>
    /// The content of the message
    /// </summary>
    [JsonPropertyName("content")]
    public object Content { get; set; }

    /// <summary>
    /// The correlation ID for tracking related messages
    /// </summary>
    [JsonPropertyName("correlation_id")]
    public string CorrelationId { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// The ID of the message this is in response to
    /// </summary>
    [JsonPropertyName("in_response_to")]
    public string InResponseTo { get; set; }

    /// <summary>
    /// Progress information for long-running operations
    /// </summary>
    [JsonPropertyName("progress")]
    public ProgressInfo Progress { get; set; }

    /// <summary>
    /// Additional metadata for the message
    /// </summary>
    [JsonPropertyName("metadata")]
    public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
}

/// <summary>
/// Represents progress information for long-running operations
/// </summary>
public class ProgressInfo
{
    /// <summary>
    /// The percentage of completion (0-100)
    /// </summary>
    [JsonPropertyName("percentage")]
    public int Percentage { get; set; }

    /// <summary>
    /// The current step in the process
    /// </summary>
    [JsonPropertyName("current_step")]
    public string CurrentStep { get; set; }

    /// <summary>
    /// The total number of steps
    /// </summary>
    [JsonPropertyName("total_steps")]
    public int TotalSteps { get; set; }

    /// <summary>
    /// Estimated time remaining in seconds
    /// </summary>
    [JsonPropertyName("estimated_time_remaining")]
    public int? EstimatedTimeRemaining { get; set; }

    /// <summary>
    /// Additional status message
    /// </summary>
    [JsonPropertyName("status_message")]
    public string StatusMessage { get; set; }
}

/// <summary>
/// Represents a knowledge transfer message
/// </summary>
public class KnowledgeTransferMessage
{
    /// <summary>
    /// The source of the knowledge
    /// </summary>
    [JsonPropertyName("source")]
    public string Source { get; set; }

    /// <summary>
    /// The knowledge items being transferred
    /// </summary>
    [JsonPropertyName("knowledge_items")]
    public List<DocumentationKnowledge> KnowledgeItems { get; set; } = new List<DocumentationKnowledge>();

    /// <summary>
    /// The context for the knowledge transfer
    /// </summary>
    [JsonPropertyName("context")]
    public string Context { get; set; }

    /// <summary>
    /// The relevance score for the knowledge (0-100)
    /// </summary>
    [JsonPropertyName("relevance_score")]
    public int RelevanceScore { get; set; }
}

/// <summary>
/// Represents a code improvement suggestion
/// </summary>
public class CodeImprovementSuggestion
{
    /// <summary>
    /// The file path to improve
    /// </summary>
    [JsonPropertyName("file_path")]
    public string FilePath { get; set; }

    /// <summary>
    /// The original content of the file
    /// </summary>
    [JsonPropertyName("original_content")]
    public string OriginalContent { get; set; }

    /// <summary>
    /// The improved content of the file
    /// </summary>
    [JsonPropertyName("improved_content")]
    public string ImprovedContent { get; set; }

    /// <summary>
    /// The changes made to the file
    /// </summary>
    [JsonPropertyName("changes")]
    public List<CodeChange> Changes { get; set; } = new List<CodeChange>();

    /// <summary>
    /// The rationale for the improvements
    /// </summary>
    [JsonPropertyName("rationale")]
    public string Rationale { get; set; }

    /// <summary>
    /// The knowledge items used for the improvement
    /// </summary>
    [JsonPropertyName("knowledge_references")]
    public List<string> KnowledgeReferences { get; set; } = new List<string>();

    /// <summary>
    /// The quality score for the improvement (0-100)
    /// </summary>
    [JsonPropertyName("quality_score")]
    public int QualityScore { get; set; }
}

/// <summary>
/// Represents a specific change to a code file
/// </summary>
public class CodeChange
{
    /// <summary>
    /// The type of change (add, remove, modify)
    /// </summary>
    [JsonPropertyName("type")]
    public string Type { get; set; }

    /// <summary>
    /// The start line of the change
    /// </summary>
    [JsonPropertyName("start_line")]
    public int StartLine { get; set; }

    /// <summary>
    /// The end line of the change
    /// </summary>
    [JsonPropertyName("end_line")]
    public int EndLine { get; set; }

    /// <summary>
    /// The original code
    /// </summary>
    [JsonPropertyName("original")]
    public string Original { get; set; }

    /// <summary>
    /// The new code
    /// </summary>
    [JsonPropertyName("new")]
    public string New { get; set; }

    /// <summary>
    /// The reason for the change
    /// </summary>
    [JsonPropertyName("reason")]
    public string Reason { get; set; }
}

/// <summary>
/// Represents feedback on a code improvement
/// </summary>
public class ImprovementFeedback
{
    /// <summary>
    /// Whether the improvement was accepted
    /// </summary>
    [JsonPropertyName("accepted")]
    public bool Accepted { get; set; }

    /// <summary>
    /// The reason for acceptance or rejection
    /// </summary>
    [JsonPropertyName("reason")]
    public string Reason { get; set; }

    /// <summary>
    /// Specific feedback on parts of the improvement
    /// </summary>
    [JsonPropertyName("specific_feedback")]
    public List<SpecificFeedback> SpecificFeedback { get; set; } = new List<SpecificFeedback>();

    /// <summary>
    /// Suggestions for future improvements
    /// </summary>
    [JsonPropertyName("suggestions")]
    public List<string> Suggestions { get; set; } = new List<string>();
}

/// <summary>
/// Represents specific feedback on a part of an improvement
/// </summary>
public class SpecificFeedback
{
    /// <summary>
    /// The start line of the feedback
    /// </summary>
    [JsonPropertyName("start_line")]
    public int StartLine { get; set; }

    /// <summary>
    /// The end line of the feedback
    /// </summary>
    [JsonPropertyName("end_line")]
    public int EndLine { get; set; }

    /// <summary>
    /// The feedback text
    /// </summary>
    [JsonPropertyName("feedback")]
    public string Feedback { get; set; }

    /// <summary>
    /// The type of feedback (positive, negative, suggestion)
    /// </summary>
    [JsonPropertyName("type")]
    public string Type { get; set; }
}
