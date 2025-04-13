using System.Text.Json.Serialization;

namespace TarsCli.Models;

/// <summary>
/// Base class for all MCP messages
/// </summary>
public abstract class McpMessage
{
    [JsonPropertyName("type")]
    public string Type { get; set; } = string.Empty;

    [JsonPropertyName("timestamp")]
    public long Timestamp { get; set; } = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

    [JsonPropertyName("id")]
    public string Id { get; set; } = Guid.NewGuid().ToString();
}



/// <summary>
/// Message for code improvement suggestions
/// </summary>
public class CodeImprovementMessage : McpMessage
{
    public CodeImprovementMessage()
    {
        Type = "code_improvement";
    }

    [JsonPropertyName("file_path")]
    public string FilePath { get; set; } = string.Empty;

    [JsonPropertyName("line_start")]
    public int LineStart { get; set; }

    [JsonPropertyName("line_end")]
    public int LineEnd { get; set; }

    [JsonPropertyName("original_code")]
    public string OriginalCode { get; set; } = string.Empty;

    [JsonPropertyName("improved_code")]
    public string ImprovedCode { get; set; } = string.Empty;

    [JsonPropertyName("improvement_type")]
    public string ImprovementType { get; set; } = string.Empty;

    [JsonPropertyName("explanation")]
    public string Explanation { get; set; } = string.Empty;

    [JsonPropertyName("confidence")]
    public double Confidence { get; set; } = 0.0;
}

/// <summary>
/// Message for progress reporting
/// </summary>
public class ProgressReportMessage : McpMessage
{
    public ProgressReportMessage()
    {
        Type = "progress_report";
    }

    [JsonPropertyName("operation")]
    public string Operation { get; set; } = string.Empty;

    [JsonPropertyName("progress_percentage")]
    public double ProgressPercentage { get; set; }

    [JsonPropertyName("status")]
    public string Status { get; set; } = string.Empty;

    [JsonPropertyName("details")]
    public string Details { get; set; } = string.Empty;

    [JsonPropertyName("estimated_completion_time")]
    public long? EstimatedCompletionTime { get; set; }
}

/// <summary>
/// Message for system handoff
/// </summary>
public class SystemHandoffMessage : McpMessage
{
    public SystemHandoffMessage()
    {
        Type = "system_handoff";
    }

    [JsonPropertyName("from_system")]
    public string FromSystem { get; set; } = string.Empty;

    [JsonPropertyName("to_system")]
    public string ToSystem { get; set; } = string.Empty;

    [JsonPropertyName("context")]
    public Dictionary<string, object> Context { get; set; } = new();

    [JsonPropertyName("action_requested")]
    public string ActionRequested { get; set; } = string.Empty;

    [JsonPropertyName("priority")]
    public string Priority { get; set; } = "normal";
}

/// <summary>
/// Message for feedback
/// </summary>
public class FeedbackMessage : McpMessage
{
    public FeedbackMessage()
    {
        Type = "feedback";
    }

    [JsonPropertyName("feedback_type")]
    public string FeedbackType { get; set; } = string.Empty;

    [JsonPropertyName("target_message_id")]
    public string TargetMessageId { get; set; } = string.Empty;

    [JsonPropertyName("rating")]
    public int? Rating { get; set; }

    [JsonPropertyName("comments")]
    public string Comments { get; set; } = string.Empty;

    [JsonPropertyName("suggestions")]
    public List<string> Suggestions { get; set; } = new();
}

/// <summary>
/// Message for workflow definition
/// </summary>
public class WorkflowDefinitionMessage : McpMessage
{
    public WorkflowDefinitionMessage()
    {
        Type = "workflow_definition";
    }

    [JsonPropertyName("workflow_name")]
    public string WorkflowName { get; set; } = string.Empty;

    [JsonPropertyName("description")]
    public string Description { get; set; } = string.Empty;

    [JsonPropertyName("coordinator")]
    public string Coordinator { get; set; } = string.Empty;

    [JsonPropertyName("steps")]
    public List<WorkflowStep> Steps { get; set; } = new();
}
