using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace TarsCli.Models;

/// <summary>
/// Configuration for the collaboration between TARS, Augment Code, and VS Code
/// </summary>
public class CollaborationConfig
{
    [JsonPropertyName("collaboration")]
    public CollaborationSettings Collaboration { get; set; } = new();
}

public class CollaborationSettings
{
    [JsonPropertyName("enabled")]
    public bool Enabled { get; set; }

    [JsonPropertyName("components")]
    public Dictionary<string, ComponentConfig> Components { get; set; } = new();

    [JsonPropertyName("workflows")]
    public List<WorkflowConfig> Workflows { get; set; } = new();
}

public class ComponentConfig
{
    [JsonPropertyName("role")]
    public string Role { get; set; } = string.Empty;

    [JsonPropertyName("capabilities")]
    public List<string> Capabilities { get; set; } = new();
}

public class WorkflowConfig
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;

    [JsonPropertyName("coordinator")]
    public string Coordinator { get; set; } = string.Empty;

    [JsonPropertyName("steps")]
    public List<WorkflowStep> Steps { get; set; } = new();
}

public class WorkflowStep
{
    [JsonPropertyName("component")]
    public string Component { get; set; } = string.Empty;

    [JsonPropertyName("action")]
    public string Action { get; set; } = string.Empty;
}
