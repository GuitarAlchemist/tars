using System.Text.Json.Serialization;

namespace TarsEngine.A2A;
/// <summary>
/// Core types for the A2A (Agent-to-Agent) protocol implementation
/// </summary>

#region Agent Card Types

/// <summary>
/// Represents an A2A Agent Card, which describes an agent's capabilities and metadata
/// </summary>
public class AgentCard
{
    [JsonPropertyName("name")]
    public string Name { get; set; }

    [JsonPropertyName("description")]
    public string Description { get; set; }

    [JsonPropertyName("url")]
    public string Url { get; set; }

    [JsonPropertyName("provider")]
    public AgentProvider Provider { get; set; }

    [JsonPropertyName("version")]
    public string Version { get; set; }

    [JsonPropertyName("documentationUrl")]
    public string DocumentationUrl { get; set; }

    [JsonPropertyName("capabilities")]
    public AgentCapabilities Capabilities { get; set; }

    [JsonPropertyName("authentication")]
    public AgentAuthentication Authentication { get; set; }

    [JsonPropertyName("defaultInputModes")]
    public List<string> DefaultInputModes { get; set; } = new() { "text" };

    [JsonPropertyName("defaultOutputModes")]
    public List<string> DefaultOutputModes { get; set; } = new() { "text" };

    [JsonPropertyName("skills")]
    public List<AgentSkill> Skills { get; set; } = new();
}

/// <summary>
/// Represents an agent provider information
/// </summary>
public class AgentProvider
{
    [JsonPropertyName("organization")]
    public string Organization { get; set; }

    [JsonPropertyName("url")]
    public string Url { get; set; }
}

/// <summary>
/// Represents agent capabilities
/// </summary>
public class AgentCapabilities
{
    [JsonPropertyName("streaming")]
    public bool Streaming { get; set; } = false;

    [JsonPropertyName("pushNotifications")]
    public bool PushNotifications { get; set; } = false;

    [JsonPropertyName("stateTransitionHistory")]
    public bool StateTransitionHistory { get; set; } = false;
}

/// <summary>
/// Represents agent authentication requirements
/// </summary>
public class AgentAuthentication
{
    [JsonPropertyName("schemes")]
    public List<string> Schemes { get; set; } = new();

    [JsonPropertyName("credentials")]
    public string Credentials { get; set; }
}

/// <summary>
/// Represents an agent skill
/// </summary>
public class AgentSkill
{
    [JsonPropertyName("id")]
    public string Id { get; set; }

    [JsonPropertyName("name")]
    public string Name { get; set; }

    [JsonPropertyName("description")]
    public string Description { get; set; }

    [JsonPropertyName("tags")]
    public List<string> Tags { get; set; }

    [JsonPropertyName("examples")]
    public List<string> Examples { get; set; }

    [JsonPropertyName("inputModes")]
    public List<string> InputModes { get; set; }

    [JsonPropertyName("outputModes")]
    public List<string> OutputModes { get; set; }
}

#endregion

#region Task Types

/// <summary>
/// Represents the status of a task
/// </summary>
public enum TaskStatus
{
    Submitted,
    Working,
    InputRequired,
    Completed,
    Failed,
    Canceled
}

/// <summary>
/// Represents a task in the A2A protocol
/// </summary>
public class Task
{
    [JsonPropertyName("taskId")]
    public string TaskId { get; set; }

    [JsonPropertyName("status")]
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public TaskStatus Status { get; set; }

    [JsonPropertyName("messages")]
    public List<Message> Messages { get; set; } = new();

    [JsonPropertyName("artifacts")]
    public List<Artifact> Artifacts { get; set; } = new();

    [JsonPropertyName("stateTransitionHistory")]
    public List<StateTransition> StateTransitionHistory { get; set; }

    [JsonPropertyName("metadata")]
    public Dictionary<string, object> Metadata { get; set; }
}

/// <summary>
/// Represents a state transition in a task's history
/// </summary>
public class StateTransition
{
    [JsonPropertyName("fromState")]
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public TaskStatus FromState { get; set; }

    [JsonPropertyName("toState")]
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public TaskStatus ToState { get; set; }

    [JsonPropertyName("timestamp")]
    public DateTime Timestamp { get; set; }

    [JsonPropertyName("reason")]
    public string Reason { get; set; }
}

/// <summary>
/// Represents a message in the A2A protocol
/// </summary>
public class Message
{
    [JsonPropertyName("role")]
    public string Role { get; set; }

    [JsonPropertyName("parts")]
    public List<Part> Parts { get; set; } = new();

    [JsonPropertyName("metadata")]
    public Dictionary<string, object> Metadata { get; set; }
}

/// <summary>
/// Base class for message parts
/// </summary>
[JsonPolymorphic(TypeDiscriminatorPropertyName = "type")]
[JsonDerivedType(typeof(TextPart), typeDiscriminator: "text")]
[JsonDerivedType(typeof(FilePart), typeDiscriminator: "file")]
[JsonDerivedType(typeof(DataPart), typeDiscriminator: "data")]
public abstract class Part
{
    [JsonPropertyName("metadata")]
    public Dictionary<string, object> Metadata { get; set; }
}

/// <summary>
/// Represents a text part in a message
/// </summary>
public class TextPart : Part
{
    [JsonPropertyName("text")]
    public string Text { get; set; }
}

/// <summary>
/// Represents a file part in a message
/// </summary>
public class FilePart : Part
{
    [JsonPropertyName("file")]
    public FileContent File { get; set; }
}

/// <summary>
/// Represents a data part in a message
/// </summary>
public class DataPart : Part
{
    [JsonPropertyName("data")]
    public Dictionary<string, object> Data { get; set; }
}

/// <summary>
/// Represents file content in a message
/// </summary>
public class FileContent
{
    [JsonPropertyName("name")]
    public string Name { get; set; }

    [JsonPropertyName("mimeType")]
    public string MimeType { get; set; }

    [JsonPropertyName("bytes")]
    public string Bytes { get; set; }

    [JsonPropertyName("uri")]
    public string Uri { get; set; }
}

/// <summary>
/// Represents an artifact in the A2A protocol
/// </summary>
public class Artifact
{
    [JsonPropertyName("name")]
    public string Name { get; set; }

    [JsonPropertyName("description")]
    public string Description { get; set; }

    [JsonPropertyName("parts")]
    public List<Part> Parts { get; set; } = new();

    [JsonPropertyName("index")]
    public int Index { get; set; }

    [JsonPropertyName("append")]
    public bool? Append { get; set; }

    [JsonPropertyName("lastChunk")]
    public bool? LastChunk { get; set; }

    [JsonPropertyName("metadata")]
    public Dictionary<string, object> Metadata { get; set; }
}

#endregion

#region JSON-RPC Types

/// <summary>
/// Base class for JSON-RPC requests
/// </summary>
[JsonConverter(typeof(JsonRpcRequestConverter))]
public abstract class JsonRpcRequest
{
    [JsonPropertyName("jsonrpc")]
    public string JsonRpc { get; set; } = "2.0";

    [JsonPropertyName("id")]
    public string Id { get; set; } = Guid.NewGuid().ToString();

    [JsonPropertyName("method")]
    public abstract string Method { get; }

    [JsonPropertyName("params")]
    public abstract object Params { get; }
}

/// <summary>
/// Base class for JSON-RPC responses
/// </summary>
public class JsonRpcResponse
{
    [JsonPropertyName("jsonrpc")]
    public string JsonRpc { get; set; } = "2.0";

    [JsonPropertyName("id")]
    public string Id { get; set; }

    [JsonPropertyName("result")]
    public object Result { get; set; }

    [JsonPropertyName("error")]
    public JsonRpcError Error { get; set; }
}

/// <summary>
/// Represents a JSON-RPC error
/// </summary>
public class JsonRpcError
{
    [JsonPropertyName("code")]
    public int Code { get; set; }

    [JsonPropertyName("message")]
    public string Message { get; set; }

    [JsonPropertyName("data")]
    public object Data { get; set; }
}

#endregion

#region Request/Response Types

/// <summary>
/// Parameters for sending a task
/// </summary>
public class SendTaskParams
{
    [JsonPropertyName("taskId")]
    public string TaskId { get; set; } = Guid.NewGuid().ToString();

    [JsonPropertyName("message")]
    public Message Message { get; set; }

    [JsonPropertyName("metadata")]
    public Dictionary<string, object> Metadata { get; set; }
}

/// <summary>
/// Request to send a task
/// </summary>
public class SendTaskRequest : JsonRpcRequest
{
    public override string Method => "tasks/send";

    [JsonPropertyName("params")]
    public override object Params => TaskParams;

    public SendTaskParams TaskParams { get; set; }
}

/// <summary>
/// Request to send a task with streaming
/// </summary>
public class SendTaskStreamingRequest : JsonRpcRequest
{
    public override string Method => "tasks/sendSubscribe";

    [JsonPropertyName("params")]
    public override object Params => TaskParams;

    public SendTaskParams TaskParams { get; set; }
}

/// <summary>
/// Parameters for getting a task
/// </summary>
public class TaskIdParams
{
    [JsonPropertyName("taskId")]
    public string TaskId { get; set; }
}

/// <summary>
/// Request to get a task
/// </summary>
public class GetTaskRequest : JsonRpcRequest
{
    public override string Method => "tasks/get";

    [JsonPropertyName("params")]
    public override object Params => TaskParams;

    public TaskIdParams TaskParams { get; set; }
}

/// <summary>
/// Request to cancel a task
/// </summary>
public class CancelTaskRequest : JsonRpcRequest
{
    public override string Method => "tasks/cancel";

    [JsonPropertyName("params")]
    public override object Params => TaskParams;

    public TaskIdParams TaskParams { get; set; }
}

/// <summary>
/// Parameters for setting up push notifications
/// </summary>
public class PushNotificationParams
{
    [JsonPropertyName("taskId")]
    public string TaskId { get; set; }

    [JsonPropertyName("callbackUrl")]
    public string CallbackUrl { get; set; }

    [JsonPropertyName("authentication")]
    public AgentAuthentication Authentication { get; set; }
}

/// <summary>
/// Request to set up push notifications
/// </summary>
public class SetPushNotificationRequest : JsonRpcRequest
{
    public override string Method => "tasks/pushNotification/set";

    [JsonPropertyName("params")]
    public override object Params => NotificationParams;

    public PushNotificationParams NotificationParams { get; set; }
}

#endregion