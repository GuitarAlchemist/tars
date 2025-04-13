using System.Text.Json;
using System.Text.Json.Serialization;

namespace TarsEngine.A2A;

/// <summary>
/// JSON converter for JsonRpcRequest
/// </summary>
public class JsonRpcRequestConverter : JsonConverter<JsonRpcRequest>
{
    /// <summary>
    /// Reads and converts the JSON to a JsonRpcRequest
    /// </summary>
    public override JsonRpcRequest Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException("Expected start of object");
        }

        string jsonRpc = null;
        string id = null;
        string method = null;
        JsonElement paramsElement = default;

        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
            {
                break;
            }

            if (reader.TokenType != JsonTokenType.PropertyName)
            {
                throw new JsonException("Expected property name");
            }

            string propertyName = reader.GetString();
            reader.Read();

            switch (propertyName)
            {
                case "jsonrpc":
                    jsonRpc = reader.GetString();
                    break;
                case "id":
                    id = reader.GetString();
                    break;
                case "method":
                    method = reader.GetString();
                    break;
                case "params":
                    paramsElement = JsonDocument.ParseValue(ref reader).RootElement;
                    break;
                default:
                    reader.Skip();
                    break;
            }
        }

        if (string.IsNullOrEmpty(method))
        {
            throw new JsonException("Required property 'method' not found");
        }

        // Create the appropriate request type based on the method
        JsonRpcRequest request = method switch
        {
            "tasks/send" => new SendTaskRequest(),
            "tasks/sendSubscribe" => new SendTaskStreamingRequest(),
            "tasks/get" => new GetTaskRequest(),
            "tasks/cancel" => new CancelTaskRequest(),
            "tasks/pushNotification/set" => new SetPushNotificationRequest(),
            "tasks/pushNotification/get" => new GetTaskRequest(),
            _ => throw new JsonException($"Unknown method: {method}")
        };

        // Set common properties
        request.JsonRpc = jsonRpc ?? "2.0";
        request.Id = id ?? Guid.NewGuid().ToString();

        // Set specific properties based on the request type
        switch (request)
        {
            case SendTaskRequest sendRequest:
                sendRequest.TaskParams = JsonSerializer.Deserialize<SendTaskParams>(paramsElement.GetRawText(), options);
                break;
            case SendTaskStreamingRequest streamingRequest:
                streamingRequest.TaskParams = JsonSerializer.Deserialize<SendTaskParams>(paramsElement.GetRawText(), options);
                break;
            case GetTaskRequest getRequest:
                getRequest.TaskParams = JsonSerializer.Deserialize<TaskIdParams>(paramsElement.GetRawText(), options);
                break;
            case CancelTaskRequest cancelRequest:
                cancelRequest.TaskParams = JsonSerializer.Deserialize<TaskIdParams>(paramsElement.GetRawText(), options);
                break;
            case SetPushNotificationRequest notificationRequest:
                notificationRequest.NotificationParams = JsonSerializer.Deserialize<PushNotificationParams>(paramsElement.GetRawText(), options);
                break;
        }

        return request;
    }

    /// <summary>
    /// Writes a JsonRpcRequest as JSON
    /// </summary>
    public override void Write(Utf8JsonWriter writer, JsonRpcRequest value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WriteString("jsonrpc", value.JsonRpc);
        writer.WriteString("id", value.Id);
        writer.WriteString("method", value.Method);
            
        writer.WritePropertyName("params");
        JsonSerializer.Serialize(writer, value.Params, options);
            
        writer.WriteEndObject();
    }
}