using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Net.Http.Headers;

namespace TarsEngine.A2A;

/// <summary>
/// Client for interacting with A2A protocol servers
/// </summary>
public class A2AClient
{
    private readonly HttpClient _httpClient;
    private readonly string _agentUrl;
    private readonly JsonSerializerOptions _jsonOptions;

    /// <summary>
    /// Initializes a new instance of the A2AClient class
    /// </summary>
    /// <param name="agentCard">The agent card containing the URL</param>
    public A2AClient(AgentCard agentCard)
    {
        if (agentCard == null)
            throw new ArgumentNullException(nameof(agentCard));

        _agentUrl = agentCard.Url;
        _httpClient = new HttpClient();
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Converters = { new JsonStringEnumConverter() }
        };
    }

    /// <summary>
    /// Initializes a new instance of the A2AClient class
    /// </summary>
    /// <param name="agentUrl">The URL of the agent</param>
    public A2AClient(string agentUrl)
    {
        if (string.IsNullOrEmpty(agentUrl))
            throw new ArgumentNullException(nameof(agentUrl));

        _agentUrl = agentUrl;
        _httpClient = new HttpClient();
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Converters = { new JsonStringEnumConverter() }
        };
    }

    /// <summary>
    /// Sends a task to the agent
    /// </summary>
    /// <param name="message">The message to send</param>
    /// <param name="taskId">Optional task ID (will be generated if not provided)</param>
    /// <param name="metadata">Optional metadata</param>
    /// <returns>The task response</returns>
    public async Task<Task> SendTaskAsync(Message message, string taskId = null, Dictionary<string, object> metadata = null)
    {
        var request = new SendTaskRequest
        {
            TaskParams = new SendTaskParams
            {
                TaskId = taskId ?? Guid.NewGuid().ToString(),
                Message = message,
                Metadata = metadata
            }
        };

        var response = await SendRequestAsync<JsonRpcResponse>(request);
        return JsonSerializer.Deserialize<Task>(JsonSerializer.Serialize(response.Result, _jsonOptions), _jsonOptions);
    }

    /// <summary>
    /// Gets a task by ID
    /// </summary>
    /// <param name="taskId">The task ID</param>
    /// <returns>The task</returns>
    public async Task<Task> GetTaskAsync(string taskId)
    {
        var request = new GetTaskRequest
        {
            TaskParams = new TaskIdParams
            {
                TaskId = taskId
            }
        };

        var response = await SendRequestAsync<JsonRpcResponse>(request);
        return JsonSerializer.Deserialize<Task>(JsonSerializer.Serialize(response.Result, _jsonOptions), _jsonOptions);
    }

    /// <summary>
    /// Cancels a task
    /// </summary>
    /// <param name="taskId">The task ID</param>
    /// <returns>The canceled task</returns>
    public async Task<Task> CancelTaskAsync(string taskId)
    {
        var request = new CancelTaskRequest
        {
            TaskParams = new TaskIdParams
            {
                TaskId = taskId
            }
        };

        var response = await SendRequestAsync<JsonRpcResponse>(request);
        return JsonSerializer.Deserialize<Task>(JsonSerializer.Serialize(response.Result, _jsonOptions), _jsonOptions);
    }

    /// <summary>
    /// Sets up push notifications for a task
    /// </summary>
    /// <param name="taskId">The task ID</param>
    /// <param name="callbackUrl">The callback URL</param>
    /// <param name="authentication">Optional authentication information</param>
    /// <returns>True if successful</returns>
    public async Task<bool> SetPushNotificationAsync(string taskId, string callbackUrl, AgentAuthentication authentication = null)
    {
        var request = new SetPushNotificationRequest
        {
            NotificationParams = new PushNotificationParams
            {
                TaskId = taskId,
                CallbackUrl = callbackUrl,
                Authentication = authentication
            }
        };

        var response = await SendRequestAsync<JsonRpcResponse>(request);
        return response.Error == null;
    }

    /// <summary>
    /// Sends a task with streaming support
    /// </summary>
    /// <param name="message">The message to send</param>
    /// <param name="taskId">Optional task ID (will be generated if not provided)</param>
    /// <param name="metadata">Optional metadata</param>
    /// <param name="onUpdate">Callback for task updates</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>The final task state</returns>
    public async Task<Task> SendTaskStreamingAsync(
        Message message,
        string taskId = null,
        Dictionary<string, object> metadata = null,
        Action<Task> onUpdate = null,
        CancellationToken cancellationToken = default)
    {
        var request = new SendTaskStreamingRequest
        {
            TaskParams = new SendTaskParams
            {
                TaskId = taskId ?? Guid.NewGuid().ToString(),
                Message = message,
                Metadata = metadata
            }
        };

        var requestJson = JsonSerializer.Serialize(request, _jsonOptions);
        var content = new StringContent(requestJson, Encoding.UTF8, "application/json");

        var request2 = new HttpRequestMessage(HttpMethod.Post, _agentUrl)
        {
            Content = content
        };
        request2.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("text/event-stream"));

        using var response = await _httpClient.SendAsync(request2, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
        response.EnsureSuccessStatusCode();

        using var stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);

        Task latestTask = null;
        string line;
        StringBuilder eventData = new StringBuilder();

        while ((line = await reader.ReadLineAsync()) != null && !cancellationToken.IsCancellationRequested)
        {
            if (string.IsNullOrEmpty(line))
            {
                if (eventData.Length > 0)
                {
                    var eventText = eventData.ToString();
                    if (eventText.StartsWith("data:"))
                    {
                        var jsonData = eventText.Substring(5).Trim();
                        var eventResponse = JsonSerializer.Deserialize<JsonRpcResponse>(jsonData, _jsonOptions);

                        if (eventResponse.Result != null)
                        {
                            latestTask = JsonSerializer.Deserialize<Task>(
                                JsonSerializer.Serialize(eventResponse.Result, _jsonOptions),
                                _jsonOptions);

                            onUpdate?.Invoke(latestTask);

                            // If we've reached a terminal state, we can exit
                            if (latestTask.Status == TaskStatus.Completed ||
                                latestTask.Status == TaskStatus.Failed ||
                                latestTask.Status == TaskStatus.Canceled)
                            {
                                break;
                            }
                        }
                    }
                    eventData.Clear();
                }
            }
            else if (line.StartsWith("data:"))
            {
                eventData.AppendLine(line);
            }
        }

        return latestTask;
    }

    /// <summary>
    /// Sends a JSON-RPC request to the agent
    /// </summary>
    /// <typeparam name="T">The response type</typeparam>
    /// <param name="request">The request</param>
    /// <returns>The response</returns>
    private async Task<T> SendRequestAsync<T>(object request)
    {
        var requestJson = JsonSerializer.Serialize(request, _jsonOptions);
        var content = new StringContent(requestJson, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync(_agentUrl, content);
        response.EnsureSuccessStatusCode();

        var responseJson = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<T>(responseJson, _jsonOptions);
    }
}