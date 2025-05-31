using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;

namespace TarsEngine.A2A;

/// <summary>
/// Server for implementing the A2A protocol
/// </summary>
public class A2AServer
{
    private readonly HttpListener _listener;
    private readonly AgentCard _agentCard;
    private readonly ILogger<A2AServer> _logger;
    private readonly JsonSerializerOptions _jsonOptions;
    private readonly Dictionary<string, Task> _tasks = new();
    private readonly Dictionary<string, Func<Message, CancellationToken, Task<Task>>> _taskHandlers = new();
    private readonly Dictionary<string, string> _pushNotificationCallbacks = new();
    private bool _isRunning;
    private CancellationTokenSource _cancellationTokenSource;

    /// <summary>
    /// Initializes a new instance of the A2AServer class
    /// </summary>
    /// <param name="agentCard">The agent card</param>
    /// <param name="logger">Logger instance</param>
    public A2AServer(AgentCard agentCard, ILogger<A2AServer> logger)
    {
        _agentCard = agentCard ?? throw new ArgumentNullException(nameof(agentCard));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _listener = new HttpListener();

        var uri = new Uri(agentCard.Url);
        _listener.Prefixes.Add(uri.GetLeftPart(UriPartial.Authority) + uri.AbsolutePath);

        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Converters = { new JsonStringEnumConverter() }
        };
    }

    /// <summary>
    /// Registers a task handler for a specific skill
    /// </summary>
    /// <param name="skillId">The skill ID</param>
    /// <param name="handler">The handler function</param>
    public void RegisterTaskHandler(string skillId, Func<Message, CancellationToken, Task<Task>> handler)
    {
        if (string.IsNullOrEmpty(skillId))
            throw new ArgumentNullException(nameof(skillId));

        if (handler == null)
            throw new ArgumentNullException(nameof(handler));

        _taskHandlers[skillId] = handler;
        _logger.LogInformation($"Registered task handler for skill: {skillId}");
    }

    /// <summary>
    /// Starts the server
    /// </summary>
    public void Start()
    {
        if (_isRunning)
            return;

        _isRunning = true;
        _cancellationTokenSource = new CancellationTokenSource();
        _listener.Start();
        _logger.LogInformation($"A2A server started at {_agentCard.Url}");

        // Start listening for requests
        System.Threading.Tasks.Task.Run(async () => await ListenForRequestsAsync(_cancellationTokenSource.Token));

        // Set up the well-known agent card endpoint
        var assemblyLocation = System.Reflection.Assembly.GetExecutingAssembly().Location;
        var assemblyDirectory = Path.GetDirectoryName(assemblyLocation) ?? string.Empty;
        var wellKnownPath = Path.Combine(assemblyDirectory, ".well-known");
        Directory.CreateDirectory(wellKnownPath);
        File.WriteAllText(Path.Combine(wellKnownPath, "agent.json"), JsonSerializer.Serialize(_agentCard, _jsonOptions));
    }

    /// <summary>
    /// Stops the server
    /// </summary>
    public void Stop()
    {
        if (!_isRunning)
            return;

        _isRunning = false;
        _cancellationTokenSource.Cancel();
        _listener.Stop();
        _logger.LogInformation("A2A server stopped");
    }

    /// <summary>
    /// Listens for incoming requests
    /// </summary>
    private async System.Threading.Tasks.Task ListenForRequestsAsync(CancellationToken cancellationToken)
    {
        while (_isRunning && !cancellationToken.IsCancellationRequested)
        {
            try
            {
                var context = await _listener.GetContextAsync();
                _ = ProcessRequestAsync(context, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                // This is expected during shutdown, no need to log an error
                break;
            }
            catch (HttpListenerException ex) when (ex.ErrorCode == 995) // The I/O operation has been aborted
            {
                // This is expected during shutdown, no need to log an error
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing request");
            }
        }
    }

    /// <summary>
    /// Processes an incoming request
    /// </summary>
    private async System.Threading.Tasks.Task ProcessRequestAsync(HttpListenerContext context, CancellationToken cancellationToken)
    {
        try
        {
            if (context.Request.HttpMethod == "GET" && context.Request.Url.AbsolutePath.EndsWith("/.well-known/agent.json"))
            {
                await HandleAgentCardRequestAsync(context);
                return;
            }

            if (context.Request.HttpMethod != "POST")
            {
                context.Response.StatusCode = 405; // Method Not Allowed
                context.Response.Close();
                return;
            }

            // Read the request body
            string requestBody;
            using (var reader = new StreamReader(context.Request.InputStream, context.Request.ContentEncoding))
            {
                requestBody = await reader.ReadToEndAsync();
            }

            // Parse the JSON-RPC request
            var request = JsonSerializer.Deserialize<JsonRpcRequest>(requestBody, _jsonOptions);

            // Handle the request based on the method
            switch (request.Method)
            {
                case "tasks/send":
                    await HandleSendTaskAsync(context, requestBody, false, cancellationToken);
                    break;

                case "tasks/sendSubscribe":
                    await HandleSendTaskAsync(context, requestBody, true, cancellationToken);
                    break;

                case "tasks/get":
                    await HandleGetTaskAsync(context, requestBody);
                    break;

                case "tasks/cancel":
                    await HandleCancelTaskAsync(context, requestBody, cancellationToken);
                    break;

                case "tasks/pushNotification/set":
                    await HandleSetPushNotificationAsync(context, requestBody);
                    break;

                case "tasks/pushNotification/get":
                    await HandleGetPushNotificationAsync(context, requestBody);
                    break;

                default:
                    await SendErrorResponseAsync(context, -32601, "Method not found", request.Id);
                    break;
            }
        }
        catch (JsonException ex)
        {
            await SendErrorResponseAsync(context, -32700, "Parse error", null, ex.Message);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing request");
            await SendErrorResponseAsync(context, -32603, "Internal error", null, ex.Message);
        }
    }

    /// <summary>
    /// Handles a request for the agent card
    /// </summary>
    private async System.Threading.Tasks.Task HandleAgentCardRequestAsync(HttpListenerContext context)
    {
        var agentCardJson = JsonSerializer.Serialize(_agentCard, _jsonOptions);
        var buffer = Encoding.UTF8.GetBytes(agentCardJson);

        context.Response.ContentType = "application/json";
        context.Response.ContentLength64 = buffer.Length;
        await context.Response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
        context.Response.Close();
    }

    /// <summary>
    /// Handles a send task request
    /// </summary>
    private async System.Threading.Tasks.Task HandleSendTaskAsync(HttpListenerContext context, string requestBody, bool streaming, CancellationToken cancellationToken)
    {
        var request = JsonSerializer.Deserialize<SendTaskRequest>(requestBody, _jsonOptions);
        var taskParams = request.TaskParams;

        // Check if we have a handler for the skill
        string skillId = null;
        if (taskParams.Message.Metadata != null && taskParams.Message.Metadata.TryGetValue("skillId", out var skillIdObj))
        {
            skillId = skillIdObj.ToString();
        }

        if (string.IsNullOrEmpty(skillId) || !_taskHandlers.TryGetValue(skillId, out var handler))
        {
            await SendErrorResponseAsync(context, -32602, "Invalid skill ID", request.Id);
            return;
        }

        // Create a new task or update an existing one
        Task task;
        var isNewTask = false;

        lock (_tasks)
        {
            if (_tasks.TryGetValue(taskParams.TaskId, out var existingTask))
            {
                // Update existing task with new message
                existingTask.Messages.Add(taskParams.Message);
                task = existingTask;
            }
            else
            {
                // Create a new task
                task = new Task
                {
                    TaskId = taskParams.TaskId,
                    Status = TaskStatus.Submitted,
                    Messages = [taskParams.Message],
                    Artifacts = [],
                    StateTransitionHistory = _agentCard.Capabilities.StateTransitionHistory ? [] : null,
                    Metadata = taskParams.Metadata
                };

                _tasks[taskParams.TaskId] = task;
                isNewTask = true;
            }
        }

        if (streaming)
        {
            // Set up streaming response
            context.Response.ContentType = "text/event-stream";
            context.Response.Headers.Add("Cache-Control", "no-cache");
            context.Response.Headers.Add("Connection", "keep-alive");

            // Create a linked cancellation token that will be canceled if the client disconnects
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

            // Send initial task state
            await SendEventAsync(context, task, request.Id);

            if (isNewTask)
            {
                // Process the task asynchronously
                _ = ProcessTaskAsync(task, handler, taskParams.Message, linkedCts.Token, async (updatedTask) =>
                {
                    try
                    {
                        await SendEventAsync(context, updatedTask, request.Id);

                        // If the task is in a terminal state, close the connection
                        if (updatedTask.Status == TaskStatus.Completed ||
                            updatedTask.Status == TaskStatus.Failed ||
                            updatedTask.Status == TaskStatus.Canceled)
                        {
                            context.Response.Close();
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error sending event");
                        linkedCts.Cancel();
                    }
                });
            }
        }
        else
        {
            // For non-streaming, process the task and wait for completion
            if (isNewTask)
            {
                task = await ProcessTaskAsync(task, handler, taskParams.Message, cancellationToken);
            }

            // Send the response
            var response = new JsonRpcResponse
            {
                Id = request.Id,
                Result = task
            };

            await SendJsonResponseAsync(context, response);
        }
    }

    /// <summary>
    /// Processes a task asynchronously
    /// </summary>
    private async Task<Task> ProcessTaskAsync(
        Task task,
        Func<Message, CancellationToken, Task<Task>> handler,
        Message message,
        CancellationToken cancellationToken,
        Func<Task, System.Threading.Tasks.Task> onUpdate = null)
    {
        try
        {
            // Update task status to Working
            UpdateTaskStatus(task, TaskStatus.Working);

            if (onUpdate != null)
            {
                await onUpdate(task);
            }

            // Process the task
            var result = await handler(message, cancellationToken);

            // Update the task with the result
            lock (_tasks)
            {
                task.Status = result.Status;
                task.Artifacts = result.Artifacts;

                if (result.Messages != null && result.Messages.Count > 0)
                {
                    // Add any new messages from the result
                    foreach (var resultMessage in result.Messages)
                    {
                        if (!task.Messages.Contains(resultMessage))
                        {
                            task.Messages.Add(resultMessage);
                        }
                    }
                }
            }

            if (onUpdate != null)
            {
                await onUpdate(task);
            }

            // Send push notification if configured
            await SendPushNotificationAsync(task);

            return task;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error processing task {task.TaskId}");

            // Update task status to Failed
            UpdateTaskStatus(task, TaskStatus.Failed);

            // Add error message
            var errorMessage = new Message
            {
                Role = "agent",
                Parts = [new TextPart { Text = $"Error: {ex.Message}" }]
            };

            task.Messages.Add(errorMessage);

            if (onUpdate != null)
            {
                await onUpdate(task);
            }

            // Send push notification if configured
            await SendPushNotificationAsync(task);

            return task;
        }
    }

    /// <summary>
    /// Updates a task's status and adds a state transition if enabled
    /// </summary>
    private void UpdateTaskStatus(Task task, TaskStatus newStatus)
    {
        var oldStatus = task.Status;
        task.Status = newStatus;

        if (_agentCard.Capabilities.StateTransitionHistory && task.StateTransitionHistory != null)
        {
            task.StateTransitionHistory.Add(new StateTransition
            {
                FromState = oldStatus,
                ToState = newStatus,
                Timestamp = DateTime.UtcNow
            });
        }
    }

    /// <summary>
    /// Handles a get task request
    /// </summary>
    private async System.Threading.Tasks.Task HandleGetTaskAsync(HttpListenerContext context, string requestBody)
    {
        var request = JsonSerializer.Deserialize<GetTaskRequest>(requestBody, _jsonOptions);
        var taskParams = request.TaskParams;

        // Check if the task exists
        if (!_tasks.TryGetValue(taskParams.TaskId, out var task))
        {
            await SendErrorResponseAsync(context, -32602, $"Task {taskParams.TaskId} not found", request.Id);
            return;
        }

        // Send the response
        var response = new JsonRpcResponse
        {
            Id = request.Id,
            Result = task
        };

        await SendJsonResponseAsync(context, response);
    }

    /// <summary>
    /// Handles a cancel task request
    /// </summary>
    private async System.Threading.Tasks.Task HandleCancelTaskAsync(HttpListenerContext context, string requestBody, CancellationToken cancellationToken)
    {
        var request = JsonSerializer.Deserialize<CancelTaskRequest>(requestBody, _jsonOptions);
        var taskParams = request.TaskParams;

        // Check if the task exists
        if (!_tasks.TryGetValue(taskParams.TaskId, out var task))
        {
            await SendErrorResponseAsync(context, -32602, $"Task {taskParams.TaskId} not found", request.Id);
            return;
        }

        // Update task status to Canceled
        UpdateTaskStatus(task, TaskStatus.Canceled);

        // Send the response
        var response = new JsonRpcResponse
        {
            Id = request.Id,
            Result = task
        };

        await SendJsonResponseAsync(context, response);

        // Send push notification if configured
        await SendPushNotificationAsync(task);
    }

    /// <summary>
    /// Handles a set push notification request
    /// </summary>
    private async System.Threading.Tasks.Task HandleSetPushNotificationAsync(HttpListenerContext context, string requestBody)
    {
        if (!_agentCard.Capabilities.PushNotifications)
        {
            await SendErrorResponseAsync(context, -32003, "Push Notification is not supported", null);
            return;
        }

        var request = JsonSerializer.Deserialize<SetPushNotificationRequest>(requestBody, _jsonOptions);
        var notificationParams = request.NotificationParams;

        // Store the callback URL
        _pushNotificationCallbacks[notificationParams.TaskId] = notificationParams.CallbackUrl;

        // Send the response
        var response = new JsonRpcResponse
        {
            Id = request.Id,
            Result = true
        };

        await SendJsonResponseAsync(context, response);
    }

    /// <summary>
    /// Handles a get push notification request
    /// </summary>
    private async System.Threading.Tasks.Task HandleGetPushNotificationAsync(HttpListenerContext context, string requestBody)
    {
        if (!_agentCard.Capabilities.PushNotifications)
        {
            await SendErrorResponseAsync(context, -32003, "Push Notification is not supported", null);
            return;
        }

        var request = JsonSerializer.Deserialize<GetTaskRequest>(requestBody, _jsonOptions);
        var taskParams = request.TaskParams;

        // Check if there's a callback URL for this task
        if (!_pushNotificationCallbacks.TryGetValue(taskParams.TaskId, out var callbackUrl))
        {
            await SendErrorResponseAsync(context, -32602, $"No push notification configured for task {taskParams.TaskId}", request.Id);
            return;
        }

        // Send the response
        var response = new JsonRpcResponse
        {
            Id = request.Id,
            Result = new
            {
                TaskId = taskParams.TaskId,
                CallbackUrl = callbackUrl
            }
        };

        await SendJsonResponseAsync(context, response);
    }

    /// <summary>
    /// Sends a push notification for a task
    /// </summary>
    private async System.Threading.Tasks.Task SendPushNotificationAsync(Task task)
    {
        if (!_agentCard.Capabilities.PushNotifications ||
            !_pushNotificationCallbacks.TryGetValue(task.TaskId, out var callbackUrl))
        {
            return;
        }

        try
        {
            var notification = new
            {
                jsonrpc = "2.0",
                method = "tasks/notification",
                parameters = new
                {
                    task
                }
            };

            var json = JsonSerializer.Serialize(notification, _jsonOptions);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            using var client = new HttpClient();
            var response = await client.PostAsync(callbackUrl, content);

            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning($"Failed to send push notification for task {task.TaskId}: {response.StatusCode}");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error sending push notification for task {task.TaskId}");
        }
    }

    /// <summary>
    /// Sends a JSON response
    /// </summary>
    private async System.Threading.Tasks.Task SendJsonResponseAsync(HttpListenerContext context, object response)
    {
        var json = JsonSerializer.Serialize(response, _jsonOptions);
        var buffer = Encoding.UTF8.GetBytes(json);

        context.Response.ContentType = "application/json";
        context.Response.ContentLength64 = buffer.Length;
        await context.Response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
        context.Response.Close();
    }

    /// <summary>
    /// Sends an error response
    /// </summary>
    private async System.Threading.Tasks.Task SendErrorResponseAsync(HttpListenerContext context, int code, string message, string id, object data = null)
    {
        var response = new JsonRpcResponse
        {
            Id = id,
            Error = new JsonRpcError
            {
                Code = code,
                Message = message,
                Data = data
            }
        };

        await SendJsonResponseAsync(context, response);
    }

    /// <summary>
    /// Sends a server-sent event
    /// </summary>
    private async System.Threading.Tasks.Task SendEventAsync(HttpListenerContext context, object data, string id)
    {
        var response = new JsonRpcResponse
        {
            Id = id,
            Result = data
        };

        var json = JsonSerializer.Serialize(response, _jsonOptions);
        var eventData = $"data: {json}\n\n";
        var buffer = Encoding.UTF8.GetBytes(eventData);

        await context.Response.OutputStream.WriteAsync(buffer, 0, buffer.Length);
        await context.Response.OutputStream.FlushAsync();
    }
}