using System.Net;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Configuration;

namespace TarsCli.Services;

/// <summary>
/// Service for handling WebSocket connections for the chat bot
/// </summary>
public class ChatWebSocketService
{
    private readonly ILogger<ChatWebSocketService> _logger;
    private readonly IConfiguration _configuration;
    private readonly ChatBotService _chatBotService;
    private readonly HttpListener _listener;
    private readonly List<WebSocket> _connectedClients = new();
    private readonly int _port;
    private bool _isRunning = false;
    private readonly CancellationTokenSource _cancellationTokenSource = new();

    public ChatWebSocketService(
        ILogger<ChatWebSocketService> logger,
        IConfiguration configuration,
        ChatBotService chatBotService)
    {
        _logger = logger;
        _configuration = configuration;
        _chatBotService = chatBotService;
            
        // Get port from configuration or use default
        _port = _configuration.GetValue<int>("Tars:Chat:WebSocketPort", 8998);
            
        // Create HTTP listener
        _listener = new HttpListener();
        _listener.Prefixes.Add($"http://localhost:{_port}/");
            
        _logger.LogInformation($"ChatWebSocketService initialized with port: {_port}");
    }

    /// <summary>
    /// Start the WebSocket server
    /// </summary>
    public async Task StartAsync()
    {
        if (_isRunning)
        {
            _logger.LogWarning("WebSocket server is already running");
            return;
        }
            
        try
        {
            _listener.Start();
            _isRunning = true;
            _logger.LogInformation($"WebSocket server started on port {_port}");
                
            // Start accepting connections
            await AcceptConnectionsAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error starting WebSocket server");
            _isRunning = false;
            throw;
        }
    }

    /// <summary>
    /// Stop the WebSocket server
    /// </summary>
    public async Task StopAsync()
    {
        if (!_isRunning)
        {
            return;
        }
            
        try
        {
            _isRunning = false;
            _cancellationTokenSource.Cancel();
                
            // Close all connected clients
            foreach (var client in _connectedClients.ToArray())
            {
                if (client.State == WebSocketState.Open)
                {
                    await client.CloseAsync(WebSocketCloseStatus.NormalClosure, "Server shutting down", CancellationToken.None);
                }
            }
                
            _connectedClients.Clear();
            _listener.Stop();
                
            _logger.LogInformation("WebSocket server stopped");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping WebSocket server");
        }
    }

    /// <summary>
    /// Accept WebSocket connections
    /// </summary>
    private async Task AcceptConnectionsAsync()
    {
        while (_isRunning)
        {
            try
            {
                var context = await _listener.GetContextAsync();
                    
                if (context.Request.IsWebSocketRequest)
                {
                    var webSocketContext = await context.AcceptWebSocketAsync(null);
                    var webSocket = webSocketContext.WebSocket;
                        
                    _connectedClients.Add(webSocket);
                    _logger.LogInformation($"Client connected. Total clients: {_connectedClients.Count}");
                        
                    // Handle the WebSocket connection
                    _ = HandleWebSocketAsync(webSocket);
                }
                else
                {
                    context.Response.StatusCode = 400;
                    context.Response.Close();
                }
            }
            catch (Exception ex)
            {
                if (_isRunning)
                {
                    _logger.LogError(ex, "Error accepting WebSocket connection");
                }
            }
        }
    }

    /// <summary>
    /// Handle a WebSocket connection
    /// </summary>
    private async Task HandleWebSocketAsync(WebSocket webSocket)
    {
        var buffer = new byte[4096];
            
        try
        {
            // Send welcome message
            var welcomeMessage = new
            {
                type = "welcome",
                message = "Welcome to TARS Chat Bot!",
                examples = _chatBotService.GetExamplePrompts()
            };
                
            await SendMessageAsync(webSocket, welcomeMessage);
                
            // Receive messages
            while (webSocket.State == WebSocketState.Open && _isRunning)
            {
                var result = await webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), _cancellationTokenSource.Token);
                    
                if (result.MessageType == WebSocketMessageType.Text)
                {
                    var message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                    await ProcessMessageAsync(webSocket, message);
                }
                else if (result.MessageType == WebSocketMessageType.Close)
                {
                    await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Connection closed by client", CancellationToken.None);
                    _connectedClients.Remove(webSocket);
                    _logger.LogInformation($"Client disconnected. Total clients: {_connectedClients.Count}");
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling WebSocket connection");
        }
        finally
        {
            if (webSocket.State != WebSocketState.Closed)
            {
                try
                {
                    await webSocket.CloseAsync(WebSocketCloseStatus.InternalServerError, "Server error", CancellationToken.None);
                }
                catch
                {
                    // Ignore errors when closing the WebSocket
                }
            }
                
            _connectedClients.Remove(webSocket);
            _logger.LogInformation($"Client disconnected. Total clients: {_connectedClients.Count}");
        }
    }

    /// <summary>
    /// Process a message from a client
    /// </summary>
    private async Task ProcessMessageAsync(WebSocket webSocket, string message)
    {
        try
        {
            var request = JsonSerializer.Deserialize<Dictionary<string, string>>(message);
                
            if (request == null)
            {
                await SendErrorAsync(webSocket, "Invalid request format");
                return;
            }
                
            if (!request.TryGetValue("type", out var type))
            {
                await SendErrorAsync(webSocket, "Missing 'type' field in request");
                return;
            }
                
            switch (type)
            {
                case "message":
                    if (request.TryGetValue("content", out var content))
                    {
                        // Send typing indicator
                        await SendMessageAsync(webSocket, new { type = "typing", status = true });
                            
                        // Process the message
                        var response = await _chatBotService.SendMessageAsync(content);
                            
                        // Send typing indicator (done)
                        await SendMessageAsync(webSocket, new { type = "typing", status = false });
                            
                        // Send response
                        await SendMessageAsync(webSocket, new { type = "response", content = response });
                    }
                    else
                    {
                        await SendErrorAsync(webSocket, "Missing 'content' field in message request");
                    }
                    break;
                    
                case "model":
                    if (request.TryGetValue("name", out var modelName))
                    {
                        _chatBotService.SetModel(modelName);
                        await SendMessageAsync(webSocket, new { type = "model", name = modelName, status = "success" });
                    }
                    else
                    {
                        await SendErrorAsync(webSocket, "Missing 'name' field in model request");
                    }
                    break;
                    
                case "speech":
                    if (request.TryGetValue("enabled", out var enabledStr) && bool.TryParse(enabledStr, out var enabled))
                    {
                        _chatBotService.EnableSpeech(enabled);
                        await SendMessageAsync(webSocket, new { type = "speech", enabled = enabled, status = "success" });
                    }
                    else
                    {
                        await SendErrorAsync(webSocket, "Missing or invalid 'enabled' field in speech request");
                    }
                    break;
                    
                case "new_conversation":
                    _chatBotService.StartNewConversation();
                    await SendMessageAsync(webSocket, new { type = "new_conversation", status = "success" });
                    break;
                    
                default:
                    await SendErrorAsync(webSocket, $"Unknown request type: {type}");
                    break;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error processing message: {message}");
            await SendErrorAsync(webSocket, $"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Send a message to a client
    /// </summary>
    private async Task SendMessageAsync(WebSocket webSocket, object message)
    {
        if (webSocket.State != WebSocketState.Open)
        {
            return;
        }
            
        try
        {
            var json = JsonSerializer.Serialize(message);
            var buffer = Encoding.UTF8.GetBytes(json);
            await webSocket.SendAsync(new ArraySegment<byte>(buffer), WebSocketMessageType.Text, true, CancellationToken.None);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error sending message to client");
        }
    }

    /// <summary>
    /// Send an error message to a client
    /// </summary>
    private async Task SendErrorAsync(WebSocket webSocket, string errorMessage)
    {
        await SendMessageAsync(webSocket, new { type = "error", message = errorMessage });
    }

    /// <summary>
    /// Get the WebSocket server URL
    /// </summary>
    public string GetServerUrl()
    {
        return $"ws://localhost:{_port}/";
    }

    /// <summary>
    /// Check if the WebSocket server is running
    /// </summary>
    public bool IsRunning()
    {
        return _isRunning;
    }

    /// <summary>
    /// Get the number of connected clients
    /// </summary>
    public int GetConnectedClientCount()
    {
        return _connectedClients.Count;
    }
}