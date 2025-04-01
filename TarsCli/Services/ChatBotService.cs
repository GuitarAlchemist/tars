using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;

namespace TarsCli.Services;

/// <summary>
/// Service for handling chat bot conversations in TARS
/// </summary>
public class ChatBotService
{
    private readonly ILogger<ChatBotService> _logger;
    private readonly IConfiguration _configuration;
    private readonly OllamaService _ollamaService;
    private readonly TarsSpeechService _speechService;
    private readonly string _historyDirectory;
    private readonly List<ChatMessage> _currentConversation = new();
    private string _currentModel = "llama3";
    private bool _speechEnabled = false;

    public ChatBotService(
        ILogger<ChatBotService> logger,
        IConfiguration configuration,
        OllamaService ollamaService,
        TarsSpeechService speechService)
    {
        _logger = logger;
        _configuration = configuration;
        _ollamaService = ollamaService;
        _speechService = speechService;
            
        // Get history directory from configuration or use default
        _historyDirectory = _configuration.GetValue<string>("Tars:Chat:HistoryDirectory", "chat_history");
            
        // Ensure history directory exists
        if (!Directory.Exists(_historyDirectory))
        {
            Directory.CreateDirectory(_historyDirectory);
        }
            
        _logger.LogInformation($"ChatBotService initialized with history directory: {_historyDirectory}");
    }

    /// <summary>
    /// Start a new conversation
    /// </summary>
    public void StartNewConversation(string model = "llama3")
    {
        _currentConversation.Clear();
        _currentModel = model;
        _logger.LogInformation($"Started new conversation with model: {model}");
    }

    /// <summary>
    /// Send a message to the chat bot and get a response
    /// </summary>
    public async Task<string> SendMessageAsync(string message, bool useSpeech = false)
    {
        try
        {
            // Add user message to conversation
            _currentConversation.Add(new ChatMessage { Role = "user", Content = message });
                
            // Generate system prompt based on conversation history
            var prompt = GeneratePrompt();
                
            // Get response from Ollama
            _logger.LogInformation($"Sending message to {_currentModel}: {message}");
            var response = await _ollamaService.GenerateCompletion(prompt, _currentModel);
                
            // Add assistant response to conversation
            _currentConversation.Add(new ChatMessage { Role = "assistant", Content = response });
                
            // Save conversation history
            await SaveConversationHistoryAsync();
                
            // Speak the response if speech is enabled
            if (useSpeech || _speechEnabled)
            {
                _speechService.Speak(response);
            }
                
            return response;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error sending message to chat bot: {message}");
            return $"Error: {ex.Message}";
        }
    }

    /// <summary>
    /// Generate a prompt based on the conversation history
    /// </summary>
    private string GeneratePrompt()
    {
        var sb = new StringBuilder();
            
        // Add system message if not present
        if (!_currentConversation.Any(m => m.Role == "system"))
        {
            sb.AppendLine("<|im_start|>system");
            sb.AppendLine("You are TARS, an AI assistant that is helpful, harmless, and honest. You provide concise and accurate responses.");
            sb.AppendLine("<|im_end|>");
        }
            
        // Add conversation history
        foreach (var message in _currentConversation)
        {
            sb.AppendLine($"<|im_start|>{message.Role}");
            sb.AppendLine(message.Content);
            sb.AppendLine("<|im_end|>");
        }
            
        // Add assistant prompt
        sb.AppendLine("<|im_start|>assistant");
            
        return sb.ToString();
    }

    /// <summary>
    /// Save the current conversation history
    /// </summary>
    private async Task SaveConversationHistoryAsync()
    {
        try
        {
            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var filePath = Path.Combine(_historyDirectory, $"conversation_{timestamp}.json");
                
            var json = JsonSerializer.Serialize(_currentConversation, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(filePath, json);
                
            _logger.LogInformation($"Saved conversation history to: {filePath}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving conversation history");
        }
    }

    /// <summary>
    /// Load a conversation history from a file
    /// </summary>
    public async Task<bool> LoadConversationHistoryAsync(string filePath)
    {
        try
        {
            if (!File.Exists(filePath))
            {
                _logger.LogWarning($"Conversation history file not found: {filePath}");
                return false;
            }
                
            var json = await File.ReadAllTextAsync(filePath);
            var conversation = JsonSerializer.Deserialize<List<ChatMessage>>(json);
                
            if (conversation != null)
            {
                _currentConversation.Clear();
                _currentConversation.AddRange(conversation);
                _logger.LogInformation($"Loaded conversation history from: {filePath}");
                return true;
            }
                
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error loading conversation history from: {filePath}");
            return false;
        }
    }

    /// <summary>
    /// Get a list of available conversation history files
    /// </summary>
    public List<string> GetConversationHistoryFiles()
    {
        try
        {
            var files = Directory.GetFiles(_historyDirectory, "conversation_*.json")
                .OrderByDescending(f => f)
                .ToList();
                
            return files;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting conversation history files");
            return new List<string>();
        }
    }

    /// <summary>
    /// Get the current conversation
    /// </summary>
    public List<ChatMessage> GetCurrentConversation()
    {
        return _currentConversation.ToList();
    }

    /// <summary>
    /// Set the current model
    /// </summary>
    public void SetModel(string model)
    {
        _currentModel = model;
        _logger.LogInformation($"Set chat model to: {model}");
    }

    /// <summary>
    /// Enable or disable speech
    /// </summary>
    public void EnableSpeech(bool enabled)
    {
        _speechEnabled = enabled;
        _logger.LogInformation($"Speech {(enabled ? "enabled" : "disabled")} for chat bot");
    }

    /// <summary>
    /// Get a list of example prompts
    /// </summary>
    public List<string> GetExamplePrompts()
    {
        return new List<string>
        {
            "Hello, how are you today?",
            "What is the capital of France?",
            "Write a short poem about artificial intelligence.",
            "Explain quantum computing in simple terms.",
            "What are the main features of TARS?",
            "How can I use the TARS CLI effectively?",
            "Tell me a joke about programming.",
            "What's the difference between machine learning and deep learning?",
            "Can you help me debug this code: Console.WriteLine('Hello, World!');",
            "What are some best practices for writing clean code?"
        };
    }
}

/// <summary>
/// Represents a chat message
/// </summary>
public class ChatMessage
{
    /// <summary>
    /// The role of the message sender (system, user, or assistant)
    /// </summary>
    public string Role { get; set; } = string.Empty;
        
    /// <summary>
    /// The content of the message
    /// </summary>
    public string Content { get; set; } = string.Empty;
        
    /// <summary>
    /// The timestamp of the message
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.Now;
}