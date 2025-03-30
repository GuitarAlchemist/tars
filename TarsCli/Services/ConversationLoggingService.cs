using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for logging conversations between TARS and other AI tools like Augment Code
    /// </summary>
    public class ConversationLoggingService
    {
        private readonly ILogger<ConversationLoggingService> _logger;
        private readonly IConfiguration _configuration;
        private readonly string _logDirectory;
        private readonly string _conversationLogPath;
        private readonly string _augmentLogPath;
        private readonly object _fileLock = new object();

        public ConversationLoggingService(
            ILogger<ConversationLoggingService> logger,
            IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;
            
            // Get log directory from configuration or use default
            _logDirectory = _configuration.GetValue<string>("Tars:Logs:Directory", "logs");
            
            // Ensure log directory exists
            if (!Directory.Exists(_logDirectory))
            {
                Directory.CreateDirectory(_logDirectory);
            }
            
            // Set log file paths
            _conversationLogPath = Path.Combine(_logDirectory, "conversations.json");
            _augmentLogPath = Path.Combine(_logDirectory, "augment_conversations.md");
            
            // Create log files if they don't exist
            if (!File.Exists(_conversationLogPath))
            {
                File.WriteAllText(_conversationLogPath, "[]");
            }
            
            if (!File.Exists(_augmentLogPath))
            {
                File.WriteAllText(_augmentLogPath, "# TARS - Augment Code Conversations\n\n");
            }
            
            _logger.LogInformation($"ConversationLoggingService initialized with log directory: {_logDirectory}");
        }

        /// <summary>
        /// Log a conversation between TARS and another AI tool
        /// </summary>
        public async Task LogConversationAsync(string source, string action, object request, object response)
        {
            try
            {
                var conversation = new
                {
                    Timestamp = DateTime.Now,
                    Source = source,
                    Action = action,
                    Request = request,
                    Response = response
                };
                
                // Log to JSON file
                await LogToJsonAsync(conversation);
                
                // If source is Augment, also log to markdown file
                if (source.Equals("augment", StringComparison.OrdinalIgnoreCase))
                {
                    await LogToMarkdownAsync(conversation);
                }
                
                _logger.LogInformation($"Logged conversation from {source} with action {action}");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error logging conversation from {source}");
            }
        }

        /// <summary>
        /// Log a conversation to the JSON log file
        /// </summary>
        private async Task LogToJsonAsync(object conversation)
        {
            try
            {
                // Read existing conversations
                string json;
                lock (_fileLock)
                {
                    json = File.ReadAllText(_conversationLogPath);
                }
                
                var conversations = JsonSerializer.Deserialize<List<object>>(json) ?? new List<object>();
                
                // Add new conversation
                conversations.Add(conversation);
                
                // Write updated conversations
                var updatedJson = JsonSerializer.Serialize(conversations, new JsonSerializerOptions { WriteIndented = true });
                
                lock (_fileLock)
                {
                    File.WriteAllText(_conversationLogPath, updatedJson);
                }
                
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error logging conversation to JSON");
                throw;
            }
        }

        /// <summary>
        /// Log a conversation to the markdown log file
        /// </summary>
        private async Task LogToMarkdownAsync(object conversation)
        {
            try
            {
                // Convert conversation to markdown format
                var markdown = FormatConversationAsMarkdown(conversation);
                
                // Append to markdown file
                lock (_fileLock)
                {
                    File.AppendAllText(_augmentLogPath, markdown);
                }
                
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error logging conversation to markdown");
                throw;
            }
        }

        /// <summary>
        /// Format a conversation as markdown
        /// </summary>
        private string FormatConversationAsMarkdown(object conversation)
        {
            var json = JsonSerializer.Serialize(conversation, new JsonSerializerOptions { WriteIndented = true });
            var conversationDict = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json);
            
            var timestamp = conversationDict["Timestamp"].GetString();
            var action = conversationDict["Action"].GetString();
            var request = conversationDict["Request"].ToString();
            var response = conversationDict["Response"].ToString();
            
            var markdown = $"## Conversation on {timestamp}\n\n";
            markdown += $"### Action: {action}\n\n";
            markdown += "### Request\n\n```json\n";
            markdown += request + "\n";
            markdown += "```\n\n";
            markdown += "### Response\n\n```json\n";
            markdown += response + "\n";
            markdown += "```\n\n";
            markdown += "---\n\n";
            
            return markdown;
        }

        /// <summary>
        /// Get the path to the Augment conversation log file
        /// </summary>
        public string GetAugmentLogPath()
        {
            return _augmentLogPath;
        }

        /// <summary>
        /// Get recent conversations
        /// </summary>
        public async Task<List<object>> GetRecentConversationsAsync(int count = 10, string source = null)
        {
            try
            {
                // Read existing conversations
                string json;
                lock (_fileLock)
                {
                    json = File.ReadAllText(_conversationLogPath);
                }
                
                var conversations = JsonSerializer.Deserialize<List<Dictionary<string, JsonElement>>>(json) ?? new List<Dictionary<string, JsonElement>>();
                
                // Filter by source if specified
                if (!string.IsNullOrEmpty(source))
                {
                    conversations = conversations.FindAll(c => c["Source"].GetString().Equals(source, StringComparison.OrdinalIgnoreCase));
                }
                
                // Get the most recent conversations
                var recentConversations = conversations.Count <= count ? conversations : conversations.GetRange(conversations.Count - count, count);
                
                return recentConversations.ConvertAll(c => (object)c);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting recent conversations");
                return new List<object>();
            }
        }
    }
}
