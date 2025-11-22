# Ollama Chatbot Prototype

**User:** Stephane Pareilleux (spareilleux@gmail.com)  
**Created:** 3/20/2025 0:37  
**Updated:** 3/20/2025 0:38  
**Exported:** 3/22/2025 16:28  

## Prompt:
Please create a prototype of a Chabot using https://devblogs.microsoft.com/dotnet/announcing-dotnet-ai-template-preview1/ but for Ollanta and qwq model https://ollama.com/library/qwq:32b

## Response:
Raisonnement terminé pendant 20 secondes

Below is an example prototype of a chatbot built in C# that follows the spirit of the new .NET AI template preview while plugging into an Ollama endpoint using the qwq:32b model. In this example the chatbot:

• Uses dependency injection (via Microsoft.Extensions.DependencyInjection) to register a custom provider that implements a chat-completion interface  
• Defines a simple interface (IChatCompletion) similar to what the .NET AI template preview describes  
• Implements a provider (OllamaQwqProvider) that sends a POST request to an assumed Ollama API endpoint (adjust the URL and payload as needed)  
• Runs a conversation loop reading user input, building a conversation history (which can be used to “prime” the model), and then outputting the model’s response

You can adjust the details (such as the API endpoint, payload structure, and prompt formatting) based on the actual Ollama API specifications.

Below is the complete sample code:

```csharp
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;

namespace ChatbotPrototype
{
    // Define a simple chat completion contract similar to the .NET AI template.
    public interface IChatCompletion
    {
        Task<ChatCompletionResult> GetChatCompletionAsync(ChatCompletionRequest request, CancellationToken cancellationToken = default);
    }

    // Request object that contains the conversation history.
    public class ChatCompletionRequest
    {
        public IEnumerable<ChatMessage> Messages { get; set; }
    }

    // Response object wrapping the assistant's reply.
    public class ChatCompletionResult
    {
        public ChatMessage Message { get; set; }
    }

    // Represents a single message in the conversation.
    public class ChatMessage
    {
        public string Role { get; set; }  // For example: "user" or "assistant"
        public string Content { get; set; }
    }

    // Represents the expected response from the Ollama API.
    public class OllamaResponse
    {
        public string Result { get; set; }
    }

    // A custom provider that calls the Ollama API for the qwq:32b model.
    public class OllamaQwqProvider : IChatCompletion
    {
        private readonly HttpClient _httpClient;
        // Adjust this URL to match your actual Ollama API endpoint.
        private readonly string _endpoint = "http://localhost:11434/api/generate";

        public OllamaQwqProvider(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public async Task<ChatCompletionResult> GetChatCompletionAsync(ChatCompletionRequest request, CancellationToken cancellationToken = default)
        {
            // Build a prompt from the conversation history.
            string prompt = BuildPrompt(request.Messages);

            var payload = new
            {
                model = "qwq:32b", // Specify the desired model.
                prompt = prompt
            };

            // Send the request to the Ollama API.
            var response = await _httpClient.PostAsJsonAsync(_endpoint, payload, cancellationToken);
            if (response.IsSuccessStatusCode)
            {
                var ollamaResponse = await response.Content.ReadFromJsonAsync<OllamaResponse>(cancellationToken: cancellationToken);
                return new ChatCompletionResult
                {
                    Message = new ChatMessage
                    {
                        Role = "assistant",
                        Content = ollamaResponse?.Result ?? "No response from model."
                    }
                };
            }
            else
            {
                throw new Exception($"Error calling Ollama API: {response.StatusCode}");
            }
        }

        // A simple method to concatenate conversation messages into a prompt.
        private string BuildPrompt(IEnumerable<ChatMessage> messages)
        {
            return string.Join("\n", messages.Select(m => $"{m.Role}: {m.Content}"));
        }
    }

    class Program
    {
        static async Task Main(string[] args)
        {
            // Set up dependency injection.
            var services = new ServiceCollection();
            services.AddHttpClient<OllamaQwqProvider>();
            services.AddSingleton<IChatCompletion, OllamaQwqProvider>();
            var serviceProvider = services.BuildServiceProvider();

            var chatProvider = serviceProvider.GetRequiredService<IChatCompletion>();

            Console.WriteLine("Welcome to the Qwq:32b Chatbot!");
            var conversation = new List<ChatMessage>();

            // A simple loop to simulate an interactive conversation.
            while (true)
            {
                Console.Write("User: ");
                string userInput = Console.ReadLine();
                if (string.IsNullOrWhiteSpace(userInput))
                {
                    break; // Exit on empty input.
                }

                // Add the user's message to the conversation.
                conversation.Add(new ChatMessage { Role = "user", Content = userInput });

                try
                {
                    var chatRequest = new ChatCompletionRequest { Messages = conversation };
                    var result = await chatProvider.GetChatCompletionAsync(chatRequest);
                    // Add the model's response to the conversation history.
                    conversation.Add(result.Message);
                    Console.WriteLine("Bot: " + result.Message.Content);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Error: " + ex.Message);
                }
            }
        }
    }
}
```

### How It Works

1.  **Chat Interface Definition:**  
    The `IChatCompletion` interface and related request/response types mimic the abstraction found in the .NET AI template preview. This allows you to easily swap out or add new providers.
    
2.  **Custom Provider Implementation:**  
    The `OllamaQwqProvider` implements the interface by building a prompt from a list of chat messages, then sending a JSON payload (including the model identifier `"qwq:32b"`) to the Ollama API endpoint via an `HttpClient`.  
    _Note:_ Adjust the endpoint URL and payload details according to the actual Ollama API documentation.
    
3.  **Program Setup:**  
    The `Program` class sets up dependency injection and runs a conversation loop that reads user input, calls the custom provider to generate a response, and prints it out.
    

### Next Steps

• **Refinement:** You might refine the prompt-building logic to better support multi-turn conversations or include system instructions.  
• **Error Handling:** Enhance error handling and possibly support streaming responses if the Ollama API supports it.  
• **Configuration:** Consider moving configuration (e.g., API URL, model name) to an external configuration file or environment variables.

This prototype should give you a starting point for building a chatbot that leverages the ideas in the .NET AI template preview and integrates with the Ollama qwq:32b model.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)