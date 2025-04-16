using System.Text.Json;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for generating improvement suggestions based on code analysis
/// </summary>
public class ImprovementSuggestionGenerator
{
    private readonly ILogger<ImprovementSuggestionGenerator> _logger;
    private readonly LlmService _llmService;

    /// <summary>
    /// Initializes a new instance of the ImprovementSuggestionGenerator class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="llmService">LLM service</param>
    public ImprovementSuggestionGenerator(
        ILogger<ImprovementSuggestionGenerator> logger,
        LlmService llmService)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _llmService = llmService ?? throw new ArgumentNullException(nameof(llmService));
    }

    /// <summary>
    /// Generates improvement suggestions based on code analysis
    /// </summary>
    /// <param name="fileContent">The file content</param>
    /// <param name="analysis">The analysis result</param>
    /// <param name="language">The language</param>
    /// <param name="model">The model to use (optional)</param>
    /// <returns>The improvement suggestions</returns>
    public async Task<List<ImprovementSuggestion>> GenerateSuggestionsAsync(string fileContent, string analysis, string language, string model = "llama3")
    {
        try
        {
            _logger.LogInformation($"Generating improvement suggestions for {language} code");

            // Create a prompt for the LLM
            var prompt = $@"Analyze the following {language} code and generate improvement suggestions based on the provided analysis.

Code:
```{language}
{fileContent}
```

Analysis:
{analysis}

Generate a list of specific, actionable improvement suggestions. For each suggestion, include:
1. A clear description of the issue
2. The location in the code (line number if possible)
3. A specific recommendation for how to fix it
4. The expected benefit of making the change

Format your response as a JSON array of objects with the following structure:
[
  {{
    ""description"": ""Description of the issue"",
    ""location"": ""Location in the code (e.g., line number, method name)"",
    ""recommendation"": ""Specific recommendation for how to fix it"",
    ""benefit"": ""Expected benefit of making the change"",
    ""priority"": 5 // Priority from 1 (low) to 10 (high)
  }}
]";

            // Generate the suggestions
            var suggestionsJson = await _llmService.GetCompletionAsync(prompt, temperature: 0.2, maxTokens: 2000, model: model);

            // Parse the suggestions
            try
            {
                var suggestions = JsonSerializer.Deserialize<List<ImprovementSuggestion>>(suggestionsJson);
                return suggestions ?? new List<ImprovementSuggestion>();
            }
            catch (JsonException ex)
            {
                _logger.LogError(ex, "Failed to parse improvement suggestions");
                return new List<ImprovementSuggestion>();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating improvement suggestions");
            return new List<ImprovementSuggestion>();
        }
    }
}

// Using the existing ImprovementSuggestion class from TarsEngine.Models namespace
