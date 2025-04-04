using System.Text;
using System.Text.Json;

namespace TarsCli.Services;

/// <summary>
/// Service for AI-powered code understanding
/// </summary>
public class AiCodeUnderstandingService
{
    private readonly ILogger<AiCodeUnderstandingService> _logger;
    private readonly OllamaService _ollamaService;
    private readonly string _defaultModel = "llama3";

    public AiCodeUnderstandingService(
        ILogger<AiCodeUnderstandingService> logger,
        OllamaService ollamaService)
    {
        _logger = logger;
        _ollamaService = ollamaService;
    }

    /// <summary>
    /// Analyzes code using AI to understand its purpose and structure
    /// </summary>
    /// <param name="code">The code to analyze</param>
    /// <param name="language">The programming language</param>
    /// <param name="model">The AI model to use (optional)</param>
    public async Task<CodeUnderstandingResult> AnalyzeCodeAsync(string code, string language, string model = null)
    {
        try
        {
            _logger.LogInformation($"Analyzing {language} code using AI");
                
            // Use the specified model or the default
            model = model ?? _defaultModel;
                
            // Create the prompt
            var prompt = new StringBuilder();
            prompt.AppendLine($"Analyze the following {language} code and provide a detailed understanding of its purpose, structure, and potential issues:");
            prompt.AppendLine();
            prompt.AppendLine("```");
            prompt.AppendLine(code);
            prompt.AppendLine("```");
            prompt.AppendLine();
            prompt.AppendLine("Provide your analysis in the following JSON format:");
            prompt.AppendLine("{");
            prompt.AppendLine("  \"purpose\": \"Brief description of what the code does\",");
            prompt.AppendLine("  \"structure\": {");
            prompt.AppendLine("    \"classes\": [\"List of classes in the code\"],");
            prompt.AppendLine("    \"methods\": [\"List of methods in the code\"],");
            prompt.AppendLine("    \"dependencies\": [\"List of external dependencies\"]");
            prompt.AppendLine("  },");
            prompt.AppendLine("  \"issues\": [");
            prompt.AppendLine("    {");
            prompt.AppendLine("      \"type\": \"Issue type (e.g., 'NullReferenceRisk', 'IneffectiveCode')\",");
            prompt.AppendLine("      \"description\": \"Description of the issue\",");
            prompt.AppendLine("      \"location\": \"Location in the code\",");
            prompt.AppendLine("      \"severity\": \"Severity (Info, Warning, Error)\"");
            prompt.AppendLine("    }");
            prompt.AppendLine("  ],");
            prompt.AppendLine("  \"suggestions\": [");
            prompt.AppendLine("    {");
            prompt.AppendLine("      \"type\": \"Suggestion type (e.g., 'AddNullCheck', 'ReplaceWithLinq')\",");
            prompt.AppendLine("      \"description\": \"Description of the suggestion\",");
            prompt.AppendLine("      \"location\": \"Location in the code\",");
            prompt.AppendLine("      \"confidence\": 0.9");
            prompt.AppendLine("    }");
            prompt.AppendLine("  ]");
            prompt.AppendLine("}");
                
            // Call the AI model
            var response = await _ollamaService.GenerateAsync(model, prompt.ToString());
                
            // Extract the JSON from the response
            var jsonStart = response.IndexOf("{");
            var jsonEnd = response.LastIndexOf("}") + 1;
                
            if (jsonStart < 0 || jsonEnd <= jsonStart)
            {
                _logger.LogWarning("Failed to extract JSON from AI response");
                return new CodeUnderstandingResult
                {
                    Success = false,
                    ErrorMessage = "Failed to extract JSON from AI response"
                };
            }
                
            var json = response.Substring(jsonStart, jsonEnd - jsonStart);
                
            // Parse the JSON
            var result = JsonSerializer.Deserialize<CodeUnderstandingData>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });
                
            return new CodeUnderstandingResult
            {
                Success = true,
                Data = result
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing code using AI");
            return new CodeUnderstandingResult
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    /// <summary>
    /// Suggests improvements to code using AI
    /// </summary>
    /// <param name="code">The code to improve</param>
    /// <param name="language">The programming language</param>
    /// <param name="model">The AI model to use (optional)</param>
    public async Task<CodeImprovementResult> SuggestImprovementsAsync(string code, string language, string model = null)
    {
        try
        {
            _logger.LogInformation($"Suggesting improvements for {language} code using AI");
                
            // Use the specified model or the default
            model = model ?? _defaultModel;
                
            // Create the prompt
            var prompt = new StringBuilder();
            prompt.AppendLine($"Suggest improvements for the following {language} code:");
            prompt.AppendLine();
            prompt.AppendLine("```");
            prompt.AppendLine(code);
            prompt.AppendLine("```");
            prompt.AppendLine();
            prompt.AppendLine("Provide your suggestions in the following JSON format:");
            prompt.AppendLine("{");
            prompt.AppendLine("  \"improvements\": [");
            prompt.AppendLine("    {");
            prompt.AppendLine("      \"type\": \"Improvement type (e.g., 'AddNullCheck', 'ReplaceWithLinq')\",");
            prompt.AppendLine("      \"description\": \"Description of the improvement\",");
            prompt.AppendLine("      \"location\": \"Location in the code\",");
            prompt.AppendLine("      \"originalCode\": \"The original code snippet\",");
            prompt.AppendLine("      \"improvedCode\": \"The improved code snippet\",");
            prompt.AppendLine("      \"confidence\": 0.9");
            prompt.AppendLine("    }");
            prompt.AppendLine("  ]");
            prompt.AppendLine("}");
                
            // Call the AI model
            var response = await _ollamaService.GenerateAsync(model, prompt.ToString());
                
            // Extract the JSON from the response
            var jsonStart = response.IndexOf("{");
            var jsonEnd = response.LastIndexOf("}") + 1;
                
            if (jsonStart < 0 || jsonEnd <= jsonStart)
            {
                _logger.LogWarning("Failed to extract JSON from AI response");
                return new CodeImprovementResult
                {
                    Success = false,
                    ErrorMessage = "Failed to extract JSON from AI response"
                };
            }
                
            var json = response.Substring(jsonStart, jsonEnd - jsonStart);
                
            // Parse the JSON
            var result = JsonSerializer.Deserialize<CodeImprovementData>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });
                
            return new CodeImprovementResult
            {
                Success = true,
                Data = result
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error suggesting improvements using AI");
            return new CodeImprovementResult
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    /// <summary>
    /// Generates unit tests for code using AI
    /// </summary>
    /// <param name="code">The code to generate tests for</param>
    /// <param name="language">The programming language</param>
    /// <param name="model">The AI model to use (optional)</param>
    public async Task<TestGenerationResult> GenerateTestsAsync(string code, string language, string model = null)
    {
        try
        {
            _logger.LogInformation($"Generating tests for {language} code using AI");
                
            // Use the specified model or the default
            model = model ?? _defaultModel;
                
            // Create the prompt
            var prompt = new StringBuilder();
            prompt.AppendLine($"Generate unit tests for the following {language} code:");
            prompt.AppendLine();
            prompt.AppendLine("```");
            prompt.AppendLine(code);
            prompt.AppendLine("```");
            prompt.AppendLine();
            prompt.AppendLine("Provide your tests in the following JSON format:");
            prompt.AppendLine("{");
            prompt.AppendLine("  \"testFramework\": \"The test framework to use (e.g., 'xUnit', 'NUnit', 'MSTest')\",");
            prompt.AppendLine("  \"tests\": [");
            prompt.AppendLine("    {");
            prompt.AppendLine("      \"name\": \"Test name\",");
            prompt.AppendLine("      \"description\": \"Description of what the test verifies\",");
            prompt.AppendLine("      \"code\": \"The test code\",");
            prompt.AppendLine("      \"targetMethod\": \"The method being tested\"");
            prompt.AppendLine("    }");
            prompt.AppendLine("  ]");
            prompt.AppendLine("}");
                
            // Call the AI model
            var response = await _ollamaService.GenerateAsync(model, prompt.ToString());
                
            // Extract the JSON from the response
            var jsonStart = response.IndexOf("{");
            var jsonEnd = response.LastIndexOf("}") + 1;
                
            if (jsonStart < 0 || jsonEnd <= jsonStart)
            {
                _logger.LogWarning("Failed to extract JSON from AI response");
                return new TestGenerationResult
                {
                    Success = false,
                    ErrorMessage = "Failed to extract JSON from AI response"
                };
            }
                
            var json = response.Substring(jsonStart, jsonEnd - jsonStart);
                
            // Parse the JSON
            var result = JsonSerializer.Deserialize<TestGenerationData>(json, new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            });
                
            return new TestGenerationResult
            {
                Success = true,
                Data = result
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating tests using AI");
            return new TestGenerationResult
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }
}

/// <summary>
/// Result of code understanding
/// </summary>
public class CodeUnderstandingResult
{
    public bool Success { get; set; }
    public string ErrorMessage { get; set; }
    public CodeUnderstandingData Data { get; set; }
}

/// <summary>
/// Data from code understanding
/// </summary>
public class CodeUnderstandingData
{
    public string Purpose { get; set; }
    public CodeStructure Structure { get; set; }
    public List<CodeIssue> Issues { get; set; } = new List<CodeIssue>();
    public List<CodeSuggestion> Suggestions { get; set; } = new List<CodeSuggestion>();
}

/// <summary>
/// Structure of code
/// </summary>
public class CodeStructure
{
    public List<string> Classes { get; set; } = new List<string>();
    public List<string> Methods { get; set; } = new List<string>();
    public List<string> Dependencies { get; set; } = new List<string>();
}

/// <summary>
/// Result of code improvement
/// </summary>
public class CodeImprovementResult
{
    public bool Success { get; set; }
    public string ErrorMessage { get; set; }
    public CodeImprovementData Data { get; set; }
}

/// <summary>
/// Data from code improvement
/// </summary>
public class CodeImprovementData
{
    public List<CodeImprovement> Improvements { get; set; } = new List<CodeImprovement>();
}

/// <summary>
/// A code improvement
/// </summary>
public class CodeImprovement
{
    public string Type { get; set; }
    public string Description { get; set; }
    public string Location { get; set; }
    public string OriginalCode { get; set; }
    public string ImprovedCode { get; set; }
    public double Confidence { get; set; }
}

/// <summary>
/// Result of test generation
/// </summary>
public class TestGenerationResult
{
    public bool Success { get; set; }
    public string ErrorMessage { get; set; }
    public TestGenerationData Data { get; set; }
}

/// <summary>
/// Data from test generation
/// </summary>
public class TestGenerationData
{
    public string TestFramework { get; set; }
    public List<TestCase> Tests { get; set; } = new List<TestCase>();
}

/// <summary>
/// A test case
/// </summary>
public class TestCase
{
    public string Name { get; set; }
    public string Description { get; set; }
    public string Code { get; set; }
    public string TargetMethod { get; set; }
}