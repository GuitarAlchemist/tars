using Microsoft.Extensions.Logging;
using System.Text;
using TarsEngine.Models;
using TarsEngine.Monads;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for generating code using LLMs
/// </summary>
public class LlmBasedCodeGenerationService : ICodeGenerationService
{
    private readonly ILogger<LlmBasedCodeGenerationService> _logger;
    private readonly ILlmService _llmService;

    /// <summary>
    /// Initializes a new instance of the <see cref="LlmBasedCodeGenerationService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="llmService">The LLM service</param>
    public LlmBasedCodeGenerationService(
        ILogger<LlmBasedCodeGenerationService> logger,
        ILlmService llmService)
    {
        _logger = logger;
        _llmService = llmService;
    }

    /// <summary>
    /// Generate code based on a description
    /// </summary>
    /// <param name="description">The description of the code to generate</param>
    /// <param name="filePath">The file path where the code will be saved</param>
    /// <param name="changeType">The type of change to make</param>
    /// <returns>The result of the code generation</returns>
    public async Task<CodeGenerationResult> GenerateCodeAsync(string description, string filePath, string changeType)
    {
        _logger.LogInformation($"Generating code for {filePath} with change type {changeType}");

        try
        {
            // Determine the programming language based on the file extension
            var language = GetLanguageFromFilePath(filePath);

            // Build the prompt
            var prompt = new StringBuilder();
            prompt.AppendLine($"Generate {language} code for the following description:");
            prompt.AppendLine(description);
            prompt.AppendLine();
            prompt.AppendLine($"File path: {filePath}");
            prompt.AppendLine($"Change type: {changeType}");
            prompt.AppendLine();
            prompt.AppendLine("Please provide only the code without any explanations or markdown formatting.");

            // Generate the code
            var generatedCode = await _llmService.GetCompletionAsync(prompt.ToString(), temperature: 0.2, maxTokens: 2000);

            // Clean up the generated code
            generatedCode = CleanGeneratedCode(generatedCode);

            return new CodeGenerationResult
            {
                Success = true,
                GeneratedCode = generatedCode,
                OutputPath = filePath
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating code for {filePath}");

            return new CodeGenerationResult
            {
                Success = false,
                ErrorMessage = ex.Message,
                OutputPath = filePath
            };
        }
    }

    /// <summary>
    /// Get the programming language from a file path
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <returns>The programming language</returns>
    private string GetLanguageFromFilePath(string filePath)
    {
        var extension = Path.GetExtension(filePath).ToLowerInvariant();

        return extension switch
        {
            ".cs" => "C#",
            ".fs" => "F#",
            ".js" => "JavaScript",
            ".ts" => "TypeScript",
            ".py" => "Python",
            ".rs" => "Rust",
            _ => "C#" // Default to C#
        };
    }

    /// <summary>
    /// Clean up the generated code
    /// </summary>
    /// <param name="code">The generated code</param>
    /// <returns>The cleaned code</returns>
    private string CleanGeneratedCode(string code)
    {
        // Remove markdown code blocks if present
        if (code.StartsWith("```") && code.EndsWith("```"))
        {
            var lines = code.Split('\n');
            var startIndex = 1; // Skip the first line with ```
            var endIndex = lines.Length - 1; // Skip the last line with ```

            // If the first line contains the language specifier (e.g., ```csharp)
            if (lines[0].Length > 3)
            {
                startIndex = 1;
            }

            // Join the lines between the code block markers
            code = string.Join("\n", lines.Skip(startIndex).Take(endIndex - startIndex));
        }

        return code;
    }

    /// <summary>
    /// Generates code based on a description and project context
    /// </summary>
    /// <param name="description">Description of the code to generate</param>
    /// <param name="projectPath">Path to the project</param>
    /// <param name="language">Programming language to use</param>
    /// <param name="outputPath">Path where the generated code should be saved</param>
    /// <returns>A CodeGenerationResult containing the generated code</returns>
    public async Task<CodeGenerationResult> GenerateCodeAsync(
        string description,
        string projectPath,
        CodeLanguage language,
        Option<string> outputPath = default)
    {
        // Convert the language enum to a string
        var languageStr = language.ToString();

        // Use the other overload
        return await GenerateCodeAsync(description, outputPath.ValueOr(projectPath), "Create");
    }

    /// <summary>
    /// Generates a unit test for a given code file
    /// </summary>
    /// <param name="sourceFilePath">Path to the source file to test</param>
    /// <param name="projectPath">Path to the project</param>
    /// <param name="outputPath">Path where the generated test should be saved</param>
    /// <returns>A CodeGenerationResult containing the generated test code</returns>
    public async Task<CodeGenerationResult> GenerateUnitTestAsync(
        string sourceFilePath,
        string projectPath,
        Option<string> outputPath = default)
    {
        // Build a description for the test
        var description = $"Generate unit tests for the file at {sourceFilePath}";

        // Use the other overload
        return await GenerateCodeAsync(description, outputPath.ValueOr(sourceFilePath.Replace(".cs", "Tests.cs")), "Create");
    }
}
