using Microsoft.Extensions.Logging;
using System.Text;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for generating tests using LLMs
/// </summary>
public class LlmBasedTestGenerationService : ITestGenerationService
{
    private readonly ILogger<LlmBasedTestGenerationService> _logger;
    private readonly ILlmService _llmService;

    /// <summary>
    /// Initializes a new instance of the <see cref="LlmBasedTestGenerationService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="llmService">The LLM service</param>
    public LlmBasedTestGenerationService(
        ILogger<LlmBasedTestGenerationService> logger,
        ILlmService llmService)
    {
        _logger = logger;
        _llmService = llmService;
    }

    /// <summary>
    /// Generate tests for a component
    /// </summary>
    /// <param name="filePath">The file path of the component</param>
    /// <param name="description">A description of the component</param>
    /// <returns>The result of the test generation</returns>
    public async Task<TestGenerationResult> GenerateTestsAsync(string filePath, string description)
    {
        _logger.LogInformation($"Generating tests for {filePath}");
        
        try
        {
            // Determine the programming language based on the file extension
            var language = GetLanguageFromFilePath(filePath);
            
            // Determine the test framework based on the language
            var testFramework = GetTestFrameworkForLanguage(language);
            
            // Build the prompt
            var prompt = new StringBuilder();
            prompt.AppendLine($"Generate {language} unit tests using {testFramework} for the following component:");
            prompt.AppendLine(description);
            prompt.AppendLine();
            prompt.AppendLine($"File path: {filePath}");
            prompt.AppendLine();
            prompt.AppendLine("Please provide only the test code without any explanations or markdown formatting.");
            
            // Generate the tests
            var generatedTests = await _llmService.GetCompletionAsync(prompt.ToString(), temperature: 0.2, maxTokens: 2000);
            
            // Clean up the generated tests
            generatedTests = CleanGeneratedCode(generatedTests);
            
            // Determine the test file path
            var testFilePath = GetTestFilePath(filePath);
            
            return new TestGenerationResult
            {
                Success = true,
                GeneratedTests = generatedTests,
                TestFilePath = testFilePath
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating tests for {filePath}");
            
            return new TestGenerationResult
            {
                Success = false,
                ErrorMessage = ex.Message,
                TestFilePath = GetTestFilePath(filePath)
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
    /// Get the test framework for a language
    /// </summary>
    /// <param name="language">The programming language</param>
    /// <returns>The test framework</returns>
    private string GetTestFrameworkForLanguage(string language)
    {
        return language switch
        {
            "C#" => "xUnit",
            "F#" => "xUnit",
            "JavaScript" => "Jest",
            "TypeScript" => "Jest",
            "Python" => "pytest",
            "Rust" => "rust-test",
            _ => "xUnit" // Default to xUnit
        };
    }

    /// <summary>
    /// Get the test file path for a source file
    /// </summary>
    /// <param name="sourceFilePath">The source file path</param>
    /// <returns>The test file path</returns>
    private string GetTestFilePath(string sourceFilePath)
    {
        var directory = Path.GetDirectoryName(sourceFilePath);
        var fileName = Path.GetFileNameWithoutExtension(sourceFilePath);
        var extension = Path.GetExtension(sourceFilePath);
        
        // Replace /src/ with /tests/ in the directory path
        var testDirectory = directory
            .Replace("/src/", "/tests/")
            .Replace("\\src\\", "\\tests\\");
        
        // If the directory doesn't contain "src", add "Tests" to the end
        if (testDirectory == directory)
        {
            testDirectory = Path.Combine(directory, "Tests");
        }
        
        // Create the test file path
        return Path.Combine(testDirectory, $"{fileName}Tests{extension}");
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
    /// Generates tests for a specific file
    /// </summary>
    /// <param name="filePath">Path to the file to generate tests for</param>
    /// <param name="projectPath">Path to the project containing the file</param>
    /// <param name="testFramework">Test framework to use (e.g., xUnit, NUnit)</param>
    /// <returns>Generated test code</returns>
    public async Task<string> GenerateTestsForFileAsync(string filePath, string projectPath, string testFramework = "xUnit")
    {
        // Build a description for the test
        var description = $"Generate unit tests for the file at {filePath} in project {projectPath}";
        
        // Use the other method
        var result = await GenerateTestsAsync(filePath, description);
        
        return result.Success ? result.GeneratedTests : $"// Error generating tests: {result.ErrorMessage}";
    }

    /// <summary>
    /// Generates tests for a specific method
    /// </summary>
    /// <param name="filePath">Path to the file containing the method</param>
    /// <param name="methodName">Name of the method to generate tests for</param>
    /// <param name="projectPath">Path to the project containing the file</param>
    /// <param name="testFramework">Test framework to use (e.g., xUnit, NUnit)</param>
    /// <returns>Generated test code</returns>
    public async Task<string> GenerateTestsForMethodAsync(string filePath, string methodName, string projectPath, string testFramework = "xUnit")
    {
        // Build a description for the test
        var description = $"Generate unit tests for the method {methodName} in file {filePath} in project {projectPath}";
        
        // Use the other method
        var result = await GenerateTestsAsync(filePath, description);
        
        return result.Success ? result.GeneratedTests : $"// Error generating tests: {result.ErrorMessage}";
    }

    /// <summary>
    /// Generates tests for improved code
    /// </summary>
    /// <param name="originalCode">Original code</param>
    /// <param name="improvedCode">Improved code</param>
    /// <param name="language">Programming language</param>
    /// <param name="testFramework">Test framework to use (e.g., xUnit, NUnit)</param>
    /// <returns>Generated test code</returns>
    public async Task<string> GenerateTestsForImprovedCodeAsync(string originalCode, string improvedCode, string language, string testFramework = "xUnit")
    {
        // Build a description for the test
        var description = $"Generate unit tests for the following improved code:\n\nOriginal code:\n{originalCode}\n\nImproved code:\n{improvedCode}";
        
        // Use the other method
        var result = await GenerateTestsAsync("improved-code.cs", description);
        
        return result.Success ? result.GeneratedTests : $"// Error generating tests: {result.ErrorMessage}";
    }

    /// <summary>
    /// Suggests test cases based on code analysis
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="projectPath">Path to the project containing the file</param>
    /// <returns>List of suggested test cases</returns>
    public async Task<List<TestCase>> SuggestTestCasesAsync(string filePath, string projectPath)
    {
        // Build a description for the test
        var description = $"Suggest test cases for the file at {filePath} in project {projectPath}";
        
        // Use the other method
        var result = await GenerateTestsAsync(filePath, description);
        
        // For now, return an empty list
        return new List<TestCase>();
    }
}
