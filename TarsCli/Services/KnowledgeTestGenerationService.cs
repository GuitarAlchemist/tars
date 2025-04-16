using System.Text.Json;

namespace TarsCli.Services;

/// <summary>
/// Service for generating test cases from extracted knowledge
/// </summary>
public class KnowledgeTestGenerationService
{
    private readonly ILogger<KnowledgeTestGenerationService> _logger;
    private readonly ConsoleService _consoleService;
    private readonly OllamaService _ollamaService;
    private readonly string _knowledgeBaseFile = "knowledge_base.json";
    private readonly string _testsOutputDir = "generated_tests";

    /// <summary>
    /// Initializes a new instance of the <see cref="KnowledgeTestGenerationService"/> class.
    /// </summary>
    public KnowledgeTestGenerationService(
        ILogger<KnowledgeTestGenerationService> logger,
        ConsoleService consoleService,
        OllamaService ollamaService)
    {
        _logger = logger;
        _consoleService = consoleService;
        _ollamaService = ollamaService;
    }

    /// <summary>
    /// Generates test cases from the knowledge base
    /// </summary>
    /// <param name="targetProject">The target project for which to generate tests</param>
    /// <param name="maxTests">Maximum number of tests to generate</param>
    /// <returns>Number of tests generated</returns>
    public async Task<int> GenerateTestsAsync(string targetProject, int maxTests = 5)
    {
        try
        {
            _consoleService.WriteHeader("TARS Knowledge Test Generation");
            _consoleService.WriteInfo($"Generating tests for project: {targetProject}");

            // Ensure the tests output directory exists
            Directory.CreateDirectory(_testsOutputDir);

            // Read the knowledge base
            if (!File.Exists(_knowledgeBaseFile))
            {
                _consoleService.WriteError($"Knowledge base file not found: {_knowledgeBaseFile}");
                return 0;
            }

            var kbJson = await File.ReadAllTextAsync(_knowledgeBaseFile);
            var kb = JsonSerializer.Deserialize<JsonElement>(kbJson);

            // Extract code examples and patterns from the knowledge base
            var codeExamples = kb.GetProperty("code_examples").EnumerateArray().ToList();
            var patterns = kb.GetProperty("patterns").EnumerateArray().ToList();

            // Filter examples and patterns relevant to the target project
            var relevantExamples = FilterRelevantItems(codeExamples, targetProject);
            var relevantPatterns = FilterRelevantItems(patterns, targetProject);

            _consoleService.WriteInfo($"Found {relevantExamples.Count} relevant code examples and {relevantPatterns.Count} relevant patterns");

            // Generate tests from code examples
            var exampleTestsCount = await GenerateTestsFromCodeExamplesAsync(relevantExamples, targetProject, maxTests / 2);

            // Generate tests from patterns
            var patternTestsCount = await GenerateTestsFromPatternsAsync(relevantPatterns, targetProject, maxTests / 2);

            var totalTests = exampleTestsCount + patternTestsCount;
            _consoleService.WriteSuccess($"Generated {totalTests} tests ({exampleTestsCount} from examples, {patternTestsCount} from patterns)");

            return totalTests;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating tests from knowledge base");
            _consoleService.WriteError($"Error: {ex.Message}");
            return 0;
        }
    }

    /// <summary>
    /// Filters items relevant to the target project
    /// </summary>
    private List<JsonElement> FilterRelevantItems(List<JsonElement> items, string targetProject)
    {
        var relevantItems = new List<JsonElement>();

        foreach (var item in items)
        {
            var isRelevant = false;

            // Check if the item has a context/language property that matches the project
            if (item.TryGetProperty("context", out var context) &&
                context.ValueKind == JsonValueKind.String)
            {
                var contextValue = context.GetString() ?? "";
                isRelevant = IsContextRelevantToProject(contextValue, targetProject);
            }
            else if (item.TryGetProperty("language", out var language) &&
                     language.ValueKind == JsonValueKind.String)
            {
                var languageValue = language.GetString() ?? "";
                isRelevant = IsContextRelevantToProject(languageValue, targetProject);
            }

            // Check if the item has a description that mentions the project
            if (!isRelevant && item.TryGetProperty("description", out var description) &&
                description.ValueKind == JsonValueKind.String)
            {
                var descriptionValue = description.GetString() ?? "";
                isRelevant = descriptionValue.Contains(targetProject, StringComparison.OrdinalIgnoreCase);
            }

            if (isRelevant)
            {
                relevantItems.Add(item);
            }
        }

        return relevantItems;
    }

    /// <summary>
    /// Determines if a context is relevant to a project
    /// </summary>
    private bool IsContextRelevantToProject(string context, string project)
    {
        // Map of project names to relevant contexts
        var projectContextMap = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase)
        {
            { "TarsEngine", ["CSharp", "C#", ".NET", "TarsEngine"] },
            { "TarsEngineFSharp", ["FSharp", "F#", ".NET", "TarsEngineFSharp"] },
            { "TarsEngine.DSL", ["FSharp", "F#", ".NET", "DSL", "TarsEngine.DSL"] },
            { "TarsEngine.SelfImprovement", ["FSharp", "F#", ".NET", "SelfImprovement", "TarsEngine.SelfImprovement"] },
            { "TarsCli", ["CSharp", "C#", ".NET", "CLI", "TarsCli"] }
        };

        // Check if the project exists in the map
        if (projectContextMap.TryGetValue(project, out string[]? relevantContexts))
        {
            // Check if the context is in the list of relevant contexts
            return relevantContexts.Any(c => context.Contains(c, StringComparison.OrdinalIgnoreCase));
        }

        // If the project is not in the map, check if the context contains the project name
        return context.Contains(project, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Generates tests from code examples
    /// </summary>
    private async Task<int> GenerateTestsFromCodeExamplesAsync(List<JsonElement> examples, string targetProject, int maxTests)
    {
        var testsGenerated = 0;

        // Take only up to maxTests examples
        foreach (var example in examples.Take(maxTests))
        {
            try
            {
                var description = example.GetProperty("description").GetString() ?? "Code Example";
                var code = example.GetProperty("code").GetString() ?? "";
                var language = example.GetProperty("language").GetString() ?? "CSharp";

                if (string.IsNullOrWhiteSpace(code))
                {
                    continue;
                }

                // Generate a test for this example
                var testCode = await GenerateTestForCodeAsync(code, description, language, targetProject);

                if (!string.IsNullOrWhiteSpace(testCode))
                {
                    // Save the test to a file
                    var testFileName = SanitizeFileName($"{targetProject}_{description}_Test.cs");
                    var testFilePath = Path.Combine(_testsOutputDir, testFileName);
                    await File.WriteAllTextAsync(testFilePath, testCode);

                    _consoleService.WriteInfo($"Generated test: {testFilePath}");
                    testsGenerated++;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating test from code example");
            }
        }

        return testsGenerated;
    }

    /// <summary>
    /// Generates tests from patterns
    /// </summary>
    private async Task<int> GenerateTestsFromPatternsAsync(List<JsonElement> patterns, string targetProject, int maxTests)
    {
        var testsGenerated = 0;

        // Take only up to maxTests patterns
        foreach (var pattern in patterns.Take(maxTests))
        {
            try
            {
                var name = pattern.GetProperty("name").GetString() ?? "Pattern";
                var description = pattern.GetProperty("description").GetString() ?? "";
                var example = pattern.GetProperty("example").GetString() ?? "";
                var context = pattern.GetProperty("context").GetString() ?? "CSharp";

                if (string.IsNullOrWhiteSpace(example))
                {
                    continue;
                }

                // Generate a test for this pattern
                var testCode = await GenerateTestForPatternAsync(example, name, description, context, targetProject);

                if (!string.IsNullOrWhiteSpace(testCode))
                {
                    // Save the test to a file
                    var testFileName = SanitizeFileName($"{targetProject}_{name}_Test.cs");
                    var testFilePath = Path.Combine(_testsOutputDir, testFileName);
                    await File.WriteAllTextAsync(testFilePath, testCode);

                    _consoleService.WriteInfo($"Generated test: {testFilePath}");
                    testsGenerated++;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating test from pattern");
            }
        }

        return testsGenerated;
    }

    /// <summary>
    /// Generates a test for a code example
    /// </summary>
    private async Task<string> GenerateTestForCodeAsync(string code, string description, string language, string targetProject)
    {
        try
        {
            // Create a prompt for the LLM to generate a test
            var prompt = $@"You are an expert at writing unit tests for .NET applications.

I'll provide you with a code example from the {targetProject} project, and I'd like you to generate a comprehensive unit test for it.

The code example is described as: {description}

Here's the code:
```{language}
{code}
```

Please generate a C# unit test for this code using xUnit. The test should:
1. Be well-structured and follow best practices
2. Include appropriate assertions to verify the code's behavior
3. Use mocks or stubs where necessary
4. Include XML documentation comments
5. Be named appropriately based on the functionality being tested

The test should be compatible with the {targetProject} project structure and naming conventions.

Return ONLY the C# test code without any explanations or markdown formatting.";

            // Use the OllamaService to generate the test
            var result = await _ollamaService.GenerateCompletion(prompt, "llama3");

            if (!string.IsNullOrWhiteSpace(result))
            {
                // Clean up the result (remove any markdown code blocks if present)
                var cleanedResult = CleanupGeneratedCode(result);

                // Add a header comment with the source information
                var headerComment = $@"// Generated test for code example: {description}
// Source: Knowledge Base
// Target Project: {targetProject}
// Generated: {DateTime.Now}

";
                return headerComment + cleanedResult;
            }

            return string.Empty;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating test for code");
            return string.Empty;
        }
    }

    /// <summary>
    /// Generates a test for a pattern
    /// </summary>
    private async Task<string> GenerateTestForPatternAsync(string example, string name, string description, string context, string targetProject)
    {
        try
        {
            // Create a prompt for the LLM to generate a test
            var prompt = $@"You are an expert at writing unit tests for .NET applications.

I'll provide you with a pattern from the {targetProject} project, and I'd like you to generate a comprehensive unit test that verifies this pattern is correctly implemented.

Pattern Name: {name}
Pattern Description: {description}
Context: {context}

Example code that implements this pattern:
```{context}
{example}
```

Please generate a C# unit test for this pattern using xUnit. The test should:
1. Be well-structured and follow best practices
2. Include appropriate assertions to verify the pattern is correctly implemented
3. Use mocks or stubs where necessary
4. Include XML documentation comments
5. Be named appropriately based on the pattern being tested

The test should be compatible with the {targetProject} project structure and naming conventions.

Return ONLY the C# test code without any explanations or markdown formatting.";

            // Use the OllamaService to generate the test
            var result = await _ollamaService.GenerateCompletion(prompt, "llama3");

            if (!string.IsNullOrWhiteSpace(result))
            {
                // Clean up the result (remove any markdown code blocks if present)
                var cleanedResult = CleanupGeneratedCode(result);

                // Add a header comment with the source information
                var headerComment = $@"// Generated test for pattern: {name}
// Description: {description}
// Context: {context}
// Target Project: {targetProject}
// Generated: {DateTime.Now}

";
                return headerComment + cleanedResult;
            }

            return string.Empty;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating test for pattern");
            return string.Empty;
        }
    }

    /// <summary>
    /// Cleans up generated code by removing markdown code blocks if present
    /// </summary>
    private string CleanupGeneratedCode(string code)
    {
        // Remove markdown code blocks if present
        if (code.StartsWith("```") && code.EndsWith("```"))
        {
            // Find the first newline after the opening ```
            var startIndex = code.IndexOf('\n');
            if (startIndex != -1)
            {
                // Find the last ``` and remove everything after it
                var endIndex = code.LastIndexOf("```");
                if (endIndex != -1)
                {
                    return code.Substring(startIndex + 1, endIndex - startIndex - 1).Trim();
                }
            }
        }

        return code;
    }

    /// <summary>
    /// Sanitizes a file name by removing invalid characters
    /// </summary>
    private string SanitizeFileName(string fileName)
    {
        // Replace invalid file name characters with underscores
        var invalidChars = Path.GetInvalidFileNameChars();
        foreach (var c in invalidChars)
        {
            fileName = fileName.Replace(c, '_');
        }

        // Replace spaces with underscores
        fileName = fileName.Replace(' ', '_');

        return fileName;
    }
}
