using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for generating tests
/// </summary>
public class TestGenerationService : ITestGenerationService
{
    private readonly ILogger<TestGenerationService> _logger;
    private readonly IMetascriptService _metascriptService;

    /// <summary>
    /// Initializes a new instance of the <see cref="TestGenerationService"/> class
    /// </summary>
    public TestGenerationService(ILogger<TestGenerationService> logger, IMetascriptService metascriptService)
    {
        _logger = logger;
        _metascriptService = metascriptService;
    }

    /// <inheritdoc/>
    public async Task<string> GenerateTestsForFileAsync(string filePath, string projectPath, string testFramework = "xUnit")
    {
        try
        {
            _logger.LogInformation("Generating tests for file: {FilePath}", filePath);

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            var fileContent = await File.ReadAllTextAsync(filePath);
            var fileExtension = Path.GetExtension(filePath).ToLowerInvariant();
            var language = GetLanguageFromExtension(fileExtension);

            // Create a metascript for test generation
            var metascript = $@"
// Test generation metascript for {Path.GetFileName(filePath)}
// Language: {language}
// Test framework: {testFramework}

// Read the source file
let sourceCode = `{fileContent.Replace("`", "\\`")}`;

// Analyze the code to identify classes, methods, and properties
let codeStructure = analyzeCode(sourceCode, '{language}');

// Generate tests for each public method
let tests = generateTests(codeStructure, '{testFramework}');

// Return the generated tests
return tests;

// Helper function to analyze code
function analyzeCode(code, language) {{
    // This would be implemented with a more sophisticated code analysis
    // For now, we'll use a simple placeholder
    return {{
        classes: extractClasses(code, language),
        methods: extractMethods(code, language),
        properties: extractProperties(code, language)
    }};
}}

// Helper function to extract classes
function extractClasses(code, language) {{
    // Implementation would depend on the language
    // This is a simplified version
    let classes = [];

    if (language === 'csharp') {{
        // Simple regex to find class declarations
        const classRegex = /class\s+(\w+)/g;
        let match;
        while ((match = classRegex.exec(code)) !== null) {{
            classes.push({{
                name: match[1],
                position: match.index
            }});
        }}
    }}

    return classes;
}}

// Helper function to extract methods
function extractMethods(code, language) {{
    // Implementation would depend on the language
    // This is a simplified version
    let methods = [];

    if (language === 'csharp') {{
        // Simple regex to find method declarations
        const methodRegex = /(public|private|protected|internal)\s+(\w+)\s+(\w+)\s*\((.*?)\)/g;
        let match;
        while ((match = methodRegex.exec(code)) !== null) {{
            methods.push({{
                accessibility: match[1],
                returnType: match[2],
                name: match[3],
                parameters: match[4],
                position: match.index
            }});
        }}
    }}

    return methods;
}}

// Helper function to extract properties
function extractProperties(code, language) {{
    // Implementation would depend on the language
    // This is a simplified version
    let properties = [];

    if (language === 'csharp') {{
        // Simple regex to find property declarations
        const propertyRegex = /(public|private|protected|internal)\s+(\w+)\s+(\w+)\s*\{{\s*get;\s*set;\s*\}}/g;
        let match;
        while ((match = propertyRegex.exec(code)) !== null) {{
            properties.push({{
                accessibility: match[1],
                type: match[2],
                name: match[3],
                position: match.index
            }});
        }}
    }}

    return properties;
}}

// Helper function to generate tests
function generateTests(codeStructure, testFramework) {{
    let testCode = '';

    if (testFramework === 'xUnit') {{
        testCode += 'using System;\n';
        testCode += 'using Xunit;\n';
        testCode += 'using Moq;\n\n';

        // Generate a test class for each class
        for (const cls of codeStructure.classes) {{
            testCode += `public class ${{cls.name}}Tests\n{{\n`;

            // Generate a test method for each public method
            for (const method of codeStructure.methods.filter(m => m.accessibility === 'public')) {{
                testCode += `    [Fact]\n`;
                testCode += `    public void ${{method.name}}_ShouldReturnExpectedResult()\n    {{\n`;
                testCode += `        // Arrange\n`;
                testCode += `        // TODO: Set up test data and dependencies\n\n`;
                testCode += `        // Act\n`;
                testCode += `        // TODO: Call the method under test\n\n`;
                testCode += `        // Assert\n`;
                testCode += `        // TODO: Verify the results\n`;
                testCode += `    }}\n\n`;
            }}

            testCode += `}}\n\n`;
        }}
    }}

    return testCode;
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            return result.ToString();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating tests for file: {FilePath}", filePath);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<string> GenerateTestsForMethodAsync(string filePath, string methodName, string projectPath, string testFramework = "xUnit")
    {
        try
        {
            _logger.LogInformation("Generating tests for method: {MethodName} in file: {FilePath}", methodName, filePath);

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            var fileContent = await File.ReadAllTextAsync(filePath);
            var fileExtension = Path.GetExtension(filePath).ToLowerInvariant();
            var language = GetLanguageFromExtension(fileExtension);

            // Create a metascript for test generation
            var metascript = $@"
// Test generation metascript for method {methodName} in {Path.GetFileName(filePath)}
// Language: {language}
// Test framework: {testFramework}

// Read the source file
let sourceCode = `{fileContent.Replace("`", "\\`")}`;

// Extract the method
let methodCode = extractMethod(sourceCode, '{methodName}', '{language}');

// Generate tests for the method
let tests = generateTestsForMethod(methodCode, '{methodName}', '{testFramework}');

// Return the generated tests
return tests;

// Helper function to extract a method
function extractMethod(code, methodName, language) {{
    // Implementation would depend on the language
    // This is a simplified version
    if (language === 'csharp') {{
        // Simple regex to find the method
        const methodRegex = new RegExp(`(public|private|protected|internal)\\s+(\\w+)\\s+${methodName}\\s*\\((.*?)\\)[^{{]*{{([^{{}}]*(?:{{[^{{}}]*}}[^{{}}]*)*)}}`, 's');
        const match = methodRegex.exec(code);

        if (match) {{
            return {{
                accessibility: match[1],
                returnType: match[2],
                name: methodName,
                parameters: match[3],
                body: match[4]
            }};
        }}
    }}

    return null;
}}

// Helper function to generate tests for a method
function generateTestsForMethod(methodCode, methodName, testFramework) {{
    if (!methodCode) {{
        return `// Could not find method: ${methodName}`;
    }}

    let testCode = '';

    if (testFramework === 'xUnit') {{
        testCode += 'using System;\n';
        testCode += 'using Xunit;\n';
        testCode += 'using Moq;\n\n';

        // Extract class name from file path
        const className = 'SomeClass'; // This would be extracted from the actual code

        testCode += `public class SomeClass_${methodName}Tests\n{{\n`;

        // Generate test methods
        testCode += `    [Fact]\n`;
        testCode += `    public void ${methodName}_ShouldReturnExpectedResult()\n    {{\n`;
        testCode += `        // Arrange\n`;
        testCode += `        // TODO: Set up test data and dependencies\n\n`;
        testCode += `        // Act\n`;
        testCode += `        // TODO: Call the method under test\n\n`;
        testCode += `        // Assert\n`;
        testCode += `        // TODO: Verify the results\n`;
        testCode += `    }}\n\n`;

        // Generate additional test methods for edge cases
        testCode += `    [Fact]\n`;
        testCode += `    public void ${methodName}_ShouldHandleEdgeCases()\n    {{\n`;
        testCode += `        // Arrange\n`;
        testCode += `        // TODO: Set up edge case test data\n\n`;
        testCode += `        // Act\n`;
        testCode += `        // TODO: Call the method under test\n\n`;
        testCode += `        // Assert\n`;
        testCode += `        // TODO: Verify the results\n`;
        testCode += `    }}\n\n`;

        // Generate test method for exceptions if applicable
        if (methodCode.body.includes('throw')) {{
            testCode += `    [Fact]\n`;
            testCode += `    public void ${methodName}_ShouldThrowException_WhenInvalidInput()\n    {{\n`;
            testCode += `        // Arrange\n`;
            testCode += `        // TODO: Set up invalid input\n\n`;
            testCode += `        // Act & Assert\n`;
            testCode += `        // TODO: Verify that the expected exception is thrown\n`;
            testCode += `    }}\n\n`;
        }}

        testCode += `}}\n`;
    }}

    return testCode;
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            return result.ToString();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating tests for method: {MethodName} in file: {FilePath}", methodName, filePath);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<string> GenerateTestsForImprovedCodeAsync(string originalCode, string improvedCode, string language, string testFramework = "xUnit")
    {
        try
        {
            _logger.LogInformation("Generating tests for improved code");

            // Create a metascript for test generation
            var metascript = $@"
// Test generation metascript for improved code
// Language: {language}
// Test framework: {testFramework}

// Original code
let originalCode = `{originalCode.Replace("`", "\\`")}`;

// Improved code
let improvedCode = `{improvedCode.Replace("`", "\\`")}`;

// Analyze the differences between original and improved code
let differences = analyzeDifferences(originalCode, improvedCode);

// Generate tests that verify the improvements
let tests = generateTestsForImprovements(differences, '{testFramework}');

// Return the generated tests
return tests;

// Helper function to analyze differences
function analyzeDifferences(original, improved) {{
    // This would be implemented with a more sophisticated diff algorithm
    // For now, we'll use a simple placeholder
    return {{
        changedMethods: identifyChangedMethods(original, improved),
        addedMethods: identifyAddedMethods(original, improved),
        removedMethods: identifyRemovedMethods(original, improved)
    }};
}}

// Helper function to identify changed methods
function identifyChangedMethods(original, improved) {{
    // Implementation would depend on the language
    // This is a simplified version
    return [];
}}

// Helper function to identify added methods
function identifyAddedMethods(original, improved) {{
    // Implementation would depend on the language
    // This is a simplified version
    return [];
}}

// Helper function to identify removed methods
function identifyRemovedMethods(original, improved) {{
    // Implementation would depend on the language
    // This is a simplified version
    return [];
}}

// Helper function to generate tests for improvements
function generateTestsForImprovements(differences, testFramework) {{
    let testCode = '';

    if (testFramework === 'xUnit') {{
        testCode += 'using System;\n';
        testCode += 'using Xunit;\n';
        testCode += 'using Moq;\n\n';

        testCode += `public class ImprovementTests\n{{\n`;

        // Generate tests for changed methods
        // This would be implemented with actual method extraction
        // For now, we'll just generate a placeholder test
        testCode += `    [Fact]\n`;
        testCode += `    public void Method_ShouldBehaveSameAsOriginal()\n    {{\n`;
        testCode += `        // Arrange\n`;
        testCode += `        // TODO: Set up test data\n\n`;
        testCode += `        // Act - Call both original and improved methods\n`;
        testCode += `        // TODO: Call the methods\n\n`;
        testCode += `        // Assert - Verify that the results are the same\n`;
        testCode += `        // TODO: Verify the results\n`;
        testCode += `    }}\n\n`;
        }}

        // Generate tests for added methods
        // This would be implemented with actual method extraction
        // For now, we'll just generate a placeholder test
        testCode += `    [Fact]\n`;
        testCode += `    public void NewMethod_ShouldWorkAsExpected()\n    {{\n`;
        testCode += `        // Arrange\n`;
        testCode += `        // TODO: Set up test data\n\n`;
        testCode += `        // Act\n`;
        testCode += `        // TODO: Call the method\n\n`;
        testCode += `        // Assert\n`;
        testCode += `        // TODO: Verify the results\n`;
        testCode += `    }}\n\n`;

        testCode += `}}\n`;
    }}

    return testCode;
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);
            return result.ToString();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating tests for improved code");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<List<TestCase>> SuggestTestCasesAsync(string filePath, string projectPath)
    {
        try
        {
            _logger.LogInformation("Suggesting test cases for file: {FilePath}", filePath);

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            var fileContent = await File.ReadAllTextAsync(filePath);
            var fileExtension = Path.GetExtension(filePath).ToLowerInvariant();
            var language = GetLanguageFromExtension(fileExtension);

            // Create a metascript for test case suggestion
            var metascript = $@"
// Test case suggestion metascript for {Path.GetFileName(filePath)}
// Language: {language}

// Read the source file
let sourceCode = `{fileContent.Replace("`", "\\`")}`;

// Analyze the code to identify methods
let methods = extractMethods(sourceCode, '{language}');

// Generate test cases for each method
let testCases = [];
for (const method of methods) {{
    const methodTestCases = suggestTestCasesForMethod(method);
    testCases = testCases.concat(methodTestCases);
}}

// Return the suggested test cases
return JSON.stringify(testCases);

// Helper function to extract methods
function extractMethods(code, language) {{
    // Implementation would depend on the language
    // This is a simplified version
    let methods = [];

    if (language === 'csharp') {{
        // Simple regex to find method declarations
        const methodRegex = /(public|private|protected|internal)\s+(\w+)\s+(\w+)\s*\((.*?)\)/g;
        let match;
        while ((match = methodRegex.exec(code)) !== null) {{
            methods.push({{
                accessibility: match[1],
                returnType: match[2],
                name: match[3],
                parameters: match[4],
                position: match.index
            }});
        }}
    }}

    return methods;
}}

// Helper function to suggest test cases for a method
function suggestTestCasesForMethod(method) {{
    // Only suggest test cases for public methods
    if (method.accessibility !== 'public') {{
        return [];
    }}

    const testCases = [];

    // Add a basic test case
    testCases.push({{
        name: `${{method.name}}_ShouldReturnExpectedResult`,
        description: `Verify that ${{method.name}} returns the expected result with valid input`,
        inputs: {{}},
        expectedOutput: null,
        expectedException: null,
        priority: 'Medium',
        category: 'Functional'
    }});

    // Add an edge case test
    testCases.push({{
        name: `${{method.name}}_ShouldHandleEdgeCases`,
        description: `Verify that ${{method.name}} handles edge cases correctly`,
        inputs: {{}},
        expectedOutput: null,
        expectedException: null,
        priority: 'Medium',
        category: 'EdgeCase'
    }});

    // Add a null input test if the method has parameters
    if (method.parameters && method.parameters.trim() !== '') {{
        testCases.push({{
            name: `${{method.name}}_ShouldHandleNullInput`,
            description: `Verify that ${{method.name}} handles null input correctly`,
            inputs: {{}},
            expectedOutput: null,
            expectedException: 'ArgumentNullException',
            priority: 'High',
            category: 'Validation'
        }});
    }}

    return testCases;
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);

            // Parse the result as JSON
            var testCases = ParseTestCases(result.ToString());
            return testCases;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error suggesting test cases for file: {FilePath}", filePath);
            throw;
        }
    }

    private List<TestCase> ParseTestCases(string? json)
    {
        if (string.IsNullOrEmpty(json))
        {
            _logger.LogWarning("Received null or empty JSON for test cases");
            return [];
        }

        try
        {
            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            var testCases = JsonSerializer.Deserialize<List<TestCase>>(json, options);
            return testCases ?? [];
        }
        catch (JsonException ex)
        {
            _logger.LogError(ex, "Failed to parse test cases JSON");
            return [];
        }
    }

    private string GetLanguageFromExtension(string extension)
    {
        return extension switch
        {
            ".cs" => "csharp",
            ".fs" => "fsharp",
            ".js" => "javascript",
            ".ts" => "typescript",
            ".py" => "python",
            ".java" => "java",
            _ => "unknown"
        };
    }
}
