using System.Text;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for generating code based on project analysis and requirements
/// </summary>
public class CodeGenerationService : ICodeGenerationService
{
    private readonly ILogger<CodeGenerationService> _logger;
    private readonly IProjectAnalysisService _projectAnalysisService;
    private readonly ILlmService _llmService;

    public CodeGenerationService(
        ILogger<CodeGenerationService> logger,
        IProjectAnalysisService projectAnalysisService,
        ILlmService llmService)
    {
        _logger = logger;
        _projectAnalysisService = projectAnalysisService;
        _llmService = llmService;
    }

    /// <summary>
    /// Generates code based on a description and project context
    /// </summary>
    /// <param name="description">Description of the code to generate</param>
    /// <param name="projectPath">Path to the project</param>
    /// <param name="language">Programming language to use</param>
    /// <param name="outputPath">Path where the generated code should be saved</param>
    /// <returns>A CodeGenerationResult containing the generated code</returns>
    public virtual async Task<CodeGenerationResult> GenerateCodeAsync(
        string description,
        string projectPath,
        ProgrammingLanguage language,
        string outputPath = null)
    {
        try
        {
            _logger.LogInformation($"Generating code for: {description}");

            // Analyze the project to get context
            var projectAnalysis = await _projectAnalysisService.AnalyzeProjectAsync(projectPath);
            if (!projectAnalysis.Success)
            {
                return new CodeGenerationResult
                {
                    Success = false,
                    ErrorMessage = $"Failed to analyze project: {projectAnalysis.ErrorMessage}"
                };
            }

            // Create a prompt for the LLM
            string prompt = CreateCodeGenerationPrompt(description, projectAnalysis, language);

            // Generate code using the LLM
            var llmResponse = await _llmService.GetCompletionAsync(prompt, temperature: 0.2, maxTokens: 2000);

            // Extract the code from the LLM response
            string generatedCode = ExtractCodeFromLlmResponse(llmResponse, language);

            // Save the code if an output path is provided
            if (!string.IsNullOrEmpty(outputPath))
            {
                await SaveGeneratedCodeAsync(generatedCode, outputPath);
            }

            return new CodeGenerationResult
            {
                Success = true,
                GeneratedCode = generatedCode,
                OutputPath = outputPath
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating code: {ex.Message}");
            return new CodeGenerationResult
            {
                Success = false,
                ErrorMessage = $"Error generating code: {ex.Message}"
            };
        }
    }

    /// <summary>
    /// Generates a unit test for a given code file
    /// </summary>
    /// <param name="sourceFilePath">Path to the source file to test</param>
    /// <param name="projectPath">Path to the project</param>
    /// <param name="outputPath">Path where the generated test should be saved</param>
    /// <returns>A CodeGenerationResult containing the generated test code</returns>
    public virtual async Task<CodeGenerationResult> GenerateUnitTestAsync(
        string sourceFilePath,
        string projectPath,
        string outputPath = null)
    {
        try
        {
            _logger.LogInformation($"Generating unit test for: {sourceFilePath}");

            // Ensure the source file exists
            if (!File.Exists(sourceFilePath))
            {
                return new CodeGenerationResult
                {
                    Success = false,
                    ErrorMessage = $"Source file not found: {sourceFilePath}"
                };
            }

            // Read the source file
            string sourceCode = await File.ReadAllTextAsync(sourceFilePath);

            // Analyze the project to get context
            var projectAnalysis = await _projectAnalysisService.AnalyzeProjectAsync(projectPath);
            if (!projectAnalysis.Success)
            {
                return new CodeGenerationResult
                {
                    Success = false,
                    ErrorMessage = $"Failed to analyze project: {projectAnalysis.ErrorMessage}"
                };
            }

            // Determine the language from the file extension
            string extension = Path.GetExtension(sourceFilePath).ToLowerInvariant();
            var language = extension switch
            {
                ".cs" => ProgrammingLanguage.CSharp,
                ".fs" => ProgrammingLanguage.FSharp,
                ".js" => ProgrammingLanguage.JavaScript,
                ".ts" => ProgrammingLanguage.TypeScript,
                ".py" => ProgrammingLanguage.Python,
                ".java" => ProgrammingLanguage.Java,
                ".cpp" or ".h" or ".hpp" => ProgrammingLanguage.Cpp,
                _ => ProgrammingLanguage.Unknown
            };

            // Create a prompt for the LLM
            string prompt = CreateUnitTestGenerationPrompt(sourceCode, sourceFilePath, projectAnalysis, language);

            // Generate test code using the LLM
            var llmResponse = await _llmService.GetCompletionAsync(prompt, temperature: 0.2, maxTokens: 2000);

            // Extract the code from the LLM response
            string generatedTestCode = ExtractCodeFromLlmResponse(llmResponse, language);

            // Determine the output path if not provided
            if (string.IsNullOrEmpty(outputPath))
            {
                string fileName = Path.GetFileNameWithoutExtension(sourceFilePath);
                string directory = Path.GetDirectoryName(sourceFilePath);

                // Determine the test file name based on the language
                string testFileName = language switch
                {
                    ProgrammingLanguage.CSharp => $"{fileName}Tests.cs",
                    ProgrammingLanguage.FSharp => $"{fileName}Tests.fs",
                    ProgrammingLanguage.JavaScript => $"{fileName}.test.js",
                    ProgrammingLanguage.TypeScript => $"{fileName}.test.ts",
                    ProgrammingLanguage.Python => $"test_{fileName}.py",
                    ProgrammingLanguage.Java => $"{fileName}Test.java",
                    ProgrammingLanguage.Cpp => $"{fileName}_test.cpp",
                    _ => $"{fileName}_test{extension}"
                };

                // Try to find a test directory
                string testDirectory = FindTestDirectory(directory, projectPath);
                outputPath = Path.Combine(testDirectory, testFileName);
            }

            // Save the test code
            await SaveGeneratedCodeAsync(generatedTestCode, outputPath);

            return new CodeGenerationResult
            {
                Success = true,
                GeneratedCode = generatedTestCode,
                OutputPath = outputPath
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating unit test: {ex.Message}");
            return new CodeGenerationResult
            {
                Success = false,
                ErrorMessage = $"Error generating unit test: {ex.Message}"
            };
        }
    }

    /// <summary>
    /// Creates a prompt for code generation
    /// </summary>
    private string CreateCodeGenerationPrompt(
        string description,
        ProjectAnalysisResult projectAnalysis,
        ProgrammingLanguage language)
    {
        var sb = new StringBuilder();

        // Add system instructions
        sb.AppendLine("You are an expert software developer tasked with generating high-quality code.");
        sb.AppendLine("Generate code based on the following description and project context.");
        sb.AppendLine();

        // Add the description
        sb.AppendLine("# Description");
        sb.AppendLine(description);
        sb.AppendLine();

        // Add project context
        sb.AppendLine("# Project Context");
        sb.AppendLine($"Project Name: {projectAnalysis.ProjectName}");
        sb.AppendLine($"Project Type: {projectAnalysis.ProjectType}");
        sb.AppendLine($"Target Framework: {projectAnalysis.TargetFramework}");
        sb.AppendLine();

        // Add namespaces
        if (projectAnalysis.Namespaces.Any())
        {
            sb.AppendLine("## Namespaces");
            foreach (var ns in projectAnalysis.Namespaces.Take(10))
            {
                sb.AppendLine($"- {ns}");
            }
            sb.AppendLine();
        }

        // Add classes
        if (projectAnalysis.Classes.Any())
        {
            sb.AppendLine("## Key Classes");
            foreach (var cls in projectAnalysis.Classes.Take(10))
            {
                sb.AppendLine($"- {cls}");
            }
            sb.AppendLine();
        }

        // Add language-specific instructions
        sb.AppendLine("# Language");
        sb.AppendLine($"Generate code in {language} programming language.");
        sb.AppendLine();

        // Add output format instructions
        sb.AppendLine("# Output Format");
        sb.AppendLine("Provide only the code without any explanations or markdown formatting.");
        sb.AppendLine("The code should be complete, well-structured, and follow best practices for the language.");
        sb.AppendLine("Include appropriate comments and documentation.");

        return sb.ToString();
    }

    /// <summary>
    /// Creates a prompt for unit test generation
    /// </summary>
    private string CreateUnitTestGenerationPrompt(
        string sourceCode,
        string sourceFilePath,
        ProjectAnalysisResult projectAnalysis,
        ProgrammingLanguage language)
    {
        var sb = new StringBuilder();

        // Add system instructions
        sb.AppendLine("You are an expert software developer tasked with generating high-quality unit tests.");
        sb.AppendLine("Generate unit tests for the following source code and project context.");
        sb.AppendLine();

        // Add the source code
        sb.AppendLine("# Source Code");
        sb.AppendLine("```");
        sb.AppendLine(sourceCode);
        sb.AppendLine("```");
        sb.AppendLine();

        // Add project context
        sb.AppendLine("# Project Context");
        sb.AppendLine($"Project Name: {projectAnalysis.ProjectName}");
        sb.AppendLine($"Project Type: {projectAnalysis.ProjectType}");
        sb.AppendLine($"Target Framework: {projectAnalysis.TargetFramework}");
        sb.AppendLine();

        // Add language-specific instructions
        sb.AppendLine("# Testing Framework");

        // Determine the testing framework based on the language
        string testingFramework = language switch
        {
            ProgrammingLanguage.CSharp => "xUnit",
            ProgrammingLanguage.FSharp => "xUnit with FsUnit",
            ProgrammingLanguage.JavaScript => "Jest",
            ProgrammingLanguage.TypeScript => "Jest with TypeScript",
            ProgrammingLanguage.Python => "pytest",
            ProgrammingLanguage.Java => "JUnit 5",
            ProgrammingLanguage.Cpp => "Google Test",
            _ => "appropriate testing framework"
        };

        sb.AppendLine($"Use {testingFramework} for the unit tests.");
        sb.AppendLine();

        // Add output format instructions
        sb.AppendLine("# Output Format");
        sb.AppendLine("Provide only the test code without any explanations or markdown formatting.");
        sb.AppendLine("The tests should be comprehensive, covering different scenarios including edge cases.");
        sb.AppendLine("Include appropriate test setup and teardown if needed.");
        sb.AppendLine("Follow best practices for unit testing in the specified language and framework.");

        return sb.ToString();
    }

    /// <summary>
    /// Extracts code from an LLM response
    /// </summary>
    private string ExtractCodeFromLlmResponse(string llmResponse, ProgrammingLanguage language)
    {
        // Check if the response is wrapped in markdown code blocks
        var codeBlockStart = "```";
        var languageIdentifier = language.ToString().ToLowerInvariant();

        // Try to find a code block with the language identifier
        int startIndex = llmResponse.IndexOf($"{codeBlockStart}{languageIdentifier}");
        if (startIndex >= 0)
        {
            startIndex = llmResponse.IndexOf('\n', startIndex) + 1;
            int endIndex = llmResponse.IndexOf(codeBlockStart, startIndex);
            if (endIndex >= 0)
            {
                return llmResponse.Substring(startIndex, endIndex - startIndex).Trim();
            }
        }

        // Try to find any code block
        startIndex = llmResponse.IndexOf(codeBlockStart);
        if (startIndex >= 0)
        {
            startIndex = llmResponse.IndexOf('\n', startIndex) + 1;
            int endIndex = llmResponse.IndexOf(codeBlockStart, startIndex);
            if (endIndex >= 0)
            {
                return llmResponse.Substring(startIndex, endIndex - startIndex).Trim();
            }
        }

        // If no code block is found, return the entire response
        return llmResponse.Trim();
    }

    /// <summary>
    /// Saves generated code to a file
    /// </summary>
    private async Task SaveGeneratedCodeAsync(string code, string outputPath)
    {
        try
        {
            // Create the directory if it doesn't exist
            string directory = Path.GetDirectoryName(outputPath);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Write the code to the file
            await File.WriteAllTextAsync(outputPath, code);

            _logger.LogInformation($"Generated code saved to: {outputPath}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error saving generated code to {outputPath}: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Finds an appropriate test directory for a source file
    /// </summary>
    private string FindTestDirectory(string sourceDirectory, string projectPath)
    {
        // Check if the source directory contains "test" in its name
        if (sourceDirectory.ToLowerInvariant().Contains("test"))
        {
            return sourceDirectory;
        }

        // Try to find a test directory in the project
        string projectDirectory = Directory.Exists(projectPath) ? projectPath : Path.GetDirectoryName(projectPath);

        // Look for common test directory names
        var testDirNames = new[] { "Tests", "Test", "tests", "test", "UnitTests", "unit-tests", "unit_tests" };
        foreach (var testDirName in testDirNames)
        {
            string testDir = Path.Combine(projectDirectory, testDirName);
            if (Directory.Exists(testDir))
            {
                return testDir;
            }
        }

        // If no test directory is found, create one
        string newTestDir = Path.Combine(projectDirectory, "Tests");
        Directory.CreateDirectory(newTestDir);
        return newTestDir;
    }
}

/// <summary>
/// Represents the result of a code generation operation
/// </summary>
public class CodeGenerationResult
{
    public bool Success { get; set; }
    public string ErrorMessage { get; set; } = string.Empty;
    public string GeneratedCode { get; set; } = string.Empty;
    public string OutputPath { get; set; } = string.Empty;
}