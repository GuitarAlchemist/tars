using System.Text;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Monads;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for generating code based on project analysis and requirements
/// </summary>
public sealed class CodeGenerationService(
    ILogger<CodeGenerationService> logger,
    IProjectAnalysisService projectAnalysisService,
    ILlmService llmService)
    : ICodeGenerationService
{
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
        try
        {
            logger.LogInformation($"Generating code for: {description}");

            // Analyze the project to get context
            var projectAnalysis = await projectAnalysisService.AnalyzeProjectAsync(projectPath);

            // Create a prompt for the LLM
            var prompt = CreateCodeGenerationPrompt(description, projectAnalysis, language);

            // Generate code using the LLM
            var llmResponse = await llmService.GetCompletionAsync(prompt, temperature: 0.2, maxTokens: 2000);

            // Extract the code from the LLM response
            var generatedCode = ExtractCodeFromLlmResponse(llmResponse, language);

            // Save the code if an output path is provided
            outputPath.IfSome(async path => {
                await SaveGeneratedCodeAsync(generatedCode, path);
            });

            return new CodeGenerationResult
            {
                Success = true,
                GeneratedCode = generatedCode,
                OutputPath = outputPath.ValueOr(string.Empty)
            };
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error generating code: {ex.Message}");
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
    public async Task<CodeGenerationResult> GenerateUnitTestAsync(
        string sourceFilePath,
        string projectPath,
        Option<string> outputPath = default)
    {
        try
        {
            logger.LogInformation($"Generating unit test for: {sourceFilePath}");

            var directory = Path.GetDirectoryName(sourceFilePath);
            if (string.IsNullOrEmpty(directory))
            {
                return new CodeGenerationResult
                {
                    Success = false,
                    ErrorMessage = $"Invalid source file path: {sourceFilePath}"
                };
            }

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
            var sourceCode = await File.ReadAllTextAsync(sourceFilePath);

            // Analyze the project to get context
            var projectAnalysis = await projectAnalysisService.AnalyzeProjectAsync(projectPath);

            // Determine the language from the file extension
            var extension = Path.GetExtension(sourceFilePath).ToLowerInvariant();
            var language = extension switch
            {
                ".cs" => CodeLanguage.CSharp,
                ".fs" => CodeLanguage.FSharp,
                ".js" => CodeLanguage.JavaScript,
                ".ts" => CodeLanguage.TypeScript,
                ".py" => CodeLanguage.Python,
                _ => CodeLanguage.CSharp
            };

            // Create a prompt for the LLM
            var prompt = CreateUnitTestGenerationPrompt(sourceCode, sourceFilePath, projectAnalysis, language);

            // Generate test code using the LLM
            var llmResponse = await llmService.GetCompletionAsync(prompt, temperature: 0.2, maxTokens: 2000);

            // Extract the code from the LLM response
            var generatedTestCode = ExtractCodeFromLlmResponse(llmResponse, language);

            // Determine the output path if not provided
            var finalOutputPath = outputPath.ValueOr(() => {
                var fileName = Path.GetFileNameWithoutExtension(sourceFilePath);

                // Determine the test file name based on the language
                var testFileName = language switch
                {
                    CodeLanguage.CSharp => $"{fileName}Tests.cs",
                    CodeLanguage.FSharp => $"{fileName}Tests.fs",
                    CodeLanguage.JavaScript => $"{fileName}.test.js",
                    CodeLanguage.TypeScript => $"{fileName}.test.ts",
                    CodeLanguage.Python => $"test_{fileName}.py",
                    CodeLanguage.Rust => $"{fileName}_test.rs",
                    _ => $"{fileName}_test{extension}"
                };

                // Try to find a test directory
                var testDirectory = FindTestDirectory(directory, projectPath);
                return Path.Combine(testDirectory, testFileName);
            });

            // Save the test code
            await SaveGeneratedCodeAsync(generatedTestCode, finalOutputPath);

            return new CodeGenerationResult
            {
                Success = true,
                GeneratedCode = generatedTestCode,
                OutputPath = finalOutputPath
            };
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error generating unit test: {ex.Message}");
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
        CodeLanguage language)
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
        sb.AppendLine();

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
        CodeLanguage language)
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
        sb.AppendLine();

        // Add language-specific instructions
        sb.AppendLine("# Testing Framework");

        // Determine the testing framework based on the language
        var testingFramework = language switch
        {
            CodeLanguage.CSharp => "xUnit",
            CodeLanguage.FSharp => "xUnit with FsUnit",
            CodeLanguage.JavaScript => "Jest",
            CodeLanguage.TypeScript => "Jest with TypeScript",
            CodeLanguage.Python => "pytest",
            CodeLanguage.Rust => "Rust Test",
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
    private string ExtractCodeFromLlmResponse(string llmResponse, CodeLanguage language)
    {
        // Check if the response is wrapped in markdown code blocks
        var codeBlockStart = "```";
        var languageIdentifier = language.ToString().ToLowerInvariant();

        // Try to find a code block with the language identifier
        var startIndex = llmResponse.IndexOf($"{codeBlockStart}{languageIdentifier}");
        if (startIndex >= 0)
        {
            startIndex = llmResponse.IndexOf('\n', startIndex) + 1;
            var endIndex = llmResponse.IndexOf(codeBlockStart, startIndex);
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
            var endIndex = llmResponse.IndexOf(codeBlockStart, startIndex);
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
            var directory = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }
            else
            {
                throw new ArgumentException("Invalid output path: directory path is null or empty", nameof(outputPath));
            }

            // Write the code to the file
            await File.WriteAllTextAsync(outputPath, code);

            logger.LogInformation($"Generated code saved to: {outputPath}");
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error saving generated code to {outputPath}: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Finds an appropriate test directory for a source file
    /// </summary>
    /// <param name="sourceDirectory">The directory containing the source file</param>
    /// <param name="projectPath">The project path</param>
    /// <returns>The path to the test directory</returns>
    private string FindTestDirectory(string sourceDirectory, string projectPath)
    {
        ArgumentNullException.ThrowIfNull(sourceDirectory);
        ArgumentNullException.ThrowIfNull(projectPath);

        // Check if the source directory contains "test" in its name
        if (sourceDirectory.ToLowerInvariant().Contains("test"))
        {
            return sourceDirectory;
        }

        // Try to find a test directory in the project
        var projectDirectory = Directory.Exists(projectPath) 
            ? projectPath 
            : Path.GetDirectoryName(projectPath) ?? throw new DirectoryNotFoundException($"Invalid project path: {projectPath}");

        // Look for common test directory names
        var testDirNames = new[] { "Tests", "Test", "tests", "test", "UnitTests", "unit-tests", "unit_tests" };
        foreach (var testDirName in testDirNames)
        {
            var testDir = Path.Combine(projectDirectory, testDirName);
            if (Directory.Exists(testDir))
            {
                return testDir;
            }
        }

        // If no test directory is found, create one
        var newTestDir = Path.Combine(projectDirectory, "Tests");
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