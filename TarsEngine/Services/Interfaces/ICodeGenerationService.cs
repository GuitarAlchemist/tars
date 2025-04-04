namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the code generation service
/// </summary>
public interface ICodeGenerationService
{
    /// <summary>
    /// Generates code based on a description and project context
    /// </summary>
    /// <param name="description">Description of the code to generate</param>
    /// <param name="projectPath">Path to the project</param>
    /// <param name="language">Programming language to use</param>
    /// <param name="outputPath">Path where the generated code should be saved</param>
    /// <returns>A CodeGenerationResult containing the generated code</returns>
    Task<CodeGenerationResult> GenerateCodeAsync(
        string description,
        string projectPath,
        ProgrammingLanguage language,
        string outputPath = null);

    /// <summary>
    /// Generates a unit test for a given code file
    /// </summary>
    /// <param name="sourceFilePath">Path to the source file to test</param>
    /// <param name="projectPath">Path to the project</param>
    /// <param name="outputPath">Path where the generated test should be saved</param>
    /// <returns>A CodeGenerationResult containing the generated test code</returns>
    Task<CodeGenerationResult> GenerateUnitTestAsync(
        string sourceFilePath,
        string projectPath,
        string outputPath = null);
}