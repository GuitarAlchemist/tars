namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the code analysis service
/// </summary>
public interface ICodeAnalysisService
{
    /// <summary>
    /// Analyzes a file and extracts information about its structure
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <returns>A CodeAnalysisResult containing information about the file</returns>
    Task<CodeAnalysisResult> AnalyzeFileAsync(string filePath);

    /// <summary>
    /// Analyzes a directory and extracts information about its structure
    /// </summary>
    /// <param name="directoryPath">Path to the directory to analyze</param>
    /// <param name="recursive">Whether to analyze subdirectories</param>
    /// <returns>A list of CodeAnalysisResult containing information about each file</returns>
    Task<List<CodeAnalysisResult>> AnalyzeDirectoryAsync(string directoryPath, bool recursive = true);
}