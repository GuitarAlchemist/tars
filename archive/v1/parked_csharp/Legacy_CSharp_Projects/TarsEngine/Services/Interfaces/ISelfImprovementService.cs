namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the self-improvement service
/// </summary>
public interface ISelfImprovementService
{
    /// <summary>
    /// Analyzes a file and suggests improvements
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="projectPath">Path to the project</param>
    /// <returns>A list of improvement suggestions</returns>
    Task<List<ImprovementSuggestion>> AnalyzeFileForImprovementsAsync(
        string filePath,
        string projectPath);

    /// <summary>
    /// Applies suggested improvements to a file
    /// </summary>
    /// <param name="filePath">Path to the file to improve</param>
    /// <param name="suggestions">List of improvement suggestions to apply</param>
    /// <param name="createBackup">Whether to create a backup of the original file</param>
    /// <returns>The path to the improved file</returns>
    Task<string> ApplyImprovementsAsync(
        string filePath,
        List<ImprovementSuggestion> suggestions,
        bool createBackup = true);

    /// <summary>
    /// Generates a complete implementation for a file based on its interface or requirements
    /// </summary>
    /// <param name="filePath">Path to the file to implement</param>
    /// <param name="projectPath">Path to the project</param>
    /// <param name="requirements">Requirements for the implementation</param>
    /// <returns>The path to the implemented file</returns>
    Task<string> GenerateImplementationAsync(
        string filePath,
        string projectPath,
        string requirements);

    /// <summary>
    /// Runs a self-improvement cycle on a project
    /// </summary>
    /// <param name="projectPath">Path to the project</param>
    /// <param name="maxFiles">Maximum number of files to improve</param>
    /// <param name="createBackups">Whether to create backups of original files</param>
    /// <returns>A summary of the improvements made</returns>
    Task<SelfImprovementSummary> RunSelfImprovementCycleAsync(
        string projectPath,
        int maxFiles = 10,
        bool createBackups = true);
}