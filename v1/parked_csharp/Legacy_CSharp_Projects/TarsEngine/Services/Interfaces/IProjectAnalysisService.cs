namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the project analysis service
/// </summary>
public interface IProjectAnalysisService
{
    /// <summary>
    /// Analyzes a project and extracts information about its structure
    /// </summary>
    /// <param name="projectPath">Path to the project file or directory</param>
    /// <returns>A ProjectAnalysisResult containing information about the project</returns>
    Task<TarsEngine.Models.ProjectAnalysisResult> AnalyzeProjectAsync(string projectPath);

    /// <summary>
    /// Analyzes a solution and extracts information about its structure
    /// </summary>
    /// <param name="solutionPath">Path to the solution file</param>
    /// <returns>A SolutionAnalysisResult containing information about the solution</returns>
    Task<TarsEngine.Models.SolutionAnalysisResult> AnalyzeSolutionAsync(string solutionPath);
}