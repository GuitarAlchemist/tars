using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Simple implementation of the project analysis service
/// </summary>
public class SimpleProjectAnalysisService : IProjectAnalysisService
{
    private readonly ILogger<SimpleProjectAnalysisService> _logger;
    private readonly string _solutionPath;

    /// <summary>
    /// Initializes a new instance of the <see cref="SimpleProjectAnalysisService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="solutionPath">The path to the solution</param>
    public SimpleProjectAnalysisService(
        ILogger<SimpleProjectAnalysisService> logger,
        string solutionPath = null)
    {
        _logger = logger;
        _solutionPath = solutionPath ?? Directory.GetCurrentDirectory();
    }

    /// <summary>
    /// Get the project structure
    /// </summary>
    /// <returns>The project structure</returns>
    public async Task<ProjectStructure> GetProjectStructureAsync()
    {
        _logger.LogInformation($"Getting project structure for {_solutionPath}");
        
        try
        {
            var projectStructure = new ProjectStructure
            {
                Name = Path.GetFileName(_solutionPath),
                Path = _solutionPath
            };
            
            // Get all directories in the solution
            var directories = Directory.GetDirectories(_solutionPath, "*", SearchOption.AllDirectories);
            
            // Filter out directories that are not part of the source code
            var sourceDirectories = directories
                .Where(d => !d.Contains("bin") && !d.Contains("obj") && !d.Contains(".git") && !d.Contains(".vs"))
                .ToList();
            
            // Add the directories to the project structure
            foreach (var directory in sourceDirectories)
            {
                var projectDirectory = new ProjectDirectory
                {
                    Name = Path.GetFileName(directory),
                    Path = directory
                };
                
                // Get all files in the directory
                var files = Directory.GetFiles(directory);
                
                // Add the files to the directory
                foreach (var file in files)
                {
                    var fileInfo = new FileInfo(file);
                    
                    var projectFile = new ProjectFile
                    {
                        Name = Path.GetFileName(file),
                        Path = file,
                        FileType = Path.GetExtension(file),
                        Size = fileInfo.Length
                    };
                    
                    projectDirectory.Files.Add(projectFile);
                    projectStructure.Files.Add(projectFile);
                }
                
                projectStructure.Directories.Add(projectDirectory);
            }
            
            return projectStructure;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting project structure for {_solutionPath}");
            
            // Return a minimal project structure
            return new ProjectStructure
            {
                Name = Path.GetFileName(_solutionPath),
                Path = _solutionPath
            };
        }
    }

    /// <summary>
    /// Analyzes a project and extracts information about its structure
    /// </summary>
    /// <param name="projectPath">Path to the project file or directory</param>
    /// <returns>A ProjectAnalysisResult containing information about the project</returns>
    public async Task<ProjectAnalysisResult> AnalyzeProjectAsync(string projectPath)
    {
        // For now, return a minimal result
        return new ProjectAnalysisResult
        {
            ProjectName = Path.GetFileName(projectPath),
            ProjectPath = projectPath
        };
    }

    /// <summary>
    /// Analyzes a solution and extracts information about its structure
    /// </summary>
    /// <param name="solutionPath">Path to the solution file</param>
    /// <returns>A SolutionAnalysisResult containing information about the solution</returns>
    public async Task<SolutionAnalysisResult> AnalyzeSolutionAsync(string solutionPath)
    {
        // For now, return a minimal result
        return new SolutionAnalysisResult
        {
            SolutionName = Path.GetFileName(solutionPath),
            SolutionPath = solutionPath
        };
    }
}
