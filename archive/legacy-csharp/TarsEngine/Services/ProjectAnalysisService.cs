using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for analyzing project structure and dependencies
/// </summary>
public class ProjectAnalysisService(
    ILogger<ProjectAnalysisService> logger,
    ICodeAnalysisService codeAnalysisService)
    : IProjectAnalysisService
{
    /// <summary>
    /// Analyzes a project and extracts information about its structure
    /// </summary>
    /// <param name="projectPath">Path to the project file or directory</param>
    /// <returns>A ProjectAnalysisResult containing information about the project</returns>
    public virtual async Task<TarsEngine.Models.ProjectAnalysisResult> AnalyzeProjectAsync(string projectPath)
    {
        try
        {
            logger.LogInformation($"Analyzing project: {projectPath}");

            var result = new TarsEngine.Models.ProjectAnalysisResult
            {
                ProjectPath = projectPath
            };

            // Determine if the path is a file or directory
            var isDirectory = Directory.Exists(projectPath);
            var isFile = File.Exists(projectPath);

            if (!isDirectory && !isFile)
            {
                logger.LogError($"Project path not found: {projectPath}");
                var errorResult = new TarsEngine.Models.ProjectAnalysisResult
                {
                    ProjectPath = projectPath,
                    ProjectName = Path.GetFileName(projectPath)
                };

                // Add an issue to indicate the error
                errorResult.Issues.Add(new TarsEngine.Models.CodeIssue
                {
                    Title = "Project Not Found",
                    Description = $"Project path not found: {projectPath}",
                    Severity = TarsEngine.Models.IssueSeverity.Error
                });

                return errorResult;
            }

            // If it's a file, determine the project type
            if (isFile)
            {
                var extension = Path.GetExtension(projectPath).ToLowerInvariant();
                result.ProjectType = DetermineProjectType(extension).ToString();
                result.ProjectName = Path.GetFileNameWithoutExtension(projectPath);

                // Parse the project file
                await ParseProjectFileAsync(projectPath, result);

                // Set the directory to the parent directory of the project file
                projectPath = Path.GetDirectoryName(projectPath);
            }
            else
            {
                // Try to find project files in the directory
                var projectFiles = Directory.GetFiles(projectPath, "*.*proj", SearchOption.TopDirectoryOnly);
                if (projectFiles.Length > 0)
                {
                    // Use the first project file found
                    var projectFile = projectFiles[0];
                    var extension = Path.GetExtension(projectFile).ToLowerInvariant();
                    result.ProjectType = DetermineProjectType(extension).ToString();
                    result.ProjectName = Path.GetFileNameWithoutExtension(projectFile);

                    // Parse the project file
                    await ParseProjectFileAsync(projectFile, result);
                }
                else
                {
                    // No project file found, try to determine project type from directory structure
                    result.ProjectType = DetermineProjectTypeFromDirectory(projectPath).ToString();
                    result.ProjectName = new DirectoryInfo(projectPath).Name;
                }
            }

            // Analyze the code files in the project
            if (string.IsNullOrEmpty(projectPath))
            {
                throw new ArgumentNullException(nameof(projectPath), "Project path cannot be null or empty");
            }

            var codeAnalysisResults = await codeAnalysisService.AnalyzeDirectoryAsync(projectPath);

            // Extract file information
            result.FileCount = codeAnalysisResults.Count;
            result.Files = codeAnalysisResults.Select(r => r.FilePath).ToList();

            // Count total lines
            result.TotalLines = codeAnalysisResults.Count;

            // Extract issues
            foreach (var codeResult in codeAnalysisResults.Where(r => r.Success))
            {
                result.Issues.AddRange(codeResult.Issues);
            }

            // Analyze project dependencies
            await AnalyzeProjectDependenciesAsync(projectPath, result);

            return result;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error analyzing project {projectPath}");
            var errorResult = new TarsEngine.Models.ProjectAnalysisResult
            {
                ProjectPath = projectPath,
                ProjectName = Path.GetFileName(projectPath)
            };

            // Add an issue to indicate the error
            errorResult.Issues.Add(new TarsEngine.Models.CodeIssue
            {
                Title = "Analysis Error",
                Description = $"Error analyzing project: {ex.Message}",
                Severity = TarsEngine.Models.IssueSeverity.Error
            });

            return errorResult;
        }
    }

    /// <summary>
    /// Analyzes a solution and extracts information about its structure
    /// </summary>
    /// <param name="solutionPath">Path to the solution file</param>
    /// <returns>A SolutionAnalysisResult containing information about the solution</returns>
    public virtual async Task<TarsEngine.Models.SolutionAnalysisResult> AnalyzeSolutionAsync(string solutionPath)
    {
        try
        {
            logger.LogInformation($"Analyzing solution: {solutionPath}");

            if (!File.Exists(solutionPath))
            {
                logger.LogError($"Solution file not found: {solutionPath}");
                var errorResult = new TarsEngine.Models.SolutionAnalysisResult
                {
                    SolutionPath = solutionPath,
                    SolutionName = Path.GetFileName(solutionPath)
                };

                return errorResult;
            }

            var result = new TarsEngine.Models.SolutionAnalysisResult
            {
                SolutionPath = solutionPath,
                SolutionName = Path.GetFileNameWithoutExtension(solutionPath)
            };

            // Parse the solution file to extract project references
            var projectPaths = ParseSolutionFile(solutionPath);

            // Analyze each project
            foreach (var projectPath in projectPaths)
            {
                var projectResult = await AnalyzeProjectAsync(projectPath);
                result.Projects.Add(projectResult);
            }

            return result;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error analyzing solution {solutionPath}");
            var errorResult = new TarsEngine.Models.SolutionAnalysisResult
            {
                SolutionPath = solutionPath,
                SolutionName = Path.GetFileName(solutionPath)
            };

            return errorResult;
        }
    }

    /// <summary>
    /// Determines the project type based on the file extension
    /// </summary>
    private static ProjectType DetermineProjectType(string extension) => extension switch
    {
        ".csproj" => ProjectType.CSharp,
        ".fsproj" => ProjectType.FSharp,
        ".vbproj" => ProjectType.VisualBasic,
        ".vcxproj" => ProjectType.Cpp,
        ".pyproj" => ProjectType.Python,
        ".njsproj" => ProjectType.Node,
        ".wapproj" => ProjectType.WindowsApp,
        _ => ProjectType.Unknown
    };

    /// <summary>
    /// Determines the project type based on the directory structure
    /// </summary>
    private static ProjectType DetermineProjectTypeFromDirectory(string directoryPath)
    {
        // Check for package.json (Node.js)
        if (File.Exists(Path.Combine(directoryPath, "package.json")))
        {
            return ProjectType.Node;
        }

        // Check for requirements.txt or setup.py (Python)
        if (File.Exists(Path.Combine(directoryPath, "requirements.txt")) ||
            File.Exists(Path.Combine(directoryPath, "setup.py")))
        {
            return ProjectType.Python;
        }

        // Check for pom.xml (Java/Maven)
        if (File.Exists(Path.Combine(directoryPath, "pom.xml")))
        {
            return ProjectType.Java;
        }

        // Check for build.gradle (Java/Gradle)
        if (File.Exists(Path.Combine(directoryPath, "build.gradle")))
        {
            return ProjectType.Java;
        }

        // Check for CMakeLists.txt (C/C++)
        if (File.Exists(Path.Combine(directoryPath, "CMakeLists.txt")))
        {
            return ProjectType.Cpp;
        }

        // Check for Makefile (C/C++)
        if (File.Exists(Path.Combine(directoryPath, "Makefile")))
        {
            return ProjectType.Cpp;
        }

        // Check file extensions in the directory
        var files = Directory.GetFiles(directoryPath, "*.*", SearchOption.TopDirectoryOnly);

        var csharpCount = files.Count(f => Path.GetExtension(f).ToLowerInvariant() == ".cs");
        var fsharpCount = files.Count(f => Path.GetExtension(f).ToLowerInvariant() == ".fs");
        var vbCount = files.Count(f => Path.GetExtension(f).ToLowerInvariant() == ".vb");
        var cppCount = files.Count(f =>
            new[] { ".cpp", ".h", ".hpp", ".c" }.Contains(Path.GetExtension(f).ToLowerInvariant()));
        var pythonCount = files.Count(f => Path.GetExtension(f).ToLowerInvariant() == ".py");
        var jsCount = files.Count(f =>
            new[] { ".js", ".ts", ".jsx", ".tsx" }.Contains(Path.GetExtension(f).ToLowerInvariant()));

        // Return the project type with the most files
        var counts = new Dictionary<ProjectType, int>
        {
            { ProjectType.CSharp, csharpCount },
            { ProjectType.FSharp, fsharpCount },
            { ProjectType.VisualBasic, vbCount },
            { ProjectType.Cpp, cppCount },
            { ProjectType.Python, pythonCount },
            { ProjectType.Node, jsCount }
        };

        var maxCount = counts.Max(kvp => kvp.Value);
        if (maxCount > 0)
        {
            return counts.First(kvp => kvp.Value == maxCount).Key;
        }

        return ProjectType.Unknown;
    }

    /// <summary>
    /// Parses a project file to extract information
    /// </summary>
    private async Task ParseProjectFileAsync(string projectFilePath, TarsEngine.Models.ProjectAnalysisResult result)
    {
        try
        {
            var content = await File.ReadAllTextAsync(projectFilePath);

            // Extract package references
            var packageReferenceRegex = new Regex("<PackageReference\\s+Include=\"([^\"]+)\"\\s+Version=\"([^\"]+)\"");
            var packageReferenceMatches = packageReferenceRegex.Matches(content);
            foreach (Match match in packageReferenceMatches)
            {
                result.Packages.Add($"{match.Groups[1].Value} {match.Groups[2].Value}");
            }

            // Extract project references
            var projectReferenceRegex = new Regex("<ProjectReference\\s+Include=\"([^\"]+)\"");
            var projectReferenceMatches = projectReferenceRegex.Matches(content);
            foreach (Match match in projectReferenceMatches)
            {
                result.References.Add(match.Groups[1].Value);
            }

            // Add dependencies to the result
            result.Dependencies = result.Packages.Concat(result.References).ToList();
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error parsing project file {projectFilePath}");
        }
    }

    /// <summary>
    /// Parses a solution file to extract project references
    /// </summary>
    private List<string> ParseSolutionFile(string solutionFilePath)
    {
        var projectPaths = new List<string>();

        try
        {
            var content = File.ReadAllText(solutionFilePath);
            var solutionDir = Path.GetDirectoryName(solutionFilePath);
            
            if (solutionDir == null)
            {
                logger.LogError("Unable to determine solution directory for path: {SolutionFilePath}", solutionFilePath);
                return projectPaths;
            }

            // Extract project paths
            var projectRegex = new Regex(@"Project\(""\{[^}]+\}""\)\s*=\s*""[^""]*"",\s*""([^""]+)""");
            var projectMatches = projectRegex.Matches(content);

            foreach (Match match in projectMatches)
            {
                var relativePath = match.Groups[1].Value;
                // Convert relative path to absolute path
                var absolutePath = Path.GetFullPath(Path.Combine(solutionDir, relativePath));
                if (File.Exists(absolutePath))
                {
                    projectPaths.Add(absolutePath);
                }
            }
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Error parsing solution file {SolutionFilePath}", solutionFilePath);
        }

        return projectPaths;
    }

    /// <summary>
    /// Analyzes project dependencies
    /// </summary>
    private async Task AnalyzeProjectDependenciesAsync(string projectPath,
        TarsEngine.Models.ProjectAnalysisResult result)
    {
        try
        {
            // Extract package references from project file
            var projectFiles = Directory.GetFiles(projectPath, "*.csproj", SearchOption.TopDirectoryOnly);
            projectFiles = projectFiles.Concat(Directory.GetFiles(projectPath, "*.fsproj", SearchOption.TopDirectoryOnly)).ToArray();

            if (projectFiles.Length > 0)
            {
                var projectFile = projectFiles[0];
                var content = await File.ReadAllTextAsync(projectFile);

                // Extract package references
                var packageReferenceRegex = new Regex("<PackageReference\\s+Include=\"([^\"]+)\"\\s+Version=\"([^\"]+)\"");
                var packageReferenceMatches = packageReferenceRegex.Matches(content);

                foreach (Match match in packageReferenceMatches)
                {
                    result.Packages.Add($"{match.Groups[1].Value} {match.Groups[2].Value}");
                }

                // Extract project references
                var projectReferenceRegex = new Regex("<ProjectReference\\s+Include=\"([^\"]+)\"");
                var projectReferenceMatches = projectReferenceRegex.Matches(content);

                foreach (Match match in projectReferenceMatches)
                {
                    result.References.Add(match.Groups[1].Value);
                }
            }

            // Add dependencies to the result
            result.Dependencies = result.Packages.Concat(result.References).ToList();
        }
        catch (Exception ex)
        {
            logger.LogError(ex, $"Error analyzing project dependencies {projectPath}");
        }
    }
}

/// <summary>
/// Represents a package reference
/// </summary>
public class PackageReference
{
    public string Name { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
}

/// <summary>
/// Represents a project type
/// </summary>
public enum ProjectType
{
    Unknown,
    CSharp,
    FSharp,
    VisualBasic,
    Cpp,
    Python,
    Node,
    Java,
    WindowsApp
}
