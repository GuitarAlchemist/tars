using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services
{
    /// <summary>
    /// Service for analyzing project structure and dependencies
    /// </summary>
    public class ProjectAnalysisService : IProjectAnalysisService
    {
        private readonly ILogger<ProjectAnalysisService> _logger;
        private readonly ICodeAnalysisService _codeAnalysisService;

        public ProjectAnalysisService(
            ILogger<ProjectAnalysisService> logger,
            ICodeAnalysisService codeAnalysisService)
        {
            _logger = logger;
            _codeAnalysisService = codeAnalysisService;
        }

        /// <summary>
        /// Analyzes a project and extracts information about its structure
        /// </summary>
        /// <param name="projectPath">Path to the project file or directory</param>
        /// <returns>A ProjectAnalysisResult containing information about the project</returns>
        public virtual async Task<ProjectAnalysisResult> AnalyzeProjectAsync(string projectPath)
        {
            try
            {
                _logger.LogInformation($"Analyzing project: {projectPath}");

                var result = new ProjectAnalysisResult
                {
                    ProjectPath = projectPath,
                    Success = true
                };

                // Determine if the path is a file or directory
                bool isDirectory = Directory.Exists(projectPath);
                bool isFile = File.Exists(projectPath);

                if (!isDirectory && !isFile)
                {
                    _logger.LogError($"Project path not found: {projectPath}");
                    return new ProjectAnalysisResult
                    {
                        Success = false,
                        ErrorMessage = $"Project path not found: {projectPath}"
                    };
                }

                // If it's a file, determine the project type
                if (isFile)
                {
                    string extension = Path.GetExtension(projectPath).ToLowerInvariant();
                    result.ProjectType = DetermineProjectType(extension);
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
                        string projectFile = projectFiles[0];
                        string extension = Path.GetExtension(projectFile).ToLowerInvariant();
                        result.ProjectType = DetermineProjectType(extension);
                        result.ProjectName = Path.GetFileNameWithoutExtension(projectFile);

                        // Parse the project file
                        await ParseProjectFileAsync(projectFile, result);
                    }
                    else
                    {
                        // No project file found, try to determine project type from directory structure
                        result.ProjectType = DetermineProjectTypeFromDirectory(projectPath);
                        result.ProjectName = new DirectoryInfo(projectPath).Name;
                    }
                }

                // Analyze the code files in the project
                var codeAnalysisResults = await _codeAnalysisService.AnalyzeDirectoryAsync(projectPath);
                result.CodeAnalysisResults = codeAnalysisResults;

                // Extract namespaces, classes, etc. from code analysis results
                foreach (var codeResult in codeAnalysisResults.Where(r => r.Success))
                {
                    result.Namespaces.AddRange(codeResult.Namespaces);
                    result.Classes.AddRange(codeResult.Classes);
                    result.Interfaces.AddRange(codeResult.Interfaces);
                }

                // Remove duplicates
                result.Namespaces = result.Namespaces.Distinct().ToList();
                result.Classes = result.Classes.Distinct().ToList();
                result.Interfaces = result.Interfaces.Distinct().ToList();

                // Analyze project dependencies
                await AnalyzeProjectDependenciesAsync(projectPath, result);

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error analyzing project {projectPath}");
                return new ProjectAnalysisResult
                {
                    Success = false,
                    ErrorMessage = $"Error analyzing project: {ex.Message}"
                };
            }
        }

        /// <summary>
        /// Analyzes a solution and extracts information about its structure
        /// </summary>
        /// <param name="solutionPath">Path to the solution file</param>
        /// <returns>A SolutionAnalysisResult containing information about the solution</returns>
        public virtual async Task<SolutionAnalysisResult> AnalyzeSolutionAsync(string solutionPath)
        {
            try
            {
                _logger.LogInformation($"Analyzing solution: {solutionPath}");

                if (!File.Exists(solutionPath))
                {
                    _logger.LogError($"Solution file not found: {solutionPath}");
                    return new SolutionAnalysisResult
                    {
                        Success = false,
                        ErrorMessage = $"Solution file not found: {solutionPath}"
                    };
                }

                var result = new SolutionAnalysisResult
                {
                    SolutionPath = solutionPath,
                    SolutionName = Path.GetFileNameWithoutExtension(solutionPath),
                    Success = true
                };

                // Parse the solution file to extract project references
                var projectPaths = ParseSolutionFile(solutionPath);

                // Analyze each project
                foreach (var projectPath in projectPaths)
                {
                    var projectResult = await AnalyzeProjectAsync(projectPath);
                    result.ProjectResults.Add(projectResult);
                }

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error analyzing solution {solutionPath}");
                return new SolutionAnalysisResult
                {
                    Success = false,
                    ErrorMessage = $"Error analyzing solution: {ex.Message}"
                };
            }
        }

        /// <summary>
        /// Determines the project type based on the file extension
        /// </summary>
        private ProjectType DetermineProjectType(string extension)
        {
            return extension switch
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
        }

        /// <summary>
        /// Determines the project type based on the directory structure
        /// </summary>
        private ProjectType DetermineProjectTypeFromDirectory(string directoryPath)
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

            int csharpCount = files.Count(f => Path.GetExtension(f).ToLowerInvariant() == ".cs");
            int fsharpCount = files.Count(f => Path.GetExtension(f).ToLowerInvariant() == ".fs");
            int vbCount = files.Count(f => Path.GetExtension(f).ToLowerInvariant() == ".vb");
            int cppCount = files.Count(f => new[] { ".cpp", ".h", ".hpp", ".c" }.Contains(Path.GetExtension(f).ToLowerInvariant()));
            int pythonCount = files.Count(f => Path.GetExtension(f).ToLowerInvariant() == ".py");
            int jsCount = files.Count(f => new[] { ".js", ".ts", ".jsx", ".tsx" }.Contains(Path.GetExtension(f).ToLowerInvariant()));

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
        private async Task ParseProjectFileAsync(string projectFilePath, ProjectAnalysisResult result)
        {
            try
            {
                string content = await File.ReadAllTextAsync(projectFilePath);

                // Extract target framework
                var targetFrameworkRegex = new Regex(@"<TargetFramework>(.*?)</TargetFramework>");
                var targetFrameworkMatch = targetFrameworkRegex.Match(content);
                if (targetFrameworkMatch.Success)
                {
                    result.TargetFramework = targetFrameworkMatch.Groups[1].Value;
                }

                // Extract package references
                var packageReferenceRegex = new Regex(@"<PackageReference\s+Include=""([^""]+)""\s+Version=""([^""]+)""");
                var packageReferenceMatches = packageReferenceRegex.Matches(content);
                foreach (Match match in packageReferenceMatches)
                {
                    result.PackageReferences.Add(new PackageReference
                    {
                        Name = match.Groups[1].Value,
                        Version = match.Groups[2].Value
                    });
                }

                // Extract project references
                var projectReferenceRegex = new Regex(@"<ProjectReference\s+Include=""([^""]+)""");
                var projectReferenceMatches = projectReferenceRegex.Matches(content);
                foreach (Match match in projectReferenceMatches)
                {
                    string referencePath = match.Groups[1].Value;
                    // Convert relative path to absolute path
                    string absolutePath = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(projectFilePath), referencePath));
                    result.ProjectReferences.Add(absolutePath);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error parsing project file {projectFilePath}");
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
                string content = File.ReadAllText(solutionFilePath);
                string solutionDir = Path.GetDirectoryName(solutionFilePath);

                // Extract project paths
                var projectRegex = new Regex(@"Project\(""\{[^}]+\}""\)\s*=\s*""[^""]*"",\s*""([^""]+)""");
                var projectMatches = projectRegex.Matches(content);

                foreach (Match match in projectMatches)
                {
                    string relativePath = match.Groups[1].Value;
                    // Convert relative path to absolute path
                    string absolutePath = Path.GetFullPath(Path.Combine(solutionDir, relativePath));
                    if (File.Exists(absolutePath))
                    {
                        projectPaths.Add(absolutePath);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error parsing solution file {solutionFilePath}");
            }

            return projectPaths;
        }

        /// <summary>
        /// Analyzes project dependencies
        /// </summary>
        private async Task AnalyzeProjectDependenciesAsync(string projectPath, ProjectAnalysisResult result)
        {
            try
            {
                // Analyze dependencies between classes
                var dependencies = new Dictionary<string, List<string>>();

                foreach (var codeResult in result.CodeAnalysisResults.Where(r => r.Success))
                {
                    foreach (var className in codeResult.Classes)
                    {
                        if (!dependencies.ContainsKey(className))
                        {
                            dependencies[className] = new List<string>();
                        }

                        // Add dependencies for this class
                        dependencies[className].AddRange(codeResult.Dependencies);
                    }
                }

                // Remove self-references and duplicates
                foreach (var className in dependencies.Keys.ToList())
                {
                    dependencies[className] = dependencies[className]
                        .Where(d => d != className)
                        .Distinct()
                        .ToList();
                }

                result.ClassDependencies = dependencies;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error analyzing project dependencies {projectPath}");
            }
        }
    }

    /// <summary>
    /// Represents the result of a project analysis
    /// </summary>
    public class ProjectAnalysisResult
    {
        public string ProjectPath { get; set; }
        public string ProjectName { get; set; }
        public ProjectType ProjectType { get; set; }
        public string TargetFramework { get; set; }
        public bool Success { get; set; }
        public string ErrorMessage { get; set; }

        public List<string> Namespaces { get; set; } = new List<string>();
        public List<string> Classes { get; set; } = new List<string>();
        public List<string> Interfaces { get; set; } = new List<string>();

        public List<PackageReference> PackageReferences { get; set; } = new List<PackageReference>();
        public List<string> ProjectReferences { get; set; } = new List<string>();
        public List<CodeAnalysisResult> CodeAnalysisResults { get; set; } = new List<CodeAnalysisResult>();
        public Dictionary<string, List<string>> ClassDependencies { get; set; } = new Dictionary<string, List<string>>();
    }

    /// <summary>
    /// Represents the result of a solution analysis
    /// </summary>
    public class SolutionAnalysisResult
    {
        public string SolutionPath { get; set; }
        public string SolutionName { get; set; }
        public bool Success { get; set; }
        public string ErrorMessage { get; set; }

        public List<ProjectAnalysisResult> ProjectResults { get; set; } = new List<ProjectAnalysisResult>();
    }

    /// <summary>
    /// Represents a package reference
    /// </summary>
    public class PackageReference
    {
        public string Name { get; set; }
        public string Version { get; set; }
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
}
