using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System.Diagnostics;
using System.Text;

namespace TarsCli.Services;

/// <summary>
/// Service for validating compilation of code files
/// </summary>
public class CompilationService
{
    private readonly ILogger<CompilationService> _logger;
    private readonly string _solutionPath;

    public CompilationService(
        ILogger<CompilationService> logger,
        IConfiguration configuration)
    {
        _logger = logger;
        _solutionPath = configuration["Tars:SolutionPath"] ?? "Tars.sln";
    }

    /// <summary>
    /// Validate that a file compiles successfully
    /// </summary>
    /// <param name="filePath">Path to the file to validate</param>
    /// <returns>Compilation result</returns>
    public async Task<CompilationResult> ValidateCompilationAsync(string filePath)
    {
        _logger.LogInformation($"Validating compilation of: {Path.GetFullPath(filePath)}");

        try
        {
            // Determine the project path based on the file path
            var projectPath = GetProjectPath(filePath);
            if (string.IsNullOrEmpty(projectPath))
            {
                return new CompilationResult
                {
                    Success = false,
                    ErrorMessage = $"Could not determine project path for {filePath}"
                };
            }

            // Run dotnet build on the project
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = $"build {projectPath} --no-dependencies",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            };

            var outputBuilder = new StringBuilder();
            var errorBuilder = new StringBuilder();

            process.OutputDataReceived += (sender, args) =>
            {
                if (args.Data != null)
                {
                    outputBuilder.AppendLine(args.Data);
                }
            };

            process.ErrorDataReceived += (sender, args) =>
            {
                if (args.Data != null)
                {
                    errorBuilder.AppendLine(args.Data);
                }
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            await process.WaitForExitAsync();

            var output = outputBuilder.ToString();
            var error = errorBuilder.ToString();

            if (process.ExitCode != 0)
            {
                _logger.LogWarning($"Compilation failed: {error}");
                return new CompilationResult
                {
                    Success = false,
                    ErrorMessage = error,
                    Output = output
                };
            }

            _logger.LogInformation("Compilation succeeded");
            return new CompilationResult
            {
                Success = true,
                Output = output
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error validating compilation of {filePath}");
            return new CompilationResult
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    /// <summary>
    /// Get the project path for a file
    /// </summary>
    private string GetProjectPath(string filePath)
    {
        try
        {
            var directory = Path.GetDirectoryName(filePath);
            while (!string.IsNullOrEmpty(directory))
            {
                var projectFiles = Directory.GetFiles(directory, "*.csproj");
                if (projectFiles.Length > 0)
                {
                    return projectFiles[0];
                }

                directory = Path.GetDirectoryName(directory);
            }

            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting project path for {filePath}");
            return null;
        }
    }
}

/// <summary>
/// Result of a compilation validation
/// </summary>
public class CompilationResult
{
    /// <summary>
    /// Whether the compilation succeeded
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if compilation failed
    /// </summary>
    public string ErrorMessage { get; set; }

    /// <summary>
    /// Compilation output
    /// </summary>
    public string Output { get; set; }
}
