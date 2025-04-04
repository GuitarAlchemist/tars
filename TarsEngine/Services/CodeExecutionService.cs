using System.Diagnostics;
using System.Text;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Services;

/// <summary>
/// Service for executing code and running tests
/// </summary>
public class CodeExecutionService
{
    private readonly ILogger<CodeExecutionService> _logger;

    public CodeExecutionService(ILogger<CodeExecutionService> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Runs tests for a project
    /// </summary>
    /// <param name="projectPath">Path to the project</param>
    /// <param name="testFilter">Optional filter for tests to run</param>
    /// <returns>The test execution result</returns>
    public async Task<TestExecutionResult> RunTestsAsync(string projectPath, string testFilter = null)
    {
        try
        {
            _logger.LogInformation($"Running tests for project: {projectPath}");
                
            // Determine if the path is a directory or a file
            bool isDirectory = Directory.Exists(projectPath);
            bool isFile = File.Exists(projectPath);
                
            if (!isDirectory && !isFile)
            {
                _logger.LogError($"Project path not found: {projectPath}");
                return new TestExecutionResult
                {
                    Success = false,
                    ErrorMessage = $"Project path not found: {projectPath}"
                };
            }
                
            // Build the command
            var command = new StringBuilder("dotnet test");
                
            // Add the project path if it's a file
            if (isFile)
            {
                command.Append($" \"{projectPath}\"");
            }
                
            // Add the test filter if provided
            if (!string.IsNullOrEmpty(testFilter))
            {
                command.Append($" --filter \"{testFilter}\"");
            }
                
            // Add verbosity
            command.Append(" --verbosity normal");
                
            // Run the command
            var result = await ExecuteCommandAsync(command.ToString(), isDirectory ? projectPath : Path.GetDirectoryName(projectPath));
                
            // Parse the test results
            return ParseTestResults(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error running tests: {ex.Message}");
            return new TestExecutionResult
            {
                Success = false,
                ErrorMessage = $"Error running tests: {ex.Message}"
            };
        }
    }

    /// <summary>
    /// Builds a project
    /// </summary>
    /// <param name="projectPath">Path to the project</param>
    /// <param name="configuration">Build configuration (Debug/Release)</param>
    /// <returns>The build execution result</returns>
    public async Task<BuildExecutionResult> BuildProjectAsync(string projectPath, string configuration = "Debug")
    {
        try
        {
            _logger.LogInformation($"Building project: {projectPath}");
                
            // Determine if the path is a directory or a file
            bool isDirectory = Directory.Exists(projectPath);
            bool isFile = File.Exists(projectPath);
                
            if (!isDirectory && !isFile)
            {
                _logger.LogError($"Project path not found: {projectPath}");
                return new BuildExecutionResult
                {
                    Success = false,
                    ErrorMessage = $"Project path not found: {projectPath}"
                };
            }
                
            // Build the command
            var command = new StringBuilder("dotnet build");
                
            // Add the project path if it's a file
            if (isFile)
            {
                command.Append($" \"{projectPath}\"");
            }
                
            // Add configuration
            command.Append($" --configuration {configuration}");
                
            // Run the command
            var result = await ExecuteCommandAsync(command.ToString(), isDirectory ? projectPath : Path.GetDirectoryName(projectPath));
                
            // Parse the build results
            return ParseBuildResults(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error building project: {ex.Message}");
            return new BuildExecutionResult
            {
                Success = false,
                ErrorMessage = $"Error building project: {ex.Message}"
            };
        }
    }

    /// <summary>
    /// Runs a specific test method
    /// </summary>
    /// <param name="testProjectPath">Path to the test project</param>
    /// <param name="testName">Name of the test to run</param>
    /// <returns>The test execution result</returns>
    public async Task<TestExecutionResult> RunSpecificTestAsync(string testProjectPath, string testName)
    {
        try
        {
            _logger.LogInformation($"Running test {testName} in project: {testProjectPath}");
                
            // Ensure the test project exists
            if (!Directory.Exists(testProjectPath) && !File.Exists(testProjectPath))
            {
                _logger.LogError($"Test project not found: {testProjectPath}");
                return new TestExecutionResult
                {
                    Success = false,
                    ErrorMessage = $"Test project not found: {testProjectPath}"
                };
            }
                
            // Build the command
            var command = new StringBuilder("dotnet test");
                
            // Add the project path if it's a file
            if (File.Exists(testProjectPath))
            {
                command.Append($" \"{testProjectPath}\"");
            }
                
            // Add the test filter
            command.Append($" --filter \"FullyQualifiedName~{testName}\"");
                
            // Add verbosity
            command.Append(" --verbosity normal");
                
            // Run the command
            var result = await ExecuteCommandAsync(command.ToString(), Directory.Exists(testProjectPath) ? testProjectPath : Path.GetDirectoryName(testProjectPath));
                
            // Parse the test results
            return ParseTestResults(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error running test: {ex.Message}");
            return new TestExecutionResult
            {
                Success = false,
                ErrorMessage = $"Error running test: {ex.Message}"
            };
        }
    }

    /// <summary>
    /// Executes a command and returns the result
    /// </summary>
    private async Task<CommandExecutionResult> ExecuteCommandAsync(string command, string workingDirectory)
    {
        try
        {
            _logger.LogInformation($"Executing command: {command}");
                
            // Create the process start info
            var startInfo = new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = $"/c {command}",
                WorkingDirectory = workingDirectory,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
                
            // Create the process
            using var process = new Process { StartInfo = startInfo };
                
            // Create string builders for output and error
            var outputBuilder = new StringBuilder();
            var errorBuilder = new StringBuilder();
                
            // Set up output and error handlers
            process.OutputDataReceived += (sender, e) =>
            {
                if (e.Data != null)
                {
                    outputBuilder.AppendLine(e.Data);
                }
            };
                
            process.ErrorDataReceived += (sender, e) =>
            {
                if (e.Data != null)
                {
                    errorBuilder.AppendLine(e.Data);
                }
            };
                
            // Start the process
            process.Start();
                
            // Begin reading output and error
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
                
            // Wait for the process to exit
            await process.WaitForExitAsync();
                
            // Get the output and error
            string output = outputBuilder.ToString();
            string error = errorBuilder.ToString();
                
            // Create the result
            var result = new CommandExecutionResult
            {
                ExitCode = process.ExitCode,
                Output = output,
                Error = error,
                Success = process.ExitCode == 0
            };
                
            _logger.LogInformation($"Command executed with exit code: {result.ExitCode}");
                
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing command: {ex.Message}");
            return new CommandExecutionResult
            {
                Success = false,
                Error = ex.Message
            };
        }
    }

    /// <summary>
    /// Parses test results from command output
    /// </summary>
    private TestExecutionResult ParseTestResults(CommandExecutionResult commandResult)
    {
        var result = new TestExecutionResult
        {
            Success = commandResult.Success,
            Output = commandResult.Output,
            Error = commandResult.Error
        };
            
        // Parse the test results
        if (commandResult.Success)
        {
            // Extract test counts
            var totalTestsMatch = System.Text.RegularExpressions.Regex.Match(commandResult.Output, @"Total tests: (\d+)");
            if (totalTestsMatch.Success)
            {
                result.TotalTests = int.Parse(totalTestsMatch.Groups[1].Value);
            }
                
            var passedTestsMatch = System.Text.RegularExpressions.Regex.Match(commandResult.Output, @"Passed: (\d+)");
            if (passedTestsMatch.Success)
            {
                result.PassedTests = int.Parse(passedTestsMatch.Groups[1].Value);
            }
                
            var failedTestsMatch = System.Text.RegularExpressions.Regex.Match(commandResult.Output, @"Failed: (\d+)");
            if (failedTestsMatch.Success)
            {
                result.FailedTests = int.Parse(failedTestsMatch.Groups[1].Value);
            }
                
            var skippedTestsMatch = System.Text.RegularExpressions.Regex.Match(commandResult.Output, @"Skipped: (\d+)");
            if (skippedTestsMatch.Success)
            {
                result.SkippedTests = int.Parse(skippedTestsMatch.Groups[1].Value);
            }
                
            // Extract test duration
            var durationMatch = System.Text.RegularExpressions.Regex.Match(commandResult.Output, @"Time: (.+)");
            if (durationMatch.Success)
            {
                result.Duration = durationMatch.Groups[1].Value;
            }
        }
        else
        {
            result.ErrorMessage = commandResult.Error;
        }
            
        return result;
    }

    /// <summary>
    /// Parses build results from command output
    /// </summary>
    private BuildExecutionResult ParseBuildResults(CommandExecutionResult commandResult)
    {
        var result = new BuildExecutionResult
        {
            Success = commandResult.Success,
            Output = commandResult.Output,
            Error = commandResult.Error
        };
            
        // Parse the build results
        if (commandResult.Success)
        {
            // Extract warning count
            var warningCountMatch = System.Text.RegularExpressions.Regex.Match(commandResult.Output, @"(\d+) Warning\(s\)");
            if (warningCountMatch.Success)
            {
                result.WarningCount = int.Parse(warningCountMatch.Groups[1].Value);
            }
                
            // Extract error count
            var errorCountMatch = System.Text.RegularExpressions.Regex.Match(commandResult.Output, @"(\d+) Error\(s\)");
            if (errorCountMatch.Success)
            {
                result.ErrorCount = int.Parse(errorCountMatch.Groups[1].Value);
            }
        }
        else
        {
            result.ErrorMessage = commandResult.Error;
                
            // Extract error count
            var errorCountMatch = System.Text.RegularExpressions.Regex.Match(commandResult.Output, @"(\d+) Error\(s\)");
            if (errorCountMatch.Success)
            {
                result.ErrorCount = int.Parse(errorCountMatch.Groups[1].Value);
            }
        }
            
        return result;
    }
}

/// <summary>
/// Represents the result of a command execution
/// </summary>
public class CommandExecutionResult
{
    public bool Success { get; set; }
    public int ExitCode { get; set; }
    public string Output { get; set; }
    public string Error { get; set; }
}

/// <summary>
/// Represents the result of a test execution
/// </summary>
public class TestExecutionResult
{
    public bool Success { get; set; }
    public string ErrorMessage { get; set; }
    public string Output { get; set; }
    public string Error { get; set; }
    public int TotalTests { get; set; }
    public int PassedTests { get; set; }
    public int FailedTests { get; set; }
    public int SkippedTests { get; set; }
    public string Duration { get; set; }
}

/// <summary>
/// Represents the result of a build execution
/// </summary>
public class BuildExecutionResult
{
    public bool Success { get; set; }
    public string ErrorMessage { get; set; }
    public string Output { get; set; }
    public string Error { get; set; }
    public int WarningCount { get; set; }
    public int ErrorCount { get; set; }
}