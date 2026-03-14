using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;

namespace TarsCli.Services;

/// <summary>
/// Service for running tests related to code files
/// </summary>
public class TestRunnerService
{
    private readonly ILogger<TestRunnerService> _logger;
    private readonly CompilationService _compilationService;

    public TestRunnerService(
        ILogger<TestRunnerService> logger,
        CompilationService compilationService)
    {
        _logger = logger;
        _compilationService = compilationService;
    }

    /// <summary>
    /// Run tests for a specific file
    /// </summary>
    /// <param name="filePath">Path to the file to test</param>
    /// <returns>Test result</returns>
    public async Task<TestResult> RunTestsForFileAsync(string filePath)
    {
        _logger.LogInformation($"Running tests for: {Path.GetFullPath(filePath)}");

        try
        {
            // First, ensure the file compiles
            var compilationResult = await _compilationService.ValidateCompilationAsync(filePath);
            if (!compilationResult.Success)
            {
                return new TestResult
                {
                    Success = false,
                    ErrorMessage = $"Compilation failed: {compilationResult.ErrorMessage}"
                };
            }

            // Determine the test project path based on the file path
            var testProjectPath = GetTestProjectPath(filePath);
            if (string.IsNullOrEmpty(testProjectPath))
            {
                _logger.LogWarning($"No test project found for {filePath}");
                return new TestResult
                {
                    Success = true,
                    TestsRun = 0,
                    Message = "No tests found"
                };
            }

            // Determine the class name from the file path
            var className = Path.GetFileNameWithoutExtension(filePath);

            // Run dotnet test on the test project, filtering by the class name
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = $"test {testProjectPath} --filter FullyQualifiedName~{className}",
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

            // Parse the test results
            var testsRun = ParseTestsRun(output);
            var testsPassed = ParseTestsPassed(output);
            var testsFailed = ParseTestsFailed(output);

            if (process.ExitCode != 0 || testsFailed > 0)
            {
                _logger.LogWarning($"Tests failed: {testsFailed} of {testsRun} tests failed");
                return new TestResult
                {
                    Success = false,
                    TestsRun = testsRun,
                    TestsPassed = testsPassed,
                    TestsFailed = testsFailed,
                    ErrorMessage = error,
                    Output = output
                };
            }

            _logger.LogInformation($"Tests passed: {testsPassed} of {testsRun} tests passed");
            return new TestResult
            {
                Success = true,
                TestsRun = testsRun,
                TestsPassed = testsPassed,
                TestsFailed = testsFailed,
                Output = output
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error running tests for {filePath}");
            return new TestResult
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    /// <summary>
    /// Get the test project path for a file
    /// </summary>
    private string GetTestProjectPath(string filePath)
    {
        try
        {
            // Extract the project name from the file path
            var projectName = GetProjectName(filePath);
            if (string.IsNullOrEmpty(projectName))
            {
                return null;
            }

            // Look for a test project with a similar name
            var solutionDirectory = Path.GetDirectoryName(Path.GetFullPath("Tars.sln"));
            var testProjectCandidates = new[]
            {
                $"{projectName}.Tests",
                $"{projectName}.Test",
                $"{projectName}Tests",
                $"{projectName}Test"
            };

            foreach (var candidate in testProjectCandidates)
            {
                var testProjectPath = Path.Combine(solutionDirectory, candidate, $"{candidate}.csproj");
                if (File.Exists(testProjectPath))
                {
                    return testProjectPath;
                }
            }

            // If no specific test project is found, look for any test project
            var testProjects = Directory.GetFiles(solutionDirectory, "*.Tests.csproj", SearchOption.AllDirectories);
            if (testProjects.Length > 0)
            {
                return testProjects[0];
            }

            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting test project path for {filePath}");
            return null;
        }
    }

    /// <summary>
    /// Get the project name from a file path
    /// </summary>
    private string GetProjectName(string filePath)
    {
        try
        {
            var directory = Path.GetDirectoryName(filePath);
            while (!string.IsNullOrEmpty(directory))
            {
                var projectFiles = Directory.GetFiles(directory, "*.csproj");
                if (projectFiles.Length > 0)
                {
                    return Path.GetFileNameWithoutExtension(projectFiles[0]);
                }

                directory = Path.GetDirectoryName(directory);
            }

            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting project name for {filePath}");
            return null;
        }
    }

    /// <summary>
    /// Parse the number of tests run from the output
    /// </summary>
    private int ParseTestsRun(string output)
    {
        var match = Regex.Match(output, @"Total tests: (\d+)");
        if (match.Success && int.TryParse(match.Groups[1].Value, out var testsRun))
        {
            return testsRun;
        }

        return 0;
    }

    /// <summary>
    /// Parse the number of tests passed from the output
    /// </summary>
    private int ParseTestsPassed(string output)
    {
        var match = Regex.Match(output, @"Passed: (\d+)");
        if (match.Success && int.TryParse(match.Groups[1].Value, out var testsPassed))
        {
            return testsPassed;
        }

        return 0;
    }

    /// <summary>
    /// Parse the number of tests failed from the output
    /// </summary>
    private int ParseTestsFailed(string output)
    {
        var match = Regex.Match(output, @"Failed: (\d+)");
        if (match.Success && int.TryParse(match.Groups[1].Value, out var testsFailed))
        {
            return testsFailed;
        }

        return 0;
    }
}

/// <summary>
/// Result of a test run
/// </summary>
public class TestResult
{
    /// <summary>
    /// Whether the tests succeeded
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Number of tests run
    /// </summary>
    public int TestsRun { get; set; }

    /// <summary>
    /// Number of tests passed
    /// </summary>
    public int TestsPassed { get; set; }

    /// <summary>
    /// Number of tests failed
    /// </summary>
    public int TestsFailed { get; set; }

    /// <summary>
    /// Error message if tests failed
    /// </summary>
    public string ErrorMessage { get; set; }

    /// <summary>
    /// Test output
    /// </summary>
    public string Output { get; set; }

    /// <summary>
    /// Additional message
    /// </summary>
    public string Message { get; set; }
}
