using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;

namespace TarsCli.Services.Testing;

/// <summary>
/// Service for running tests
/// </summary>
public class TestRunnerService
{
    private readonly ILogger<TestRunnerService> _logger;

    /// <summary>
    /// Initializes a new instance of the TestRunnerService class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    public TestRunnerService(ILogger<TestRunnerService> logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <summary>
    /// Runs tests for a project
    /// </summary>
    /// <param name="projectPath">Path to the project file</param>
    /// <returns>Test run result</returns>
    public async Task<TestRunResult> RunTestsAsync(string projectPath)
    {
        _logger.LogInformation($"Running tests for project: {projectPath}");

        var result = new TestRunResult
        {
            ProjectPath = projectPath,
            Success = false
        };

        try
        {
            // Check if the project exists
            if (!File.Exists(projectPath))
            {
                _logger.LogError($"Project file not found: {projectPath}");
                result.ErrorMessage = $"Project file not found: {projectPath}";
                return result;
            }

            // Run the tests
            var (success, output, error) = await RunDotNetTestAsync(projectPath);

            // Parse the test results
            result.Success = success;
            result.Output = output;
            result.ErrorOutput = error;
            result.TestResults = ParseTestResults(output);

            _logger.LogInformation($"Test run completed for {projectPath}. Success: {result.Success}, Total tests: {result.TestResults.Count}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error running tests for {projectPath}");
            result.ErrorMessage = ex.Message;
        }

        return result;
    }

    /// <summary>
    /// Runs tests for a specific test file
    /// </summary>
    /// <param name="testFilePath">Path to the test file</param>
    /// <returns>Test run result</returns>
    public async Task<TestRunResult> RunTestFileAsync(string testFilePath)
    {
        _logger.LogInformation($"Running tests in file: {testFilePath}");

        var result = new TestRunResult
        {
            ProjectPath = testFilePath,
            Success = false
        };

        try
        {
            // Check if the test file exists
            if (!File.Exists(testFilePath))
            {
                _logger.LogError($"Test file not found: {testFilePath}");
                result.ErrorMessage = $"Test file not found: {testFilePath}";
                return result;
            }

            // Find the project file
            var projectPath = FindProjectFile(testFilePath);
            if (string.IsNullOrEmpty(projectPath))
            {
                _logger.LogError($"Could not find project file for test file: {testFilePath}");
                result.ErrorMessage = $"Could not find project file for test file: {testFilePath}";
                return result;
            }

            // Run the tests with filter
            var testName = Path.GetFileNameWithoutExtension(testFilePath);
            var (success, output, error) = await RunDotNetTestAsync(projectPath, testName);

            // Parse the test results
            result.Success = success;
            result.Output = output;
            result.ErrorOutput = error;
            result.TestResults = ParseTestResults(output);

            _logger.LogInformation($"Test run completed for {testFilePath}. Success: {result.Success}, Total tests: {result.TestResults.Count}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error running tests for {testFilePath}");
            result.ErrorMessage = ex.Message;
        }

        return result;
    }

    /// <summary>
    /// Runs a specific test
    /// </summary>
    /// <param name="projectPath">Path to the project file</param>
    /// <param name="testName">Name of the test to run</param>
    /// <returns>Test run result</returns>
    public async Task<TestRunResult> RunSpecificTestAsync(string projectPath, string testName)
    {
        _logger.LogInformation($"Running test: {testName} in project: {projectPath}");

        var result = new TestRunResult
        {
            ProjectPath = projectPath,
            Success = false
        };

        try
        {
            // Check if the project exists
            if (!File.Exists(projectPath))
            {
                _logger.LogError($"Project file not found: {projectPath}");
                result.ErrorMessage = $"Project file not found: {projectPath}";
                return result;
            }

            // Run the test
            var (success, output, error) = await RunDotNetTestAsync(projectPath, testName);

            // Parse the test results
            result.Success = success;
            result.Output = output;
            result.ErrorOutput = error;
            result.TestResults = ParseTestResults(output);

            _logger.LogInformation($"Test run completed for {testName} in {projectPath}. Success: {result.Success}, Total tests: {result.TestResults.Count}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error running test {testName} in {projectPath}");
            result.ErrorMessage = ex.Message;
        }

        return result;
    }

    /// <summary>
    /// Runs dotnet test command
    /// </summary>
    /// <param name="projectPath">Path to the project file</param>
    /// <param name="filter">Optional test filter</param>
    /// <returns>Tuple containing success flag, standard output, and error output</returns>
    private async Task<(bool Success, string Output, string Error)> RunDotNetTestAsync(string projectPath, string filter = null)
    {
        var arguments = new StringBuilder("test");
        arguments.Append($" \"{projectPath}\"");
        arguments.Append(" --verbosity normal");

        if (!string.IsNullOrEmpty(filter))
        {
            arguments.Append($" --filter FullyQualifiedName~{filter}");
        }

        var processStartInfo = new ProcessStartInfo
        {
            FileName = "dotnet",
            Arguments = arguments.ToString(),
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = new Process { StartInfo = processStartInfo };
        var outputBuilder = new StringBuilder();
        var errorBuilder = new StringBuilder();

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

        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        await process.WaitForExitAsync();

        var output = outputBuilder.ToString();
        var error = errorBuilder.ToString();
        var success = process.ExitCode == 0;

        return (success, output, error);
    }

    /// <summary>
    /// Parses test results from the output
    /// </summary>
    /// <param name="output">Command output</param>
    /// <returns>List of test results</returns>
    private List<TestResult> ParseTestResults(string output)
    {
        var results = new List<TestResult>();

        // Extract test results using regex
        var testResultRegex = new Regex(@"(Passed|Failed)\s+([^\s]+)\s+\[([^\]]+)\]");
        var matches = testResultRegex.Matches(output);

        foreach (Match match in matches)
        {
            var status = match.Groups[1].Value;
            var testName = match.Groups[2].Value;
            var duration = match.Groups[3].Value;

            results.Add(new TestResult
            {
                TestName = testName,
                Status = status == "Passed" ? TestStatus.Passed : TestStatus.Failed,
                Duration = duration
            });
        }

        // Extract error messages for failed tests
        var failedTestRegex = new Regex(@"Failed\s+([^\s]+).*?Error Message:\s*(.*?)(?=\s*Stack Trace:|$)", RegexOptions.Singleline);
        var failedMatches = failedTestRegex.Matches(output);

        foreach (Match match in failedMatches)
        {
            var testName = match.Groups[1].Value;
            var errorMessage = match.Groups[2].Value.Trim();

            var testResult = results.FirstOrDefault(r => r.TestName == testName);
            if (testResult != null)
            {
                testResult.ErrorMessage = errorMessage;
            }
        }

        return results;
    }

    /// <summary>
    /// Finds the project file for a test file
    /// </summary>
    /// <param name="testFilePath">Path to the test file</param>
    /// <returns>Path to the project file, or null if not found</returns>
    private string FindProjectFile(string testFilePath)
    {
        var directory = Path.GetDirectoryName(testFilePath);
        while (!string.IsNullOrEmpty(directory))
        {
            var projectFiles = Directory.GetFiles(directory, "*.csproj").Concat(Directory.GetFiles(directory, "*.fsproj"));
            if (projectFiles.Any())
            {
                return projectFiles.First();
            }

            directory = Path.GetDirectoryName(directory);
        }

        return null;
    }
}

/// <summary>
/// Result of a test run
/// </summary>
public class TestRunResult
{
    /// <summary>
    /// Path to the project
    /// </summary>
    public string ProjectPath { get; set; }

    /// <summary>
    /// Whether the test run was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if the test run failed
    /// </summary>
    public string ErrorMessage { get; set; }

    /// <summary>
    /// Command output
    /// </summary>
    public string Output { get; set; }

    /// <summary>
    /// Command error output
    /// </summary>
    public string ErrorOutput { get; set; }

    /// <summary>
    /// List of test results
    /// </summary>
    public List<TestResult> TestResults { get; set; } = new();

    /// <summary>
    /// Gets the number of passed tests
    /// </summary>
    public int PassedCount => TestResults.Count(r => r.Status == TestStatus.Passed);

    /// <summary>
    /// Gets the number of failed tests
    /// </summary>
    public int FailedCount => TestResults.Count(r => r.Status == TestStatus.Failed);

    /// <summary>
    /// Gets the number of skipped tests
    /// </summary>
    public int SkippedCount => TestResults.Count(r => r.Status == TestStatus.Skipped);

    /// <summary>
    /// Gets the total number of tests
    /// </summary>
    public int TotalCount => TestResults.Count;
}

/// <summary>
/// Result of a single test
/// </summary>
public class TestResult
{
    /// <summary>
    /// Name of the test
    /// </summary>
    public string TestName { get; set; }

    /// <summary>
    /// Status of the test
    /// </summary>
    public TestStatus Status { get; set; }

    /// <summary>
    /// Duration of the test
    /// </summary>
    public string Duration { get; set; }

    /// <summary>
    /// Error message if the test failed
    /// </summary>
    public string ErrorMessage { get; set; }
}

/// <summary>
/// Status of a test
/// </summary>
public enum TestStatus
{
    /// <summary>
    /// Test passed
    /// </summary>
    Passed,

    /// <summary>
    /// Test failed
    /// </summary>
    Failed,

    /// <summary>
    /// Test was skipped
    /// </summary>
    Skipped
}