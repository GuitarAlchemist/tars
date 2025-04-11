using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Executes tests to validate changes
/// </summary>
public class TestExecutor
{
    private readonly ILogger<TestExecutor> _logger;
    private readonly SafeExecutionEnvironment _safeExecutionEnvironment;

    /// <summary>
    /// Initializes a new instance of the <see cref="TestExecutor"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="safeExecutionEnvironment">The safe execution environment</param>
    public TestExecutor(ILogger<TestExecutor> logger, SafeExecutionEnvironment safeExecutionEnvironment)
    {
        _logger = logger;
        _safeExecutionEnvironment = safeExecutionEnvironment;
    }

    /// <summary>
    /// Executes tests for a project
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="projectPath">The project path</param>
    /// <param name="testFilter">The test filter</param>
    /// <param name="options">Optional test options</param>
    /// <returns>The test execution result</returns>
    public async Task<TestExecutionResult> ExecuteTestsAsync(
        string contextId,
        string projectPath,
        string? testFilter = null,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Executing tests for project: {ProjectPath}", projectPath);

            // Create result
            var result = new TestExecutionResult
            {
                ProjectPath = projectPath,
                TestFilter = testFilter,
                StartedAt = DateTime.UtcNow
            };

            // Check if project exists
            if (!File.Exists(projectPath) && !Directory.Exists(projectPath))
            {
                result.IsSuccessful = false;
                result.ErrorMessage = $"Project not found: {projectPath}";
                result.CompletedAt = DateTime.UtcNow;
                // Set the Duration property
                result.Duration = (result.CompletedAt - result.StartedAt).ToString();
                return result;
            }

            // Determine test command
            var (testCommand, testArguments) = GetTestCommand(projectPath, testFilter, options);

            // Execute tests
            var output = await _safeExecutionEnvironment.ExecuteCommandAsync(contextId, testCommand, testArguments, Path.GetDirectoryName(projectPath));

            // Parse test results
            result.Output = output;
            result.IsSuccessful = !output.Contains("Failed:") && !output.Contains("Error:") && !output.Contains("Aborted:");
            result.CompletedAt = DateTime.UtcNow;
            // Calculate duration in milliseconds
            var duration = (long)(result.CompletedAt - result.StartedAt).TotalMilliseconds;

            // Use reflection to set the DurationMs property if it's writable
            var durationMsProperty = result.GetType().GetProperty("DurationMs");
            if (durationMsProperty != null && durationMsProperty.CanWrite)
            {
                durationMsProperty.SetValue(result, duration);
            }
            else
            {
                // Try to set the Duration property as a string if DurationMs is not writable
                var durationProperty = result.GetType().GetProperty("Duration");
                if (durationProperty != null && durationProperty.CanWrite)
                {
                    var timeSpan = TimeSpan.FromMilliseconds(duration);
                    durationProperty.SetValue(result, timeSpan.ToString());
                }
            }

            // Parse test counts
            result.TotalTests = ParseTestCount(output, "Total:");
            result.PassedTests = ParseTestCount(output, "Passed:");
            result.FailedTests = ParseTestCount(output, "Failed:");
            result.SkippedTests = ParseTestCount(output, "Skipped:");

            // Parse test failures
            result.TestFailures = ParseTestFailures(output);

            _logger.LogInformation("Test execution completed. Total: {TotalTests}, Passed: {PassedTests}, Failed: {FailedTests}, Skipped: {SkippedTests}",
                result.TotalTests, result.PassedTests, result.FailedTests, result.SkippedTests);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing tests for project: {ProjectPath}", projectPath);
            return new TestExecutionResult
            {
                ProjectPath = projectPath,
                TestFilter = testFilter,
                IsSuccessful = false,
                ErrorMessage = ex.Message,
                StartedAt = DateTime.UtcNow,
                CompletedAt = DateTime.UtcNow,
                // Duration is set to 0 for error cases
                // Use reflection to set the DurationMs property if it's writable
                // This will be handled by the adapter when converting between types
            };
        }
    }

    /// <summary>
    /// Executes a specific test
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="projectPath">The project path</param>
    /// <param name="testName">The test name</param>
    /// <param name="options">Optional test options</param>
    /// <returns>The test execution result</returns>
    public async Task<TestExecutionResult> ExecuteTestAsync(
        string contextId,
        string projectPath,
        string testName,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Executing test: {TestName} in project: {ProjectPath}", testName, projectPath);

            // Create result
            var result = new TestExecutionResult
            {
                ProjectPath = projectPath,
                TestFilter = testName,
                StartedAt = DateTime.UtcNow
            };

            // Check if project exists
            if (!File.Exists(projectPath) && !Directory.Exists(projectPath))
            {
                result.IsSuccessful = false;
                result.ErrorMessage = $"Project not found: {projectPath}";
                result.CompletedAt = DateTime.UtcNow;
                // Set the Duration property
                result.Duration = (result.CompletedAt - result.StartedAt).ToString();
                return result;
            }

            // Determine test command
            var (testCommand, testArguments) = GetTestCommand(projectPath, testName, options);

            // Execute test
            var output = await _safeExecutionEnvironment.ExecuteCommandAsync(contextId, testCommand, testArguments, Path.GetDirectoryName(projectPath));

            // Parse test results
            result.Output = output;
            result.IsSuccessful = !output.Contains("Failed:") && !output.Contains("Error:") && !output.Contains("Aborted:");
            result.CompletedAt = DateTime.UtcNow;
            result.Duration = (result.CompletedAt - result.StartedAt).ToString();

            // Parse test counts
            result.TotalTests = ParseTestCount(output, "Total:");
            result.PassedTests = ParseTestCount(output, "Passed:");
            result.FailedTests = ParseTestCount(output, "Failed:");
            result.SkippedTests = ParseTestCount(output, "Skipped:");

            // Parse test failures
            result.TestFailures = ParseTestFailures(output);

            _logger.LogInformation("Test execution completed. Total: {TotalTests}, Passed: {PassedTests}, Failed: {FailedTests}, Skipped: {SkippedTests}",
                result.TotalTests, result.PassedTests, result.FailedTests, result.SkippedTests);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing test: {TestName} in project: {ProjectPath}", testName, projectPath);
            return new TestExecutionResult
            {
                ProjectPath = projectPath,
                TestFilter = testName,
                IsSuccessful = false,
                ErrorMessage = ex.Message,
                StartedAt = DateTime.UtcNow,
                CompletedAt = DateTime.UtcNow,
                // Duration is set to 0 for error cases
                // Use reflection to set the DurationMs property if it's writable
                // This will be handled by the adapter when converting between types
            };
        }
    }

    /// <summary>
    /// Gets the test command for a project
    /// </summary>
    /// <param name="projectPath">The project path</param>
    /// <param name="testFilter">The test filter</param>
    /// <param name="options">Optional test options</param>
    /// <returns>The test command and arguments</returns>
    private (string Command, string Arguments) GetTestCommand(string projectPath, string? testFilter, Dictionary<string, string>? options)
    {
        // Default command
        var command = "dotnet";
        var arguments = "test";

        // Add project path
        arguments += $" \"{projectPath}\"";

        // Add test filter
        if (!string.IsNullOrEmpty(testFilter))
        {
            arguments += $" --filter \"{testFilter}\"";
        }

        // Add verbosity
        var verbosity = options?.TryGetValue("Verbosity", out var v) == true ? v : "normal";
        arguments += $" --verbosity {verbosity}";

        // Add logger
        var logger = options?.TryGetValue("Logger", out var l) == true ? l : "console;verbosity=detailed";
        arguments += $" --logger \"{logger}\"";

        // Add configuration
        var configuration = options?.TryGetValue("Configuration", out var c) == true ? c : "Debug";
        arguments += $" --configuration {configuration}";

        // Add no build option
        var noBuild = options?.TryGetValue("NoBuild", out var nb) == true && nb.Equals("true", StringComparison.OrdinalIgnoreCase);
        if (noBuild)
        {
            arguments += " --no-build";
        }

        // Add no restore option
        var noRestore = options?.TryGetValue("NoRestore", out var nr) == true && nr.Equals("true", StringComparison.OrdinalIgnoreCase);
        if (noRestore)
        {
            arguments += " --no-restore";
        }

        // Add blame option
        var blame = options?.TryGetValue("Blame", out var b) == true && b.Equals("true", StringComparison.OrdinalIgnoreCase);
        if (blame)
        {
            arguments += " --blame";
        }

        return (command, arguments);
    }

    /// <summary>
    /// Parses the test count from the output
    /// </summary>
    /// <param name="output">The test output</param>
    /// <param name="prefix">The count prefix</param>
    /// <returns>The test count</returns>
    private int ParseTestCount(string output, string prefix)
    {
        var match = Regex.Match(output, $"{prefix}\\s+(\\d+)");
        if (match.Success && int.TryParse(match.Groups[1].Value, out var count))
        {
            return count;
        }
        return 0;
    }

    /// <summary>
    /// Parses test failures from the output
    /// </summary>
    /// <param name="output">The test output</param>
    /// <returns>The list of test failures</returns>
    private List<TestFailure> ParseTestFailures(string output)
    {
        var failures = new List<TestFailure>();
        var failureMatches = Regex.Matches(output, @"Failed\s+(.+?)\s*\r?\n\s*Error Message:\s*\r?\n\s*(.+?)\r?\n\s*Stack Trace:\s*\r?\n\s*(.+?)(?:\r?\n\s*-{3,}|$)", RegexOptions.Singleline);

        foreach (Match match in failureMatches)
        {
            if (match.Groups.Count >= 4)
            {
                failures.Add(new TestFailure
                {
                    TestName = match.Groups[1].Value.Trim(),
                    ErrorMessage = match.Groups[2].Value.Trim(),
                    StackTrace = match.Groups[3].Value.Trim()
                });
            }
        }

        return failures;
    }

    /// <summary>
    /// Converts test results to validation results
    /// </summary>
    /// <param name="testResult">The test execution result</param>
    /// <param name="filePath">The file path</param>
    /// <returns>The list of validation results</returns>
    public List<ValidationResult> ConvertToValidationResults(TestExecutionResult testResult, string filePath)
    {
        var results = new List<ValidationResult>();

        // Add overall result
        results.Add(new ValidationResult
        {
            RuleName = "TestExecution",
            IsPassed = testResult.IsSuccessful,
            Severity = testResult.IsSuccessful ? ValidationRuleSeverity.Information : ValidationRuleSeverity.Error,
            Message = testResult.IsSuccessful
                ? $"Tests passed: {testResult.PassedTests}/{testResult.TotalTests}"
                : $"Tests failed: {testResult.FailedTests}/{testResult.TotalTests}",
            Target = filePath,
            Timestamp = DateTime.UtcNow,
            Details = testResult.Output,
            Metadata = new Dictionary<string, string>
            {
                { "ProjectPath", testResult.ProjectPath },
                { "TestFilter", testResult.TestFilter ?? string.Empty },
                { "TotalTests", testResult.TotalTests.ToString() },
                { "PassedTests", testResult.PassedTests.ToString() },
                { "FailedTests", testResult.FailedTests.ToString() },
                { "SkippedTests", testResult.SkippedTests.ToString() },
                { "Duration", $"{testResult.DurationMs} ms" }
            }
        });

        // Add results for each failure
        foreach (var failure in testResult.TestFailures)
        {
            results.Add(new ValidationResult
            {
                RuleName = "TestFailure",
                IsPassed = false,
                Severity = ValidationRuleSeverity.Error,
                Message = $"Test failed: {failure.TestName}",
                Target = filePath,
                Timestamp = DateTime.UtcNow,
                Details = $"Error Message: {failure.ErrorMessage}\nStack Trace: {failure.StackTrace}",
                Metadata = new Dictionary<string, string>
                {
                    { "TestName", failure.TestName },
                    { "ErrorMessage", failure.ErrorMessage },
                    { "StackTrace", failure.StackTrace }
                }
            });
        }

        return results;
    }

    /// <summary>
    /// Gets the available test options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        return new Dictionary<string, string>
        {
            { "Verbosity", "The verbosity level (quiet, minimal, normal, detailed, diagnostic)" },
            { "Logger", "The test logger to use" },
            { "Configuration", "The configuration to use (Debug, Release)" },
            { "NoBuild", "Whether to skip building the project (true, false)" },
            { "NoRestore", "Whether to skip restoring the project (true, false)" },
            { "Blame", "Whether to enable blame mode (true, false)" }
        };
    }
}
