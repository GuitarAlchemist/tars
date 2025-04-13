using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsCli.Services.Testing;

namespace TarsCli.Services.Mcp
{
    /// <summary>
    /// Action handler for tester replica
    /// </summary>
    public class TesterReplicaActionHandler : IMcpActionHandler
    {
        private readonly ILogger<TesterReplicaActionHandler> _logger;
        private readonly TestGeneratorService _testGeneratorService;
        private readonly TestRunnerService _testRunnerService;
        private readonly TestResultAnalyzer _testResultAnalyzer;

        /// <summary>
        /// Initializes a new instance of the TesterReplicaActionHandler class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="testGeneratorService">Test generator service</param>
        /// <param name="testRunnerService">Test runner service</param>
        /// <param name="testResultAnalyzer">Test result analyzer</param>
        public TesterReplicaActionHandler(
            ILogger<TesterReplicaActionHandler> logger,
            TestGeneratorService testGeneratorService,
            TestRunnerService testRunnerService,
            TestResultAnalyzer testResultAnalyzer)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _testGeneratorService = testGeneratorService ?? throw new ArgumentNullException(nameof(testGeneratorService));
            _testRunnerService = testRunnerService ?? throw new ArgumentNullException(nameof(testRunnerService));
            _testResultAnalyzer = testResultAnalyzer ?? throw new ArgumentNullException(nameof(testResultAnalyzer));
        }

        /// <inheritdoc/>
        public string ActionName => "test_code";

        /// <inheritdoc/>
        public async Task<JsonElement> HandleActionAsync(JsonElement request)
        {
            _logger.LogInformation("Handling test_code action");

            try
            {
                // Extract parameters from the request
                var filePath = request.TryGetProperty("file_path", out var filePathElement)
                    ? filePathElement.GetString()
                    : null;

                var fileContent = request.TryGetProperty("file_content", out var fileContentElement)
                    ? fileContentElement.GetString()
                    : null;

                var operation = request.TryGetProperty("operation", out var operationElement)
                    ? operationElement.GetString()
                    : "run";

                // Validate parameters
                if (string.IsNullOrEmpty(filePath))
                {
                    return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                    {
                        success = false,
                        error = "File path is required"
                    }));
                }

                // If file content is provided, save it to a temporary file
                var tempFilePath = filePath;
                if (!string.IsNullOrEmpty(fileContent))
                {
                    tempFilePath = Path.Combine(Path.GetTempPath(), Path.GetFileName(filePath));
                    await File.WriteAllTextAsync(tempFilePath, fileContent);
                }

                // Handle different operations
                switch (operation)
                {
                    case "generate":
                        return await HandleGenerateTestsAsync(tempFilePath);

                    case "run":
                        return await HandleRunTestsAsync(tempFilePath);

                    case "analyze":
                        return await HandleAnalyzeTestsAsync(tempFilePath);

                    default:
                        return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                        {
                            success = false,
                            error = $"Unknown operation: {operation}"
                        }));
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error handling test_code action");
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = ex.Message
                }));
            }
        }

        /// <summary>
        /// Handles the generate tests operation
        /// </summary>
        /// <param name="filePath">Path to the file</param>
        /// <returns>JSON response</returns>
        private async Task<JsonElement> HandleGenerateTestsAsync(string filePath)
        {
            _logger.LogInformation($"Generating tests for {filePath}");

            // Generate tests
            var testGenerationResult = await _testGeneratorService.GenerateTestsAsync(filePath);
            if (testGenerationResult == null)
            {
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = "Failed to generate tests"
                }));
            }

            // Save the tests
            await _testGeneratorService.SaveTestsAsync(testGenerationResult);

            // Convert the test generation result to a JSON-friendly format
            var result = new
            {
                success = true,
                source_file_path = testGenerationResult.SourceFilePath,
                test_file_path = testGenerationResult.TestFilePath,
                test_file_content = testGenerationResult.TestFileContent,
                tests = ConvertTests(testGenerationResult.Tests)
            };

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(result));
        }

        /// <summary>
        /// Handles the run tests operation
        /// </summary>
        /// <param name="filePath">Path to the file</param>
        /// <returns>JSON response</returns>
        private async Task<JsonElement> HandleRunTestsAsync(string filePath)
        {
            _logger.LogInformation($"Running tests for {filePath}");

            // Find the test file
            var testFilePath = FindTestFile(filePath);
            if (string.IsNullOrEmpty(testFilePath))
            {
                // Generate tests if they don't exist
                var testGenerationResult = await _testGeneratorService.GenerateTestsAsync(filePath);
                if (testGenerationResult == null)
                {
                    return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                    {
                        success = false,
                        error = "Failed to generate tests"
                    }));
                }

                // Save the tests
                await _testGeneratorService.SaveTestsAsync(testGenerationResult);
                testFilePath = testGenerationResult.TestFilePath;
            }

            // Run the tests
            var testRunResult = await _testRunnerService.RunTestFileAsync(testFilePath);
            if (testRunResult == null)
            {
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = "Failed to run tests"
                }));
            }

            // Convert the test run result to a JSON-friendly format
            var result = new
            {
                success = true,
                project_path = testRunResult.ProjectPath,
                test_success = testRunResult.Success,
                test_results = ConvertTestResults(testRunResult.TestResults),
                passed_count = testRunResult.PassedCount,
                failed_count = testRunResult.FailedCount,
                skipped_count = testRunResult.SkippedCount,
                total_count = testRunResult.TotalCount
            };

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(result));
        }

        /// <summary>
        /// Handles the analyze tests operation
        /// </summary>
        /// <param name="filePath">Path to the file</param>
        /// <returns>JSON response</returns>
        private async Task<JsonElement> HandleAnalyzeTestsAsync(string filePath)
        {
            _logger.LogInformation($"Analyzing tests for {filePath}");

            // Find the test file
            var testFilePath = FindTestFile(filePath);
            if (string.IsNullOrEmpty(testFilePath))
            {
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = "Test file not found"
                }));
            }

            // Run the tests
            var testRunResult = await _testRunnerService.RunTestFileAsync(testFilePath);
            if (testRunResult == null)
            {
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = "Failed to run tests"
                }));
            }

            // Analyze the test results
            var testAnalysisResult = await _testResultAnalyzer.AnalyzeResultsAsync(testRunResult);
            if (testAnalysisResult == null)
            {
                return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
                {
                    success = false,
                    error = "Failed to analyze test results"
                }));
            }

            // Convert the test analysis result to a JSON-friendly format
            var result = new
            {
                success = true,
                project_path = testAnalysisResult.ProjectPath,
                test_success = testAnalysisResult.Success,
                passed_count = testAnalysisResult.PassedCount,
                failed_count = testAnalysisResult.FailedCount,
                skipped_count = testAnalysisResult.SkippedCount,
                total_count = testAnalysisResult.TotalCount,
                issues = ConvertTestIssues(testAnalysisResult.Issues),
                summary = testAnalysisResult.Summary,
                detailed_report = testAnalysisResult.DetailedReport,
                improvement_suggestions = testAnalysisResult.ImprovementSuggestions
            };

            return JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(result));
        }

        /// <summary>
        /// Finds the test file for a source file
        /// </summary>
        /// <param name="sourceFilePath">Path to the source file</param>
        /// <returns>Path to the test file, or null if not found</returns>
        private string FindTestFile(string sourceFilePath)
        {
            var directory = Path.GetDirectoryName(sourceFilePath);
            var fileName = Path.GetFileNameWithoutExtension(sourceFilePath);
            var extension = Path.GetExtension(sourceFilePath);

            // Check for test file in Tests directory
            var testDirectory = Path.Combine(directory, "Tests");
            var testFilePath = Path.Combine(testDirectory, $"{fileName}Tests{extension}");
            if (File.Exists(testFilePath))
            {
                return testFilePath;
            }

            // Check for test file in test directory
            testDirectory = Path.Combine(directory, "test");
            testFilePath = Path.Combine(testDirectory, $"{fileName}Tests{extension}");
            if (File.Exists(testFilePath))
            {
                return testFilePath;
            }

            // Check for test file in same directory
            testFilePath = Path.Combine(directory, $"{fileName}Tests{extension}");
            if (File.Exists(testFilePath))
            {
                return testFilePath;
            }

            return null;
        }

        /// <summary>
        /// Converts test cases to a JSON-friendly format
        /// </summary>
        /// <param name="tests">List of test cases</param>
        /// <returns>List of JSON-friendly test cases</returns>
        private List<object> ConvertTests(List<TestCase> tests)
        {
            var result = new List<object>();
            foreach (var test in tests)
            {
                result.Add(new
                {
                    name = test.Name,
                    description = test.Description,
                    type = test.Type.ToString().ToLowerInvariant(),
                    target_method = test.TargetMethod,
                    test_code = test.TestCode
                });
            }
            return result;
        }

        /// <summary>
        /// Converts test results to a JSON-friendly format
        /// </summary>
        /// <param name="testResults">List of test results</param>
        /// <returns>List of JSON-friendly test results</returns>
        private List<object> ConvertTestResults(List<TestResult> testResults)
        {
            var result = new List<object>();
            foreach (var testResult in testResults)
            {
                result.Add(new
                {
                    test_name = testResult.TestName,
                    status = testResult.Status.ToString().ToLowerInvariant(),
                    duration = testResult.Duration,
                    error_message = testResult.ErrorMessage
                });
            }
            return result;
        }

        /// <summary>
        /// Converts test issues to a JSON-friendly format
        /// </summary>
        /// <param name="issues">List of test issues</param>
        /// <returns>List of JSON-friendly test issues</returns>
        private List<object> ConvertTestIssues(List<TestIssue> issues)
        {
            var result = new List<object>();
            foreach (var issue in issues)
            {
                result.Add(new
                {
                    test_name = issue.TestName,
                    type = issue.Type.ToString().ToLowerInvariant(),
                    severity = issue.Severity.ToString().ToLowerInvariant(),
                    description = issue.Description,
                    error_message = issue.ErrorMessage,
                    suggestion = issue.Suggestion
                });
            }
            return result;
        }
    }
}
