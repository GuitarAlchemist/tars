using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsCli.Services.CodeGeneration;
using TarsCli.Services.Mcp;
using TarsCli.Services.Testing;

namespace TarsCli.Services.SelfCoding
{
    /// <summary>
    /// Service for processing tests in the self-coding workflow
    /// </summary>
    public class TestProcessor
    {
        private readonly ILogger<TestProcessor> _logger;
        private readonly TestGeneratorService _testGeneratorService;
        private readonly TestRunnerService _testRunnerService;
        private readonly TestResultAnalyzer _testResultAnalyzer;
        private readonly ReplicaCommunicationProtocol _replicaCommunicationProtocol;

        /// <summary>
        /// Initializes a new instance of the TestProcessor class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="testGeneratorService">Test generator service</param>
        /// <param name="testRunnerService">Test runner service</param>
        /// <param name="testResultAnalyzer">Test result analyzer</param>
        /// <param name="replicaCommunicationProtocol">Replica communication protocol</param>
        public TestProcessor(
            ILogger<TestProcessor> logger,
            TestGeneratorService testGeneratorService,
            TestRunnerService testRunnerService,
            TestResultAnalyzer testResultAnalyzer,
            ReplicaCommunicationProtocol replicaCommunicationProtocol)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _testGeneratorService = testGeneratorService ?? throw new ArgumentNullException(nameof(testGeneratorService));
            _testRunnerService = testRunnerService ?? throw new ArgumentNullException(nameof(testRunnerService));
            _testResultAnalyzer = testResultAnalyzer ?? throw new ArgumentNullException(nameof(testResultAnalyzer));
            _replicaCommunicationProtocol = replicaCommunicationProtocol ?? throw new ArgumentNullException(nameof(replicaCommunicationProtocol));
        }

        /// <summary>
        /// Generates tests for files
        /// </summary>
        /// <param name="generationResults">Code generation results</param>
        /// <returns>List of test generation results</returns>
        public async Task<List<TestGenerationResult>> GenerateTestsAsync(List<CodeGenerationResult> generationResults)
        {
            _logger.LogInformation($"Generating tests for {generationResults.Count} files");

            try
            {
                // Validate parameters
                if (generationResults == null || !generationResults.Any())
                {
                    throw new ArgumentException("Generation results are required", nameof(generationResults));
                }

                // Generate tests for each file
                var testGenerationResults = new List<TestGenerationResult>();
                foreach (var generationResult in generationResults)
                {
                    var result = await GenerateTestsForFileAsync(generationResult.FilePath);
                    if (result != null)
                    {
                        testGenerationResults.Add(result);
                    }
                }

                _logger.LogInformation($"Generated tests for {testGenerationResults.Count} files");
                return testGenerationResults;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating tests");
                throw;
            }
        }

        /// <summary>
        /// Generates tests for a file
        /// </summary>
        /// <param name="filePath">Path to the file</param>
        /// <returns>Test generation result</returns>
        public async Task<TestGenerationResult> GenerateTestsForFileAsync(string filePath)
        {
            _logger.LogInformation($"Generating tests for file: {filePath}");

            try
            {
                // Validate parameters
                if (string.IsNullOrEmpty(filePath))
                {
                    throw new ArgumentException("File path is required", nameof(filePath));
                }

                if (!File.Exists(filePath))
                {
                    throw new FileNotFoundException($"File not found: {filePath}");
                }

                // Generate tests
                var testGenerationResult = await _testGeneratorService.GenerateTestsAsync(filePath);
                if (testGenerationResult == null)
                {
                    _logger.LogWarning($"Failed to generate tests for file: {filePath}");
                    return null;
                }

                // Save the tests
                await _testGeneratorService.SaveTestsAsync(testGenerationResult);

                _logger.LogInformation($"Generated tests for file: {filePath}, with {testGenerationResult.Tests.Count} tests");
                return testGenerationResult;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating tests for file: {filePath}");
                throw;
            }
        }

        /// <summary>
        /// Generates tests for a file using a tester replica
        /// </summary>
        /// <param name="filePath">Path to the file</param>
        /// <param name="replicaUrl">URL of the tester replica</param>
        /// <returns>Test generation result</returns>
        public async Task<TestGenerationResult> GenerateTestsWithReplicaAsync(string filePath, string replicaUrl)
        {
            _logger.LogInformation($"Generating tests for file with replica: {filePath}");

            try
            {
                // Validate parameters
                if (string.IsNullOrEmpty(filePath))
                {
                    throw new ArgumentException("File path is required", nameof(filePath));
                }

                if (!File.Exists(filePath))
                {
                    throw new FileNotFoundException($"File not found: {filePath}");
                }

                if (string.IsNullOrEmpty(replicaUrl))
                {
                    throw new ArgumentException("Replica URL is required", nameof(replicaUrl));
                }

                // Read the file content
                var fileContent = await File.ReadAllTextAsync(filePath);

                // Create the request
                var request = new
                {
                    file_path = filePath,
                    file_content = fileContent,
                    operation = "generate"
                };

                // Send the request to the replica
                var response = await _replicaCommunicationProtocol.SendMessageAsync(replicaUrl, "test_code", request);

                // Check if the response was successful
                var success = response.TryGetProperty("success", out var successElement) && successElement.GetBoolean();
                if (!success)
                {
                    // Extract the error message
                    var error = response.TryGetProperty("error", out var errorElement)
                        ? errorElement.GetString()
                        : "Unknown error";

                    _logger.LogError($"Error generating tests with replica: {error}");
                    return null;
                }

                // Convert the response to a TestGenerationResult
                var testGenerationResult = ConvertResponseToTestGenerationResult(response, filePath);

                // Save the tests
                await _testGeneratorService.SaveTestsAsync(testGenerationResult);

                _logger.LogInformation($"Generated tests for file with replica: {filePath}, with {testGenerationResult.Tests.Count} tests");
                return testGenerationResult;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating tests for file with replica: {filePath}");
                throw;
            }
        }

        /// <summary>
        /// Runs tests for files
        /// </summary>
        /// <param name="testGenerationResults">Test generation results</param>
        /// <returns>List of test run results</returns>
        public async Task<List<TestRunResult>> RunTestsAsync(List<TestGenerationResult> testGenerationResults)
        {
            _logger.LogInformation($"Running tests for {testGenerationResults.Count} files");

            try
            {
                // Validate parameters
                if (testGenerationResults == null || !testGenerationResults.Any())
                {
                    throw new ArgumentException("Test generation results are required", nameof(testGenerationResults));
                }

                // Run tests for each file
                var testRunResults = new List<TestRunResult>();
                foreach (var testGenerationResult in testGenerationResults)
                {
                    var result = await RunTestsForFileAsync(testGenerationResult.TestFilePath);
                    if (result != null)
                    {
                        testRunResults.Add(result);
                    }
                }

                _logger.LogInformation($"Ran tests for {testRunResults.Count} files");
                return testRunResults;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error running tests");
                throw;
            }
        }

        /// <summary>
        /// Runs tests for a file
        /// </summary>
        /// <param name="testFilePath">Path to the test file</param>
        /// <returns>Test run result</returns>
        public async Task<TestRunResult> RunTestsForFileAsync(string testFilePath)
        {
            _logger.LogInformation($"Running tests for file: {testFilePath}");

            try
            {
                // Validate parameters
                if (string.IsNullOrEmpty(testFilePath))
                {
                    throw new ArgumentException("Test file path is required", nameof(testFilePath));
                }

                if (!File.Exists(testFilePath))
                {
                    throw new FileNotFoundException($"Test file not found: {testFilePath}");
                }

                // Run the tests
                var testRunResult = await _testRunnerService.RunTestFileAsync(testFilePath);
                if (testRunResult == null)
                {
                    _logger.LogWarning($"Failed to run tests for file: {testFilePath}");
                    return null;
                }

                _logger.LogInformation($"Ran tests for file: {testFilePath}, Success: {testRunResult.Success}, Total tests: {testRunResult.TotalCount}");
                return testRunResult;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error running tests for file: {testFilePath}");
                throw;
            }
        }

        /// <summary>
        /// Runs tests for a file using a tester replica
        /// </summary>
        /// <param name="testFilePath">Path to the test file</param>
        /// <param name="replicaUrl">URL of the tester replica</param>
        /// <returns>Test run result</returns>
        public async Task<TestRunResult> RunTestsWithReplicaAsync(string testFilePath, string replicaUrl)
        {
            _logger.LogInformation($"Running tests for file with replica: {testFilePath}");

            try
            {
                // Validate parameters
                if (string.IsNullOrEmpty(testFilePath))
                {
                    throw new ArgumentException("Test file path is required", nameof(testFilePath));
                }

                if (!File.Exists(testFilePath))
                {
                    throw new FileNotFoundException($"Test file not found: {testFilePath}");
                }

                if (string.IsNullOrEmpty(replicaUrl))
                {
                    throw new ArgumentException("Replica URL is required", nameof(replicaUrl));
                }

                // Read the test file content
                var testFileContent = await File.ReadAllTextAsync(testFilePath);

                // Create the request
                var request = new
                {
                    file_path = testFilePath,
                    file_content = testFileContent,
                    operation = "run"
                };

                // Send the request to the replica
                var response = await _replicaCommunicationProtocol.SendMessageAsync(replicaUrl, "test_code", request);

                // Check if the response was successful
                var success = response.TryGetProperty("success", out var successElement) && successElement.GetBoolean();
                if (!success)
                {
                    // Extract the error message
                    var error = response.TryGetProperty("error", out var errorElement)
                        ? errorElement.GetString()
                        : "Unknown error";

                    _logger.LogError($"Error running tests with replica: {error}");
                    return null;
                }

                // Convert the response to a TestRunResult
                var testRunResult = ConvertResponseToTestRunResult(response, testFilePath);

                _logger.LogInformation($"Ran tests for file with replica: {testFilePath}, Success: {testRunResult.Success}, Total tests: {testRunResult.TotalCount}");
                return testRunResult;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error running tests for file with replica: {testFilePath}");
                throw;
            }
        }

        /// <summary>
        /// Analyzes test results
        /// </summary>
        /// <param name="testRunResult">Test run result</param>
        /// <returns>Test analysis result</returns>
        public async Task<TestAnalysisResult> AnalyzeTestResultsAsync(TestRunResult testRunResult)
        {
            _logger.LogInformation($"Analyzing test results for {testRunResult.ProjectPath}");

            try
            {
                // Validate parameters
                if (testRunResult == null)
                {
                    throw new ArgumentNullException(nameof(testRunResult));
                }

                // Analyze the test results
                var testAnalysisResult = await _testResultAnalyzer.AnalyzeResultsAsync(testRunResult);
                if (testAnalysisResult == null)
                {
                    _logger.LogWarning($"Failed to analyze test results for {testRunResult.ProjectPath}");
                    return null;
                }

                _logger.LogInformation($"Analyzed test results for {testRunResult.ProjectPath}, Found {testAnalysisResult.Issues.Count} issues");
                return testAnalysisResult;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error analyzing test results for {testRunResult.ProjectPath}");
                throw;
            }
        }

        /// <summary>
        /// Analyzes test results using a tester replica
        /// </summary>
        /// <param name="testRunResult">Test run result</param>
        /// <param name="replicaUrl">URL of the tester replica</param>
        /// <returns>Test analysis result</returns>
        public async Task<TestAnalysisResult> AnalyzeTestResultsWithReplicaAsync(TestRunResult testRunResult, string replicaUrl)
        {
            _logger.LogInformation($"Analyzing test results for {testRunResult.ProjectPath} with replica");

            try
            {
                // Validate parameters
                if (testRunResult == null)
                {
                    throw new ArgumentNullException(nameof(testRunResult));
                }

                if (string.IsNullOrEmpty(replicaUrl))
                {
                    throw new ArgumentException("Replica URL is required", nameof(replicaUrl));
                }

                // Create the request
                var request = new
                {
                    file_path = testRunResult.ProjectPath,
                    operation = "analyze"
                };

                // Send the request to the replica
                var response = await _replicaCommunicationProtocol.SendMessageAsync(replicaUrl, "test_code", request);

                // Check if the response was successful
                var success = response.TryGetProperty("success", out var successElement) && successElement.GetBoolean();
                if (!success)
                {
                    // Extract the error message
                    var error = response.TryGetProperty("error", out var errorElement)
                        ? errorElement.GetString()
                        : "Unknown error";

                    _logger.LogError($"Error analyzing test results with replica: {error}");
                    return null;
                }

                // Convert the response to a TestAnalysisResult
                var testAnalysisResult = ConvertResponseToTestAnalysisResult(response, testRunResult.ProjectPath);

                _logger.LogInformation($"Analyzed test results for {testRunResult.ProjectPath} with replica, Found {testAnalysisResult.Issues.Count} issues");
                return testAnalysisResult;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error analyzing test results for {testRunResult.ProjectPath} with replica");
                throw;
            }
        }

        /// <summary>
        /// Converts a response from a tester replica to a TestGenerationResult
        /// </summary>
        /// <param name="response">Response from the replica</param>
        /// <param name="filePath">Path to the file</param>
        /// <returns>TestGenerationResult</returns>
        private TestGenerationResult ConvertResponseToTestGenerationResult(System.Text.Json.JsonElement response, string filePath)
        {
            var result = new TestGenerationResult
            {
                SourceFilePath = filePath,
                Success = true
            };

            // Extract test_file_path
            if (response.TryGetProperty("test_file_path", out var testFilePathElement))
            {
                result.TestFilePath = testFilePathElement.GetString();
            }
            else
            {
                // Generate a default test file path
                var directory = Path.GetDirectoryName(filePath);
                var fileName = Path.GetFileNameWithoutExtension(filePath);
                var extension = Path.GetExtension(filePath);
                result.TestFilePath = Path.Combine(directory, $"{fileName}Tests{extension}");
            }

            // Extract test_file_content
            if (response.TryGetProperty("test_file_content", out var testFileContentElement))
            {
                result.TestFileContent = testFileContentElement.GetString();
            }

            // Extract tests
            if (response.TryGetProperty("tests", out var testsElement) && testsElement.ValueKind == System.Text.Json.JsonValueKind.Array)
            {
                foreach (var testElement in testsElement.EnumerateArray())
                {
                    var test = new TestCase();

                    // Extract name
                    if (testElement.TryGetProperty("name", out var nameElement))
                    {
                        test.Name = nameElement.GetString();
                    }

                    // Extract description
                    if (testElement.TryGetProperty("description", out var descriptionElement))
                    {
                        test.Description = descriptionElement.GetString();
                    }

                    // Extract type
                    if (testElement.TryGetProperty("type", out var typeElement))
                    {
                        var typeString = typeElement.GetString();
                        if (Enum.TryParse<TestType>(typeString, true, out var type))
                        {
                            test.Type = type;
                        }
                    }

                    // Extract target_method
                    if (testElement.TryGetProperty("target_method", out var targetMethodElement))
                    {
                        test.TargetMethod = targetMethodElement.GetString();
                    }

                    // Extract test_code
                    if (testElement.TryGetProperty("test_code", out var testCodeElement))
                    {
                        test.TestCode = testCodeElement.GetString();
                    }

                    result.Tests.Add(test);
                }
            }

            return result;
        }

        /// <summary>
        /// Converts a response from a tester replica to a TestRunResult
        /// </summary>
        /// <param name="response">Response from the replica</param>
        /// <param name="testFilePath">Path to the test file</param>
        /// <returns>TestRunResult</returns>
        private TestRunResult ConvertResponseToTestRunResult(System.Text.Json.JsonElement response, string testFilePath)
        {
            var result = new TestRunResult
            {
                ProjectPath = testFilePath
            };

            // Extract test_success
            if (response.TryGetProperty("test_success", out var testSuccessElement))
            {
                result.Success = testSuccessElement.GetBoolean();
            }

            // Extract passed_count
            if (response.TryGetProperty("passed_count", out var passedCountElement))
            {
                // No need to set PassedCount directly, it's calculated from TestResults
            }

            // Extract failed_count
            if (response.TryGetProperty("failed_count", out var failedCountElement))
            {
                // No need to set FailedCount directly, it's calculated from TestResults
            }

            // Extract total_count
            if (response.TryGetProperty("total_count", out var totalCountElement))
            {
                // No need to set TotalCount directly, it's calculated from TestResults
            }

            // Extract test_results
            if (response.TryGetProperty("test_results", out var testResultsElement) && testResultsElement.ValueKind == System.Text.Json.JsonValueKind.Array)
            {
                foreach (var testResultElement in testResultsElement.EnumerateArray())
                {
                    var testResult = new TestResult();

                    // Extract test_name
                    if (testResultElement.TryGetProperty("test_name", out var testNameElement))
                    {
                        testResult.TestName = testNameElement.GetString();
                    }

                    // Extract status
                    if (testResultElement.TryGetProperty("status", out var statusElement))
                    {
                        var statusString = statusElement.GetString();
                        if (Enum.TryParse<TestStatus>(statusString, true, out var status))
                        {
                            testResult.Status = status;
                        }
                    }

                    // Extract duration
                    if (testResultElement.TryGetProperty("duration", out var durationElement))
                    {
                        testResult.Duration = durationElement.GetString();
                    }

                    // Extract error_message
                    if (testResultElement.TryGetProperty("error_message", out var errorMessageElement))
                    {
                        testResult.ErrorMessage = errorMessageElement.GetString();
                    }

                    result.TestResults.Add(testResult);
                }
            }

            return result;
        }

        /// <summary>
        /// Converts a response from a tester replica to a TestAnalysisResult
        /// </summary>
        /// <param name="response">Response from the replica</param>
        /// <param name="projectPath">Path to the project</param>
        /// <returns>TestAnalysisResult</returns>
        private TestAnalysisResult ConvertResponseToTestAnalysisResult(System.Text.Json.JsonElement response, string projectPath)
        {
            var result = new TestAnalysisResult
            {
                ProjectPath = projectPath
            };

            // Extract test_success
            if (response.TryGetProperty("test_success", out var testSuccessElement))
            {
                result.Success = testSuccessElement.GetBoolean();
            }

            // Extract passed_count
            if (response.TryGetProperty("passed_count", out var passedCountElement))
            {
                result.PassedCount = passedCountElement.GetInt32();
            }

            // Extract failed_count
            if (response.TryGetProperty("failed_count", out var failedCountElement))
            {
                result.FailedCount = failedCountElement.GetInt32();
            }

            // Extract skipped_count
            if (response.TryGetProperty("skipped_count", out var skippedCountElement))
            {
                result.SkippedCount = skippedCountElement.GetInt32();
            }

            // Extract total_count
            if (response.TryGetProperty("total_count", out var totalCountElement))
            {
                result.TotalCount = totalCountElement.GetInt32();
            }

            // Extract issues
            if (response.TryGetProperty("issues", out var issuesElement) && issuesElement.ValueKind == System.Text.Json.JsonValueKind.Array)
            {
                foreach (var issueElement in issuesElement.EnumerateArray())
                {
                    var issue = new TestIssue();

                    // Extract test_name
                    if (issueElement.TryGetProperty("test_name", out var testNameElement))
                    {
                        issue.TestName = testNameElement.GetString();
                    }

                    // Extract type
                    if (issueElement.TryGetProperty("type", out var typeElement))
                    {
                        var typeString = typeElement.GetString();
                        if (Enum.TryParse<TestIssueType>(typeString, true, out var type))
                        {
                            issue.Type = type;
                        }
                    }

                    // Extract severity
                    if (issueElement.TryGetProperty("severity", out var severityElement))
                    {
                        var severityString = severityElement.GetString();
                        if (Enum.TryParse<TestIssueSeverity>(severityString, true, out var severity))
                        {
                            issue.Severity = severity;
                        }
                    }

                    // Extract description
                    if (issueElement.TryGetProperty("description", out var descriptionElement))
                    {
                        issue.Description = descriptionElement.GetString();
                    }

                    // Extract error_message
                    if (issueElement.TryGetProperty("error_message", out var errorMessageElement))
                    {
                        issue.ErrorMessage = errorMessageElement.GetString();
                    }

                    // Extract suggestion
                    if (issueElement.TryGetProperty("suggestion", out var suggestionElement))
                    {
                        issue.Suggestion = suggestionElement.GetString();
                    }

                    result.Issues.Add(issue);
                }
            }

            // Extract summary
            if (response.TryGetProperty("summary", out var summaryElement))
            {
                result.Summary = summaryElement.GetString();
            }

            // Extract detailed_report
            if (response.TryGetProperty("detailed_report", out var detailedReportElement))
            {
                result.DetailedReport = detailedReportElement.GetString();
            }

            // Extract improvement_suggestions
            if (response.TryGetProperty("improvement_suggestions", out var improvementSuggestionsElement))
            {
                result.ImprovementSuggestions = improvementSuggestionsElement.GetString();
            }

            return result;
        }
    }
}
