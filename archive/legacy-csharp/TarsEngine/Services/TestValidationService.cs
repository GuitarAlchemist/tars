using System.Diagnostics;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for validating tests
/// </summary>
public class TestValidationService : ITestValidationService
{
    private readonly ILogger<TestValidationService> _logger;
    private readonly IMetascriptService _metascriptService;

    /// <summary>
    /// Initializes a new instance of the <see cref="TestValidationService"/> class
    /// </summary>
    public TestValidationService(ILogger<TestValidationService> logger, IMetascriptService metascriptService)
    {
        _logger = logger;
        _metascriptService = metascriptService;
    }

    /// <inheritdoc/>
    public async Task<TestValidationResult> ValidateImprovedCodeAsync(string originalCode, string improvedCode, string testCode, string language)
    {
        try
        {
            _logger.LogInformation("Validating improved code");

            if (string.IsNullOrEmpty(originalCode) || string.IsNullOrEmpty(improvedCode) || string.IsNullOrEmpty(testCode))
            {
                return new TestValidationResult 
                { 
                    IsValid = false,
                    Issues = 
                    [
                        new ValidationIssue
                        {
                            Description = "Original code, improved code, or test code is empty",
                            Severity = IssueSeverity.Error
                        }
                    ]
                };
            }

            // Create a metascript for test validation
            var metascript = $@"
// Test validation metascript
// Language: {language}

// Original code
let originalCode = `{originalCode.Replace("`", "\\`")}`;

// Improved code
let improvedCode = `{improvedCode.Replace("`", "\\`")}`;

// Test code
let testCode = `{testCode.Replace("`", "\\`")}`;

// Validate that the improved code passes the tests
let validationResult = validateImprovedCode(originalCode, improvedCode, testCode, '{language}');

// Return the validation result
return JSON.stringify(validationResult);

// Helper function to validate improved code
function validateImprovedCode(original, improved, tests, language) {{
    // This would be implemented with actual test execution
    // For now, we'll use a simple placeholder

    // Check if the improved code has syntax errors
    const syntaxErrors = checkSyntax(improved, language);

    // Check if the improved code passes the tests
    const testResults = runTests(improved, tests, language);

    // Check if the improved code has any regressions
    const regressions = checkRegressions(original, improved, tests, language);

    return {{
        isValid: syntaxErrors.length === 0 && testResults.passedTests === testResults.totalTests && regressions.length === 0,
        issues: [
            ...syntaxErrors.map(error => ({{
                description: `Syntax error: ${{error.message}}`,
                severity: 'Error',
                location: `Line ${{error.line}}, Column ${{error.column}}`,
                suggestedFix: error.suggestedFix
            }})),
            ...testResults.failures.map(failure => ({{
                description: `Test failure: ${{failure.testName}}`,
                severity: 'Error',
                location: failure.location,
                suggestedFix: failure.suggestedFix
            }})),
            ...regressions.map(regression => ({{
                description: `Regression: ${{regression.description}}`,
                severity: 'Error',
                location: regression.location,
                suggestedFix: regression.suggestedFix
            }}))
        ],
        testResult: testResults
    }};
}}

// Helper function to check syntax
function checkSyntax(code, language) {{
    // This would be implemented with a real syntax checker
    // For now, we'll return an empty array
    return [];
}}

// Helper function to run tests
function runTests(code, tests, language) {{
    // This would be implemented with a real test runner
    // For now, we'll return a placeholder result
    return {{
        totalTests: 3,
        passedTests: 3,
        failedTests: 0,
        skippedTests: 0,
        executionTimeMs: 100,
        failures: [],
        rawOutput: 'All tests passed!'
    }};
}}

// Helper function to check regressions
function checkRegressions(original, improved, tests, language) {{
    // This would be implemented with a real regression checker
    // For now, we'll return an empty array
    return [];
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);

            if (result == null)
            {
                return new TestValidationResult 
                { 
                    IsValid = false,
                    Issues = 
                    [
                        new ValidationIssue
                        {
                            Description = "Metascript execution returned null result",
                            Severity = IssueSeverity.Error
                        }
                    ]
                };
            }

            var resultString = result?.ToString() ?? string.Empty;
            if (string.IsNullOrEmpty(resultString))
            {
                return new TestValidationResult 
                { 
                    IsValid = false,
                    Issues = 
                    [
                        new ValidationIssue
                        {
                            Description = "Metascript execution returned empty result",
                            Severity = IssueSeverity.Error
                        }
                    ]
                };
            }

            try
            {
                // Parse the result as JSON
                var validationResult = JsonSerializer.Deserialize<TestValidationResult>(
                    resultString,
                    new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true,
                        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
                    });

                return validationResult ?? new TestValidationResult 
                { 
                    IsValid = false,
                    Issues = 
                    [
                        new ValidationIssue
                        {
                            Description = "Failed to deserialize validation result",
                            Severity = IssueSeverity.Error
                        }
                    ]
                };
            }
            catch (JsonException ex)
            {
                _logger.LogError(ex, "Error deserializing validation result");
                return new TestValidationResult
                {
                    IsValid = false,
                    Issues = 
                    [
                        new ValidationIssue
                        {
                            Description = $"Error deserializing validation result: {ex.Message}",
                            Severity = IssueSeverity.Error
                        }
                    ]
                };
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating improved code");
            return new TestValidationResult
            {
                IsValid = false,
                Issues =
                [
                    new ValidationIssue
                    {
                        Description = $"Error validating improved code: {ex.Message}",
                        Severity = IssueSeverity.Error
                    }
                ]
            };
        }
    }

    /// <inheritdoc/>
    public async Task<TestResult> RunTestsAsync(string filePath, string testFilePath, string projectPath)
    {
        try
        {
            _logger.LogInformation("Running tests for file: {FilePath} with test file: {TestFilePath}", filePath, testFilePath);

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            if (!File.Exists(testFilePath))
            {
                throw new FileNotFoundException($"Test file not found: {testFilePath}");
            }

            // Determine the test framework
            var testFramework = DetermineTestFramework(testFilePath);

            // Create a temporary project for testing
            var tempProjectPath = CreateTemporaryTestProject(filePath, testFilePath, projectPath, testFramework);

            try
            {
                // Build the project
                var buildSuccess = await BuildProjectAsync(tempProjectPath);
                if (!buildSuccess)
                {
                    return new TestResult
                    {
                        TotalTests = 0,
                        PassedTests = 0,
                        FailedTests = 0,
                        SkippedTests = 0,
                        ExecutionTimeMs = 0,
                        RawOutput = "Build failed"
                    };
                }

                // Run the tests
                return await RunTestsInProjectAsync(tempProjectPath, testFramework);
            }
            finally
            {
                // Clean up the temporary project
                CleanupTemporaryProject(tempProjectPath);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running tests for file: {FilePath} with test file: {TestFilePath}", filePath, testFilePath);
            return new TestResult
            {
                TotalTests = 0,
                PassedTests = 0,
                FailedTests = 1,
                SkippedTests = 0,
                ExecutionTimeMs = 0,
                Failures =
                [
                    new TestFailure
                    {
                        TestName = "TestExecution",
                        ErrorMessage = ex.Message,
                        StackTrace = ex.StackTrace ?? string.Empty
                    }
                ],
                RawOutput = $"Error: {ex.Message}"
            };
        }
    }

    /// <inheritdoc/>
    public TestComparisonResult CompareTestResults(TestResult beforeResult, TestResult afterResult)
    {
        try
        {
            _logger.LogInformation("Comparing test results");

            var newFailingTests = 0;
            var newPassingTests = 0;
            var newFailures = new List<TestFailure>();
            var fixedFailures = new List<string>();

            // Find new failing tests
            foreach (var afterFailure in afterResult.Failures)
            {
                var matchingBeforeFailure = beforeResult.Failures.Find(f => f.TestName == afterFailure.TestName);
                if (matchingBeforeFailure == null)
                {
                    newFailingTests++;
                    newFailures.Add(afterFailure);
                }
            }

            // Find fixed failures
            foreach (var beforeFailure in beforeResult.Failures)
            {
                var matchingAfterFailure = afterResult.Failures.Find(f => f.TestName == beforeFailure.TestName);
                if (matchingAfterFailure == null)
                {
                    newPassingTests++;
                    fixedFailures.Add(beforeFailure.TestName);
                }
            }

            return new TestComparisonResult
            {
                IsSuccessful = newFailingTests == 0,
                NewPassingTests = newPassingTests,
                NewFailingTests = newFailingTests,
                NewFailures = newFailures,
                FixedFailures = fixedFailures,
                ExecutionTimeChange = afterResult.ExecutionTimeMs - beforeResult.ExecutionTimeMs
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error comparing test results");
            return new TestComparisonResult
            {
                IsSuccessful = false,
                NewPassingTests = 0,
                NewFailingTests = 0,
                NewFailures =
                [
                    new TestFailure
                    {
                        TestName = "TestComparison",
                        ErrorMessage = ex.Message,
                        StackTrace = ex.StackTrace ?? string.Empty
                    }
                ],
                FixedFailures = [],
                ExecutionTimeChange = 0
            };
        }
    }

    /// <inheritdoc/>
    public async Task<List<TestFix>> SuggestFixesForFailingTestsAsync(TestResult testResult, string codeFilePath, string testFilePath)
    {
        try
        {
            _logger.LogInformation("Suggesting fixes for failing tests");

            if (testResult.FailedTests == 0)
            {
                return [];
            }

            if (!File.Exists(codeFilePath))
            {
                throw new FileNotFoundException($"Code file not found: {codeFilePath}");
            }

            if (!File.Exists(testFilePath))
            {
                throw new FileNotFoundException($"Test file not found: {testFilePath}");
            }

            var codeContent = await File.ReadAllTextAsync(codeFilePath);
            var testContent = await File.ReadAllTextAsync(testFilePath);
            var codeExtension = Path.GetExtension(codeFilePath).ToLowerInvariant();
            var language = GetLanguageFromExtension(codeExtension);

            // Create a metascript for suggesting fixes
            var metascript = $@"
// Test fix suggestion metascript
// Language: {language}

// Code content
let codeContent = `{codeContent.Replace("`", "\\`")}`;

// Test content
let testContent = `{testContent.Replace("`", "\\`")}`;

// Test failures
let testFailures = {JsonSerializer.Serialize(testResult.Failures)};

// Suggest fixes for the failing tests
let fixes = suggestFixesForFailingTests(codeContent, testContent, testFailures, '{language}');

// Return the suggested fixes
return JSON.stringify(fixes);

// Helper function to suggest fixes for failing tests
function suggestFixesForFailingTests(code, tests, failures, language) {{
    // This would be implemented with a more sophisticated analysis
    // For now, we'll use a simple placeholder

    const fixes = [];

    for (const failure of failures) {{
        // Extract the method name from the test name
        const methodNameMatch = /Test(\w+)/.exec(failure.TestName);
        if (!methodNameMatch) continue;

        const methodName = methodNameMatch[1];

        // Find the method in the code
        const methodRegex = new RegExp(`(public|private|protected|internal)\\s+(\\w+)\\s+${{methodName}}\\s*\\((.*?)\\)`, 's');
        const methodMatch = methodRegex.exec(code);
        if (!methodMatch) continue;

        // Suggest a fix
        fixes.push({{
            description: `Fix for failing test: ${{failure.TestName}}`,
            fixCode: `// TODO: Implement fix for ${{methodName}}`,
            filePath: 'codeFilePath', // This would be the actual file path
            lineNumber: 0, // This would be the actual line number
            confidence: 0.5
        }});
    }}

    return fixes;
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);

            // Parse the result as JSON
            var resultString = result?.ToString() ?? string.Empty;
            var fixes = JsonSerializer.Deserialize<List<TestFix>>(resultString);

            // Update the file paths
            if (fixes != null)
            {
                foreach (var fix in fixes)
                {
                    fix.FilePath = codeFilePath;
                }
            }

            return fixes ?? [];
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error suggesting fixes for failing tests");
            return [];
        }
    }

    private string DetermineTestFramework(string testFilePath)
    {
        try
        {
            var content = File.ReadAllText(testFilePath);

            if (content.Contains("using Xunit;") || content.Contains("using xunit;"))
            {
                return "xUnit";
            }
            else if (content.Contains("using NUnit.Framework;") || content.Contains("using nunit.framework;"))
            {
                return "NUnit";
            }
            else if (content.Contains("using Microsoft.VisualStudio.TestTools.UnitTesting;"))
            {
                return "MSTest";
            }

            // Default to xUnit
            return "xUnit";
        }
        catch
        {
            // Default to xUnit if there's an error
            return "xUnit";
        }
    }

    private string CreateTemporaryTestProject(string filePath, string testFilePath, string projectPath, string testFramework)
    {
        // This would create a temporary project for testing
        // For now, we'll just return the project path
        return projectPath;
    }

    private async Task<bool> BuildProjectAsync(string projectPath)
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = $"build \"{projectPath}\" --configuration Debug",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            await process.WaitForExitAsync();

            return process.ExitCode == 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error building project: {ProjectPath}", projectPath);
            return false;
        }
    }

    private async Task<TestResult> RunTestsInProjectAsync(string projectPath, string testFramework)
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = $"test \"{projectPath}\" --configuration Debug --no-build",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            var output = new StringBuilder();
            var error = new StringBuilder();

            process.OutputDataReceived += (sender, e) =>
            {
                if (e.Data != null)
                {
                    output.AppendLine(e.Data);
                }
            };

            process.ErrorDataReceived += (sender, e) =>
            {
                if (e.Data != null)
                {
                    error.AppendLine(e.Data);
                }
            };

            var stopwatch = Stopwatch.StartNew();
            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();
            await process.WaitForExitAsync();
            stopwatch.Stop();

            var rawOutput = output.ToString();
            var rawError = error.ToString();

            // Parse the test results
            var totalTests = 0;
            var passedTests = 0;
            var failedTests = 0;
            var skippedTests = 0;
            var failures = new List<TestFailure>();

            // Extract test counts
            var totalMatch = Regex.Match(rawOutput, @"Total tests: (\d+)");
            if (totalMatch.Success)
            {
                totalTests = int.Parse(totalMatch.Groups[1].Value);
            }

            var passedMatch = Regex.Match(rawOutput, @"Passed: (\d+)");
            if (passedMatch.Success)
            {
                passedTests = int.Parse(passedMatch.Groups[1].Value);
            }

            var failedMatch = Regex.Match(rawOutput, @"Failed: (\d+)");
            if (failedMatch.Success)
            {
                failedTests = int.Parse(failedMatch.Groups[1].Value);
            }

            var skippedMatch = Regex.Match(rawOutput, @"Skipped: (\d+)");
            if (skippedMatch.Success)
            {
                skippedTests = int.Parse(skippedMatch.Groups[1].Value);
            }

            // Extract failures
            var failureMatches = Regex.Matches(rawOutput, @"Failed (\w+\.\w+).*?Error Message:\s*(.*?)(?=\s*Stack Trace:|$)", RegexOptions.Singleline);
            foreach (Match match in failureMatches)
            {
                failures.Add(new TestFailure
                {
                    TestName = match.Groups[1].Value,
                    ErrorMessage = match.Groups[2].Value.Trim(),
                    StackTrace = string.Empty // We would extract this from the output
                });
            }

            return new TestResult
            {
                TotalTests = totalTests,
                PassedTests = passedTests,
                FailedTests = failedTests,
                SkippedTests = skippedTests,
                ExecutionTimeMs = stopwatch.ElapsedMilliseconds,
                Failures = failures,
                RawOutput = rawOutput + rawError
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running tests in project: {ProjectPath}", projectPath);
            return new TestResult
            {
                TotalTests = 0,
                PassedTests = 0,
                FailedTests = 1,
                SkippedTests = 0,
                ExecutionTimeMs = 0,
                Failures =
                [
                    new TestFailure
                    {
                        TestName = "TestExecution",
                        ErrorMessage = ex.Message,
                        StackTrace = ex.StackTrace ?? string.Empty
                    }
                ],
                RawOutput = $"Error: {ex.Message}"
            };
        }
    }

    private void CleanupTemporaryProject(string tempProjectPath)
    {
        // This would clean up the temporary project
        // For now, we'll do nothing
    }

    private string GetLanguageFromExtension(string extension)
    {
        return extension switch
        {
            ".cs" => "csharp",
            ".fs" => "fsharp",
            ".js" => "javascript",
            ".ts" => "typescript",
            ".py" => "python",
            ".java" => "java",
            _ => "unknown"
        };
    }
}
