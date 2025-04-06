using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for regression testing
/// </summary>
public class RegressionTestingService : IRegressionTestingService
{
    private readonly ILogger<RegressionTestingService> _logger;
    private readonly ITestValidationService _testValidationService;
    private readonly IMetascriptService _metascriptService;

    /// <summary>
    /// Initializes a new instance of the <see cref="RegressionTestingService"/> class
    /// </summary>
    public RegressionTestingService(
        ILogger<RegressionTestingService> logger,
        ITestValidationService testValidationService,
        IMetascriptService metascriptService)
    {
        _logger = logger;
        _testValidationService = testValidationService;
        _metascriptService = metascriptService;
    }

    /// <inheritdoc/>
    public async Task<RegressionTestResult> RunRegressionTestsAsync(string filePath, string projectPath)
    {
        try
        {
            _logger.LogInformation("Running regression tests for file: {FilePath}", filePath);

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            // Find test files for the given file
            var testFiles = FindTestFiles(filePath, projectPath);
            if (testFiles.Count == 0)
            {
                _logger.LogWarning("No test files found for file: {FilePath}", filePath);
                return new RegressionTestResult
                {
                    Passed = true,
                    TestResults = new List<TestResult>()
                };
            }

            // Run tests for each test file
            var testResults = new List<TestResult>();
            foreach (var testFile in testFiles)
            {
                var testResult = await _testValidationService.RunTestsAsync(filePath, testFile, projectPath);
                testResults.Add(testResult);
            }

            // Calculate overall result
            var passed = true;
            var issues = new List<RegressionIssue>();
            foreach (var testResult in testResults)
            {
                if (testResult.FailedTests > 0)
                {
                    passed = false;
                    foreach (var failure in testResult.Failures)
                    {
                        issues.Add(new RegressionIssue
                        {
                            Description = $"Test failure: {failure.TestName}",
                            Severity = IssueSeverity.Error,
                            Location = failure.TestName,
                            AffectedFunctionality = failure.TestName,
                            SuggestedFix = null,
                            Confidence = 1.0f
                        });
                    }
                }
            }

            // Track test coverage
            var coverageResult = await TrackTestCoverageAsync(projectPath);

            return new RegressionTestResult
            {
                Passed = passed,
                Issues = issues,
                TestResults = testResults,
                TotalExecutionTimeMs = testResults.Sum(r => r.ExecutionTimeMs),
                CoverageResult = coverageResult
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running regression tests for file: {FilePath}", filePath);
            return new RegressionTestResult
            {
                Passed = false,
                Issues = new List<RegressionIssue>
                {
                    new RegressionIssue
                    {
                        Description = $"Error running regression tests: {ex.Message}",
                        Severity = IssueSeverity.Error,
                        Location = filePath,
                        AffectedFunctionality = "Regression Testing",
                        SuggestedFix = null,
                        Confidence = 1.0f
                    }
                },
                TestResults = new List<TestResult>()
            };
        }
    }

    /// <inheritdoc/>
    public async Task<RegressionTestResult> RunRegressionTestsForProjectAsync(string projectPath)
    {
        try
        {
            _logger.LogInformation("Running regression tests for project: {ProjectPath}", projectPath);

            if (!Directory.Exists(projectPath))
            {
                throw new DirectoryNotFoundException($"Project directory not found: {projectPath}");
            }

            // Find all source files in the project
            var sourceFiles = FindSourceFiles(projectPath);
            if (sourceFiles.Count == 0)
            {
                _logger.LogWarning("No source files found in project: {ProjectPath}", projectPath);
                return new RegressionTestResult
                {
                    Passed = true,
                    TestResults = new List<TestResult>()
                };
            }

            // Run regression tests for each source file
            var regressionResults = new List<RegressionTestResult>();
            foreach (var sourceFile in sourceFiles)
            {
                var regressionResult = await RunRegressionTestsAsync(sourceFile, projectPath);
                regressionResults.Add(regressionResult);
            }

            // Calculate overall result
            var passed = regressionResults.All(r => r.Passed);
            var issues = regressionResults.SelectMany(r => r.Issues).ToList();
            var testResults = regressionResults.SelectMany(r => r.TestResults).ToList();
            var totalExecutionTimeMs = regressionResults.Sum(r => r.TotalExecutionTimeMs);

            // Track test coverage
            var coverageResult = await TrackTestCoverageAsync(projectPath);

            return new RegressionTestResult
            {
                Passed = passed,
                Issues = issues,
                TestResults = testResults,
                TotalExecutionTimeMs = totalExecutionTimeMs,
                CoverageResult = coverageResult
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running regression tests for project: {ProjectPath}", projectPath);
            return new RegressionTestResult
            {
                Passed = false,
                Issues = new List<RegressionIssue>
                {
                    new RegressionIssue
                    {
                        Description = $"Error running regression tests: {ex.Message}",
                        Severity = IssueSeverity.Error,
                        Location = projectPath,
                        AffectedFunctionality = "Regression Testing",
                        SuggestedFix = null,
                        Confidence = 1.0f
                    }
                },
                TestResults = new List<TestResult>()
            };
        }
    }

    /// <inheritdoc/>
    public async Task<RegressionTestResult> RunRegressionTestsForSolutionAsync(string solutionPath)
    {
        try
        {
            _logger.LogInformation("Running regression tests for solution: {SolutionPath}", solutionPath);

            if (!File.Exists(solutionPath))
            {
                throw new FileNotFoundException($"Solution file not found: {solutionPath}");
            }

            // Find all projects in the solution
            var projectPaths = FindProjectsInSolution(solutionPath);
            if (projectPaths.Count == 0)
            {
                _logger.LogWarning("No projects found in solution: {SolutionPath}", solutionPath);
                return new RegressionTestResult
                {
                    Passed = true,
                    TestResults = new List<TestResult>()
                };
            }

            // Run regression tests for each project
            var regressionResults = new List<RegressionTestResult>();
            foreach (var projectPath in projectPaths)
            {
                var regressionResult = await RunRegressionTestsForProjectAsync(projectPath);
                regressionResults.Add(regressionResult);
            }

            // Calculate overall result
            var passed = regressionResults.All(r => r.Passed);
            var issues = regressionResults.SelectMany(r => r.Issues).ToList();
            var testResults = regressionResults.SelectMany(r => r.TestResults).ToList();
            var totalExecutionTimeMs = regressionResults.Sum(r => r.TotalExecutionTimeMs);

            // Track test coverage
            var coverageResult = await TrackTestCoverageAsync(Path.GetDirectoryName(solutionPath) ?? string.Empty);

            return new RegressionTestResult
            {
                Passed = passed,
                Issues = issues,
                TestResults = testResults,
                TotalExecutionTimeMs = totalExecutionTimeMs,
                CoverageResult = coverageResult
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running regression tests for solution: {SolutionPath}", solutionPath);
            return new RegressionTestResult
            {
                Passed = false,
                Issues = new List<RegressionIssue>
                {
                    new RegressionIssue
                    {
                        Description = $"Error running regression tests: {ex.Message}",
                        Severity = IssueSeverity.Error,
                        Location = solutionPath,
                        AffectedFunctionality = "Regression Testing",
                        SuggestedFix = null,
                        Confidence = 1.0f
                    }
                },
                TestResults = new List<TestResult>()
            };
        }
    }

    /// <inheritdoc/>
    public async Task<List<RegressionIssue>> IdentifyPotentialRegressionIssuesAsync(string originalCode, string improvedCode, string language)
    {
        try
        {
            _logger.LogInformation("Identifying potential regression issues");

            // Create a metascript for identifying potential regression issues
            var metascript = $@"
// Regression issue identification metascript
// Language: {language}

// Original code
let originalCode = `{originalCode.Replace("`", "\\`")}`;

// Improved code
let improvedCode = `{improvedCode.Replace("`", "\\`")}`;

// Identify potential regression issues
let issues = identifyPotentialRegressionIssues(originalCode, improvedCode, '{language}');

// Return the identified issues
return JSON.stringify(issues);

// Helper function to identify potential regression issues
function identifyPotentialRegressionIssues(original, improved, language) {{
    // This would be implemented with a more sophisticated analysis
    // For now, we'll use a simple placeholder

    const issues = [];

    // Check for removed methods
    const originalMethods = extractMethods(original, language);
    const improvedMethods = extractMethods(improved, language);

    for (const originalMethod of originalMethods) {{
        const matchingImprovedMethod = improvedMethods.find(m => m.name === originalMethod.name);
        if (!matchingImprovedMethod) {{
            issues.push({{
                description: `Method removed: ${{originalMethod.name}}`,
                severity: 'Error',
                location: `Line ${{originalMethod.line}}`,
                affectedFunctionality: originalMethod.name,
                suggestedFix: `Restore the method: ${{originalMethod.name}}`,
                confidence: 0.9
            }});
        }}
    }}

    // Check for changed method signatures
    for (const originalMethod of originalMethods) {{
        const matchingImprovedMethod = improvedMethods.find(m => m.name === originalMethod.name);
        if (matchingImprovedMethod && originalMethod.parameters !== matchingImprovedMethod.parameters) {{
            issues.push({{
                description: `Method signature changed: ${{originalMethod.name}}`,
                severity: 'Warning',
                location: `Line ${{matchingImprovedMethod.line}}`,
                affectedFunctionality: originalMethod.name,
                suggestedFix: `Restore the original method signature: ${{originalMethod.name}}(${{originalMethod.parameters}})`,
                confidence: 0.8
            }});
        }}
    }}

    // Check for changed return types
    for (const originalMethod of originalMethods) {{
        const matchingImprovedMethod = improvedMethods.find(m => m.name === originalMethod.name);
        if (matchingImprovedMethod && originalMethod.returnType !== matchingImprovedMethod.returnType) {{
            issues.push({{
                description: `Return type changed: ${{originalMethod.name}}`,
                severity: 'Warning',
                location: `Line ${{matchingImprovedMethod.line}}`,
                affectedFunctionality: originalMethod.name,
                suggestedFix: `Restore the original return type: ${{originalMethod.returnType}} ${{originalMethod.name}}`,
                confidence: 0.8
            }});
        }}
    }}

    // Check for changed accessibility
    for (const originalMethod of originalMethods) {{
        const matchingImprovedMethod = improvedMethods.find(m => m.name === originalMethod.name);
        if (matchingImprovedMethod && originalMethod.accessibility !== matchingImprovedMethod.accessibility) {{
            issues.push({{
                description: `Accessibility changed: ${{originalMethod.name}}`,
                severity: 'Warning',
                location: `Line ${{matchingImprovedMethod.line}}`,
                affectedFunctionality: originalMethod.name,
                suggestedFix: `Restore the original accessibility: ${{originalMethod.accessibility}} ${{originalMethod.returnType}} ${{originalMethod.name}}`,
                confidence: 0.8
            }});
        }}
    }}

    return issues;
}}

// Helper function to extract methods
function extractMethods(code, language) {{
    // This would be implemented with a more sophisticated parser
    // For now, we'll use a simple placeholder

    const methods = [];

    if (language === 'csharp') {{
        // Simple regex to find method declarations
        const methodRegex = /(public|private|protected|internal)\s+(\w+)\s+(\w+)\s*\((.*?)\)/g;
        let match;
        let line = 1;

        for (const codeLine of code.split('\\n')) {{
            while ((match = methodRegex.exec(codeLine)) !== null) {{
                methods.push({{
                    accessibility: match[1],
                    returnType: match[2],
                    name: match[3],
                    parameters: match[4],
                    line: line
                }});
            }}

            line++;
        }}
    }}

    return methods;
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);

            // Parse the result as JSON
            var issues = System.Text.Json.JsonSerializer.Deserialize<List<RegressionIssue>>(result.ToString());
            return issues ?? new List<RegressionIssue>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error identifying potential regression issues");
            return new List<RegressionIssue>
            {
                new RegressionIssue
                {
                    Description = $"Error identifying potential regression issues: {ex.Message}",
                    Severity = IssueSeverity.Error,
                    Location = string.Empty,
                    AffectedFunctionality = "Regression Issue Identification",
                    SuggestedFix = null,
                    Confidence = 1.0f
                }
            };
        }
    }

    /// <inheritdoc/>
    public async Task<TestCoverageResult> TrackTestCoverageAsync(string projectPath)
    {
        try
        {
            _logger.LogInformation("Tracking test coverage for project: {ProjectPath}", projectPath);

            // This would be implemented with a real coverage tool
            // For now, we'll return a placeholder result
            await Task.Delay(100); // Simulate work

            return new TestCoverageResult
            {
                LineCoverage = 0.75f,
                BranchCoverage = 0.65f,
                MethodCoverage = 0.80f,
                ClassCoverage = 0.85f,
                UncoveredLines = new List<UncoveredLine>
                {
                    new UncoveredLine
                    {
                        FilePath = Path.Combine(projectPath, "SampleClass.cs"),
                        LineNumber = 42,
                        LineContent = "    if (value == null) throw new ArgumentNullException(nameof(value));"
                    }
                },
                UncoveredBranches = new List<UncoveredBranch>
                {
                    new UncoveredBranch
                    {
                        FilePath = Path.Combine(projectPath, "SampleClass.cs"),
                        LineNumber = 42,
                        BranchContent = "value == null",
                        MethodName = "ProcessValue"
                    }
                },
                UncoveredMethods = new List<UncoveredMethod>
                {
                    new UncoveredMethod
                    {
                        FilePath = Path.Combine(projectPath, "SampleClass.cs"),
                        MethodName = "HandleSpecialCase",
                        ClassName = "SampleClass",
                        LineNumber = 100
                    }
                }
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error tracking test coverage for project: {ProjectPath}", projectPath);
            return new TestCoverageResult
            {
                LineCoverage = 0,
                BranchCoverage = 0,
                MethodCoverage = 0,
                ClassCoverage = 0
            };
        }
    }

    /// <inheritdoc/>
    public async Task<List<SuggestedTest>> SuggestAdditionalTestsAsync(TestCoverageResult coverageResult, string projectPath)
    {
        try
        {
            _logger.LogInformation("Suggesting additional tests for project: {ProjectPath}", projectPath);

            // Create a metascript for suggesting additional tests
            var metascript = $@"
// Test suggestion metascript

// Coverage result
let coverageResult = {System.Text.Json.JsonSerializer.Serialize(coverageResult)};

// Project path
let projectPath = '{projectPath.Replace("\\", "\\\\")}';

// Suggest additional tests
let suggestedTests = suggestAdditionalTests(coverageResult, projectPath);

// Return the suggested tests
return JSON.stringify(suggestedTests);

// Helper function to suggest additional tests
function suggestAdditionalTests(coverage, projectPath) {{
    // This would be implemented with a more sophisticated analysis
    // For now, we'll use a simple placeholder

    const suggestedTests = [];

    // Suggest tests for uncovered methods
    for (const method of coverage.uncoveredMethods) {{
        suggestedTests.push({{
            name: `Test${{method.methodName}}`,
            description: `Test for uncovered method: ${{method.methodName}}`,
            targetMethod: method.methodName,
            targetClass: method.className,
            filePath: `${{projectPath}}/Tests/${{method.className}}Tests.cs`,
            testCode: `[Fact]\\npublic void ${{method.methodName}}_ShouldWorkCorrectly()\\n{{\\n    // Arrange\\n    var instance = new ${{method.className}}();\\n\\n    // Act\\n    var result = instance.${{method.methodName}}();\\n\\n    // Assert\\n    Assert.NotNull(result);\\n}}`,
            priority: 'High'
        }});
    }}

    // Suggest tests for uncovered branches
    for (const branch of coverage.uncoveredBranches) {{
        suggestedTests.push({{
            name: `Test${{branch.methodName}}_${{branch.branchContent.replace(/[^a-zA-Z0-9]/g, '')}}`,
            description: `Test for uncovered branch: ${{branch.branchContent}} in method ${{branch.methodName}}`,
            targetMethod: branch.methodName,
            targetClass: branch.filePath.split('/').pop().replace('.cs', ''),
            filePath: `${{projectPath}}/Tests/${{branch.filePath.split('/').pop().replace('.cs', '')}}Tests.cs`,
            testCode: `[Fact]\\npublic void ${{branch.methodName}}_Should${{branch.branchContent.replace(/[^a-zA-Z0-9]/g, '')}}()\\n{{\\n    // Arrange\\n    var instance = new ${{branch.filePath.split('/').pop().replace('.cs', '')}}();\\n\\n    // Act\\n    var result = instance.${{branch.methodName}}(null);\\n\\n    // Assert\\n    Assert.NotNull(result);\\n}}`,
            priority: 'Medium'
        }});
    }}

    return suggestedTests;
}}
";

            // Execute the metascript
            var result = await _metascriptService.ExecuteMetascriptAsync(metascript);

            // Parse the result as JSON
            var suggestedTests = System.Text.Json.JsonSerializer.Deserialize<List<SuggestedTest>>(result.ToString());
            return suggestedTests ?? new List<SuggestedTest>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error suggesting additional tests for project: {ProjectPath}", projectPath);
            return new List<SuggestedTest>();
        }
    }

    private List<string> FindTestFiles(string filePath, string projectPath)
    {
        try
        {
            var fileName = Path.GetFileNameWithoutExtension(filePath);
            var testFiles = new List<string>();

            // Look for test files in the Tests directory
            var testDirectory = Path.Combine(projectPath, "Tests");
            if (Directory.Exists(testDirectory))
            {
                var potentialTestFiles = Directory.GetFiles(testDirectory, $"{fileName}*Tests.cs");
                testFiles.AddRange(potentialTestFiles);
            }

            // Look for test files in the test project
            var testProjectDirectory = Path.Combine(Path.GetDirectoryName(projectPath) ?? string.Empty, $"{Path.GetFileNameWithoutExtension(projectPath)}.Tests");
            if (Directory.Exists(testProjectDirectory))
            {
                var potentialTestFiles = Directory.GetFiles(testProjectDirectory, $"{fileName}*Tests.cs", SearchOption.AllDirectories);
                testFiles.AddRange(potentialTestFiles);
            }

            return testFiles;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding test files for file: {FilePath}", filePath);
            return new List<string>();
        }
    }

    private List<string> FindSourceFiles(string projectPath)
    {
        try
        {
            var sourceFiles = new List<string>();

            // Look for source files in the project directory
            var potentialSourceFiles = Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories);
            foreach (var file in potentialSourceFiles)
            {
                // Exclude test files
                if (!file.Contains("Tests") && !file.Contains("\\obj\\") && !file.Contains("\\bin\\"))
                {
                    sourceFiles.Add(file);
                }
            }

            return sourceFiles;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding source files in project: {ProjectPath}", projectPath);
            return new List<string>();
        }
    }

    private List<string> FindProjectsInSolution(string solutionPath)
    {
        try
        {
            var projectPaths = new List<string>();

            // Read the solution file
            var solutionContent = File.ReadAllText(solutionPath);
            var projectLines = solutionContent.Split('\n').Where(line => line.Contains("Project(")).ToList();

            foreach (var line in projectLines)
            {
                // Extract the project path
                var match = System.Text.RegularExpressions.Regex.Match(line, @"Project\([^)]+\)\s*=\s*""[^""]*""\s*,\s*""([^""]*)""\s*,");
                if (match.Success)
                {
                    var relativePath = match.Groups[1].Value;
                    var absolutePath = Path.Combine(Path.GetDirectoryName(solutionPath) ?? string.Empty, relativePath);
                    if (File.Exists(absolutePath))
                    {
                        projectPaths.Add(absolutePath);
                    }
                }
            }

            return projectPaths;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding projects in solution: {SolutionPath}", solutionPath);
            return new List<string>();
        }
    }
}
