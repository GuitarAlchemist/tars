using System;
using System.CommandLine;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for testing framework operations
/// </summary>
public class TestingFrameworkCommand : TarsCommand
{
    private readonly IServiceProvider? _serviceProvider;
    private readonly ConsoleService _consoleService;

    /// <summary>
    /// Initializes a new instance of the <see cref="TestingFrameworkCommand"/> class
    /// </summary>
    public TestingFrameworkCommand(IServiceProvider? serviceProvider) : base("testing", "Testing framework operations")
    {
        _consoleService = new ConsoleService();
        _serviceProvider = serviceProvider;

        // Add generate-tests command
        var generateTestsCommand = new Command("generate-tests", "Generate tests for a file");
        var fileOption = new Option<string>("--file", "Path to the file to generate tests for") { IsRequired = true };
        var projectOption = new Option<string>("--project", "Path to the project containing the file");
        var frameworkOption = new Option<string>("--framework", "Test framework to use (xUnit, NUnit, MSTest)") { IsRequired = false };
        frameworkOption.SetDefaultValue("xUnit");
        generateTestsCommand.AddOption(fileOption);
        generateTestsCommand.AddOption(projectOption);
        generateTestsCommand.AddOption(frameworkOption);
        generateTestsCommand.SetHandler(GenerateTests, fileOption, projectOption, frameworkOption);
        AddCommand(generateTestsCommand);

        // Add validate-tests command
        var validateTestsCommand = new Command("validate-tests", "Validate tests for a file");
        var codeFileOption = new Option<string>("--code-file", "Path to the code file") { IsRequired = true };
        var testFileOption = new Option<string>("--test-file", "Path to the test file") { IsRequired = true };
        var projectOption2 = new Option<string>("--project", "Path to the project containing the files");
        validateTestsCommand.AddOption(codeFileOption);
        validateTestsCommand.AddOption(testFileOption);
        validateTestsCommand.AddOption(projectOption2);
        validateTestsCommand.SetHandler(ValidateTests, codeFileOption, testFileOption, projectOption2);
        AddCommand(validateTestsCommand);

        // Add analyze-quality command
        var analyzeQualityCommand = new Command("analyze-quality", "Analyze code quality");
        var fileOption2 = new Option<string>("--file", "Path to the file to analyze") { IsRequired = true };
        var languageOption = new Option<string>("--language", "Programming language");
        analyzeQualityCommand.AddOption(fileOption2);
        analyzeQualityCommand.AddOption(languageOption);
        analyzeQualityCommand.SetHandler(AnalyzeQuality, fileOption2, languageOption);
        AddCommand(analyzeQualityCommand);

        // Add analyze-complexity command
        var analyzeComplexityCommand = new Command("analyze-complexity", "Analyze code complexity");
        var fileOption3 = new Option<string>("--file", "Path to the file to analyze") { IsRequired = true };
        var languageOption2 = new Option<string>("--language", "Programming language");
        var thresholdOption = new Option<int>("--threshold", "Complexity threshold");
        thresholdOption.SetDefaultValue(10);
        analyzeComplexityCommand.AddOption(fileOption3);
        analyzeComplexityCommand.AddOption(languageOption2);
        analyzeComplexityCommand.AddOption(thresholdOption);
        analyzeComplexityCommand.SetHandler(AnalyzeComplexity, fileOption3, languageOption2, thresholdOption);
        AddCommand(analyzeComplexityCommand);

        // Add analyze-readability command
        var analyzeReadabilityCommand = new Command("analyze-readability", "Analyze code readability");
        var fileOption4 = new Option<string>("--file", "Path to the file to analyze") { IsRequired = true };
        var languageOption3 = new Option<string>("--language", "Programming language");
        analyzeReadabilityCommand.AddOption(fileOption4);
        analyzeReadabilityCommand.AddOption(languageOption3);
        analyzeReadabilityCommand.SetHandler(AnalyzeReadability, fileOption4, languageOption3);
        AddCommand(analyzeReadabilityCommand);
    }

    private async Task GenerateTests(string file, string? project, string framework)
    {
        try
        {
            _consoleService.WriteHeader("Generate Tests");

            if (_serviceProvider == null)
            {
                _consoleService.WriteColorLine("Service provider is not available", ConsoleColor.Red);
                return;
            }

            var testGenerationService = _serviceProvider.GetRequiredService<ITestGenerationService>();
            var logger = _serviceProvider.GetRequiredService<ILogger<TestingFrameworkCommand>>();

            // Resolve the project path
            string projectPath = ResolveProjectPath(file, project);

            // Generate tests
            logger.LogInformation("Generating tests for file: {File} in project: {Project} with framework: {Framework}", file, projectPath, framework);
            var testCode = await testGenerationService.GenerateTestsForFileAsync(file, projectPath, framework);

            // Determine the output file path
            string fileName = Path.GetFileNameWithoutExtension(file);
            string testFileName = $"{fileName}Tests.cs";
            string testDirectory = Path.Combine(projectPath, "Tests");
            if (!Directory.Exists(testDirectory))
            {
                Directory.CreateDirectory(testDirectory);
            }
            string testFilePath = Path.Combine(testDirectory, testFileName);

            // Write the test code to the file
            await File.WriteAllTextAsync(testFilePath, testCode);

            _consoleService.WriteColorLine($"Tests generated successfully: {testFilePath}", ConsoleColor.Green);
            Console.WriteLine();
            Console.WriteLine("Test code:");
            Console.WriteLine(testCode);
        }
        catch (Exception ex)
        {
            _consoleService.WriteColorLine($"Error generating tests: {ex.Message}", ConsoleColor.Red);
        }
    }

    private async Task ValidateTests(string codeFile, string testFile, string? project)
    {
        try
        {
            _consoleService.WriteHeader("Validate Tests");

            if (_serviceProvider == null)
            {
                _consoleService.WriteColorLine("Service provider is not available", ConsoleColor.Red);
                return;
            }

            var testValidationService = _serviceProvider.GetRequiredService<ITestValidationService>();
            var logger = _serviceProvider.GetRequiredService<ILogger<TestingFrameworkCommand>>();

            // Resolve the project path
            string projectPath = ResolveProjectPath(codeFile, project);

            // Run the tests
            logger.LogInformation("Running tests for file: {CodeFile} with test file: {TestFile} in project: {Project}", codeFile, testFile, projectPath);
            var testResult = await testValidationService.RunTestsAsync(codeFile, testFile, projectPath);

            // Display the test results
            _consoleService.WriteColorLine($"Test Results:", ConsoleColor.Cyan);
            _consoleService.WriteColorLine($"Total Tests: {testResult.TotalTests}", ConsoleColor.White);
            _consoleService.WriteColorLine($"Passed Tests: {testResult.PassedTests}", ConsoleColor.Green);
            _consoleService.WriteColorLine($"Failed Tests: {testResult.FailedTests}", ConsoleColor.Red);
            _consoleService.WriteColorLine($"Skipped Tests: {testResult.SkippedTests}", ConsoleColor.Yellow);
            _consoleService.WriteColorLine($"Execution Time: {testResult.ExecutionTimeMs}ms", ConsoleColor.White);

            if (testResult.FailedTests > 0)
            {
                _consoleService.WriteColorLine("Failures:", ConsoleColor.Red);
                foreach (var failure in testResult.Failures)
                {
                    _consoleService.WriteColorLine($"  {failure.TestName}: {failure.ErrorMessage}", ConsoleColor.Red);
                }

                // Suggest fixes for failing tests
                var fixes = await testValidationService.SuggestFixesForFailingTestsAsync(testResult, codeFile, testFile);
                if (fixes.Count > 0)
                {
                    _consoleService.WriteColorLine("Suggested Fixes:", ConsoleColor.Yellow);
                    foreach (var fix in fixes)
                    {
                        _consoleService.WriteColorLine($"  {fix.Description}", ConsoleColor.Yellow);
                        _consoleService.WriteColorLine($"    File: {fix.FilePath}", ConsoleColor.Yellow);
                        _consoleService.WriteColorLine($"    Line: {fix.LineNumber}", ConsoleColor.Yellow);
                        _consoleService.WriteColorLine($"    Fix: {fix.FixCode}", ConsoleColor.Yellow);
                        _consoleService.WriteColorLine($"    Confidence: {fix.Confidence:P0}", ConsoleColor.Yellow);
                    }
                }
            }
            else
            {
                _consoleService.WriteColorLine("All tests passed!", ConsoleColor.Green);
            }
        }
        catch (Exception ex)
        {
            _consoleService.WriteColorLine($"Error validating tests: {ex.Message}", ConsoleColor.Red);
        }
    }

    private async Task AnalyzeQuality(string file, string? language)
    {
        try
        {
            _consoleService.WriteHeader("Analyze Code Quality");

            if (_serviceProvider == null)
            {
                _consoleService.WriteColorLine("Service provider is not available", ConsoleColor.Red);
                return;
            }

            var codeQualityService = _serviceProvider.GetRequiredService<ICodeQualityService>();
            var logger = _serviceProvider.GetRequiredService<ILogger<TestingFrameworkCommand>>();

            // Determine the language
            string lang = language ?? GetLanguageFromExtension(Path.GetExtension(file));

            // Analyze code quality
            logger.LogInformation("Analyzing code quality for file: {File} with language: {Language}", file, lang);
            var qualityResult = await codeQualityService.AnalyzeCodeQualityAsync(file, lang);

            // Display the quality results
            _consoleService.WriteColorLine($"Code Quality Results:", ConsoleColor.Cyan);
            _consoleService.WriteColorLine($"Overall Score: {qualityResult.OverallScore:F1}/100", GetScoreColor(qualityResult.OverallScore));
            _consoleService.WriteColorLine($"Maintainability Score: {qualityResult.MaintainabilityScore:F1}/100", GetScoreColor(qualityResult.MaintainabilityScore));
            _consoleService.WriteColorLine($"Reliability Score: {qualityResult.ReliabilityScore:F1}/100", GetScoreColor(qualityResult.ReliabilityScore));
            _consoleService.WriteColorLine($"Security Score: {qualityResult.SecurityScore:F1}/100", GetScoreColor(qualityResult.SecurityScore));
            _consoleService.WriteColorLine($"Performance Score: {qualityResult.PerformanceScore:F1}/100", GetScoreColor(qualityResult.PerformanceScore));

            if (qualityResult.Issues.Count > 0)
            {
                _consoleService.WriteColorLine("Issues:", ConsoleColor.Yellow);
                foreach (var issue in qualityResult.Issues)
                {
                    ConsoleColor color = ConsoleColor.White;
                    if (issue.Severity.ToString() == "Info")
                        color = ConsoleColor.Blue;
                    else if (issue.Severity.ToString() == "Warning")
                        color = ConsoleColor.Yellow;
                    else if (issue.Severity.ToString() == "Error")
                        color = ConsoleColor.Red;
                    else if (issue.Severity.ToString() == "Critical")
                        color = ConsoleColor.DarkRed;

                    _consoleService.WriteColorLine($"  [{issue.Severity}] {issue.Description}", color);
                    _consoleService.WriteColorLine($"    Location: {issue.Location}", color);
                    if (!string.IsNullOrEmpty(issue.SuggestedFix))
                    {
                        _consoleService.WriteColorLine($"    Suggested Fix: {issue.SuggestedFix}", color);
                    }
                }
            }
            else
            {
                _consoleService.WriteColorLine("No issues found!", ConsoleColor.Green);
            }

            // Display complexity metrics
            _consoleService.WriteColorLine("Complexity Metrics:", ConsoleColor.Cyan);
            _consoleService.WriteColorLine($"  Average Cyclomatic Complexity: {qualityResult.ComplexityMetrics.AverageCyclomaticComplexity:F1}", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Maximum Cyclomatic Complexity: {qualityResult.ComplexityMetrics.MaxCyclomaticComplexity}", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Average Cognitive Complexity: {qualityResult.ComplexityMetrics.AverageCognitiveComplexity:F1}", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Maximum Cognitive Complexity: {qualityResult.ComplexityMetrics.MaxCognitiveComplexity}", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Average Method Length: {qualityResult.ComplexityMetrics.AverageMethodLength:F1} lines", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Maximum Method Length: {qualityResult.ComplexityMetrics.MaxMethodLength} lines", ConsoleColor.White);

            // Display readability metrics
            _consoleService.WriteColorLine("Readability Metrics:", ConsoleColor.Cyan);
            _consoleService.WriteColorLine($"  Average Identifier Length: {qualityResult.ReadabilityMetrics.AverageIdentifierLength:F1} characters", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Comment Density: {qualityResult.ReadabilityMetrics.CommentDensity:F2} comments per line", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Documentation Coverage: {qualityResult.ReadabilityMetrics.DocumentationCoverage:P0}", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Average Parameter Count: {qualityResult.ReadabilityMetrics.AverageParameterCount:F1}", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Maximum Parameter Count: {qualityResult.ReadabilityMetrics.MaxParameterCount}", ConsoleColor.White);

            // Display duplication metrics
            _consoleService.WriteColorLine("Duplication Metrics:", ConsoleColor.Cyan);
            _consoleService.WriteColorLine($"  Duplication Percentage: {qualityResult.DuplicationMetrics.DuplicationPercentage:P0}", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Duplicated Blocks: {qualityResult.DuplicationMetrics.DuplicatedBlocks}", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Duplicated Lines: {qualityResult.DuplicationMetrics.DuplicatedLines}", ConsoleColor.White);
        }
        catch (Exception ex)
        {
            _consoleService.WriteColorLine($"Error analyzing code quality: {ex.Message}", ConsoleColor.Red);
        }
    }

    private async Task AnalyzeComplexity(string file, string? language, int threshold)
    {
        try
        {
            _consoleService.WriteHeader("Analyze Code Complexity");

            if (_serviceProvider == null)
            {
                _consoleService.WriteColorLine("Service provider is not available", ConsoleColor.Red);
                return;
            }

            var complexityAnalysisService = _serviceProvider.GetRequiredService<IComplexityAnalysisService>();
            var logger = _serviceProvider.GetRequiredService<ILogger<TestingFrameworkCommand>>();

            // Determine the language
            string lang = language ?? GetLanguageFromExtension(Path.GetExtension(file));

            // Analyze code complexity
            logger.LogInformation("Analyzing code complexity for file: {File} with language: {Language} and threshold: {Threshold}", file, lang, threshold);
            var complexityResult = await complexityAnalysisService.AnalyzeComplexityAsync(file, lang);

            // Display the complexity results
            _consoleService.WriteColorLine($"Code Complexity Results:", ConsoleColor.Cyan);
            _consoleService.WriteColorLine($"Average Cyclomatic Complexity: {complexityResult.AverageCyclomaticComplexity:F1}", ConsoleColor.White);
            _consoleService.WriteColorLine($"Maximum Cyclomatic Complexity: {complexityResult.MaxCyclomaticComplexity}", ConsoleColor.White);
            _consoleService.WriteColorLine($"Average Cognitive Complexity: {complexityResult.AverageCognitiveComplexity:F1}", ConsoleColor.White);
            _consoleService.WriteColorLine($"Maximum Cognitive Complexity: {complexityResult.MaxCognitiveComplexity}", ConsoleColor.White);
            _consoleService.WriteColorLine($"Average Halstead Complexity: {complexityResult.AverageHalsteadComplexity:F1}", ConsoleColor.White);
            _consoleService.WriteColorLine($"Maximum Halstead Complexity: {complexityResult.MaxHalsteadComplexity:F1}", ConsoleColor.White);
            _consoleService.WriteColorLine($"Average Maintainability Index: {complexityResult.AverageMaintainabilityIndex:F1}", ConsoleColor.White);
            _consoleService.WriteColorLine($"Minimum Maintainability Index: {complexityResult.MinMaintainabilityIndex:F1}", ConsoleColor.White);

            // Display complex methods
            if (complexityResult.ComplexMethods.Count > 0)
            {
                _consoleService.WriteColorLine("Complex Methods:", ConsoleColor.Yellow);
                foreach (var method in complexityResult.ComplexMethods)
                {
                    _consoleService.WriteColorLine($"  {method.ClassName}.{method.MethodName}", ConsoleColor.Yellow);
                    _consoleService.WriteColorLine($"    File: {method.FilePath}", ConsoleColor.Yellow);
                    _consoleService.WriteColorLine($"    Line: {method.LineNumber}", ConsoleColor.Yellow);
                    _consoleService.WriteColorLine($"    Cyclomatic Complexity: {method.CyclomaticComplexity}", ConsoleColor.Yellow);
                    _consoleService.WriteColorLine($"    Cognitive Complexity: {method.CognitiveComplexity}", ConsoleColor.Yellow);
                    _consoleService.WriteColorLine($"    Method Length: {method.MethodLength} lines", ConsoleColor.Yellow);
                }
            }
            else
            {
                _consoleService.WriteColorLine("No complex methods found!", ConsoleColor.Green);
            }

            // Identify complex code sections
            var complexSections = await complexityAnalysisService.IdentifyComplexCodeAsync(file, lang, threshold);
            if (complexSections.Count > 0)
            {
                _consoleService.WriteColorLine($"Complex Code Sections (Threshold: {threshold}):", ConsoleColor.Yellow);
                foreach (var section in complexSections)
                {
                    _consoleService.WriteColorLine($"  {section.ComplexityType} Complexity: {section.ComplexityValue}", ConsoleColor.Yellow);
                    _consoleService.WriteColorLine($"    File: {section.FilePath}", ConsoleColor.Yellow);
                    _consoleService.WriteColorLine($"    Lines: {section.StartLine}-{section.EndLine}", ConsoleColor.Yellow);
                    if (!string.IsNullOrEmpty(section.MethodName))
                    {
                        _consoleService.WriteColorLine($"    Method: {section.MethodName}", ConsoleColor.Yellow);
                    }
                    if (!string.IsNullOrEmpty(section.ClassName))
                    {
                        _consoleService.WriteColorLine($"    Class: {section.ClassName}", ConsoleColor.Yellow);
                    }

                    // Suggest simplifications
                    var simplifications = await complexityAnalysisService.SuggestSimplificationsAsync(section);
                    if (simplifications.Count > 0)
                    {
                        _consoleService.WriteColorLine("    Suggested Simplifications:", ConsoleColor.Green);
                        foreach (var simplification in simplifications)
                        {
                            _consoleService.WriteColorLine($"      {simplification.Description}", ConsoleColor.Green);
                            _consoleService.WriteColorLine($"        Complexity Reduction: {simplification.ComplexityReduction}", ConsoleColor.Green);
                            _consoleService.WriteColorLine($"        Confidence: {simplification.Confidence:P0}", ConsoleColor.Green);
                            if (simplification.PotentialRisks.Count > 0)
                            {
                                _consoleService.WriteColorLine("        Potential Risks:", ConsoleColor.Red);
                                foreach (var risk in simplification.PotentialRisks)
                                {
                                    _consoleService.WriteColorLine($"          {risk}", ConsoleColor.Red);
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                _consoleService.WriteColorLine($"No complex code sections found (Threshold: {threshold})!", ConsoleColor.Green);
            }
        }
        catch (Exception ex)
        {
            _consoleService.WriteColorLine($"Error analyzing code complexity: {ex.Message}", ConsoleColor.Red);
        }
    }

    private async Task AnalyzeReadability(string file, string? language)
    {
        try
        {
            _consoleService.WriteHeader("Analyze Code Readability");

            if (_serviceProvider == null)
            {
                _consoleService.WriteColorLine("Service provider is not available", ConsoleColor.Red);
                return;
            }

            var readabilityService = _serviceProvider.GetRequiredService<IReadabilityService>();
            var logger = _serviceProvider.GetRequiredService<ILogger<TestingFrameworkCommand>>();

            // Determine the language
            string lang = language ?? GetLanguageFromExtension(Path.GetExtension(file));

            // Analyze code readability
            logger.LogInformation("Analyzing code readability for file: {File} with language: {Language}", file, lang);
            var readabilityResult = await readabilityService.AnalyzeReadabilityAsync(file, lang);

            // Display the readability results
            _consoleService.WriteColorLine($"Code Readability Results:", ConsoleColor.Cyan);
            _consoleService.WriteColorLine($"Overall Score: {readabilityResult.OverallScore:F1}/100", GetScoreColor(readabilityResult.OverallScore));
            _consoleService.WriteColorLine($"Naming Convention Score: {readabilityResult.NamingConventionScore:F1}/100", GetScoreColor(readabilityResult.NamingConventionScore));
            _consoleService.WriteColorLine($"Comment Quality Score: {readabilityResult.CommentQualityScore:F1}/100", GetScoreColor(readabilityResult.CommentQualityScore));
            _consoleService.WriteColorLine($"Code Formatting Score: {readabilityResult.CodeFormattingScore:F1}/100", GetScoreColor(readabilityResult.CodeFormattingScore));
            _consoleService.WriteColorLine($"Documentation Score: {readabilityResult.DocumentationScore:F1}/100", GetScoreColor(readabilityResult.DocumentationScore));

            // Display readability metrics
            _consoleService.WriteColorLine("Readability Metrics:", ConsoleColor.Cyan);
            _consoleService.WriteColorLine($"  Average Identifier Length: {readabilityResult.Metrics.AverageIdentifierLength:F1} characters", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Comment Density: {readabilityResult.Metrics.CommentDensity:F2} comments per line", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Documentation Coverage: {readabilityResult.Metrics.DocumentationCoverage:P0}", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Average Parameter Count: {readabilityResult.Metrics.AverageParameterCount:F1}", ConsoleColor.White);
            _consoleService.WriteColorLine($"  Maximum Parameter Count: {readabilityResult.Metrics.MaxParameterCount}", ConsoleColor.White);

            // Display readability issues
            if (readabilityResult.Issues.Count > 0)
            {
                _consoleService.WriteColorLine("Readability Issues:", ConsoleColor.Yellow);
                foreach (var issue in readabilityResult.Issues)
                {
                    _consoleService.WriteColorLine($"  {issue.Description}", ConsoleColor.Yellow);
                    _consoleService.WriteColorLine($"    Location: {issue.Location}", ConsoleColor.Yellow);
                    if (!string.IsNullOrEmpty(issue.SuggestedFix))
                    {
                        _consoleService.WriteColorLine($"    Suggested Fix: {issue.SuggestedFix}", ConsoleColor.Yellow);
                    }
                    _consoleService.WriteColorLine($"    Score Impact: {issue.ScoreImpact:F1}", ConsoleColor.Yellow);
                }

                // Identify readability issues
                var readabilityIssues = await readabilityService.IdentifyReadabilityIssuesAsync(file, lang);
                if (readabilityIssues.Count > 0)
                {
                    _consoleService.WriteColorLine("Detailed Readability Issues:", ConsoleColor.Yellow);
                    foreach (var issue in readabilityIssues)
                    {
                        _consoleService.WriteColorLine($"  {issue.Description}", ConsoleColor.Yellow);
                        _consoleService.WriteColorLine($"    Location: {issue.Location}", ConsoleColor.Yellow);
                        if (!string.IsNullOrEmpty(issue.SuggestedFix))
                        {
                            _consoleService.WriteColorLine($"    Suggested Fix: {issue.SuggestedFix}", ConsoleColor.Yellow);
                        }
                        _consoleService.WriteColorLine($"    Score Impact: {issue.ScoreImpact:F1}", ConsoleColor.Yellow);

                        // Suggest improvements
                        var improvements = await readabilityService.SuggestReadabilityImprovementsAsync(issue);
                        if (improvements.Count > 0)
                        {
                            _consoleService.WriteColorLine("    Suggested Improvements:", ConsoleColor.Green);
                            foreach (var improvement in improvements)
                            {
                                _consoleService.WriteColorLine($"      {improvement.Description}", ConsoleColor.Green);
                                _consoleService.WriteColorLine($"        Category: {improvement.Category}", ConsoleColor.Green);
                                _consoleService.WriteColorLine($"        Readability Improvement: {improvement.ReadabilityScore:F1}", ConsoleColor.Green);
                                _consoleService.WriteColorLine($"        Confidence: {improvement.Confidence:P0}", ConsoleColor.Green);
                            }
                        }
                    }
                }
            }
            else
            {
                _consoleService.WriteColorLine("No readability issues found!", ConsoleColor.Green);
            }
        }
        catch (Exception ex)
        {
            _consoleService.WriteColorLine($"Error analyzing code readability: {ex.Message}", ConsoleColor.Red);
        }
    }

    private string ResolveProjectPath(string filePath, string? projectPath)
    {
        if (!string.IsNullOrEmpty(projectPath))
        {
            return projectPath;
        }

        // Try to find the project file in the same directory as the file
        string directory = Path.GetDirectoryName(filePath) ?? string.Empty;
        string[] projectFiles = Directory.GetFiles(directory, "*.csproj");
        if (projectFiles.Length > 0)
        {
            return projectFiles[0];
        }

        // Try to find the project file in the parent directory
        string? parentDirectory = Path.GetDirectoryName(directory);
        if (!string.IsNullOrEmpty(parentDirectory))
        {
            projectFiles = Directory.GetFiles(parentDirectory, "*.csproj");
            if (projectFiles.Length > 0)
            {
                return projectFiles[0];
            }
        }

        // If no project file is found, return the directory
        return directory;
    }

    private string GetLanguageFromExtension(string extension)
    {
        return extension.ToLowerInvariant() switch
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

    private ConsoleColor GetScoreColor(float score)
    {
        return score switch
        {
            >= 90 => ConsoleColor.Green,
            >= 70 => ConsoleColor.DarkGreen,
            >= 50 => ConsoleColor.Yellow,
            >= 30 => ConsoleColor.DarkYellow,
            _ => ConsoleColor.Red
        };
    }
}


