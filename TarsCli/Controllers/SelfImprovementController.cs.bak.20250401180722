using System;
using System.Collections.Generic;
using System.CommandLine;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsCli.Services;
using TarsEngine.Services;
using TarsEngine.Services.Interfaces;
using EngineSelfImprovementService = TarsEngine.Services.SelfImprovementService;
using ProgrammingLanguage = TarsEngine.Services.ProgrammingLanguage;

namespace TarsCli.Controllers
{
    /// <summary>
    /// Controller for self-improvement commands
    /// </summary>
    public class SelfImprovementController
    {
        private readonly ILogger<SelfImprovementController> _logger;
        private readonly ISelfImprovementService _selfImprovementService;
        private readonly ICodeAnalysisService _codeAnalysisService;
        private readonly IProjectAnalysisService _projectAnalysisService;
        private readonly ICodeGenerationService _codeGenerationService;
        private readonly CodeExecutionService _codeExecutionService;
        private readonly LearningService _learningService;

        public SelfImprovementController(
            ILogger<SelfImprovementController> logger,
            ISelfImprovementService selfImprovementService,
            ICodeAnalysisService codeAnalysisService,
            IProjectAnalysisService projectAnalysisService,
            ICodeGenerationService codeGenerationService,
            CodeExecutionService codeExecutionService,
            LearningService learningService)
        {
            _logger = logger;
            _selfImprovementService = selfImprovementService;
            _codeAnalysisService = codeAnalysisService;
            _projectAnalysisService = projectAnalysisService;
            _codeGenerationService = codeGenerationService;
            _codeExecutionService = codeExecutionService;
            _learningService = learningService;
        }

        /// <summary>
        /// Registers self-improvement commands
        /// </summary>
        /// <param name="rootCommand">The root command to add to</param>
        public void RegisterCommands(Command rootCommand)
        {
            // Create the self-improvement command
            var selfImprovementCommand = new Command("self-improve", "Self-improvement commands");
            rootCommand.AddCommand(selfImprovementCommand);

            // Add analyze command
            var analyzeCommand = new Command("analyze", "Analyze code for improvements");
            var pathArgument = new Argument<string>("path", "Path to the file or project to analyze");
            var projectOption = new Option<string>(new[] { "--project", "-p" }, "Path to the project (if analyzing a single file)");
            var recursiveOption = new Option<bool>(new[] { "--recursive", "-r" }, "Analyze recursively (for directories)");
            var maxFilesOption = new Option<int>(new[] { "--max-files", "-m" }, () => 10, "Maximum number of files to analyze");

            analyzeCommand.AddArgument(pathArgument);
            analyzeCommand.AddOption(projectOption);
            analyzeCommand.AddOption(recursiveOption);
            analyzeCommand.AddOption(maxFilesOption);
            analyzeCommand.SetHandler(AnalyzeCodeAsync, pathArgument, projectOption, recursiveOption, maxFilesOption);
            selfImprovementCommand.AddCommand(analyzeCommand);

            // Add improve command
            var improveCommand = new Command("improve", "Improve code based on analysis");
            var improvePath = new Argument<string>("path", "Path to the file or project to improve");
            var improveProject = new Option<string>(new[] { "--project", "-p" }, "Path to the project (if improving a single file)");
            var improveRecursive = new Option<bool>(new[] { "--recursive", "-r" }, "Improve recursively (for directories)");
            var improveMaxFiles = new Option<int>(new[] { "--max-files", "-m" }, () => 5, "Maximum number of files to improve");
            var improveBackup = new Option<bool>(new[] { "--backup", "-b" }, () => true, "Create backups of original files");

            improveCommand.AddArgument(improvePath);
            improveCommand.AddOption(improveProject);
            improveCommand.AddOption(improveRecursive);
            improveCommand.AddOption(improveMaxFiles);
            improveCommand.AddOption(improveBackup);
            improveCommand.SetHandler(ImproveCodeAsync, improvePath, improveProject, improveRecursive, improveMaxFiles, improveBackup);
            selfImprovementCommand.AddCommand(improveCommand);

            // Add generate command
            var generateCommand = new Command("generate", "Generate code based on requirements");
            var outputArg = new Argument<string>("output", "Path to the output file");
            var projectOpt = new Option<string>(new[] { "--project", "-p" }, "Path to the project");
            var requirementsOpt = new Option<string>(new[] { "--requirements", "-r" }, "Requirements for the code");
            var languageOpt = new Option<string>(new[] { "--language", "-l" }, "Programming language");

            generateCommand.AddArgument(outputArg);
            generateCommand.AddOption(projectOpt);
            generateCommand.AddOption(requirementsOpt);
            generateCommand.AddOption(languageOpt);
            generateCommand.SetHandler(GenerateCodeAsync, outputArg, projectOpt, requirementsOpt, languageOpt);
            selfImprovementCommand.AddCommand(generateCommand);

            // Add test command
            var testCommand = new Command("test", "Generate and run tests");
            var testPath = new Argument<string>("path", "Path to the file or project to test");
            var testProject = new Option<string>(new[] { "--project", "-p" }, "Path to the project (if testing a single file)");
            var testOutput = new Option<string>(new[] { "--output", "-o" }, "Path to the output test file");

            testCommand.AddArgument(testPath);
            testCommand.AddOption(testProject);
            testCommand.AddOption(testOutput);
            testCommand.SetHandler(GenerateAndRunTestsAsync, testPath, testProject, testOutput);
            selfImprovementCommand.AddCommand(testCommand);

            // Add cycle command
            var cycleCommand = new Command("cycle", "Run a complete self-improvement cycle");
            var cyclePath = new Argument<string>("path", "Path to the project");
            var cycleMaxFiles = new Option<int>(new[] { "--max-files", "-m" }, () => 10, "Maximum number of files to improve");
            var cycleBackup = new Option<bool>(new[] { "--backup", "-b" }, () => true, "Create backups of original files");
            var cycleTest = new Option<bool>(new[] { "--test", "-t" }, () => true, "Run tests after improvements");

            cycleCommand.AddArgument(cyclePath);
            cycleCommand.AddOption(cycleMaxFiles);
            cycleCommand.AddOption(cycleBackup);
            cycleCommand.AddOption(cycleTest);
            cycleCommand.SetHandler(RunSelfImprovementCycleAsync, cyclePath, cycleMaxFiles, cycleBackup, cycleTest);
            selfImprovementCommand.AddCommand(cycleCommand);

            // Add feedback command
            var feedbackCommand = new Command("feedback", "Record feedback on code generation or improvement");
            var feedbackPath = new Argument<string>("path", "Path to the file to provide feedback on");
            var feedbackRating = new Option<int>(new[] { "--rating", "-r" }, "Rating (1-5)");
            var feedbackComment = new Option<string>(new[] { "--comment", "-c" }, "Comment");
            var feedbackType = new Option<string>(new[] { "--type", "-t" }, "Feedback type (Generation, Improvement, Test)");

            feedbackCommand.AddArgument(feedbackPath);
            feedbackCommand.AddOption(feedbackRating);
            feedbackCommand.AddOption(feedbackComment);
            feedbackCommand.AddOption(feedbackType);
            feedbackCommand.SetHandler(RecordFeedbackAsync, feedbackPath, feedbackRating, feedbackComment, feedbackType);
            selfImprovementCommand.AddCommand(feedbackCommand);

            // Add stats command
            var statsCommand = new Command("stats", "Show learning statistics");
            statsCommand.SetHandler(ShowLearningStatisticsAsync);
            selfImprovementCommand.AddCommand(statsCommand);
        }

        /// <summary>
        /// Analyzes code for improvements
        /// </summary>
        private async Task<int> AnalyzeCodeAsync(string path, string project, bool recursive, int maxFiles)
        {
            try
            {
                Console.WriteLine($"Analyzing code: {path}");

                // Determine if the path is a file or directory
                bool isDirectory = Directory.Exists(path);
                bool isFile = File.Exists(path);

                if (!isDirectory && !isFile)
                {
                    Console.WriteLine($"Error: Path not found: {path}");
                    return 1;
                }

                // If it's a file, analyze it
                if (isFile)
                {
                    // If no project path is provided, use the directory of the file
                    string projectPath = !string.IsNullOrEmpty(project) ? project : Path.GetDirectoryName(path);

                    // Analyze the file
                    var suggestions = await _selfImprovementService.AnalyzeFileForImprovementsAsync(path, projectPath);

                    // Display the suggestions
                    DisplaySuggestions(suggestions, path);
                }
                else
                {
                    // It's a directory, analyze all code files
                    string projectPath = path;

                    // Get all code files
                    var codeFiles = GetCodeFiles(path, recursive);

                    // Limit the number of files
                    codeFiles = codeFiles.Take(maxFiles).ToList();

                    Console.WriteLine($"Found {codeFiles.Count} code files to analyze");

                    // Analyze each file
                    int fileCount = 0;
                    foreach (var file in codeFiles)
                    {
                        fileCount++;
                        Console.WriteLine($"Analyzing file {fileCount}/{codeFiles.Count}: {file}");

                        try
                        {
                            var suggestions = await _selfImprovementService.AnalyzeFileForImprovementsAsync(file, projectPath);
                            DisplaySuggestions(suggestions, file);
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error analyzing file {file}: {ex.Message}");
                        }
                    }
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                _logger.LogError(ex, $"Error analyzing code: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Improves code based on analysis
        /// </summary>
        private async Task<int> ImproveCodeAsync(string path, string project, bool recursive, int maxFiles, bool backup)
        {
            try
            {
                Console.WriteLine($"Improving code: {path}");

                // Determine if the path is a file or directory
                bool isDirectory = Directory.Exists(path);
                bool isFile = File.Exists(path);

                if (!isDirectory && !isFile)
                {
                    Console.WriteLine($"Error: Path not found: {path}");
                    return 1;
                }

                // If it's a file, improve it
                if (isFile)
                {
                    // If no project path is provided, use the directory of the file
                    string projectPath = !string.IsNullOrEmpty(project) ? project : Path.GetDirectoryName(path);

                    // Analyze the file
                    var suggestions = await _selfImprovementService.AnalyzeFileForImprovementsAsync(path, projectPath);

                    if (suggestions.Any())
                    {
                        // Display the suggestions
                        DisplaySuggestions(suggestions, path);

                        // Confirm the improvements
                        if (ConfirmAction($"Apply {suggestions.Count} improvements to {path}?"))
                        {
                            // Apply the improvements
                            string improvedFilePath = await _selfImprovementService.ApplyImprovementsAsync(path, suggestions, backup);
                            Console.WriteLine($"Improvements applied to: {improvedFilePath}");
                        }
                    }
                    else
                    {
                        Console.WriteLine($"No improvements found for {path}");
                    }
                }
                else
                {
                    // It's a directory, improve all code files
                    string projectPath = path;

                    // Get all code files
                    var codeFiles = GetCodeFiles(path, recursive);

                    // Limit the number of files
                    codeFiles = codeFiles.Take(maxFiles).ToList();

                    Console.WriteLine($"Found {codeFiles.Count} code files to improve");

                    // Improve each file
                    int fileCount = 0;
                    int improvedCount = 0;
                    foreach (var file in codeFiles)
                    {
                        fileCount++;
                        Console.WriteLine($"Analyzing file {fileCount}/{codeFiles.Count}: {file}");

                        try
                        {
                            var suggestions = await _selfImprovementService.AnalyzeFileForImprovementsAsync(file, projectPath);

                            if (suggestions.Any())
                            {
                                // Display the suggestions
                                DisplaySuggestions(suggestions, file);

                                // Apply the improvements
                                string improvedFilePath = await _selfImprovementService.ApplyImprovementsAsync(file, suggestions, backup);
                                Console.WriteLine($"Improvements applied to: {improvedFilePath}");
                                improvedCount++;
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"Error improving file {file}: {ex.Message}");
                        }
                    }

                    Console.WriteLine($"Improved {improvedCount} out of {fileCount} files");
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                _logger.LogError(ex, $"Error improving code: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Generates code based on requirements
        /// </summary>
        private async Task<int> GenerateCodeAsync(string output, string project, string requirements, string language)
        {
            try
            {
                Console.WriteLine($"Generating code to: {output}");

                // If no requirements are provided, prompt for them
                if (string.IsNullOrEmpty(requirements))
                {
                    Console.WriteLine("Enter requirements (press Enter twice to finish):");
                    var sb = new System.Text.StringBuilder();
                    string line;
                    while (!string.IsNullOrEmpty(line = Console.ReadLine()))
                    {
                        sb.AppendLine(line);
                    }
                    requirements = sb.ToString();
                }

                // If no project path is provided, use the directory of the output file
                string projectPath = !string.IsNullOrEmpty(project) ? project : Path.GetDirectoryName(output);

                // Determine the language from the file extension if not provided
                if (string.IsNullOrEmpty(language))
                {
                    string extension = Path.GetExtension(output).ToLowerInvariant();
                    language = extension switch
                    {
                        ".cs" => "CSharp",
                        ".fs" => "FSharp",
                        ".js" => "JavaScript",
                        ".ts" => "TypeScript",
                        ".py" => "Python",
                        ".java" => "Java",
                        ".cpp" or ".h" or ".hpp" => "Cpp",
                        _ => "Unknown"
                    };
                }

                // Parse the language
                if (!Enum.TryParse<ProgrammingLanguage>(language, true, out var programmingLanguageEnum))
                {
                    Console.WriteLine($"Error: Invalid language: {language}");
                    return 1;
                }

                // Generate the code
                var result = await _codeGenerationService.GenerateCodeAsync(
                    requirements,
                    projectPath,
                    programmingLanguageEnum,
                    output);

                if (result.Success)
                {
                    Console.WriteLine($"Code generated successfully to: {result.OutputPath}");
                    Console.WriteLine();
                    Console.WriteLine("Generated code:");
                    Console.WriteLine(new string('-', 80));
                    Console.WriteLine(result.GeneratedCode);
                    Console.WriteLine(new string('-', 80));
                }
                else
                {
                    Console.WriteLine($"Error generating code: {result.ErrorMessage}");
                    return 1;
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                _logger.LogError(ex, $"Error generating code: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Generates and runs tests
        /// </summary>
        private async Task<int> GenerateAndRunTestsAsync(string path, string project, string output)
        {
            try
            {
                Console.WriteLine($"Generating tests for: {path}");

                // Ensure the file exists
                if (!File.Exists(path))
                {
                    Console.WriteLine($"Error: File not found: {path}");
                    return 1;
                }

                // If no project path is provided, use the directory of the file
                string projectPath = !string.IsNullOrEmpty(project) ? project : Path.GetDirectoryName(path);

                // Generate the test
                var result = await _codeGenerationService.GenerateUnitTestAsync(path, projectPath, output);

                if (result.Success)
                {
                    Console.WriteLine($"Test generated successfully to: {result.OutputPath}");
                    Console.WriteLine();
                    Console.WriteLine("Generated test code:");
                    Console.WriteLine(new string('-', 80));
                    Console.WriteLine(result.GeneratedCode);
                    Console.WriteLine(new string('-', 80));

                    // Ask if the user wants to run the test
                    if (ConfirmAction("Run the generated test?"))
                    {
                        // Build the project
                        Console.WriteLine("Building the project...");
                        var buildResult = await _codeExecutionService.BuildProjectAsync(projectPath);

                        if (buildResult.Success)
                        {
                            Console.WriteLine("Build successful");

                            // Run the test
                            Console.WriteLine("Running the test...");
                            var testResult = await _codeExecutionService.RunTestsAsync(projectPath);

                            if (testResult.Success)
                            {
                                Console.WriteLine($"Test execution successful: {testResult.PassedTests}/{testResult.TotalTests} tests passed");
                            }
                            else
                            {
                                Console.WriteLine($"Test execution failed: {testResult.ErrorMessage}");
                                Console.WriteLine(testResult.Output);
                                return 1;
                            }
                        }
                        else
                        {
                            Console.WriteLine($"Build failed: {buildResult.ErrorMessage}");
                            Console.WriteLine(buildResult.Output);
                            return 1;
                        }
                    }
                }
                else
                {
                    Console.WriteLine($"Error generating test: {result.ErrorMessage}");
                    return 1;
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                _logger.LogError(ex, $"Error generating and running tests: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Runs a complete self-improvement cycle
        /// </summary>
        private async Task<int> RunSelfImprovementCycleAsync(string path, int maxFiles, bool backup, bool test)
        {
            try
            {
                Console.WriteLine($"Running self-improvement cycle on: {path}");

                // Ensure the path exists
                if (!Directory.Exists(path) && !File.Exists(path))
                {
                    Console.WriteLine($"Error: Path not found: {path}");
                    return 1;
                }

                // Run the self-improvement cycle
                var summary = await _selfImprovementService.RunSelfImprovementCycleAsync(path, maxFiles, backup);

                // Display the summary
                Console.WriteLine();
                Console.WriteLine("Self-improvement cycle summary:");
                Console.WriteLine($"Project: {summary.ProjectPath}");
                Console.WriteLine($"Duration: {summary.Duration}");
                Console.WriteLine($"Files improved: {summary.ImprovedFiles.Count}");

                if (summary.ImprovedFiles.Any())
                {
                    Console.WriteLine();
                    Console.WriteLine("Improved files:");
                    foreach (var file in summary.ImprovedFiles)
                    {
                        Console.WriteLine($"- {file.FilePath} ({file.SuggestionsApplied} improvements)");
                    }
                }

                if (summary.Errors.Any())
                {
                    Console.WriteLine();
                    Console.WriteLine("Errors:");
                    foreach (var error in summary.Errors)
                    {
                        Console.WriteLine($"- {error}");
                    }
                }

                // Run tests if requested
                if (test)
                {
                    Console.WriteLine();
                    Console.WriteLine("Running tests...");

                    // Build the project
                    Console.WriteLine("Building the project...");
                    var buildResult = await _codeExecutionService.BuildProjectAsync(path);

                    if (buildResult.Success)
                    {
                        Console.WriteLine("Build successful");

                        // Run the tests
                        Console.WriteLine("Running tests...");
                        var testResult = await _codeExecutionService.RunTestsAsync(path);

                        if (testResult.Success)
                        {
                            Console.WriteLine($"Test execution successful: {testResult.PassedTests}/{testResult.TotalTests} tests passed");
                        }
                        else
                        {
                            Console.WriteLine($"Test execution failed: {testResult.ErrorMessage}");
                            Console.WriteLine(testResult.Output);
                        }
                    }
                    else
                    {
                        Console.WriteLine($"Build failed: {buildResult.ErrorMessage}");
                        Console.WriteLine(buildResult.Output);
                    }
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                _logger.LogError(ex, $"Error running self-improvement cycle: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Records feedback on code generation or improvement
        /// </summary>
        private async Task<int> RecordFeedbackAsync(string path, int rating, string comment, string type)
        {
            try
            {
                Console.WriteLine($"Recording feedback for: {path}");

                // Ensure the file exists
                if (!File.Exists(path))
                {
                    Console.WriteLine($"Error: File not found: {path}");
                    return 1;
                }

                // If no rating is provided, prompt for it
                if (rating < 1 || rating > 5)
                {
                    Console.Write("Enter rating (1-5): ");
                    if (!int.TryParse(Console.ReadLine(), out rating) || rating < 1 || rating > 5)
                    {
                        Console.WriteLine("Error: Invalid rating");
                        return 1;
                    }
                }

                // If no comment is provided, prompt for it
                if (string.IsNullOrEmpty(comment))
                {
                    Console.Write("Enter comment (optional): ");
                    comment = Console.ReadLine();
                }

                // If no type is provided, prompt for it
                if (string.IsNullOrEmpty(type))
                {
                    Console.WriteLine("Enter feedback type:");
                    Console.WriteLine("1. Generation");
                    Console.WriteLine("2. Improvement");
                    Console.WriteLine("3. Test");
                    Console.Write("Select type (1-3): ");

                    if (int.TryParse(Console.ReadLine(), out int typeNum))
                    {
                        type = typeNum switch
                        {
                            1 => "Generation",
                            2 => "Improvement",
                            3 => "Test",
                            _ => ""
                        };
                    }

                    if (string.IsNullOrEmpty(type))
                    {
                        Console.WriteLine("Error: Invalid type");
                        return 1;
                    }
                }

                // Read the file content
                string code = await File.ReadAllTextAsync(path);

                // Determine the context (language)
                string extension = Path.GetExtension(path).ToLowerInvariant();
                string context = extension switch
                {
                    ".cs" => "CSharp",
                    ".fs" => "FSharp",
                    ".js" => "JavaScript",
                    ".ts" => "TypeScript",
                    ".py" => "Python",
                    ".java" => "Java",
                    ".cpp" or ".h" or ".hpp" => "Cpp",
                    _ => "Unknown"
                };

                // Create the feedback
                var feedback = new CodeFeedback
                {
                    Type = type,
                    Context = context,
                    Code = code,
                    Rating = rating,
                    Comment = comment
                };

                // Record the feedback
                bool success = await _learningService.RecordFeedbackAsync(feedback);

                if (success)
                {
                    Console.WriteLine("Feedback recorded successfully");
                }
                else
                {
                    Console.WriteLine("Error recording feedback");
                    return 1;
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                _logger.LogError(ex, $"Error recording feedback: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Shows learning statistics
        /// </summary>
        private async Task<int> ShowLearningStatisticsAsync()
        {
            try
            {
                Console.WriteLine("Learning statistics:");

                // Get the statistics
                var stats = _learningService.GetStatistics();

                // Display the statistics
                Console.WriteLine($"Total feedback count: {stats.TotalFeedbackCount}");
                Console.WriteLine($"Total pattern count: {stats.TotalPatternCount}");
                Console.WriteLine($"Average feedback rating: {stats.AverageFeedbackRating:F2}");

                if (stats.TopPatterns.Any())
                {
                    Console.WriteLine();
                    Console.WriteLine("Top patterns:");
                    foreach (var pattern in stats.TopPatterns)
                    {
                        Console.WriteLine($"- {pattern.Description} (Score: {pattern.Score:F2}, Used: {pattern.UsageCount} times)");
                    }
                }

                if (stats.FeedbackByType.Any())
                {
                    Console.WriteLine();
                    Console.WriteLine("Feedback by type:");
                    foreach (var kvp in stats.FeedbackByType)
                    {
                        Console.WriteLine($"- {kvp.Key}: {kvp.Value}");
                    }
                }

                if (stats.FeedbackByRating.Any())
                {
                    Console.WriteLine();
                    Console.WriteLine("Feedback by rating:");
                    for (int i = 1; i <= 5; i++)
                    {
                        int count = stats.FeedbackByRating.ContainsKey(i) ? stats.FeedbackByRating[i] : 0;
                        Console.WriteLine($"- {i}: {count}");
                    }
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                _logger.LogError(ex, $"Error showing learning statistics: {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Displays improvement suggestions
        /// </summary>
        private void DisplaySuggestions(List<ImprovementSuggestion> suggestions, string filePath)
        {
            if (!suggestions.Any())
            {
                Console.WriteLine($"No improvements found for {filePath}");
                return;
            }

            Console.WriteLine($"Found {suggestions.Count} improvements for {filePath}:");

            foreach (var suggestion in suggestions)
            {
                Console.WriteLine();
                Console.WriteLine($"Line {suggestion.LineNumber}: {suggestion.Issue}");
                Console.WriteLine($"Improvement: {suggestion.Improvement}");

                if (!string.IsNullOrEmpty(suggestion.ReplacementCode))
                {
                    Console.WriteLine("Replacement code:");
                    Console.WriteLine(new string('-', 40));
                    Console.WriteLine(suggestion.ReplacementCode);
                    Console.WriteLine(new string('-', 40));
                }
            }
        }

        /// <summary>
        /// Gets all code files in a directory
        /// </summary>
        private List<string> GetCodeFiles(string directoryPath, bool recursive)
        {
            var searchOption = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

            // Get all code files
            var codeExtensions = new[] { ".cs", ".fs", ".js", ".ts", ".py", ".java", ".cpp", ".h", ".hpp" };
            var files = Directory.GetFiles(directoryPath, "*.*", searchOption)
                .Where(f => codeExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
                .ToList();

            return files;
        }

        /// <summary>
        /// Confirms an action with the user
        /// </summary>
        private bool ConfirmAction(string message)
        {
            Console.Write($"{message} (y/n): ");
            string response = Console.ReadLine()?.ToLowerInvariant();
            return response == "y" || response == "yes";
        }
    }
}
