namespace TarsCli.Commands;

using Services;

/// <summary>
/// Command for managing the retroaction loop
/// </summary>
public class RetroactionLoopCommand : Command
{
    public RetroactionLoopCommand(
        ILogger<RetroactionLoopCommand> logger,
        RetroactionLoopService retroactionLoopService,
        ConsoleService consoleService)
        : base("retroaction-loop", "Manage the retroaction loop")
    {
        // Add subcommands
        AddCommand(new RunCommand(logger, retroactionLoopService, consoleService));
        AddCommand(new CreatePatternCommand(logger, retroactionLoopService, consoleService));
        AddCommand(new StatsCommand(logger, retroactionLoopService, consoleService));
        AddCommand(new ApplyCommand(logger, retroactionLoopService, consoleService));
    }

    /// <summary>
    /// Command to run the retroaction loop
    /// </summary>
    private class RunCommand : Command
    {
        private readonly ILogger _logger;
        private readonly RetroactionLoopService _retroactionLoopService;
        private readonly ConsoleService _consoleService;

        public RunCommand(
            ILogger logger,
            RetroactionLoopService retroactionLoopService,
            ConsoleService consoleService)
            : base("run", "Run the retroaction loop")
        {
            _logger = logger;
            _retroactionLoopService = retroactionLoopService;
            _consoleService = consoleService;

            this.SetHandler(async () =>
            {
                await RunRetroactionLoopAsync();
            });
        }

        private async Task<int> RunRetroactionLoopAsync()
        {
            try
            {
                var success = await _retroactionLoopService.RunRetroactionLoopAsync();
                return success ? 0 : 1;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running retroaction loop");
                _consoleService.WriteError($"Error: {ex.Message}");
                return 1;
            }
        }
    }

    /// <summary>
    /// Command to create a new pattern
    /// </summary>
    private class CreatePatternCommand : Command
    {
        private readonly ILogger _logger;
        private readonly RetroactionLoopService _retroactionLoopService;
        private readonly ConsoleService _consoleService;

        public CreatePatternCommand(
            ILogger logger,
            RetroactionLoopService retroactionLoopService,
            ConsoleService consoleService)
            : base("create-pattern", "Create a new pattern")
        {
            _logger = logger;
            _retroactionLoopService = retroactionLoopService;
            _consoleService = consoleService;

            // Add options
            var nameOption = new Option<string>(
                ["--name", "-n"],
                "Pattern name")
            {
                IsRequired = true
            };

            var descriptionOption = new Option<string>(
                ["--description", "-d"],
                "Pattern description")
            {
                IsRequired = true
            };

            var patternOption = new Option<string>(
                ["--pattern", "-p"],
                "Regex pattern")
            {
                IsRequired = true
            };

            var replacementOption = new Option<string>(
                ["--replacement", "-r"],
                "Replacement string")
            {
                IsRequired = true
            };

            var contextOption = new Option<string>(
                ["--context", "-c"],
                "Context (language)")
            {
                IsRequired = true
            };

            AddOption(nameOption);
            AddOption(descriptionOption);
            AddOption(patternOption);
            AddOption(replacementOption);
            AddOption(contextOption);

            this.SetHandler(async (string name, string description, string pattern, string replacement, string context) =>
            {
                await CreatePatternAsync(name, description, pattern, replacement, context);
            }, nameOption, descriptionOption, patternOption, replacementOption, contextOption);
        }

        private async Task<int> CreatePatternAsync(string name, string description, string pattern, string replacement, string context)
        {
            try
            {
                var success = await _retroactionLoopService.CreatePatternAsync(name, description, pattern, replacement, context);
                return success ? 0 : 1;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating pattern");
                _consoleService.WriteError($"Error: {ex.Message}");
                return 1;
            }
        }
    }

    /// <summary>
    /// Command to get statistics about the retroaction loop
    /// </summary>
    private class StatsCommand : Command
    {
        private readonly ILogger _logger;
        private readonly RetroactionLoopService _retroactionLoopService;
        private readonly ConsoleService _consoleService;

        public StatsCommand(
            ILogger logger,
            RetroactionLoopService retroactionLoopService,
            ConsoleService consoleService)
            : base("stats", "Get statistics about the retroaction loop")
        {
            _logger = logger;
            _retroactionLoopService = retroactionLoopService;
            _consoleService = consoleService;

            // Add options
            var detailedOption = new Option<bool>(
                ["--detailed", "-d"],
                "Show detailed statistics");

            AddOption(detailedOption);

            this.SetHandler(async (bool detailed) =>
            {
                await GetStatisticsAsync(detailed);
            }, detailedOption);
        }

        private async Task<int> GetStatisticsAsync(bool detailed)
        {
            try
            {
                var stats = await _retroactionLoopService.GetStatisticsAsync();
                if (stats == null)
                {
                    _consoleService.WriteError("Error getting statistics");
                    return 1;
                }

                _consoleService.WriteHeader("TARS Retroaction Loop - Statistics");
                _consoleService.WriteInfo($"Total patterns: {stats.TotalPatterns}");
                _consoleService.WriteInfo($"Active patterns: {stats.ActivePatterns}");
                _consoleService.WriteInfo($"Success rate: {stats.SuccessRate:P2}");
                _consoleService.WriteInfo($"Average pattern score: {stats.AveragePatternScore:F2}");
                _consoleService.WriteInfo($"Last updated: {stats.LastUpdated}");

                if (!detailed) return 0;
                _consoleService.WriteInfo("\nPatterns by context:");
                if (stats.PatternsByContext != null)
                {
                    foreach (var item in stats.PatternsByContext)
                    {
                        _consoleService.WriteInfo($"  {item.Key}: {item.Value}");
                    }
                }

                _consoleService.WriteInfo("\nEvents by type:");
                if (stats.EventsByType != null)
                {
                    foreach (var item in stats.EventsByType)
                    {
                        _consoleService.WriteInfo($"  {item.Key}: {item.Value}");
                    }
                }

                return 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting statistics");
                _consoleService.WriteError($"Error: {ex.Message}");
                return 1;
            }
        }
    }

    /// <summary>
    /// Command to apply patterns to a file
    /// </summary>
    private class ApplyCommand : Command
    {
        private readonly ILogger _logger;
        private readonly RetroactionLoopService _retroactionLoopService;
        private readonly ConsoleService _consoleService;

        public ApplyCommand(
            ILogger logger,
            RetroactionLoopService retroactionLoopService,
            ConsoleService consoleService)
            : base("apply", "Apply patterns to a file")
        {
            _logger = logger;
            _retroactionLoopService = retroactionLoopService;
            _consoleService = consoleService;

            // Add options
            var fileOption = new Option<string>(
                ["--file", "-f"],
                "File to apply patterns to")
            {
                IsRequired = true
            };

            var contextOption = new Option<string>(
                ["--context", "-c"],
                "Context (language)");

            var outputOption = new Option<string>(
                ["--output", "-o"],
                "Output file");

            var backupOption = new Option<bool>(
                ["--backup", "-b"],
                "Create a backup of the original file");

            AddOption(fileOption);
            AddOption(contextOption);
            AddOption(outputOption);
            AddOption(backupOption);

            this.SetHandler(async (string file, string context, string output, bool backup) =>
            {
                await ApplyPatternsAsync(file, context, output, backup);
            }, fileOption, contextOption, outputOption, backupOption);
        }

        private async Task<int> ApplyPatternsAsync(string file, string context, string output, bool backup)
        {
            try
            {
                // Ensure file exists
                if (!File.Exists(file))
                {
                    _consoleService.WriteError($"File not found: {file}");
                    return 1;
                }

                // Read the file
                var code = await File.ReadAllTextAsync(file);

                // Determine the context if not provided
                if (string.IsNullOrEmpty(context))
                {
                    var extension = Path.GetExtension(file).ToLowerInvariant();
                    context = extension switch
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

                _consoleService.WriteHeader("TARS Retroaction Loop - Apply Patterns");
                _consoleService.WriteInfo($"Applying patterns to {file} with context {context}...");

                // Apply patterns
                var (improvedCode, appliedPatterns) = await _retroactionLoopService.ApplyPatternsAsync(code, context);

                // Check if any patterns were applied
                if (appliedPatterns.Count == 0)
                {
                    _consoleService.WriteInfo("No patterns were applied");
                    return 0;
                }

                _consoleService.WriteSuccess($"Applied {appliedPatterns.Count} patterns");

                // Create a backup if requested
                if (backup)
                {
                    var backupFile = $"{file}.bak";
                    await File.WriteAllTextAsync(backupFile, code);
                    _consoleService.WriteInfo($"Created backup: {backupFile}");
                }

                // Write the improved code
                var outputFile = string.IsNullOrEmpty(output) ? file : output;
                await File.WriteAllTextAsync(outputFile, improvedCode);
                _consoleService.WriteSuccess($"Wrote improved code to {outputFile}");

                return 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error applying patterns");
                _consoleService.WriteError($"Error: {ex.Message}");
                return 1;
            }
        }
    }
}