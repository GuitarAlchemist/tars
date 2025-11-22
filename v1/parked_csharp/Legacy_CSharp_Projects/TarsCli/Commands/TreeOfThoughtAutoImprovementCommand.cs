using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Metascript;

namespace TarsCli.Commands
{
    /// <summary>
    /// Command for running the Tree-of-Thought auto-improvement pipeline.
    /// </summary>
    public class TreeOfThoughtAutoImprovementCommand : Command
    {
        private readonly ILogger<TreeOfThoughtAutoImprovementCommand> _logger;
        private readonly IMetascriptExecutor _metascriptExecutor;

        /// <summary>
        /// Initializes a new instance of the <see cref="TreeOfThoughtAutoImprovementCommand"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="metascriptExecutor">The metascript executor.</param>
        public TreeOfThoughtAutoImprovementCommand(
            ILogger<TreeOfThoughtAutoImprovementCommand> logger,
            IMetascriptExecutor metascriptExecutor)
            : base("tot-auto-improve", "Run the Tree-of-Thought auto-improvement pipeline")
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _metascriptExecutor = metascriptExecutor ?? throw new ArgumentNullException(nameof(metascriptExecutor));

            // Add options
            var fileOption = new Option<string>(
                aliases: new[] { "--file", "-f" },
                description: "The file to improve");
            fileOption.IsRequired = true;

            var typeOption = new Option<string>(
                aliases: new[] { "--type", "-t" },
                description: "The type of improvement to make (performance, maintainability, error_handling)",
                getDefaultValue: () => "performance");

            var outputOption = new Option<string>(
                aliases: new[] { "--output-dir", "-o" },
                description: "The output directory for reports and improved code",
                getDefaultValue: () => "tot_output");

            // Add options to the command
            AddOption(fileOption);
            AddOption(typeOption);
            AddOption(outputOption);

            // Set the handler
            Handler = CommandHandler.Create<string, string, string>(HandleCommandAsync);
        }

        private async Task<int> HandleCommandAsync(string file, string type, string outputDir)
        {
            try
            {
                _logger.LogInformation("Running Tree-of-Thought auto-improvement pipeline");
                _logger.LogInformation("File: {File}", file);
                _logger.LogInformation("Improvement type: {Type}", type);
                _logger.LogInformation("Output directory: {OutputDir}", outputDir);

                // Check if the file exists
                if (!File.Exists(file))
                {
                    _logger.LogError("File does not exist: {File}", file);
                    return 1;
                }

                // Create the output directory if it doesn't exist
                if (!Directory.Exists(outputDir))
                {
                    Directory.CreateDirectory(outputDir);
                }

                // Load the metascript
                var metascriptPath = "Metascripts/TreeOfThought/CodeImprovement.tars";
                if (!File.Exists(metascriptPath))
                {
                    _logger.LogError("Metascript not found: {MetascriptPath}", metascriptPath);
                    return 1;
                }

                var metascript = await File.ReadAllTextAsync(metascriptPath);

                // Set the variables
                var variables = new System.Collections.Generic.Dictionary<string, string>
                {
                    { "default_target_file", file },
                    { "default_improvement_type", type },
                    { "default_output_dir", outputDir }
                };

                // Execute the metascript
                _logger.LogInformation("Executing metascript: {MetascriptPath}", metascriptPath);
                var result = await _metascriptExecutor.ExecuteAsync(metascript, variables);

                if (result.Success)
                {
                    _logger.LogInformation("Metascript executed successfully");
                    _logger.LogInformation("Output: {Output}", result.Output);
                    _logger.LogInformation("Reports saved to: {OutputDir}", outputDir);
                    return 0;
                }
                else
                {
                    _logger.LogError("Metascript execution failed");
                    _logger.LogError("Errors: {Errors}", string.Join(Environment.NewLine, result.Errors));
                    return 1;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running Tree-of-Thought auto-improvement pipeline");
                return 1;
            }
        }
    }
}
