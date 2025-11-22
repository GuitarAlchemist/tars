using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.TreeOfThought;

namespace TarsCli.Commands
{
    /// <summary>
    /// Command for running a demonstration auto-improvement pipeline using Tree-of-Thought reasoning.
    /// </summary>
    public class DemoAutoImprovementCommand : Command
    {
        private readonly ILogger<DemoAutoImprovementCommand> _logger;
        private readonly DemoTreeOfThoughtService _demoTreeOfThoughtService;

        /// <summary>
        /// Initializes a new instance of the <see cref="DemoAutoImprovementCommand"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="demoTreeOfThoughtService">The demo Tree-of-Thought service.</param>
        public DemoAutoImprovementCommand(
            ILogger<DemoAutoImprovementCommand> logger,
            DemoTreeOfThoughtService demoTreeOfThoughtService)
            : base("demo-auto-improve", "Run a demonstration auto-improvement pipeline using Tree-of-Thought reasoning")
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _demoTreeOfThoughtService = demoTreeOfThoughtService ?? throw new ArgumentNullException(nameof(demoTreeOfThoughtService));

            // Add options
            var fileOption = new Option<string>(
                aliases: new[] { "--file", "-f" },
                description: "The file to improve");
            fileOption.IsRequired = true;

            var typeOption = new Option<string>(
                aliases: new[] { "--type", "-t" },
                description: "The type of improvement to make",
                getDefaultValue: () => "performance");

            var outputOption = new Option<string>(
                aliases: new[] { "--output", "-o" },
                description: "The output file for the report",
                getDefaultValue: () => "demo_auto_improvement_report.md");

            // Add options to the command
            AddOption(fileOption);
            AddOption(typeOption);
            AddOption(outputOption);

            // Set the handler
            Handler = CommandHandler.Create<string, string, string>(HandleCommandAsync);
        }

        private async Task<int> HandleCommandAsync(string file, string type, string output)
        {
            try
            {
                _logger.LogInformation("Running demo auto-improvement pipeline using Tree-of-Thought reasoning");
                _logger.LogInformation("File: {File}", file);
                _logger.LogInformation("Improvement type: {Type}", type);
                _logger.LogInformation("Output: {Output}", output);

                // Check if the file exists
                if (!File.Exists(file))
                {
                    _logger.LogError("File does not exist: {File}", file);
                    return 1;
                }

                // Read the file content
                var code = await File.ReadAllTextAsync(file);

                // Run the auto-improvement pipeline
                var result = await _demoTreeOfThoughtService.RunAutoImprovementPipelineAsync(file, type);

                // Save the report
                await File.WriteAllTextAsync(output, result);
                _logger.LogInformation("Report saved to {Output}", output);

                _logger.LogInformation("Demo auto-improvement pipeline completed successfully");
                return 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running demo auto-improvement pipeline");
                return 1;
            }
        }
    }
}
