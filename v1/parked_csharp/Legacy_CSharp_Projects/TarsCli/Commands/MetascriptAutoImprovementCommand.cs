using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.AutoImprovement;

namespace TarsCli.Commands
{
    /// <summary>
    /// Command for running the auto-improvement pipeline using Metascript Tree-of-Thought reasoning.
    /// </summary>
    public class MetascriptAutoImprovementCommand : Command
    {
        private readonly ILogger<MetascriptAutoImprovementCommand> _logger;
        private readonly MetascriptTreeOfThoughtIntegration _metascriptTreeOfThoughtIntegration;

        /// <summary>
        /// Initializes a new instance of the <see cref="MetascriptAutoImprovementCommand"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="metascriptTreeOfThoughtIntegration">The metascript Tree-of-Thought integration.</param>
        public MetascriptAutoImprovementCommand(
            ILogger<MetascriptAutoImprovementCommand> logger,
            MetascriptTreeOfThoughtIntegration metascriptTreeOfThoughtIntegration)
            : base("metascript-auto-improve", "Run the auto-improvement pipeline using Metascript Tree-of-Thought reasoning")
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _metascriptTreeOfThoughtIntegration = metascriptTreeOfThoughtIntegration ?? throw new ArgumentNullException(nameof(metascriptTreeOfThoughtIntegration));

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
                getDefaultValue: () => "auto_improvement_report.md");

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
                _logger.LogInformation("Running auto-improvement pipeline using Metascript Tree-of-Thought reasoning");
                _logger.LogInformation("File: {File}", file);
                _logger.LogInformation("Improvement type: {Type}", type);
                _logger.LogInformation("Output: {Output}", output);

                // Check if the file exists
                if (!File.Exists(file))
                {
                    _logger.LogError("File does not exist: {File}", file);
                    return 1;
                }

                // Run the auto-improvement pipeline
                var result = await _metascriptTreeOfThoughtIntegration.RunAutoImprovementPipelineAsync(file, type);

                // Save the report
                await File.WriteAllTextAsync(output, result);
                _logger.LogInformation("Report saved to {Output}", output);

                _logger.LogInformation("Auto-improvement pipeline completed successfully");
                return 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running auto-improvement pipeline");
                return 1;
            }
        }
    }
}
