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
    /// Command for running the enhanced Tree-of-Thought auto-improvement pipeline.
    /// </summary>
    public class EnhancedTreeOfThoughtCommand : Command
    {
        private readonly ILogger<EnhancedTreeOfThoughtCommand> _logger;
        private readonly EnhancedTreeOfThoughtService _enhancedTreeOfThoughtService;

        /// <summary>
        /// Initializes a new instance of the <see cref="EnhancedTreeOfThoughtCommand"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="enhancedTreeOfThoughtService">The enhanced Tree-of-Thought service.</param>
        public EnhancedTreeOfThoughtCommand(
            ILogger<EnhancedTreeOfThoughtCommand> logger,
            EnhancedTreeOfThoughtService enhancedTreeOfThoughtService)
            : base("enhanced-tot", "Run the enhanced Tree-of-Thought auto-improvement pipeline")
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _enhancedTreeOfThoughtService = enhancedTreeOfThoughtService ?? throw new ArgumentNullException(nameof(enhancedTreeOfThoughtService));

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
                aliases: new[] { "--output", "-o" },
                description: "The output file for the report",
                getDefaultValue: () => "enhanced_tot_report.md");

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
                _logger.LogInformation("Running enhanced Tree-of-Thought auto-improvement pipeline");
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
                var result = await _enhancedTreeOfThoughtService.RunAutoImprovementPipelineAsync(file, type);

                // Save the report
                await File.WriteAllTextAsync(output, result);
                _logger.LogInformation("Report saved to {Output}", output);

                _logger.LogInformation("Enhanced Tree-of-Thought auto-improvement pipeline completed successfully");
                return 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running enhanced Tree-of-Thought auto-improvement pipeline");
                return 1;
            }
        }
    }
}
