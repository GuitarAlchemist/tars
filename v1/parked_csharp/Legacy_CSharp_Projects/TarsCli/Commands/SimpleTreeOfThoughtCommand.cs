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
    /// A simplified command for testing the Tree-of-Thought reasoning.
    /// </summary>
    public class SimpleTreeOfThoughtCommand : Command
    {
        private readonly ILogger<SimpleTreeOfThoughtCommand> _logger;
        private readonly SimpleTreeOfThoughtService _treeOfThoughtService;

        /// <summary>
        /// Initializes a new instance of the <see cref="SimpleTreeOfThoughtCommand"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="treeOfThoughtService">The Tree-of-Thought service.</param>
        public SimpleTreeOfThoughtCommand(ILogger<SimpleTreeOfThoughtCommand> logger, SimpleTreeOfThoughtService treeOfThoughtService)
            : base("simple-tot", "Test the Simple Tree-of-Thought reasoning")
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _treeOfThoughtService = treeOfThoughtService ?? throw new ArgumentNullException(nameof(treeOfThoughtService));

            // Add options
            var fileOption = new Option<string>(
                aliases: new[] { "--file", "-f" },
                description: "The file to analyze");
            
            var outputOption = new Option<string>(
                aliases: new[] { "--output", "-o" },
                description: "The output directory for results",
                getDefaultValue: () => "tot_results");
            
            var modeOption = new Option<string>(
                aliases: new[] { "--mode", "-m" },
                description: "The mode to run (analyze, generate, apply)",
                getDefaultValue: () => "analyze");
            modeOption.FromAmong("analyze", "generate", "apply");
            
            // Add options to the command
            AddOption(fileOption);
            AddOption(outputOption);
            AddOption(modeOption);

            // Set the handler
            Handler = CommandHandler.Create<string, string, string>(HandleCommandAsync);
        }

        private async Task<int> HandleCommandAsync(string file, string output, string mode)
        {
            try
            {
                _logger.LogInformation("Starting Simple Tree-of-Thought test");
                _logger.LogInformation("File: {File}", file);
                _logger.LogInformation("Output directory: {Output}", output);
                _logger.LogInformation("Mode: {Mode}", mode);

                // Ensure the output directory exists
                Directory.CreateDirectory(output);

                // Check if the file exists
                if (!File.Exists(file))
                {
                    _logger.LogError("File does not exist: {File}", file);
                    return 1;
                }

                // Read the file content
                var content = await File.ReadAllTextAsync(file);

                // Process the file based on the mode
                string result;
                switch (mode.ToLower())
                {
                    case "analyze":
                        result = await _treeOfThoughtService.AnalyzeCodeAsync(content);
                        break;
                    case "generate":
                        result = await _treeOfThoughtService.GenerateFixesAsync(content);
                        break;
                    case "apply":
                        result = await _treeOfThoughtService.ApplyFixAsync(content);
                        break;
                    default:
                        _logger.LogError("Invalid mode: {Mode}", mode);
                        return 1;
                }

                // Save the result
                var resultPath = Path.Combine(output, $"{Path.GetFileNameWithoutExtension(file)}_{mode}_result.json");
                await File.WriteAllTextAsync(resultPath, result);
                _logger.LogInformation("Result saved to {Path}", resultPath);

                // Select the best approach
                var bestApproach = await _treeOfThoughtService.SelectBestApproachAsync(result);
                var bestApproachPath = Path.Combine(output, $"{Path.GetFileNameWithoutExtension(file)}_{mode}_best_approach.json");
                await File.WriteAllTextAsync(bestApproachPath, bestApproach);
                _logger.LogInformation("Best approach saved to {Path}", bestApproachPath);

                _logger.LogInformation("Simple Tree-of-Thought test completed successfully");
                return 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running Simple Tree-of-Thought test");
                return 1;
            }
        }
    }
}
