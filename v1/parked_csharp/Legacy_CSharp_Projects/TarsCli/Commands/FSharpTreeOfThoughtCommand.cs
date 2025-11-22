using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.FSharp.Adapters.TreeOfThought;
using TarsEngine.Services.Abstractions.TreeOfThought;

namespace TarsCli.Commands
{
    /// <summary>
    /// Command for running the F# Tree-of-Thought implementation.
    /// </summary>
    public class FSharpTreeOfThoughtCommand : Command
    {
        private readonly ITreeOfThoughtService _totService;
        private readonly ILogger<FSharpTreeOfThoughtCommand> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="FSharpTreeOfThoughtCommand"/> class.
        /// </summary>
        /// <param name="totService">The Tree-of-Thought service.</param>
        /// <param name="logger">The logger.</param>
        public FSharpTreeOfThoughtCommand(ITreeOfThoughtService totService, ILogger<FSharpTreeOfThoughtCommand> logger)
            : base("fsharp-tot", "Run the F# Tree-of-Thought implementation")
        {
            _totService = totService ?? throw new ArgumentNullException(nameof(totService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));

            // Add options
            var problemOption = new Option<string>(
                "--problem",
                "The problem to solve")
            {
                IsRequired = true
            };

            var approachesOption = new Option<string[]>(
                "--approaches",
                "The approaches to consider")
            {
                IsRequired = true,
                AllowMultipleArgumentsPerToken = true
            };

            var outputOption = new Option<string>(
                "--output",
                "The output file path")
            {
                IsRequired = false
            };

            AddOption(problemOption);
            AddOption(approachesOption);
            AddOption(outputOption);

            // Set the handler
            this.SetHandler(HandleCommandAsync, problemOption, approachesOption, outputOption);
        }

        private async Task HandleCommandAsync(string problem, string[] approaches, string? output)
        {
            try
            {
                _logger.LogInformation("Running F# Tree-of-Thought for problem: {Problem}", problem);

                // Create the options
                var options = new TreeCreationOptions
                {
                    Approaches = new List<string>(approaches)
                };

                // Create the thought tree
                var root = await _totService.CreateThoughtTreeAsync(problem, options);

                // Add some evaluations
                var random = new Random();
                foreach (var approach in approaches)
                {
                    var metrics = new EvaluationMetrics(
                        random.NextDouble(),
                        random.NextDouble(),
                        random.NextDouble(),
                        random.NextDouble(),
                        random.NextDouble());

                    options.ApproachEvaluations[approach] = metrics;
                }

                // Create the thought tree again with evaluations
                root = await _totService.CreateThoughtTreeAsync(problem, options);

                // Select the best node
                var bestNode = await _totService.SelectBestNodeAsync(root);
                _logger.LogInformation("Best approach: {Approach} (Score: {Score})", bestNode.Thought, bestNode.Score);

                // Add some child nodes to the best approach
                await _totService.AddChildAsync(bestNode, "Implementation 1");
                await _totService.AddChildAsync(bestNode, "Implementation 2");
                await _totService.AddChildAsync(bestNode, "Implementation 3");

                // Generate a report
                var report = await _totService.GenerateReportAsync(root, $"Tree-of-Thought Analysis: {problem}");

                // Print the report
                Console.WriteLine("\nReport:");
                Console.WriteLine(report);

                // Save the report if output is specified
                if (!string.IsNullOrEmpty(output))
                {
                    await _totService.SaveReportAsync(root, $"Tree-of-Thought Analysis: {problem}", output);
                    _logger.LogInformation("Report saved to: {Output}", output);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running F# Tree-of-Thought");
                throw;
            }
        }
    }
}
