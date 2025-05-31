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
    /// Command for running the Tree-of-Thought auto-improvement pipeline.
    /// </summary>
    public class TreeOfThoughtCommand : Command
    {
        private readonly ILogger<TreeOfThoughtCommand> _logger;
        private readonly TreeOfThoughtService _treeOfThoughtService;

        /// <summary>
        /// Initializes a new instance of the <see cref="TreeOfThoughtCommand"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="treeOfThoughtService">The Tree-of-Thought service.</param>
        public TreeOfThoughtCommand(ILogger<TreeOfThoughtCommand> logger, TreeOfThoughtService treeOfThoughtService)
            : base("tot", "Run the Tree-of-Thought auto-improvement pipeline")
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _treeOfThoughtService = treeOfThoughtService ?? throw new ArgumentNullException(nameof(treeOfThoughtService));

            // Add options
            var targetOption = new Option<string>(
                aliases: new[] { "--target", "-t" },
                description: "The target file or directory to analyze");
            targetOption.IsRequired = true;
            AddOption(targetOption);

            var outputOption = new Option<string>(
                aliases: new[] { "--output", "-o" },
                description: "The output directory for reports",
                getDefaultValue: () => "tot_reports");
            AddOption(outputOption);

            var branchingFactorOption = new Option<int>(
                aliases: new[] { "--branching-factor", "-b" },
                description: "The branching factor for Tree-of-Thought reasoning",
                getDefaultValue: () => 3);
            AddOption(branchingFactorOption);

            var beamWidthOption = new Option<int>(
                aliases: new[] { "--beam-width", "-w" },
                description: "The beam width for Tree-of-Thought reasoning",
                getDefaultValue: () => 2);
            AddOption(beamWidthOption);

            var selectionStrategyOption = new Option<string>(
                aliases: new[] { "--selection-strategy", "-s" },
                description: "The selection strategy for Tree-of-Thought reasoning",
                getDefaultValue: () => "bestFirst");
            selectionStrategyOption.AddAlias("--strategy");
            selectionStrategyOption.FromAmong("bestFirst", "diversityBased", "confidenceBased", "hybridSelection");
            AddOption(selectionStrategyOption);

            var dryRunOption = new Option<bool>(
                aliases: new[] { "--dry-run", "-d" },
                description: "Perform a dry run without applying fixes",
                getDefaultValue: () => false);
            AddOption(dryRunOption);

            // Set the handler
            Handler = CommandHandler.Create<string, string, int, int, string, bool>(HandleCommandAsync);
        }

        private async Task<int> HandleCommandAsync(string target, string output, int branchingFactor, int beamWidth, string selectionStrategy, bool dryRun)
        {
            try
            {
                _logger.LogInformation("Starting Tree-of-Thought auto-improvement pipeline");
                _logger.LogInformation("Target: {Target}", target);
                _logger.LogInformation("Output directory: {Output}", output);
                _logger.LogInformation("Branching factor: {BranchingFactor}", branchingFactor);
                _logger.LogInformation("Beam width: {BeamWidth}", beamWidth);
                _logger.LogInformation("Selection strategy: {SelectionStrategy}", selectionStrategy);
                _logger.LogInformation("Dry run: {DryRun}", dryRun);

                // Ensure the output directory exists
                Directory.CreateDirectory(output);

                // Check if the target exists
                if (!File.Exists(target) && !Directory.Exists(target))
                {
                    _logger.LogError("Target does not exist: {Target}", target);
                    return 1;
                }

                // Get the target files
                var targetFiles = GetTargetFiles(target);
                _logger.LogInformation("Found {Count} target files", targetFiles.Length);

                // Process each file
                foreach (var file in targetFiles)
                {
                    await ProcessFileAsync(file, output, branchingFactor, beamWidth, selectionStrategy, dryRun);
                }

                _logger.LogInformation("Tree-of-Thought auto-improvement pipeline completed successfully");
                return 0;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running Tree-of-Thought auto-improvement pipeline");
                return 1;
            }
        }

        private string[] GetTargetFiles(string target)
        {
            if (File.Exists(target))
            {
                return new[] { target };
            }
            else if (Directory.Exists(target))
            {
                return Directory.GetFiles(target, "*.cs", SearchOption.AllDirectories);
            }
            else
            {
                return Array.Empty<string>();
            }
        }

        private async Task ProcessFileAsync(string filePath, string outputDir, int branchingFactor, int beamWidth, string selectionStrategy, bool dryRun)
        {
            _logger.LogInformation("Processing file: {FilePath}", filePath);

            try
            {
                // Read the file content
                var code = await File.ReadAllTextAsync(filePath);

                // Create the analysis thought tree
                _logger.LogInformation("Creating analysis thought tree");
                var analysisTreeJson = await _treeOfThoughtService.CreateAnalysisThoughtTreeAsync(code, branchingFactor, beamWidth);

                // Save the analysis thought tree
                var analysisTreePath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(filePath)}_analysis_tree.json");
                await File.WriteAllTextAsync(analysisTreePath, analysisTreeJson);
                _logger.LogInformation("Analysis thought tree saved to {Path}", analysisTreePath);

                // Select the best approach
                _logger.LogInformation("Selecting best approach");
                var bestApproachJson = await _treeOfThoughtService.SelectBestApproachAsync(analysisTreeJson, selectionStrategy);

                // Save the best approach
                var bestApproachPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(filePath)}_best_approach.json");
                await File.WriteAllTextAsync(bestApproachPath, bestApproachJson);
                _logger.LogInformation("Best approach saved to {Path}", bestApproachPath);

                // TODO: Generate fixes based on the best approach
                // TODO: Apply fixes if not a dry run

                _logger.LogInformation("File processed successfully: {FilePath}", filePath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing file: {FilePath}", filePath);
            }
        }
    }
}
