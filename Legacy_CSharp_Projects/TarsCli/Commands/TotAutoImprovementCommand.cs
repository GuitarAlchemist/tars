using System;
using System.CommandLine;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Metascripts;

namespace TarsCli.Commands
{
    /// <summary>
    /// Command for running the Tree-of-Thought auto-improvement pipeline.
    /// </summary>
    public class TotAutoImprovementCommand : Command
    {
        private readonly ILogger<TotAutoImprovementCommand> _logger;
        private readonly IMetascriptExecutor _metascriptExecutor;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="TotAutoImprovementCommand"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="metascriptExecutor">The metascript executor.</param>
        public TotAutoImprovementCommand(
            ILogger<TotAutoImprovementCommand> logger,
            IMetascriptExecutor metascriptExecutor)
            : base("tot-auto-improve", "Run the Tree-of-Thought auto-improvement pipeline")
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _metascriptExecutor = metascriptExecutor ?? throw new ArgumentNullException(nameof(metascriptExecutor));
            
            // Add options
            var targetOption = new Option<string>(
                "--target",
                () => "all",
                "The target to improve (all, code-quality, documentation, tests)");
            AddOption(targetOption);
            
            var dryRunOption = new Option<bool>(
                "--dry-run",
                () => false,
                "Run in dry-run mode (don't apply changes)");
            AddOption(dryRunOption);
            
            var verboseOption = new Option<bool>(
                "--verbose",
                () => false,
                "Enable verbose logging");
            AddOption(verboseOption);
            
            var branchingFactorOption = new Option<int>(
                "--branching-factor",
                () => 3,
                "The branching factor for Tree-of-Thought reasoning");
            AddOption(branchingFactorOption);
            
            var maxDepthOption = new Option<int>(
                "--max-depth",
                () => 3,
                "The maximum depth for Tree-of-Thought reasoning");
            AddOption(maxDepthOption);
            
            var beamWidthOption = new Option<int>(
                "--beam-width",
                () => 2,
                "The beam width for Tree-of-Thought pruning");
            AddOption(beamWidthOption);
            
            // Set the handler
            this.SetHandler(async (string target, bool dryRun, bool verbose, int branchingFactor, int maxDepth, int beamWidth) =>
            {
                await RunTotAutoImprovement(target, dryRun, verbose, branchingFactor, maxDepth, beamWidth);
            });
        }
        
        private async Task RunTotAutoImprovement(string target, bool dryRun, bool verbose, int branchingFactor, int maxDepth, int beamWidth)
        {
            try
            {
                _logger.LogInformation("Starting Tree-of-Thought auto-improvement pipeline");
                _logger.LogInformation($"Target: {target}");
                _logger.LogInformation($"Dry run: {dryRun}");
                _logger.LogInformation($"Verbose: {verbose}");
                _logger.LogInformation($"ToT Parameters: Branching factor={branchingFactor}, Max depth={maxDepth}, Beam width={beamWidth}");
                
                // Determine which metascript to run based on the target
                string metascriptPath;
                switch (target.ToLowerInvariant())
                {
                    case "all":
                        metascriptPath = "TarsCli/Metascripts/Improvements/tot_auto_improvement_pipeline.tars";
                        break;
                    case "code-quality":
                        metascriptPath = "TarsCli/Metascripts/Core/tree_of_thought_generator.tars";
                        break;
                    default:
                        _logger.LogError($"Unknown target: {target}");
                        return;
                }
                
                // Check if the metascript exists
                if (!File.Exists(metascriptPath))
                {
                    _logger.LogError($"Metascript not found: {metascriptPath}");
                    return;
                }
                
                // Execute the metascript
                var result = await _metascriptExecutor.ExecuteMetascriptAsync(metascriptPath, new
                {
                    DryRun = dryRun,
                    Verbose = verbose,
                    TotParams = new
                    {
                        BranchingFactor = branchingFactor,
                        MaxDepth = maxDepth,
                        BeamWidth = beamWidth,
                        EvaluationMetrics = new[] { "relevance", "feasibility", "impact", "novelty" },
                        PruningStrategy = "beam_search"
                    }
                });
                
                // Log the result
                if (result.Success)
                {
                    _logger.LogInformation("Tree-of-Thought auto-improvement pipeline completed successfully");
                    
                    // If we have a summary report, display its location
                    if (File.Exists("tot_auto_improvement_summary_report.md"))
                    {
                        _logger.LogInformation("Summary report: tot_auto_improvement_summary_report.md");
                    }
                }
                else
                {
                    _logger.LogError("Tree-of-Thought auto-improvement pipeline failed");
                    _logger.LogError($"Error: {result.ErrorMessage}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running Tree-of-Thought auto-improvement pipeline");
            }
        }
    }
}
