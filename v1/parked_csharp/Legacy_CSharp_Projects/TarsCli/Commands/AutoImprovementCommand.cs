using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Metascripts;

namespace TarsCli.Commands
{
    /// <summary>
    /// Command for running the auto-improvement pipeline.
    /// </summary>
    public class AutoImprovementCommand : Command
    {
        private readonly ILogger<AutoImprovementCommand> _logger;
        private readonly IMetascriptExecutor _metascriptExecutor;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="AutoImprovementCommand"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="metascriptExecutor">The metascript executor.</param>
        public AutoImprovementCommand(
            ILogger<AutoImprovementCommand> logger,
            IMetascriptExecutor metascriptExecutor)
            : base("auto-improve", "Run the auto-improvement pipeline")
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
            
            // Set the handler
            Handler = CommandHandler.Create<string, bool, bool>(RunAutoImprovement);
        }
        
        private async Task RunAutoImprovement(string target, bool dryRun, bool verbose)
        {
            try
            {
                _logger.LogInformation("Starting auto-improvement pipeline");
                _logger.LogInformation($"Target: {target}");
                _logger.LogInformation($"Dry run: {dryRun}");
                _logger.LogInformation($"Verbose: {verbose}");
                
                // Determine which metascript to run based on the target
                string metascriptPath;
                switch (target.ToLowerInvariant())
                {
                    case "all":
                        metascriptPath = "TarsCli/Metascripts/Improvements/auto_improvement_pipeline.tars";
                        break;
                    case "code-quality":
                        metascriptPath = "TarsCli/Metascripts/Improvements/code_quality_analyzer.tars";
                        break;
                    case "documentation":
                        metascriptPath = "TarsCli/Metascripts/Improvements/documentation_generator.tars";
                        break;
                    case "tests":
                        metascriptPath = "TarsCli/Metascripts/Improvements/test_generator.tars";
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
                    Verbose = verbose
                });
                
                // Log the result
                if (result.Success)
                {
                    _logger.LogInformation("Auto-improvement pipeline completed successfully");
                    
                    // If we have a summary report, display its location
                    if (File.Exists("auto_improvement_summary_report.md"))
                    {
                        _logger.LogInformation("Summary report: auto_improvement_summary_report.md");
                    }
                }
                else
                {
                    _logger.LogError("Auto-improvement pipeline failed");
                    _logger.LogError($"Error: {result.ErrorMessage}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error running auto-improvement pipeline");
            }
        }
    }
}
