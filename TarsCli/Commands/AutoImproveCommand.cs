using System.CommandLine;
using Microsoft.Extensions.Logging;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for running autonomous improvement using metascripts
/// </summary>
public class AutoImproveCommand : Command
{
    private readonly ILogger<AutoImproveCommand> _logger;
    private readonly DslService _dslService;
    private readonly ConsoleService _consoleService;

    /// <summary>
    /// Initializes a new instance of the <see cref="AutoImproveCommand"/> class.
    /// </summary>
    public AutoImproveCommand(
        ILogger<AutoImproveCommand> logger,
        DslService dslService,
        ConsoleService consoleService)
        : base("auto-improve", "Run autonomous improvement using metascripts")
    {
        _logger = logger;
        _dslService = dslService;
        _consoleService = consoleService;

        // Add options
        var timeOption = new Option<int>(
            new[] { "--time", "-t" },
            () => 60,
            "Time limit in minutes");

        var modelOption = new Option<string>(
            new[] { "--model", "-m" },
            () => "llama3",
            "Model to use for improvements");

        var statusOption = new Option<bool>(
            new[] { "--status", "-s" },
            "Show status of the autonomous improvement process");

        var stopOption = new Option<bool>(
            new[] { "--stop" },
            "Stop the autonomous improvement process");

        var reportOption = new Option<bool>(
            new[] { "--report", "-r" },
            "Generate a report of the autonomous improvement process");

        // Add options to command
        AddOption(timeOption);
        AddOption(modelOption);
        AddOption(statusOption);
        AddOption(stopOption);
        AddOption(reportOption);

        // Set handler
        this.SetHandler(async (int time, string model, bool status, bool stop, bool report) =>
        {
            if (status)
            {
                await ShowStatusAsync();
                return;
            }

            if (stop)
            {
                await StopAsync();
                return;
            }

            if (report)
            {
                await GenerateReportAsync();
                return;
            }

            await StartAsync(time, model);
        }, timeOption, modelOption, statusOption, stopOption, reportOption);
    }

    /// <summary>
    /// Start the autonomous improvement process
    /// </summary>
    private async Task StartAsync(int timeLimit, string model)
    {
        try
        {
            _consoleService.WriteHeader("TARS Autonomous Improvement");
            _consoleService.WriteInfo($"Starting autonomous improvement with time limit of {timeLimit} minutes using model {model}");

            // Path to the metascript
            string metascriptPath = Path.Combine("Examples", "metascripts", "autonomous_improvement.tars");

            if (!File.Exists(metascriptPath))
            {
                _consoleService.WriteError($"Metascript not found: {metascriptPath}");
                return;
            }

            // Execute the metascript
            _consoleService.WriteInfo($"Executing metascript: {metascriptPath}");

            // Set environment variables for the metascript
            Environment.SetEnvironmentVariable("TARS_AUTO_IMPROVE_TIME_LIMIT", timeLimit.ToString());
            Environment.SetEnvironmentVariable("TARS_AUTO_IMPROVE_MODEL", model);

            int result = await _dslService.RunDslFileAsync(metascriptPath, true);

            if (result == 0)
            {
                _consoleService.WriteSuccess("Autonomous improvement completed successfully");
            }
            else
            {
                _consoleService.WriteError($"Autonomous improvement failed with exit code {result}");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error starting autonomous improvement");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Show the status of the autonomous improvement process
    /// </summary>
    private async Task ShowStatusAsync()
    {
        try
        {
            _consoleService.WriteHeader("TARS Autonomous Improvement - Status");

            // Check if state file exists
            string stateFilePath = "autonomous_improvement_state.json";
            if (!File.Exists(stateFilePath))
            {
                _consoleService.WriteInfo("No autonomous improvement process has been run yet");
                return;
            }

            // Read the state file
            string stateJson = await File.ReadAllTextAsync(stateFilePath);
            var state = System.Text.Json.JsonSerializer.Deserialize<System.Text.Json.JsonElement>(stateJson);

            // Display status
            _consoleService.WriteInfo($"Last Updated: {state.GetProperty("last_updated").GetString()}");
            _consoleService.WriteInfo($"Total Files Processed: {state.GetProperty("processed_files").GetArrayLength()}");
            _consoleService.WriteInfo($"Files Remaining: {state.GetProperty("pending_files").GetArrayLength()}");
            _consoleService.WriteInfo($"Total Improvements: {state.GetProperty("total_improvements").GetInt32()}");

            // Display current file if any
            if (state.TryGetProperty("current_file", out var currentFile) && currentFile.ValueKind != System.Text.Json.JsonValueKind.Null)
            {
                _consoleService.WriteInfo($"Current File: {currentFile.GetString()}");
            }

            // Display last improved file if any
            if (state.TryGetProperty("last_improved_file", out var lastImprovedFile) && lastImprovedFile.ValueKind != System.Text.Json.JsonValueKind.Null)
            {
                _consoleService.WriteInfo($"Last Improved File: {lastImprovedFile.GetString()}");
            }

            // Display recent improvements
            if (state.TryGetProperty("improvement_history", out var improvementHistory) && improvementHistory.GetArrayLength() > 0)
            {
                _consoleService.WriteInfo("\nRecent Improvements:");

                // Get the last 5 improvements
                var recentImprovements = improvementHistory.EnumerateArray()
                    .Reverse()
                    .Take(5);

                foreach (var improvement in recentImprovements)
                {
                    string filePath = improvement.GetProperty("file_path").GetString();
                    string timestamp = improvement.GetProperty("timestamp").GetString();
                    int improvements = improvement.GetProperty("improvements").GetInt32();

                    _consoleService.WriteInfo($"- {Path.GetFileName(filePath)}: {improvements} improvements at {DateTime.Parse(timestamp).ToLocalTime()}");
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error showing autonomous improvement status");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Stop the autonomous improvement process
    /// </summary>
    private async Task StopAsync()
    {
        try
        {
            _consoleService.WriteHeader("TARS Autonomous Improvement - Stop");
            _consoleService.WriteInfo("Stopping autonomous improvement process...");

            // In a real implementation, we would signal the process to stop
            // For now, we'll just create a stop file that the metascript can check
            await File.WriteAllTextAsync("autonomous_improvement_stop", DateTime.Now.ToString());

            _consoleService.WriteSuccess("Stop signal sent to autonomous improvement process");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping autonomous improvement");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Generate a report of the autonomous improvement process
    /// </summary>
    private async Task GenerateReportAsync()
    {
        try
        {
            _consoleService.WriteHeader("TARS Autonomous Improvement - Report");

            // Check if state file exists
            string stateFilePath = "autonomous_improvement_state.json";
            if (!File.Exists(stateFilePath))
            {
                _consoleService.WriteInfo("No autonomous improvement process has been run yet");
                return;
            }

            // Read the state file
            string stateJson = await File.ReadAllTextAsync(stateFilePath);
            var state = System.Text.Json.JsonSerializer.Deserialize<System.Text.Json.JsonElement>(stateJson);

            // Generate report
            string reportTimestamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
            string reportPath = $"autonomous_improvement_report_{reportTimestamp}.md";

            var report = new System.Text.StringBuilder();
            report.AppendLine("# TARS Autonomous Improvement Report");
            report.AppendLine();
            report.AppendLine("## Summary");
            report.AppendLine($"- **Date:** {DateTime.Now}");
            report.AppendLine($"- **Total Files Processed:** {state.GetProperty("processed_files").GetArrayLength()}");
            report.AppendLine($"- **Total Improvements:** {state.GetProperty("total_improvements").GetInt32()}");
            report.AppendLine();

            // Add improved files
            report.AppendLine("## Improved Files");
            foreach (var file in state.GetProperty("improved_files").EnumerateArray())
            {
                report.AppendLine($"- {file.GetString()}");
            }
            report.AppendLine();

            // Add improvement history
            report.AppendLine("## Improvement History");
            foreach (var improvement in state.GetProperty("improvement_history").EnumerateArray())
            {
                string filePath = improvement.GetProperty("file_path").GetString();
                string timestamp = improvement.GetProperty("timestamp").GetString();
                int improvements = improvement.GetProperty("improvements").GetInt32();
                string description = improvement.GetProperty("description").GetString();

                report.AppendLine($"### {filePath}");
                report.AppendLine($"- **Time:** {DateTime.Parse(timestamp).ToLocalTime()}");
                report.AppendLine($"- **Improvements:** {improvements}");
                report.AppendLine($"- **Description:** {description}");
                report.AppendLine();
            }

            // Save the report
            await File.WriteAllTextAsync(reportPath, report.ToString());

            _consoleService.WriteSuccess($"Report generated: {reportPath}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating autonomous improvement report");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }
}
