using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for improving TARS Explorations documentation
/// </summary>
public class ExplorationsImproveCommand : Command
{
    private readonly ILogger<ExplorationsImproveCommand> _logger;
    private readonly DslService _dslService;
    private readonly ConsoleService _consoleService;

    /// <summary>
    /// Initializes a new instance of the <see cref="ExplorationsImproveCommand"/> class.
    /// </summary>
    public ExplorationsImproveCommand(
        ILogger<ExplorationsImproveCommand> logger,
        DslService dslService,
        ConsoleService consoleService)
        : base("improve-explorations", "Improve TARS Explorations documentation using metascripts")
    {
        _logger = logger;
        _dslService = dslService;
        _consoleService = consoleService;

        // Add options
        var timeOption = new Option<int>(
            ["--time", "-t"],
            () => 60,
            "Time limit in minutes");

        var modelOption = new Option<string>(
            ["--model", "-m"],
            () => "llama3",
            "Model to use for improvements");

        var statusOption = new Option<bool>(
            ["--status", "-s"],
            "Show status of the improvement process");

        var stopOption = new Option<bool>(
            ["--stop"],
            "Stop the improvement process");

        var reportOption = new Option<bool>(
            ["--report", "-r"],
            "Generate a report of the improvement process");

        var chatsOnlyOption = new Option<bool>(
            ["--chats-only"],
            "Only improve files in the Chats directory");

        var reflectionsOnlyOption = new Option<bool>(
            ["--reflections-only"],
            "Only improve files in the Reflections directory");

        var fileOption = new Option<string>(
            ["--file", "-f"],
            "Improve a specific file");

        // Add options to command
        AddOption(timeOption);
        AddOption(modelOption);
        AddOption(statusOption);
        AddOption(stopOption);
        AddOption(reportOption);
        AddOption(chatsOnlyOption);
        AddOption(reflectionsOnlyOption);
        AddOption(fileOption);

        // Set handler
        this.SetHandler(HandleCommand, timeOption, modelOption, statusOption, stopOption, reportOption, chatsOnlyOption, reflectionsOnlyOption, fileOption);
    }

    /// <summary>
    /// Handle the command execution
    /// </summary>
    private async Task HandleCommand(int time, string model, bool status, bool stop, bool report, bool chatsOnly, bool reflectionsOnly, string file)
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

        await StartAsync(time, model, chatsOnly, reflectionsOnly, file);
    }

    /// <summary>
    /// Start the improvement process
    /// </summary>
    private async Task StartAsync(int timeLimit, string model, bool chatsOnly, bool reflectionsOnly, string file)
    {
        try
        {
            _consoleService.WriteHeader("TARS Explorations Improvement");
            _consoleService.WriteInfo($"Starting improvement process with time limit: {timeLimit} minutes");
            _consoleService.WriteInfo($"Using model: {model}");

            if (chatsOnly)
            {
                _consoleService.WriteInfo("Improving only files in the Chats directory");
            }
            else if (reflectionsOnly)
            {
                _consoleService.WriteInfo("Improving only files in the Reflections directory");
            }
            else if (!string.IsNullOrEmpty(file))
            {
                _consoleService.WriteInfo($"Improving specific file: {file}");
            }
            else
            {
                _consoleService.WriteInfo("Improving all files in Chats and Reflections directories");
            }

            // Find the metascript
            var metascriptPath = Path.Combine("TarsCli", "Metascripts", "explorations_improvement.tars");
            if (!File.Exists(metascriptPath))
            {
                metascriptPath = "explorations_improvement.tars";
                if (!File.Exists(metascriptPath))
                {
                    _consoleService.WriteError("Metascript not found: explorations_improvement.tars");
                    return;
                }
            }

            // Execute the metascript
            _consoleService.WriteInfo($"Executing metascript: {metascriptPath}");

            // Set environment variables for the metascript
            Environment.SetEnvironmentVariable("TARS_AUTO_IMPROVE_TIME_LIMIT", timeLimit.ToString());
            Environment.SetEnvironmentVariable("TARS_AUTO_IMPROVE_MODEL", model);

            // Set additional environment variables based on options
            if (chatsOnly)
            {
                Environment.SetEnvironmentVariable("TARS_IMPROVE_CHATS_ONLY", "true");
            }
            else if (reflectionsOnly)
            {
                Environment.SetEnvironmentVariable("TARS_IMPROVE_REFLECTIONS_ONLY", "true");
            }

            if (!string.IsNullOrEmpty(file))
            {
                Environment.SetEnvironmentVariable("TARS_IMPROVE_SPECIFIC_FILE", file);
            }

            int result = await _dslService.RunDslFileAsync(metascriptPath, true);

            if (result == 0)
            {
                _consoleService.WriteSuccess("Explorations improvement completed successfully");

                // Check if a report was generated
                if (File.Exists("explorations_improvement_report.md"))
                {
                    _consoleService.WriteInfo("Report generated: explorations_improvement_report.md");

                    // Display a summary of the report
                    var reportContent = await File.ReadAllTextAsync("explorations_improvement_report.md");
                    var summaryLines = reportContent.Split('\n')
                        .Where(line => line.StartsWith("- **"))
                        .ToList();

                    _consoleService.WriteInfo("Summary:");
                    foreach (var line in summaryLines)
                    {
                        _consoleService.WriteInfo(line);
                    }
                }
            }
            else
            {
                _consoleService.WriteError($"Explorations improvement failed with exit code {result}");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error starting explorations improvement");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Show the status of the improvement process
    /// </summary>
    private async Task ShowStatusAsync()
    {
        try
        {
            _consoleService.WriteHeader("TARS Explorations Improvement - Status");

            // Check if the state file exists
            var stateFilePath = "explorations_improvement_state.json";
            if (!File.Exists(stateFilePath))
            {
                _consoleService.WriteInfo("No active improvement process found");
                return;
            }

            // Read the state file
            var stateJson = await File.ReadAllTextAsync(stateFilePath);
            var state = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(stateJson);

            // Display the status
            _consoleService.WriteInfo($"Status: {state["status"]}");
            _consoleService.WriteInfo($"Start time: {state["start_time"]}");
            _consoleService.WriteInfo($"Files processed: {state["files_processed"]}");
            _consoleService.WriteInfo($"Files improved: {state["files_improved"]}");
            _consoleService.WriteInfo($"Improvements made: {state["improvements_made"]}");

            // Display current file if any
            if (state["current_file"] != null && state["current_file"].ToString() != "null")
            {
                _consoleService.WriteInfo($"Currently processing: {state["current_file"]}");
            }

            // Display recently improved files
            var improvedFiles = System.Text.Json.JsonSerializer.Deserialize<List<string>>(state["improved_files"].ToString());
            if (improvedFiles.Count > 0)
            {
                _consoleService.WriteInfo("Recently improved files:");
                foreach (var file in improvedFiles.TakeLast(Math.Min(5, improvedFiles.Count)))
                {
                    _consoleService.WriteInfo($"- {file}");
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error showing status");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Stop the improvement process
    /// </summary>
    private async Task StopAsync()
    {
        try
        {
            _consoleService.WriteHeader("TARS Explorations Improvement - Stop");
            _consoleService.WriteInfo("Stopping improvement process...");

            // Create a stop file that the metascript can check
            await File.WriteAllTextAsync("autonomous_improvement_stop", DateTime.Now.ToString());

            _consoleService.WriteSuccess("Stop signal sent to improvement process");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error stopping improvement process");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }

    /// <summary>
    /// Generate a report of the improvement process
    /// </summary>
    private async Task GenerateReportAsync()
    {
        try
        {
            _consoleService.WriteHeader("TARS Explorations Improvement - Report");

            // Check if the state file exists
            var stateFilePath = "explorations_improvement_state.json";
            if (!File.Exists(stateFilePath))
            {
                _consoleService.WriteInfo("No improvement process data found");
                return;
            }

            // Read the state file
            var stateJson = await File.ReadAllTextAsync(stateFilePath);
            var state = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(stateJson);

            // Generate a report
            var reportBuilder = new System.Text.StringBuilder();
            reportBuilder.AppendLine("# TARS Explorations Improvement Report");
            reportBuilder.AppendLine();
            reportBuilder.AppendLine("## Summary");
            reportBuilder.AppendLine();
            reportBuilder.AppendLine($"- **Status:** {state["status"]}");
            reportBuilder.AppendLine($"- **Start Time:** {state["start_time"]}");
            reportBuilder.AppendLine($"- **End Time:** {state["end_time"]}");
            reportBuilder.AppendLine($"- **Files Processed:** {state["files_processed"]}");
            reportBuilder.AppendLine($"- **Files Improved:** {state["files_improved"]}");
            reportBuilder.AppendLine($"- **Improvements Made:** {state["improvements_made"]}");
            reportBuilder.AppendLine();

            // Add improved files
            reportBuilder.AppendLine("## Improved Files");
            reportBuilder.AppendLine();
            var improvedFiles = System.Text.Json.JsonSerializer.Deserialize<List<string>>(state["improved_files"].ToString());
            foreach (var file in improvedFiles)
            {
                reportBuilder.AppendLine($"- {file}");
            }
            reportBuilder.AppendLine();

            // Add next steps
            reportBuilder.AppendLine("## Next Steps");
            reportBuilder.AppendLine();
            reportBuilder.AppendLine("1. Review the improved files to ensure the changes are appropriate");
            reportBuilder.AppendLine("2. Run the improvement process again to continue improving more files");
            reportBuilder.AppendLine("3. Consider extracting key insights from the improved files to apply to the codebase");
            reportBuilder.AppendLine();

            // Save the report
            var reportPath = "explorations_improvement_report.md";
            await File.WriteAllTextAsync(reportPath, reportBuilder.ToString());

            _consoleService.WriteSuccess($"Report generated: {reportPath}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating report");
            _consoleService.WriteError($"Error: {ex.Message}");
        }
    }
}
