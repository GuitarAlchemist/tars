using System.Diagnostics;
using Microsoft.Extensions.DependencyInjection;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for running comprehensive self-diagnostics
/// </summary>
public class SelfDiagnoseCommand : Command
{
    /// <summary>
    /// Create a new self-diagnose command
    /// </summary>
    public SelfDiagnoseCommand() : base("self-diagnose", "Run comprehensive self-diagnostics and generate a report")
    {
        // Add options
        var testsOption = new Option<bool>(
            aliases: ["--tests", "-t"],
            description: "Run unit tests as part of the diagnostics",
            getDefaultValue: () => true);

        var demosOption = new Option<bool>(
            aliases: ["--demos", "-d"],
            description: "Run demos as part of the diagnostics",
            getDefaultValue: () => true);

        var modelOption = new Option<string>(
            aliases: ["--model", "-m"],
            description: "Model to use for demos",
            getDefaultValue: () => "llama3");

        var openOption = new Option<bool>(
            aliases: ["--open", "-o"],
            description: "Open the report in the default browser after generation",
            getDefaultValue: () => true);

        AddOption(testsOption);
        AddOption(demosOption);
        AddOption(modelOption);
        AddOption(openOption);

        this.SetHandler(HandleCommand, testsOption, demosOption, modelOption, openOption);
    }

    /// <summary>
    /// Handle the command execution
    /// </summary>
    private async Task HandleCommand(bool runTests, bool runDemos, string model, bool openReport)
    {
        try
        {
            // Get required services
            var serviceProvider = new ServiceCollection().BuildServiceProvider();
            var logger = serviceProvider.GetRequiredService<ILogger<SelfDiagnoseCommand>>();
            var diagnosticReportService = serviceProvider.GetRequiredService<DiagnosticReportService>();
            var consoleService = serviceProvider.GetRequiredService<ConsoleService>();

            // Display diagnostic information
            consoleService.WriteHeader("=== TARS Self-Diagnostic ===");
            consoleService.WriteInfo("Running comprehensive self-diagnostics...");
            consoleService.WriteInfo($"Tests: {(runTests ? "Enabled" : "Disabled")}");
            consoleService.WriteInfo($"Demos: {(runDemos ? "Enabled" : "Disabled")}");
            if (runDemos)
            {
                consoleService.WriteInfo($"Model: {model}");
            }
            Console.WriteLine();

            // Run the diagnostic report
            var reportPath = await diagnosticReportService.RunDiagnosticReportAsync(runTests, runDemos, model);

            // Display results
            consoleService.WriteSuccess("Self-diagnostic completed successfully!");
            consoleService.WriteInfo($"Report saved to: {Path.GetFullPath(reportPath)}");

            // Open the report if requested
            if (openReport)
            {
                consoleService.WriteInfo("Opening report...");
                var process = new Process();
                process.StartInfo.FileName = Path.GetFullPath(reportPath);
                process.StartInfo.UseShellExecute = true;
                process.Start();
            }
        }
        catch (Exception ex)
        {
            // Handle errors
            Console.Error.WriteLine($"Error running self-diagnostic: {ex.Message}");
        }
    }
}
