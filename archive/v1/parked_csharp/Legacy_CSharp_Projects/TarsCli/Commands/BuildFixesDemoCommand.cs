using System.CommandLine.Invocation;
using Microsoft.Extensions.DependencyInjection;
using TarsCli.Services;

namespace TarsCli.Commands;

/// <summary>
/// Command for demonstrating the build fixes
/// </summary>
public class BuildFixesDemoCommand : Command
{
    private readonly ILogger<BuildFixesDemoCommand>? _logger;
    private readonly IServiceProvider? _serviceProvider;
    private ConsoleService? _consoleService;

    /// <summary>
    /// Initializes a new instance of the BuildFixesDemoCommand class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="serviceProvider">Service provider</param>
    public BuildFixesDemoCommand(ILogger<BuildFixesDemoCommand>? logger = null, IServiceProvider? serviceProvider = null)
        : base("build-fixes-demo", "Demonstrate build fixes implementation")
    {
        _logger = logger;
        _serviceProvider = serviceProvider;

        if (_serviceProvider != null)
        {
            _consoleService = _serviceProvider.GetService<ConsoleService>();
        }

        // Add handler
        this.SetHandler(HandleCommand);
    }

    private async Task HandleCommand(InvocationContext context)
    {
        try
        {
            var serviceProvider = _serviceProvider ?? context.BindingContext.GetService<IServiceProvider>();
            if (serviceProvider == null)
            {
                Console.WriteLine("Error: Service provider not found. Please run the command through the TarsCli application.");
                context.ExitCode = 1;
                return;
            }

            var consoleService = _consoleService ?? serviceProvider.GetRequiredService<ConsoleService>();
            var logger = _logger ?? serviceProvider.GetRequiredService<ILogger<BuildFixesDemoCommand>>();

            // Run the demo
            await RunBuildFixesDemoAsync(consoleService, logger);
            context.ExitCode = 0;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Error running build fixes demo");
            Console.WriteLine($"Error running build fixes demo: {ex.Message}");
            context.ExitCode = 1;
        }
    }

    private async Task RunBuildFixesDemoAsync(ConsoleService consoleService, ILogger<BuildFixesDemoCommand> logger)
    {
        consoleService.WriteHeader("TARS Build Fixes Demo");
        consoleService.WriteInfo("This demo showcases the recent build fixes implemented in the TARS project.");
        consoleService.WriteInfo("It demonstrates how to identify and resolve common build issues in a complex .NET solution.");
        Console.WriteLine();

        // Step 1: Identify Build Errors
        consoleService.WriteSubHeader("Step 1: Identify Build Errors");
        consoleService.WriteInfo("The first step is to identify the build errors by running the build command:");
        consoleService.WriteColorLine("dotnet build", ConsoleColor.DarkGray);
        consoleService.WriteInfo("This will show errors like:");
        consoleService.WriteError("error CS1061: 'TestRunnerService' does not contain a definition for 'RunTestsAsync'");
        consoleService.WriteError("error CS8767: Nullability of reference types in type of parameter 'exception' doesn't match");
        await Task.Delay(1000);
        Console.WriteLine();

        // Step 2: Fix Model Class Compatibility Issues
        consoleService.WriteSubHeader("Step 2: Fix Model Class Compatibility Issues");
        consoleService.WriteInfo("To fix model class compatibility issues, we created adapter classes:");
        consoleService.WriteColorLine(@"public static class CodeIssueAdapter", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"{", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    public static TarsEngine.Models.CodeIssue ToEngineCodeIssue(this TarsCli.Models.CodeIssue cliIssue)", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    {", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"        return new TarsEngine.Models.CodeIssue", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"        {", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"            Description = cliIssue.Message,", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"            CodeSnippet = cliIssue.Code,", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"            SuggestedFix = cliIssue.Suggestion,", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"            // ... other properties", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"        };", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    }", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"}", ConsoleColor.DarkGray);
        await Task.Delay(1000);
        Console.WriteLine();

        // Step 3: Fix Service Conflicts
        consoleService.WriteSubHeader("Step 3: Fix Service Conflicts");
        consoleService.WriteInfo("To fix service conflicts, we updated references to use fully qualified names:");
        consoleService.WriteColorLine(@"// Before", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"private readonly TestRunnerService _testRunnerService;", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"// After", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"private readonly Testing.TestRunnerService _testRunnerService;", ConsoleColor.DarkGray);
        consoleService.WriteInfo("And updated method calls to use the correct methods:");
        consoleService.WriteColorLine(@"// Before", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"var testRunResult = await _testRunnerService.RunTestsAsync(testFilePath);", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"// After", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"var testRunResult = await _testRunnerService.RunTestFileAsync(testFilePath);", ConsoleColor.DarkGray);
        await Task.Delay(1000);
        Console.WriteLine();

        // Step 4: Fix Nullability Warnings
        consoleService.WriteSubHeader("Step 4: Fix Nullability Warnings");
        consoleService.WriteInfo("To fix nullability warnings, we implemented interface methods explicitly:");
        consoleService.WriteColorLine(@"public class LoggerAdapter<T> : ILogger<T>", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"{", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    private readonly ILogger _logger;", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    public LoggerAdapter(ILogger logger)", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    {", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"        _logger = logger ?? throw new ArgumentNullException(nameof(logger));", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    }", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    // Explicit interface implementation with correct nullability", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    IDisposable ILogger.BeginScope<TState>(TState state) => _logger.BeginScope(state);", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    public bool IsEnabled(LogLevel logLevel) => _logger.IsEnabled(logLevel);", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    // Explicit interface implementation with correct nullability", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    void ILogger.Log<TState>(LogLevel logLevel, EventId eventId, TState state, ", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"        Exception? exception, Func<TState, Exception?, string> formatter)", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    {", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"        _logger.Log(logLevel, eventId, state, exception, formatter);", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"    }", ConsoleColor.DarkGray);
        consoleService.WriteColorLine(@"}", ConsoleColor.DarkGray);
        await Task.Delay(1000);
        Console.WriteLine();

        // Step 5: Verify Fixes
        consoleService.WriteSubHeader("Step 5: Verify Fixes");
        consoleService.WriteInfo("Finally, we verify the fixes by running the build command again:");
        consoleService.WriteColorLine("dotnet build", ConsoleColor.DarkGray);
        consoleService.WriteSuccess("Build succeeded with 0 errors");
        await Task.Delay(1000);
        Console.WriteLine();

        // Summary
        consoleService.WriteSubHeader("Summary");
        consoleService.WriteInfo("In this demo, we've shown how to fix common build issues in a .NET solution:");
        consoleService.WriteInfo("1. Model class compatibility issues using adapter classes");
        consoleService.WriteInfo("2. Service conflicts using fully qualified names");
        consoleService.WriteInfo("3. Nullability warnings using explicit interface implementation");
        consoleService.WriteInfo("These fixes have successfully resolved all build errors in the TARS solution.");
        Console.WriteLine();

        consoleService.WriteInfo("For more information, see the following documentation:");
        consoleService.WriteInfo("- docs/build-fixes.md");
        consoleService.WriteInfo("- docs/demos/Build-Fixes-Demo.md");
        consoleService.WriteInfo("- scripts/run-build-fixes-demo.cmd");
        Console.WriteLine();

        consoleService.WriteSuccess("Demo completed successfully!");
    }
}
