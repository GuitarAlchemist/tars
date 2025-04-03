using System.CommandLine;
using System.CommandLine.Invocation;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsCli.Services;
using TarsEngine.DSL;

namespace TarsCli.Commands;

/// <summary>
/// Command for working with TARS metascripts
/// </summary>
public class MetascriptCommand : Command
{
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the <see cref="MetascriptCommand"/> class.
    /// </summary>
    public MetascriptCommand(IServiceProvider serviceProvider = null) : base("metascript", "Work with TARS metascripts")
    {
        _serviceProvider = serviceProvider;

        // Add subcommands
        this.AddCommand(new ExecuteCommand(_serviceProvider));
        this.AddCommand(new ValidateCommand(_serviceProvider));
    }

    /// <summary>
    /// Command for executing a TARS metascript
    /// </summary>
    private class ExecuteCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        /// <summary>
        /// Initializes a new instance of the <see cref="ExecuteCommand"/> class.
        /// </summary>
        public ExecuteCommand(IServiceProvider serviceProvider) : base("execute", "Execute a TARS metascript")
        {
            _serviceProvider = serviceProvider;
            // Add arguments
            var fileArgument = new Argument<string>("file", "The TARS metascript file to execute");
            this.AddArgument(fileArgument);

            // Add options
            var verboseOption = new Option<bool>("--verbose", "Enable verbose output");
            this.AddOption(verboseOption);

            // Set handler
            this.SetHandler(async (context) =>
            {
                var file = context.ParseResult.GetValueForArgument(fileArgument);
                var verbose = context.ParseResult.GetValueForOption(verboseOption);

                var logger = _serviceProvider?.GetService<ILogger<MetascriptCommand>>();
                var consoleService = _serviceProvider?.GetService<ConsoleService>();

                // If no service provider is available, create a console service
                if (consoleService == null)
                {
                    consoleService = new ConsoleService();
                }

                try
                {
                    consoleService.WriteHeader("TARS Metascript Execution");
                    consoleService.WriteInfo($"Executing TARS metascript: {file}");

                    // Check if the file exists as specified
                    if (!File.Exists(file))
                    {
                        // Try to find the file in the Metascripts directory
                        var metascriptsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Metascripts");
                        var metascriptFile = Path.Combine(metascriptsDir, file);

                        if (File.Exists(metascriptFile))
                        {
                            file = metascriptFile;
                        }
                        else
                        {
                            consoleService.WriteError($"File not found: {file}");
                            consoleService.WriteInfo("Try using one of the built-in metascripts in the Metascripts directory:");

                            if (Directory.Exists(metascriptsDir))
                            {
                                foreach (var metascript in Directory.GetFiles(metascriptsDir, "*.tars"))
                                {
                                    consoleService.WriteInfo($"  - {Path.GetFileName(metascript)}");
                                }
                            }

                            context.ExitCode = 1;
                            return;
                        }
                    }

                    // Read the file
                    var script = await File.ReadAllTextAsync(file);

                    // Parse the script using the simplified DSL
                    var program = SimpleDsl.parseProgram(script);

                    // Execute the program
                    var result = SimpleDsl.executeProgram(program);

                    // Display the result
                    switch (result)
                    {
                        case SimpleDsl.ExecutionResult.Success success:
                            consoleService.WriteSuccess("Metascript executed successfully");
                            if (verbose)
                            {
                                consoleService.WriteInfo("Result:");
                                consoleService.WriteInfo(success.ToString());
                            }
                            break;
                        case SimpleDsl.ExecutionResult.Error error:
                            consoleService.WriteError($"Error executing metascript: {error}");
                            context.ExitCode = 1;
                            break;
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error executing TARS metascript");
                    consoleService.WriteError($"Error: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }

    /// <summary>
    /// Command for validating a TARS metascript
    /// </summary>
    private class ValidateCommand : Command
    {
        private readonly IServiceProvider _serviceProvider;

        /// <summary>
        /// Initializes a new instance of the <see cref="ValidateCommand"/> class.
        /// </summary>
        public ValidateCommand(IServiceProvider serviceProvider) : base("validate", "Validate a TARS metascript")
        {
            _serviceProvider = serviceProvider;
            // Add arguments
            var fileArgument = new Argument<string>("file", "The TARS metascript file to validate");
            this.AddArgument(fileArgument);

            // Add options
            var verboseOption = new Option<bool>("--verbose", "Enable verbose output");
            this.AddOption(verboseOption);

            // Set handler
            this.SetHandler(async (context) =>
            {
                var file = context.ParseResult.GetValueForArgument(fileArgument);
                var verbose = context.ParseResult.GetValueForOption(verboseOption);

                var logger = _serviceProvider?.GetService<ILogger<MetascriptCommand>>();
                var consoleService = _serviceProvider?.GetService<ConsoleService>();

                // If no service provider is available, create a console service
                if (consoleService == null)
                {
                    consoleService = new ConsoleService();
                }

                try
                {
                    consoleService.WriteHeader("TARS Metascript Validation");
                    consoleService.WriteInfo($"Validating TARS metascript: {file}");

                    // Check if the file exists as specified
                    if (!File.Exists(file))
                    {
                        // Try to find the file in the Metascripts directory
                        var metascriptsDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Metascripts");
                        var metascriptFile = Path.Combine(metascriptsDir, file);

                        if (File.Exists(metascriptFile))
                        {
                            file = metascriptFile;
                        }
                        else
                        {
                            consoleService.WriteError($"File not found: {file}");
                            consoleService.WriteInfo("Try using one of the built-in metascripts in the Metascripts directory:");

                            if (Directory.Exists(metascriptsDir))
                            {
                                foreach (var metascript in Directory.GetFiles(metascriptsDir, "*.tars"))
                                {
                                    consoleService.WriteInfo($"  - {Path.GetFileName(metascript)}");
                                }
                            }

                            context.ExitCode = 1;
                            return;
                        }
                    }

                    // Read the file
                    var script = await File.ReadAllTextAsync(file);

                    // Parse the script using the simplified DSL
                    var program = SimpleDsl.parseProgram(script);

                    // Display the result
                    consoleService.WriteSuccess("Metascript is valid");
                    if (verbose)
                    {
                        var blocks = program.Blocks;
                        int blockCount = blocks.Length;
                        consoleService.WriteInfo($"Found {blockCount} top-level blocks");
                        foreach (var block in program.Blocks)
                        {
                            consoleService.WriteInfo($"- {block.Type} block");
                        }
                    }
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "Error validating TARS metascript");
                    consoleService.WriteError($"Error: {ex.Message}");
                    context.ExitCode = 1;
                }
            });
        }
    }
}
