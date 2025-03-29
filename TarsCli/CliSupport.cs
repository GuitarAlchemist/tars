using System.CommandLine;
using System.CommandLine.Invocation;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TarsCli.Services;
using TarsCli.Mcp;

namespace TarsCli;

public static class CliSupport
{
    private static IServiceProvider? _serviceProvider;
    // Color output helpers
    public static void WriteColorLine(string text, ConsoleColor color)
    {
        var originalColor = Console.ForegroundColor;
        Console.ForegroundColor = color;
        Console.WriteLine(text);
        Console.ForegroundColor = originalColor;
    }

    public static void WriteHeader(string text)
    {
        Console.WriteLine();
        WriteColorLine(text, ConsoleColor.Cyan);
        WriteColorLine(new string('=', text.Length), ConsoleColor.Cyan);
    }

    public static void WriteCommand(string command, string description)
    {
        Console.Write("  ");
        var originalColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Green;
        Console.Write(command.PadRight(12));
        Console.ForegroundColor = originalColor;
        Console.WriteLine($"- {description}");
    }

    public static void WriteExample(string example)
    {
        Console.Write("  ");
        var originalColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine(example);
        Console.ForegroundColor = originalColor;
    }

    // CLI setup
    public static RootCommand SetupCommandLine(
        IConfiguration configuration,
        ILogger logger,
        ILoggerFactory loggerFactory,
        DiagnosticsService diagnosticsService,
        RetroactionService retroactionService,
        OllamaSetupService ollamaSetupService,
        IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
        // Setup command line options
        var fileOption = new Option<string>(
            name: "--file",
            description: "Path to the file to process")
        {
            IsRequired = true
        };

        var taskOption = new Option<string>(
            name: "--task",
            description: "Description of the task to perform",
            getDefaultValue: () => "Improve code quality and performance");

        var modelOption = new Option<string>(
            name: "--model",
            description: "Ollama model to use",
            getDefaultValue: () => configuration["Ollama:DefaultModel"] ?? "codellama:13b-code");

        // Add validation for file option
        fileOption.AddValidator(result =>
        {
            var filePath = result.GetValueForOption(fileOption);
            if (filePath != null && !File.Exists(filePath))
            {
                result.ErrorMessage = $"File does not exist: {filePath}";
            }
        });

        // Add validation for model option
        modelOption.AddValidator(result =>
        {
            var model = result.GetValueForOption(modelOption);
            if (string.IsNullOrWhiteSpace(model))
            {
                result.ErrorMessage = "Model name cannot be empty";
            }
        });

        // Create root command
        var rootCommand = new RootCommand("TARS CLI - Transformative Autonomous Reasoning System");

        // Add help option
        var helpOption = new Option<bool>(
            aliases: ["--help", "-h"],
            description: "Display detailed help information about TARS CLI");

        rootCommand.AddGlobalOption(helpOption);

        // Create help command
        var helpCommand = new Command("help", "Display detailed help information about TARS CLI");
        helpCommand.SetHandler(() =>
        {
            WriteHeader("=== TARS CLI Help ===");
            Console.WriteLine("TARS (Transformative Autonomous Reasoning System) is a tool for processing files through AI models.");

            WriteHeader("Available Commands");
            WriteCommand("process", "Process a file through the TARS retroaction loop");
            WriteCommand("docs", "Process documentation files in the docs directory");
            WriteCommand("diagnostics", "Run system diagnostics and check environment setup");
            WriteCommand("help", "Display this help information");
            WriteCommand("init", "Initialize a new TARS session");
            WriteCommand("run", "Run a defined agent workflow from DSL script");
            WriteCommand("trace", "View trace logs for a completed run");
            WriteCommand("self-analyze", "Analyze a file for potential improvements");
            WriteCommand("self-propose", "Propose improvements for a file");
            WriteCommand("self-rewrite", "Analyze, propose, and apply improvements to a file");
            WriteCommand("workflow", "Run a multi-agent workflow for a task");

            WriteHeader("Global Options");
            WriteCommand("--help, -h", "Display help information");

            WriteHeader("Examples");
            WriteExample("tarscli process --file path/to/file.cs --task \"Refactor this code\"");
            WriteExample("tarscli docs --task \"Improve documentation clarity\"");
            WriteExample("tarscli diagnostics");
            WriteExample("tarscli init my-session");
            WriteExample("tarscli run --session my-session my-plan.fsx");
            WriteExample("tarscli trace --session my-session last");
            WriteExample("tarscli self-analyze --file path/to/file.cs --model llama3");
            WriteExample("tarscli self-propose --file path/to/file.cs --model codellama:13b-code");
            WriteExample("tarscli self-rewrite --file path/to/file.cs --model codellama:13b-code --auto-apply");
            WriteExample("tarscli workflow --task \"Create a simple web API in C#\"");

            Console.WriteLine("\nFor more information, visit: https://github.com/yourusername/tars");
        });

        // Create process command
        var processCommand = new Command("process", "Process a file through the TARS retroaction loop")
        {
            fileOption,
            taskOption,
            modelOption
        };

        processCommand.SetHandler(async (file, task, model) =>
        {
            WriteHeader("Processing File");
            Console.WriteLine($"File: {file}");
            Console.WriteLine($"Task: {task}");
            Console.WriteLine($"Model: {model}");
            Console.WriteLine();

            var success = await retroactionService.ProcessFile(file, task, model);

            if (success)
            {
                WriteColorLine("Processing completed successfully", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Processing failed", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, fileOption, taskOption, modelOption);

        // Create docs command
        var docsCommand = new Command("docs", "Process documentation files in the docs directory")
        {
            taskOption,
            modelOption
        };

        var docsPathOption = new Option<string>(
            name: "--path",
            description: "Specific path within the docs directory to process",
            getDefaultValue: () => "");

        docsCommand.AddOption(docsPathOption);

        docsCommand.SetHandler(async (task, model, path) =>
        {
            var docsPath = Path.Combine(configuration["Tars:ProjectRoot"] ?? "", "docs");

            if (!string.IsNullOrEmpty(path))
            {
                docsPath = Path.Combine(docsPath, path);
            }

            if (!Directory.Exists(docsPath))
            {
                WriteColorLine($"Directory not found: {docsPath}", ConsoleColor.Red);
                Environment.Exit(1);
                return;
            }

            WriteHeader("Processing Documentation");
            Console.WriteLine($"Path: {docsPath}");
            Console.WriteLine($"Task: {task}");
            Console.WriteLine($"Model: {model}");
            Console.WriteLine();

            // Process all markdown files in the directory
            var files = Directory.GetFiles(docsPath, "*.md", SearchOption.AllDirectories);
            var successCount = 0;

            foreach (var file in files)
            {
                Console.WriteLine($"Processing file: {file}");
                var success = await retroactionService.ProcessFile(file, task, model);

                if (success)
                {
                    successCount++;
                }
            }

            WriteColorLine($"Processing completed. {successCount}/{files.Length} files processed successfully.",
                successCount == files.Length ? ConsoleColor.Green : ConsoleColor.Yellow);

        }, taskOption, modelOption, docsPathOption);

        // Create diagnostics command
        var diagnosticsCommand = new Command("diagnostics", "Run system diagnostics and check environment setup");
        diagnosticsCommand.SetHandler(async () =>
        {
            try
            {
                WriteHeader("=== Running TARS Diagnostics ===");
                Console.WriteLine("Checking system configuration, Ollama setup, and required models...");

                var diagnosticsResult = await diagnosticsService.RunInitialDiagnosticsAsync(verbose: true);

                WriteHeader("=== TARS Diagnostics Report ===");

                WriteColorLine("--- System Information ---", ConsoleColor.Cyan);
                Console.WriteLine($"  Operating System: {diagnosticsResult.SystemInfo.OperatingSystem}");
                Console.WriteLine($"  CPU Cores: {diagnosticsResult.SystemInfo.ProcessorCores}");
                Console.WriteLine($"  Available Memory: {diagnosticsResult.SystemInfo.AvailableMemoryGB:F2} GB");
                Console.WriteLine();

                WriteColorLine("--- Ollama Configuration ---", ConsoleColor.Cyan);
                Console.WriteLine($"  Base URL: {diagnosticsResult.OllamaConfig.BaseUrl}");
                Console.WriteLine($"  Default Model: {diagnosticsResult.OllamaConfig.DefaultModel}");
                Console.WriteLine();

                WriteColorLine("--- Required Models ---", ConsoleColor.Cyan);
                foreach (var model in diagnosticsResult.ModelStatus)
                {
                    var statusColor = model.Value ? ConsoleColor.Green : ConsoleColor.Red;
                    var statusText = model.Value ? "Available ✓" : "Not Available ✗";
                    Console.Write($"  {model.Key}: ");
                    WriteColorLine(statusText, statusColor);
                }
                Console.WriteLine();

                WriteColorLine("--- Project Configuration ---", ConsoleColor.Cyan);
                Console.WriteLine($"  Project Root: {diagnosticsResult.ProjectConfig.ProjectRoot}");
                Console.WriteLine();

                Console.Write($"Overall Status: ");
                WriteColorLine(diagnosticsResult.IsReady ? "Ready ✓" : "Not Ready ✗",
                              diagnosticsResult.IsReady ? ConsoleColor.Green : ConsoleColor.Red);

                if (!diagnosticsResult.IsReady)
                {
                    Console.WriteLine();
                    WriteColorLine("Recommendations:", ConsoleColor.Yellow);
                    if (diagnosticsResult.ModelStatus.Any(m => !m.Value))
                    {
                        var missingModels = diagnosticsResult.ModelStatus
                            .Where(m => !m.Value)
                            .Select(m => m.Key)
                            .ToList();

                        WriteColorLine($"  - Missing required models: {string.Join(", ", missingModels)}", ConsoleColor.Yellow);
                        WriteColorLine("  - Run the following command to install missing models:", ConsoleColor.Yellow);
                        WriteColorLine("    tarscli models install", ConsoleColor.White);
                        WriteColorLine("  - Or use the Install-Prerequisites.ps1 script:", ConsoleColor.Yellow);
                        WriteColorLine("    .\\TarsCli\\Scripts\\Install-Prerequisites.ps1", ConsoleColor.White);
                    }
                }

                WriteColorLine("===========================", ConsoleColor.Cyan);
                WriteColorLine("Diagnostics completed successfully.", ConsoleColor.Green);
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error running diagnostics: {ex.Message}", ConsoleColor.Red);
                WriteColorLine($"Stack trace: {ex.StackTrace}", ConsoleColor.Red);
            }
        });

        // Add a setup command
        var setupCommand = new Command("setup", "Run the prerequisites setup script");
        setupCommand.SetHandler(async () =>
        {
            WriteHeader("Running Prerequisites Setup");
            var scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Scripts", "Install-Prerequisites.ps1");

            if (!File.Exists(scriptPath))
            {
                WriteColorLine($"Setup script not found: {scriptPath}", ConsoleColor.Red);
                Environment.Exit(1);
                return;
            }

            // Check if PowerShell Core is installed
            if (!diagnosticsService.IsPowerShellCoreInstalled())
            {
                WriteColorLine("PowerShell Core (pwsh) is not installed.", ConsoleColor.Yellow);
                WriteColorLine("For better cross-platform compatibility, we recommend installing PowerShell Core:", ConsoleColor.Yellow);
                WriteColorLine("", ConsoleColor.White);
                WriteColorLine("Windows: Install from Microsoft Store or download from:", ConsoleColor.White);
                WriteColorLine("https://github.com/PowerShell/PowerShell/releases", ConsoleColor.Cyan);
                WriteColorLine("", ConsoleColor.White);
                WriteColorLine("macOS: brew install --cask powershell", ConsoleColor.White);
                WriteColorLine("", ConsoleColor.White);
                WriteColorLine("Linux: See installation instructions at:", ConsoleColor.White);
                WriteColorLine("https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-linux", ConsoleColor.Cyan);
                WriteColorLine("", ConsoleColor.White);
                WriteColorLine("Continuing with Windows PowerShell...", ConsoleColor.Yellow);
                Console.WriteLine();
            }

            var success = await diagnosticsService.RunPowerShellScript(scriptPath);

            if (success)
            {
                WriteColorLine("Setup completed successfully", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Setup failed", ConsoleColor.Red);
                Environment.Exit(1);
            }
        });

        // Add a models command with install subcommand
        var modelsCommand = new Command("models", "Manage Ollama models");
        var installCommand = new Command("install", "Install required models");

        installCommand.SetHandler(async () =>
        {
            try
            {
                Console.WriteLine("Installing required models...");

                var requiredModels = await ollamaSetupService.GetRequiredModelsAsync();
                Console.WriteLine($"Required models: {string.Join(", ", requiredModels)}");

                var missingModels = await ollamaSetupService.GetMissingModelsAsync();

                if (missingModels.Count == 0)
                {
                    Console.WriteLine("All required models are already installed.");
                    return;
                }

                Console.WriteLine($"Found {missingModels.Count} missing models: {string.Join(", ", missingModels)}");

                foreach (var model in missingModels)
                {
                    Console.WriteLine($"Installing {model}...");
                    var success = await ollamaSetupService.InstallModelAsync(model);

                    Console.WriteLine(success ? $"Successfully installed {model}" : $"Failed to install {model}");
                }

                Console.WriteLine("Model installation completed.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error installing models: {ex.Message}");
            }
        });

        modelsCommand.Add(installCommand);
        rootCommand.Add(modelsCommand);

        // Create MCP command
        var mcpCommand = new Command("mcp", "Machine Control Panel - control your system");

        // Add subcommands for MCP
        var runCommand = new Command("run", "Run an application")
        {
            new Argument<string>("application", "Path or name of the application to run")
        };

        var processesCommand = new Command("processes", "List running processes");
        var statusCommand = new Command("status", "Show system status");

        // Set handlers for MCP subcommands
        runCommand.SetHandler(async (application) =>
        {
            WriteHeader("MCP - Run Application");
            var mcpController = new McpController(loggerFactory.CreateLogger<TarsCli.Mcp.McpController>());
            var result = await mcpController.ExecuteCommand("run", application);
            Console.WriteLine(result);
        }, new Argument<string>("application"));

        processesCommand.SetHandler(async () =>
        {
            WriteHeader("MCP - Processes");
            var mcpController = new McpController(loggerFactory.CreateLogger<TarsCli.Mcp.McpController>());
            var result = await mcpController.ExecuteCommand("processes");
            Console.WriteLine(result);
        });

        statusCommand.SetHandler(async () =>
        {
            WriteHeader("MCP - System Status");
            var mcpController = new McpController(loggerFactory.CreateLogger<TarsCli.Mcp.McpController>());
            var result = await mcpController.ExecuteCommand("status");
            Console.WriteLine(result);
        });

        // Add subcommands to MCP command
        mcpCommand.AddCommand(runCommand);
        mcpCommand.AddCommand(processesCommand);
        mcpCommand.AddCommand(statusCommand);

        // Add MCP command to root command
        rootCommand.AddCommand(mcpCommand);

        // Create init command
        var initCommand = new Command("init", "Initialize a new TARS session");
        var sessionNameArgument = new Argument<string>("session-name", "Name of the session to initialize");
        initCommand.AddArgument(sessionNameArgument);

        initCommand.SetHandler(async (string sessionName) =>
        {
            WriteHeader("Initializing TARS Session");
            Console.WriteLine($"Session name: {sessionName}");

            var sessionService = _serviceProvider!.GetRequiredService<SessionService>();
            var success = await sessionService.InitializeSession(sessionName);

            if (success)
            {
                WriteColorLine("Session initialized successfully", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Failed to initialize session", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, sessionNameArgument);

        // Create run command
        var runPlanCommand = new Command("run", "Run a defined agent workflow from DSL script");
        var sessionOption = new Option<string>("--session", "Name of the session to use");
        var planArgument = new Argument<string>("plan", "Name of the plan file to run");

        runPlanCommand.AddOption(sessionOption);
        runPlanCommand.AddArgument(planArgument);

        runPlanCommand.SetHandler(async (string session, string plan) =>
        {
            WriteHeader("Running TARS Plan");
            Console.WriteLine($"Session: {session}");
            Console.WriteLine($"Plan: {plan}");

            var sessionService = _serviceProvider!.GetRequiredService<SessionService>();
            var success = await sessionService.RunPlan(session, plan);

            if (success)
            {
                WriteColorLine("Plan executed successfully", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Failed to execute plan", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, sessionOption, planArgument);

        // Create trace command
        var traceCommand = new Command("trace", "View trace logs for a completed run");
        var traceSessionOption = new Option<string>("--session", "Name of the session to use");
        var traceIdArgument = new Argument<string>("trace-id", "ID of the trace to view (or 'last' for the most recent)");
        traceIdArgument.SetDefaultValue("last");

        traceCommand.AddOption(traceSessionOption);
        traceCommand.AddArgument(traceIdArgument);

        traceCommand.SetHandler(async (string session, string traceId) =>
        {
            var sessionService = _serviceProvider!.GetRequiredService<SessionService>();
            var success = await sessionService.ViewTrace(session, traceId);

            if (!success)
            {
                Environment.Exit(1);
            }
        }, traceSessionOption, traceIdArgument);

        // Create self-analyze command
        var selfAnalyzeCommand = new Command("self-analyze", "Analyze a file for potential improvements");
        selfAnalyzeCommand.AddOption(fileOption);
        selfAnalyzeCommand.AddOption(modelOption);

        selfAnalyzeCommand.SetHandler(async (string file, string model) =>
        {
            WriteHeader("TARS Self-Analysis");
            Console.WriteLine($"File: {file}");
            Console.WriteLine($"Model: {model}");

            var selfImprovementService = _serviceProvider!.GetRequiredService<SelfImprovementService>();
            var success = await selfImprovementService.AnalyzeFile(file, model);

            if (success)
            {
                WriteColorLine("Analysis completed successfully", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Analysis failed", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, fileOption, modelOption);

        // Create self-propose command
        var selfProposeCommand = new Command("self-propose", "Propose improvements for a file");
        selfProposeCommand.AddOption(fileOption);
        selfProposeCommand.AddOption(modelOption);

        selfProposeCommand.SetHandler(async (string file, string model) =>
        {
            WriteHeader("TARS Self-Improvement Proposal");
            Console.WriteLine($"File: {file}");
            Console.WriteLine($"Model: {model}");

            var selfImprovementService = _serviceProvider!.GetRequiredService<SelfImprovementService>();
            var success = await selfImprovementService.ProposeImprovement(file, model);

            if (success)
            {
                WriteColorLine("Proposal generated successfully", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Failed to generate proposal", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, fileOption, modelOption);

        // Create self-rewrite command
        var selfRewriteCommand = new Command("self-rewrite", "Analyze, propose, and apply improvements to a file");
        selfRewriteCommand.AddOption(fileOption);
        selfRewriteCommand.AddOption(modelOption);
        var autoApplyOption = new Option<bool>("--auto-apply", "Automatically apply the proposed changes");
        selfRewriteCommand.AddOption(autoApplyOption);

        selfRewriteCommand.SetHandler(async (string file, string model, bool autoApply) =>
        {
            WriteHeader("TARS Self-Rewrite");
            Console.WriteLine($"File: {file}");
            Console.WriteLine($"Model: {model}");
            Console.WriteLine($"Auto-apply: {autoApply}");

            var selfImprovementService = _serviceProvider!.GetRequiredService<SelfImprovementService>();
            var success = await selfImprovementService.SelfRewrite(file, model, autoApply);

            if (success)
            {
                WriteColorLine("Self-rewrite completed successfully", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Self-rewrite failed", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, fileOption, modelOption, autoApplyOption);

        // Create workflow command
        var workflowCommand = new Command("workflow", "Run a multi-agent workflow for a task");
        var workflowTaskOption = new Option<string>("--task", "Description of the task to perform");
        workflowTaskOption.IsRequired = true;
        workflowCommand.AddOption(workflowTaskOption);

        workflowCommand.SetHandler(async (string task) =>
        {
            WriteHeader("TARS Workflow");
            Console.WriteLine($"Task: {task}");

            var workflowService = _serviceProvider!.GetRequiredService<WorkflowCoordinationService>();
            var success = await workflowService.RunWorkflow(task);

            if (success)
            {
                WriteColorLine("Workflow completed successfully", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Workflow failed", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, workflowTaskOption);

        // Add commands to root command
        rootCommand.AddCommand(helpCommand);
        rootCommand.AddCommand(processCommand);
        rootCommand.AddCommand(docsCommand);
        rootCommand.AddCommand(diagnosticsCommand);
        rootCommand.AddCommand(setupCommand);
        rootCommand.AddCommand(initCommand);
        rootCommand.AddCommand(runPlanCommand);
        rootCommand.AddCommand(traceCommand);
        rootCommand.AddCommand(selfAnalyzeCommand);
        rootCommand.AddCommand(selfProposeCommand);
        rootCommand.AddCommand(selfRewriteCommand);
        rootCommand.AddCommand(workflowCommand);

        // Add default handler for root command
        rootCommand.SetHandler((InvocationContext context) =>
        {
            if (context.ParseResult.GetValueForOption(helpOption))
            {
                helpCommand.Handler?.Invoke(context);
            }
            else
            {
                WriteColorLine("Please specify a command. Use --help for more information.", ConsoleColor.Yellow);
                Environment.ExitCode = 1;
            }
        });

        return rootCommand;
    }
}