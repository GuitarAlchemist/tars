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
        WriteColorLine(text, color, true);
    }

    public static void WriteColorLine(string text, ConsoleColor color, bool addNewLine)
    {
        var originalColor = Console.ForegroundColor;
        Console.ForegroundColor = color;

        if (addNewLine)
            Console.WriteLine(text);
        else
            Console.Write(text);

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
            WriteCommand("learning", "View and manage learning data");
            WriteCommand("template", "Manage TARS templates");
            WriteCommand("workflow", "Run a multi-agent workflow for a task");
            WriteCommand("huggingface", "Interact with Hugging Face models");
            WriteCommand("language", "Generate and manage language specifications");
            WriteCommand("docs-explore", "Explore TARS documentation");
            WriteCommand("demo", "Run a demonstration of TARS capabilities");
            WriteCommand("secrets", "Manage API keys and other secrets");
            WriteCommand("auto-improve", "Run autonomous self-improvement");
            WriteCommand("slack", "Manage Slack integration");
            WriteCommand("speech", "Text-to-speech functionality");

            WriteHeader("Global Options");
            WriteCommand("--help, -h", "Display help information");

            WriteHeader("Examples");
            WriteExample("tarscli process --file path/to/file.cs --task \"Refactor this code\"");
            WriteExample("tarscli docs --task \"Improve documentation clarity\"");
            WriteExample("tarscli diagnostics");
            WriteExample("tarscli diagnostics gpu");
            WriteExample("tarscli init my-session");
            WriteExample("tarscli run --session my-session my-plan.fsx");
            WriteExample("tarscli trace --session my-session last");
            WriteExample("tarscli self-analyze --file path/to/file.cs --model llama3");
            WriteExample("tarscli self-propose --file path/to/file.cs --model codellama:13b-code");
            WriteExample("tarscli self-propose --file path/to/file.cs --model codellama:13b-code --auto-accept");
            WriteExample("tarscli self-rewrite --file path/to/file.cs --model codellama:13b-code --auto-apply");
            WriteExample("tarscli learning stats");
            WriteExample("tarscli learning events --count 5");
            WriteExample("tarscli template list");
            WriteExample("tarscli template create --name my_template.json --file path/to/template.json");
            WriteExample("tarscli workflow --task \"Create a simple web API in C#\"");
            WriteExample("tarscli mcp execute \"echo Hello, World!\"");
            WriteExample("tarscli mcp code path/to/file.cs \"public class MyClass { }\"");
            WriteExample("tarscli mcp triple-code path/to/file.cs \"using System;\n\npublic class Program\n{\n    public static void Main()\n    {\n        Console.WriteLine(\"Hello, World!\");\n    }\n}\"");
            WriteExample("tarscli mcp augment sqlite uvx --args mcp-server-sqlite --db-path /path/to/test.db");
            WriteExample("tarscli huggingface search --query \"code generation\" --task text-generation --limit 5");
            WriteExample("tarscli huggingface best --limit 3");
            WriteExample("tarscli huggingface details --model microsoft/phi-2");
            WriteExample("tarscli huggingface download --model microsoft/phi-2");
            WriteExample("tarscli huggingface install --model microsoft/phi-2 --name phi2");
            WriteExample("tarscli docs-explore --list");
            WriteExample("tarscli docs-explore --search \"self-improvement\"");
            WriteExample("tarscli docs-explore --path index.md");
            WriteExample("tarscli demo --type self-improvement --model llama3");
            WriteExample("tarscli demo --type code-generation --model codellama");
            WriteExample("tarscli demo --type all");
            WriteExample("tarscli secrets list");
            WriteExample("tarscli secrets set --key HuggingFace:ApiKey");
            WriteExample("tarscli secrets remove --key OpenAI:ApiKey");
            WriteExample("tarscli auto-improve --time-limit 60 --model llama3");
            WriteExample("tarscli auto-improve --status");
            WriteExample("tarscli auto-improve --stop");
            WriteExample("tarscli slack set-webhook --url https://hooks.slack.com/services/XXX/YYY/ZZZ");
            WriteExample("tarscli slack test --message \"Hello from TARS\" --channel #tars");
            WriteExample("tarscli slack announce --title \"New Release\" --message \"TARS v1.0 is now available!\"");
            WriteExample("tarscli slack feature --name \"GPU Acceleration\" --description \"TARS now supports GPU acceleration for faster processing.\"");
            WriteExample("tarscli slack milestone --name \"1000 Users\" --description \"TARS has reached 1000 active users!\"");
            WriteExample("tarscli speech speak --text \"Hello, I am TARS\"");
            WriteExample("tarscli speech speak --text \"Bonjour, je suis TARS\" --language fr");
            WriteExample("tarscli speech speak --text \"Hello\" --speaker-wav reference.wav");
            WriteExample("tarscli speech list-voices");
            WriteExample("tarscli speech configure --default-voice \"tts_models/en/ljspeech/tacotron2-DDC\" --default-language en");

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

        // Create GPU subcommand
        var gpuCommand = new Command("gpu", "Check GPU capabilities and Ollama GPU acceleration");
        gpuCommand.SetHandler(() =>
        {
            try
            {
                WriteHeader("=== TARS GPU Diagnostics ===");
                Console.WriteLine("Checking GPU capabilities and Ollama GPU acceleration...");

                var diagnosticsService = _serviceProvider!.GetRequiredService<DiagnosticsService>();
                var gpuDiagnostics = diagnosticsService.GetGpuDiagnostics();

                WriteColorLine("--- GPU Information ---", ConsoleColor.Cyan);
                Console.Write("  GPU Acceleration: ");
                WriteColorLine(gpuDiagnostics.IsGpuAvailable ? "Available ✓" : "Not Available ✗",
                              gpuDiagnostics.IsGpuAvailable ? ConsoleColor.Green : ConsoleColor.Yellow);
                Console.WriteLine();

                if (gpuDiagnostics.GpuInfo.Count > 0)
                {
                    WriteColorLine("--- Detected GPUs ---", ConsoleColor.Cyan);
                    foreach (var gpu in gpuDiagnostics.GpuInfo)
                    {
                        var isCompatible = (gpu.Type == GpuType.Nvidia && gpu.MemoryMB >= 4000) ||
                                         (gpu.Type == GpuType.Amd && gpu.MemoryMB >= 8000) ||
                                         (gpu.Type == GpuType.Apple && gpu.MemoryMB >= 4000);

                        var statusColor = isCompatible ? ConsoleColor.Green : ConsoleColor.Yellow;
                        var statusText = isCompatible ? "Compatible ✓" : "Not Compatible ✗";

                        Console.WriteLine($"  {gpu.Name} ({gpu.MemoryMB}MB):");
                        Console.Write($"    Type: {gpu.Type}, Status: ");
                        WriteColorLine(statusText, statusColor);
                    }
                    Console.WriteLine();
                }
                else
                {
                    WriteColorLine("  No GPUs detected", ConsoleColor.Yellow);
                    Console.WriteLine();
                }

                if (gpuDiagnostics.IsGpuAvailable && gpuDiagnostics.OllamaGpuParameters.Count > 0)
                {
                    WriteColorLine("--- Ollama GPU Configuration ---", ConsoleColor.Cyan);
                    foreach (var param in gpuDiagnostics.OllamaGpuParameters)
                    {
                        Console.WriteLine($"  {param.Key}: {param.Value}");
                    }
                    Console.WriteLine();

                    WriteColorLine("GPU acceleration is enabled for Ollama", ConsoleColor.Green);
                    Console.WriteLine();

                    // Run a quick test with Ollama
                    WriteColorLine("--- Running GPU Test ---", ConsoleColor.Cyan);
                    Console.WriteLine("Sending a test request to Ollama to verify GPU acceleration...");
                    Console.WriteLine("This may take a few seconds...");

                    var ollamaService = _serviceProvider!.GetRequiredService<OllamaService>();
                    var testPrompt = "What is the capital of France? Keep it short.";
                    var testModel = "llama3";

                    try
                    {
                        var startTime = DateTime.Now;
                        var response = ollamaService.GenerateCompletion(testPrompt, testModel).GetAwaiter().GetResult();
                        var endTime = DateTime.Now;
                        var duration = (endTime - startTime).TotalSeconds;

                        Console.WriteLine();
                        WriteColorLine("Test Results:", ConsoleColor.Cyan);
                        Console.WriteLine($"  Prompt: {testPrompt}");
                        Console.WriteLine($"  Response: {response}");
                        Console.WriteLine($"  Duration: {duration:F2} seconds");
                        Console.WriteLine();

                        WriteColorLine("GPU test completed successfully", ConsoleColor.Green);
                    }
                    catch (Exception ex)
                    {
                        WriteColorLine($"Error running GPU test: {ex.Message}", ConsoleColor.Red);
                    }
                }
                else if (gpuDiagnostics.IsGpuAvailable)
                {
                    WriteColorLine("GPU is available but not configured for Ollama", ConsoleColor.Yellow);
                }
                else
                {
                    WriteColorLine("GPU acceleration is not available for Ollama", ConsoleColor.Yellow);
                    Console.WriteLine("Reasons could include:");
                    Console.WriteLine("  - No compatible GPU detected");
                    Console.WriteLine("  - GPU drivers not installed or outdated");
                    Console.WriteLine("  - GPU disabled in configuration");
                }

                if (!string.IsNullOrEmpty(gpuDiagnostics.ErrorMessage))
                {
                    Console.WriteLine();
                    WriteColorLine("Error during GPU diagnostics:", ConsoleColor.Red);
                    WriteColorLine($"  {gpuDiagnostics.ErrorMessage}", ConsoleColor.Red);
                }
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error running GPU diagnostics: {ex.Message}", ConsoleColor.Red);
                WriteColorLine($"Stack trace: {ex.StackTrace}", ConsoleColor.Red);
            }
        });

        // Add GPU subcommand to diagnostics command
        diagnosticsCommand.AddCommand(gpuCommand);

        // Main diagnostics command handler
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

        // Add more MCP subcommands for enhanced features
        var executeCommand = new Command("execute", "Execute a terminal command without asking for permission")
        {
            new Argument<string>("command", "The command to execute")
        };

        executeCommand.SetHandler(async (command) =>
        {
            WriteHeader("MCP - Execute Command");
            Console.WriteLine($"Command: {command}");

            var mcpController = _serviceProvider!.GetRequiredService<TarsCli.Mcp.McpController>();
            var result = await mcpController.ExecuteCommand("execute", command);
            Console.WriteLine(result);
        }, new Argument<string>("command"));

        var codeCommand = new Command("code", "Generate and save code without asking for permission");
        var fileArgument = new Argument<string>("file", "Path to the file to create or update");
        var contentArgument = new Argument<string>("content", "The content to write to the file");

        codeCommand.AddArgument(fileArgument);
        codeCommand.AddArgument(contentArgument);

        codeCommand.SetHandler(async (string file, string content) =>
        {
            WriteHeader("MCP - Generate Code");
            Console.WriteLine($"File: {file}");

            // Check if the content is triple-quoted
            if (content.StartsWith("\"\"\"") && content.EndsWith("\"\"\""))
            {
                // Remove the triple quotes
                content = content.Substring(3, content.Length - 6);
                Console.WriteLine("Using triple-quoted syntax for code generation.");
            }

            // Create a direct parameter string with file path and content
            var codeSpec = $"{file}:::{content}";

            var mcpController = _serviceProvider!.GetRequiredService<TarsCli.Mcp.McpController>();
            var result = await mcpController.ExecuteCommand("code", codeSpec);
            Console.WriteLine(result);
        }, fileArgument, contentArgument);

        var augmentCommand = new Command("augment", "Configure Augment Code MCP server")
        {
            new Argument<string>("name", "Name of the MCP server"),
            new Argument<string>("command", "Command to execute"),
            new Option<string[]>("--args", "Arguments to pass to the command") { AllowMultipleArgumentsPerToken = true }
        };

        augmentCommand.SetHandler(async (name, command, args) =>
        {
            WriteHeader("MCP - Configure Augment");
            Console.WriteLine($"Server Name: {name}");
            Console.WriteLine($"Command: {command}");
            Console.WriteLine($"Args: {string.Join(", ", args)}");

            // Create a JSON specification for the Augment configuration
            var configSpec = System.Text.Json.JsonSerializer.Serialize(new
            {
                serverName = name,
                command = command,
                args = args
            });

            var mcpController = _serviceProvider!.GetRequiredService<TarsCli.Mcp.McpController>();
            var result = await mcpController.ExecuteCommand("augment", configSpec);
            Console.WriteLine(result);
        }, new Argument<string>("name"), new Argument<string>("command"), new Option<string[]>("--args"));



        // Create a dedicated command for triple-quoted code
        var tripleCodeCommand = new Command("triple-code", "Generate code using triple-quoted syntax");
        var fileArg = new Argument<string>("file", "Path to the file to create or update");
        var contentArg = new Argument<string>("content", "The content to write to the file");

        tripleCodeCommand.AddArgument(fileArg);
        tripleCodeCommand.AddArgument(contentArg);

        tripleCodeCommand.SetHandler(async (string file, string content) =>
        {
            WriteHeader("MCP - Triple-Quoted Code");
            Console.WriteLine($"File: {file}");

            // Create a direct parameter string with file path and triple-quoted content
            var codeSpec = $"{file}:::-triple-quoted:::{content}";

            var mcpController = _serviceProvider!.GetRequiredService<TarsCli.Mcp.McpController>();
            var result = await mcpController.ExecuteCommand("code", codeSpec);
            Console.WriteLine(result);
        }, fileArg, contentArg);

        // Add subcommands to MCP command
        mcpCommand.AddCommand(runCommand);
        mcpCommand.AddCommand(processesCommand);
        mcpCommand.AddCommand(statusCommand);
        mcpCommand.AddCommand(executeCommand);
        mcpCommand.AddCommand(codeCommand);
        mcpCommand.AddCommand(tripleCodeCommand);
        mcpCommand.AddCommand(augmentCommand);

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
        var autoAcceptOption = new Option<bool>("--auto-accept", "Automatically accept and apply the proposed changes");
        selfProposeCommand.AddOption(autoAcceptOption);

        selfProposeCommand.SetHandler(async (string file, string model, bool autoAccept) =>
        {
            WriteHeader("TARS Self-Improvement Proposal");
            Console.WriteLine($"File: {file}");
            Console.WriteLine($"Model: {model}");
            Console.WriteLine($"Auto-accept: {autoAccept}");

            var selfImprovementService = _serviceProvider!.GetRequiredService<SelfImprovementService>();
            var success = await selfImprovementService.ProposeImprovement(file, model, autoAccept);

            if (success)
            {
                WriteColorLine("Proposal generated successfully", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Failed to generate proposal", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, fileOption, modelOption, autoAcceptOption);

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

        // Create template command
        var templateCommand = new Command("template", "Manage TARS templates");

        // Create template list command
        var templateListCommand = new Command("list", "List available templates");
        templateListCommand.SetHandler(() =>
        {
            WriteHeader("TARS Templates");

            var templateService = _serviceProvider!.GetRequiredService<TemplateService>();
            var templates = templateService.ListTemplates();

            if (templates.Count == 0)
            {
                WriteColorLine("No templates found", ConsoleColor.Yellow);
                return;
            }

            foreach (var template in templates)
            {
                Console.WriteLine($"- {template}");
            }
        });

        // Create template create command
        var templateCreateCommand = new Command("create", "Create a new template");
        var templateNameOption = new Option<string>("--name", "Name of the template");
        var templateFileOption = new Option<string>("--file", "Path to the template file");
        templateNameOption.IsRequired = true;
        templateFileOption.IsRequired = true;

        templateCreateCommand.AddOption(templateNameOption);
        templateCreateCommand.AddOption(templateFileOption);

        templateCreateCommand.SetHandler(async (string name, string file) =>
        {
            WriteHeader("Create TARS Template");
            Console.WriteLine($"Template name: {name}");
            Console.WriteLine($"Template file: {file}");

            if (!File.Exists(file))
            {
                WriteColorLine($"File not found: {file}", ConsoleColor.Red);
                Environment.Exit(1);
                return;
            }

            var templateService = _serviceProvider!.GetRequiredService<TemplateService>();
            await templateService.InitializeTemplatesDirectory();

            var content = await File.ReadAllTextAsync(file);
            var success = await templateService.CreateTemplate(name, content);

            if (success)
            {
                WriteColorLine($"Template '{name}' created successfully", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine($"Failed to create template '{name}'", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, templateNameOption, templateFileOption);

        // Add template subcommands
        templateCommand.AddCommand(templateListCommand);
        templateCommand.AddCommand(templateCreateCommand);

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
        // Create learning command
        var learningCommand = new Command("learning", "View and manage learning data");

        // Create learning stats command
        var learningStatsCommand = new Command("stats", "View learning statistics");
        learningStatsCommand.SetHandler(async () =>
        {
            WriteHeader("TARS Learning Statistics");

            var selfImprovementService = _serviceProvider!.GetRequiredService<SelfImprovementService>();
            var statistics = await selfImprovementService.GetLearningStatistics();

            Console.WriteLine(statistics);
        });

        // Create learning events command
        var learningEventsCommand = new Command("events", "View recent learning events");
        var countOption = new Option<int>("--count", () => 10, "Number of events to show");
        learningEventsCommand.AddOption(countOption);

        learningEventsCommand.SetHandler(async (int count) =>
        {
            WriteHeader("TARS Learning Events");

            var selfImprovementService = _serviceProvider!.GetRequiredService<SelfImprovementService>();
            var events = await selfImprovementService.GetRecentLearningEvents(count);

            if (events.Count == 0)
            {
                WriteColorLine("No learning events found", ConsoleColor.Yellow);
                return;
            }

            foreach (var evt in events)
            {
                dynamic eventObj = evt;
                WriteColorLine($"Event ID: {eventObj.Id}", ConsoleColor.Cyan);
                Console.WriteLine($"Type: {eventObj.EventType}");
                Console.WriteLine($"File: {eventObj.FileName}");
                Console.WriteLine($"Time: {eventObj.Timestamp:yyyy-MM-dd HH:mm:ss}");
                Console.WriteLine($"Success: {eventObj.Success}");

                // Access feedback property
                try
                {
                    var feedback = eventObj.Feedback;
                    if (feedback != null)
                    {
                        var isSome = Microsoft.FSharp.Core.FSharpOption<string>.get_IsSome(feedback);
                        if (isSome)
                        {
                            Console.WriteLine($"Feedback: {feedback.Value}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    // Ignore errors accessing properties
                    Console.WriteLine("(No feedback available)");
                }

                Console.WriteLine();
            }
        }, countOption);

        // Create learning clear command
        var learningClearCommand = new Command("clear", "Clear learning database");
        learningClearCommand.SetHandler(async () =>
        {
            WriteHeader("TARS Learning Database Clear");

            Console.Write("Are you sure you want to clear the learning database? (y/n): ");
            var response = Console.ReadLine()?.ToLower();

            if (response == "y" || response == "yes")
            {
                var selfImprovementService = _serviceProvider!.GetRequiredService<SelfImprovementService>();
                var success = await selfImprovementService.ClearLearningDatabase();

                if (success)
                {
                    WriteColorLine("Learning database cleared successfully", ConsoleColor.Green);
                }
                else
                {
                    WriteColorLine("Failed to clear learning database", ConsoleColor.Red);
                }
            }
            else
            {
                WriteColorLine("Operation cancelled", ConsoleColor.Yellow);
            }
        });

        // Add learning subcommands
        learningCommand.AddCommand(learningStatsCommand);
        learningCommand.AddCommand(learningEventsCommand);
        learningCommand.AddCommand(learningClearCommand);

        // Create huggingface command
        var huggingFaceCommand = new Command("huggingface", "Interact with Hugging Face models");

        // Create huggingface search command
        var hfSearchCommand = new Command("search", "Search for models on Hugging Face");
        var queryOption = new Option<string>("--query", "Search query") { IsRequired = true };
        var hfTaskOption = new Option<string>("--task", () => "text-generation", "Task type (e.g., text-generation, code-generation)");
        var limitOption = new Option<int>("--limit", () => 10, "Maximum number of results to return");

        hfSearchCommand.AddOption(queryOption);
        hfSearchCommand.AddOption(hfTaskOption);
        hfSearchCommand.AddOption(limitOption);

        hfSearchCommand.SetHandler(async (string query, string task, int limit) =>
        {
            WriteHeader("Hugging Face - Search Models");
            Console.WriteLine($"Query: {query}");
            Console.WriteLine($"Task: {task}");
            Console.WriteLine($"Limit: {limit}");

            var huggingFaceService = _serviceProvider!.GetRequiredService<HuggingFaceService>();
            var models = await huggingFaceService.SearchModelsAsync(query, task, limit);

            if (models.Count == 0)
            {
                WriteColorLine("No models found matching the search criteria", ConsoleColor.Yellow);
                return;
            }

            WriteColorLine($"Found {models.Count} models:", ConsoleColor.Green);
            Console.WriteLine();

            foreach (var model in models)
            {
                WriteColorLine(model.Id, ConsoleColor.Cyan);
                Console.WriteLine($"Author: {model.Author}");
                Console.WriteLine($"Downloads: {model.Downloads:N0}");
                Console.WriteLine($"Likes: {model.Likes:N0}");
                Console.WriteLine($"Tags: {string.Join(", ", model.Tags)}");
                Console.WriteLine();
            }
        }, queryOption, hfTaskOption, limitOption);

        // Create huggingface best command
        var hfBestCommand = new Command("best", "Get the best coding models from Hugging Face");
        hfBestCommand.AddOption(limitOption);

        hfBestCommand.SetHandler(async (int limit) =>
        {
            WriteHeader("Hugging Face - Best Coding Models");
            Console.WriteLine($"Limit: {limit}");

            var huggingFaceService = _serviceProvider!.GetRequiredService<HuggingFaceService>();
            var models = await huggingFaceService.GetBestCodingModelsAsync(limit);

            if (models.Count == 0)
            {
                WriteColorLine("No models found", ConsoleColor.Yellow);
                return;
            }

            WriteColorLine($"Found {models.Count} best coding models:", ConsoleColor.Green);
            Console.WriteLine();

            foreach (var model in models)
            {
                WriteColorLine(model.Id, ConsoleColor.Cyan);
                Console.WriteLine($"Author: {model.Author}");
                Console.WriteLine($"Downloads: {model.Downloads:N0}");
                Console.WriteLine($"Likes: {model.Likes:N0}");
                Console.WriteLine($"Tags: {string.Join(", ", model.Tags)}");
                Console.WriteLine();
            }
        }, limitOption);

        // Create huggingface details command
        var hfDetailsCommand = new Command("details", "Get detailed information about a model");
        var modelIdOption = new Option<string>("--model", "Model ID") { IsRequired = true };

        hfDetailsCommand.AddOption(modelIdOption);

        hfDetailsCommand.SetHandler(async (string modelId) =>
        {
            WriteHeader("Hugging Face - Model Details");
            Console.WriteLine($"Model ID: {modelId}");

            var huggingFaceService = _serviceProvider!.GetRequiredService<HuggingFaceService>();
            var modelDetails = await huggingFaceService.GetModelDetailsAsync(modelId);

            if (string.IsNullOrEmpty(modelDetails.Id))
            {
                WriteColorLine($"No details found for model: {modelId}", ConsoleColor.Yellow);
                return;
            }

            WriteColorLine(modelDetails.Id, ConsoleColor.Cyan);
            Console.WriteLine($"Author: {modelDetails.Author}");
            Console.WriteLine($"Downloads: {modelDetails.Downloads:N0}");
            Console.WriteLine($"Likes: {modelDetails.Likes:N0}");
            Console.WriteLine($"Last Modified: {modelDetails.LastModified}");
            Console.WriteLine($"License: {modelDetails.CardData.License}");
            Console.WriteLine($"Tags: {string.Join(", ", modelDetails.Tags)}");
            Console.WriteLine($"Languages: {string.Join(", ", modelDetails.CardData.Languages)}");
            Console.WriteLine($"Datasets: {string.Join(", ", modelDetails.CardData.Datasets)}");

            if (modelDetails.Siblings.Count > 0)
            {
                Console.WriteLine("\nFiles:");
                foreach (var file in modelDetails.Siblings)
                {
                    Console.WriteLine($"- {file.Filename} ({file.Size / 1024.0 / 1024.0:N2} MB)");
                }
            }
        }, modelIdOption);

        // Create huggingface download command
        var hfDownloadCommand = new Command("download", "Download a model from Hugging Face");
        hfDownloadCommand.AddOption(modelIdOption);

        hfDownloadCommand.SetHandler(async (string modelId) =>
        {
            WriteHeader("Hugging Face - Download Model");
            Console.WriteLine($"Model ID: {modelId}");

            var huggingFaceService = _serviceProvider!.GetRequiredService<HuggingFaceService>();
            var success = await huggingFaceService.DownloadModelAsync(modelId);

            if (success)
            {
                WriteColorLine($"Successfully downloaded model: {modelId}", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine($"Failed to download model: {modelId}", ConsoleColor.Red);
            }
        }, modelIdOption);

        // Create huggingface install command
        var hfInstallCommand = new Command("install", "Install a model from Hugging Face to Ollama");
        hfInstallCommand.AddOption(modelIdOption);
        var ollamaNameOption = new Option<string>("--name", "Name to use in Ollama");
        hfInstallCommand.AddOption(ollamaNameOption);

        hfInstallCommand.SetHandler(async (string modelId, string ollamaName) =>
        {
            WriteHeader("Hugging Face - Install Model");
            Console.WriteLine($"Model ID: {modelId}");

            if (!string.IsNullOrEmpty(ollamaName))
            {
                Console.WriteLine($"Ollama Name: {ollamaName}");
            }

            var huggingFaceService = _serviceProvider!.GetRequiredService<HuggingFaceService>();
            var success = await huggingFaceService.InstallModelAsync(modelId, ollamaName);

            if (success)
            {
                WriteColorLine($"Successfully installed model: {modelId}", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine($"Failed to install model: {modelId}", ConsoleColor.Red);
            }
        }, modelIdOption, ollamaNameOption);

        // Create huggingface list command
        var hfListCommand = new Command("list", "List installed models");

        hfListCommand.SetHandler(() =>
        {
            WriteHeader("Hugging Face - Installed Models");

            var huggingFaceService = _serviceProvider!.GetRequiredService<HuggingFaceService>();
            var installedModels = huggingFaceService.GetInstalledModels();

            if (installedModels.Count == 0)
            {
                WriteColorLine("No models installed", ConsoleColor.Yellow);
                return;
            }

            WriteColorLine($"Found {installedModels.Count} installed models:", ConsoleColor.Green);
            Console.WriteLine();

            foreach (var model in installedModels)
            {
                WriteColorLine(model, ConsoleColor.Cyan);
            }
        });

        // Add subcommands to huggingface command
        huggingFaceCommand.AddCommand(hfSearchCommand);
        huggingFaceCommand.AddCommand(hfBestCommand);
        huggingFaceCommand.AddCommand(hfDetailsCommand);
        huggingFaceCommand.AddCommand(hfDownloadCommand);
        huggingFaceCommand.AddCommand(hfInstallCommand);
        huggingFaceCommand.AddCommand(hfListCommand);

        // Create language command
        var languageCommand = new Command("language", "Generate and manage language specifications");

        // Create language ebnf command
        var languageEbnfCommand = new Command("ebnf", "Generate EBNF specification for TARS DSL");
        var outputOption = new Option<string>("--output", "Output file path");
        languageEbnfCommand.AddOption(outputOption);

        languageEbnfCommand.SetHandler(async (string output) =>
        {
            WriteHeader("TARS Language - EBNF Specification");

            var languageService = _serviceProvider!.GetRequiredService<LanguageSpecificationService>();
            var ebnf = await languageService.GenerateEbnfAsync();

            if (!string.IsNullOrEmpty(output))
            {
                var success = await languageService.SaveSpecificationToFileAsync(ebnf, output);
                if (success)
                {
                    WriteColorLine($"EBNF specification saved to: {output}", ConsoleColor.Green);
                }
                else
                {
                    WriteColorLine($"Failed to save EBNF specification to: {output}", ConsoleColor.Red);
                    Environment.Exit(1);
                }
            }
            else
            {
                // Print to console
                Console.WriteLine(ebnf);
            }
        }, outputOption);

        // Create language bnf command
        var languageBnfCommand = new Command("bnf", "Generate BNF specification for TARS DSL");
        languageBnfCommand.AddOption(outputOption);

        languageBnfCommand.SetHandler(async (string output) =>
        {
            WriteHeader("TARS Language - BNF Specification");

            var languageService = _serviceProvider!.GetRequiredService<LanguageSpecificationService>();
            var bnf = await languageService.GenerateBnfAsync();

            if (!string.IsNullOrEmpty(output))
            {
                var success = await languageService.SaveSpecificationToFileAsync(bnf, output);
                if (success)
                {
                    WriteColorLine($"BNF specification saved to: {output}", ConsoleColor.Green);
                }
                else
                {
                    WriteColorLine($"Failed to save BNF specification to: {output}", ConsoleColor.Red);
                    Environment.Exit(1);
                }
            }
            else
            {
                // Print to console
                Console.WriteLine(bnf);
            }
        }, outputOption);

        // Create language json-schema command
        var languageJsonSchemaCommand = new Command("json-schema", "Generate JSON schema for TARS DSL");
        languageJsonSchemaCommand.AddOption(outputOption);

        languageJsonSchemaCommand.SetHandler(async (string output) =>
        {
            WriteHeader("TARS Language - JSON Schema");

            var languageService = _serviceProvider!.GetRequiredService<LanguageSpecificationService>();
            var schema = await languageService.GenerateJsonSchemaAsync();

            if (!string.IsNullOrEmpty(output))
            {
                var success = await languageService.SaveSpecificationToFileAsync(schema, output);
                if (success)
                {
                    WriteColorLine($"JSON schema saved to: {output}", ConsoleColor.Green);
                }
                else
                {
                    WriteColorLine($"Failed to save JSON schema to: {output}", ConsoleColor.Red);
                    Environment.Exit(1);
                }
            }
            else
            {
                // Print to console
                Console.WriteLine(schema);
            }
        }, outputOption);

        // Create language docs command
        var languageDocsCommand = new Command("docs", "Generate markdown documentation for TARS DSL");
        languageDocsCommand.AddOption(outputOption);

        languageDocsCommand.SetHandler(async (string output) =>
        {
            WriteHeader("TARS Language - Documentation");

            var languageService = _serviceProvider!.GetRequiredService<LanguageSpecificationService>();
            var docs = await languageService.GenerateMarkdownDocumentationAsync();

            if (!string.IsNullOrEmpty(output))
            {
                var success = await languageService.SaveSpecificationToFileAsync(docs, output);
                if (success)
                {
                    WriteColorLine($"Documentation saved to: {output}", ConsoleColor.Green);
                }
                else
                {
                    WriteColorLine($"Failed to save documentation to: {output}", ConsoleColor.Red);
                    Environment.Exit(1);
                }
            }
            else
            {
                // Print to console
                Console.WriteLine(docs);
            }
        }, outputOption);

        // Add language subcommands
        languageCommand.AddCommand(languageEbnfCommand);
        languageCommand.AddCommand(languageBnfCommand);
        languageCommand.AddCommand(languageJsonSchemaCommand);
        languageCommand.AddCommand(languageDocsCommand);

        // Create docs-explore command
        var docsExploreCommand = new Command("docs-explore", "Explore TARS documentation");
        var searchOption = new Option<string>("--search", "Search query");
        var pathOption = new Option<string>("--path", "Documentation path");
        var listOption = new Option<bool>("--list", "List all documentation");

        docsExploreCommand.AddOption(searchOption);
        docsExploreCommand.AddOption(pathOption);
        docsExploreCommand.AddOption(listOption);

        docsExploreCommand.SetHandler((string search, string path, bool list) =>
        {
            WriteHeader("TARS Documentation Explorer");

            var docService = _serviceProvider!.GetRequiredService<DocumentationService>();

            if (!string.IsNullOrEmpty(path))
            {
                // Display a specific document
                var entry = docService.GetDocEntry(path);
                if (entry == null)
                {
                    WriteColorLine($"Documentation not found: {path}", ConsoleColor.Red);
                    return;
                }

                WriteColorLine(entry.Title, ConsoleColor.Cyan);
                Console.WriteLine();

                var formattedContent = docService.FormatMarkdownForConsole(entry.Content);
                Console.WriteLine(formattedContent);
            }
            else if (list || string.IsNullOrEmpty(search))
            {
                // List all documents
                var entries = docService.GetAllDocEntries();
                entries = entries.OrderBy(e => e.Path).ToList();

                WriteColorLine($"Found {entries.Count} documentation files:", ConsoleColor.Green);
                Console.WriteLine();

                foreach (var entry in entries)
                {
                    WriteColorLine(entry.Title, ConsoleColor.Cyan);
                    Console.WriteLine($"Path: {entry.Path}");
                    if (!string.IsNullOrEmpty(entry.Summary))
                    {
                        Console.WriteLine(entry.Summary);
                    }
                    Console.WriteLine();
                }

                Console.WriteLine("To view a specific document, use:");
                Console.WriteLine($"  tarscli docs --path <path>");
                Console.WriteLine("Example:");
                if (entries.Count > 0)
                {
                    Console.WriteLine($"  tarscli docs --path {entries[0].Path}");
                }
            }
            else
            {
                // Search documents
                var entries = docService.SearchDocEntries(search);
                entries = entries.OrderBy(e => e.Path).ToList();

                WriteColorLine($"Found {entries.Count} documents matching '{search}':", ConsoleColor.Green);
                Console.WriteLine();

                foreach (var entry in entries)
                {
                    WriteColorLine(entry.Title, ConsoleColor.Cyan);
                    Console.WriteLine($"Path: {entry.Path}");
                    if (!string.IsNullOrEmpty(entry.Summary))
                    {
                        Console.WriteLine(entry.Summary);
                    }
                    Console.WriteLine();
                }

                if (entries.Count > 0)
                {
                    Console.WriteLine("To view a specific document, use:");
                    Console.WriteLine($"  tarscli docs --path <path>");
                    Console.WriteLine("Example:");
                    Console.WriteLine($"  tarscli docs --path {entries[0].Path}");
                }
            }
        }, searchOption, pathOption, listOption);

        // Create demo command
        var demoCommand = new Command("demo", "Run a demonstration of TARS capabilities");
        var demoTypeOption = new Option<string>("--type", () => "all", "Type of demo to run (self-improvement, code-generation, language-specs, all)");
        var demoModelOption = new Option<string>("--model", () => "llama3", "Model to use for the demo");

        demoCommand.AddOption(demoTypeOption);
        demoCommand.AddOption(demoModelOption);

        demoCommand.SetHandler(async (string type, string model) =>
        {
            var demoService = _serviceProvider!.GetRequiredService<DemoService>();
            var success = await demoService.RunDemoAsync(type, model);

            if (!success)
            {
                Environment.Exit(1);
            }
        }, demoTypeOption, demoModelOption);

        // Create secrets command
        var secretsCommand = new Command("secrets", "Manage API keys and other secrets");

        // List secrets command
        var listSecretsCommand = new Command("list", "List all secret keys");
        listSecretsCommand.SetHandler(async () =>
        {
            WriteHeader("TARS Secrets - List");

            var secretsService = _serviceProvider!.GetRequiredService<SecretsService>();
            var keys = await secretsService.ListSecretKeysAsync();

            if (keys.Count == 0)
            {
                WriteColorLine("No secrets found.", ConsoleColor.Yellow);
                return;
            }

            WriteColorLine($"Found {keys.Count} secret(s):", ConsoleColor.Green);
            Console.WriteLine();

            foreach (var key in keys)
            {
                WriteColorLine(key, ConsoleColor.Cyan);
            }
        });

        // Set secret command
        var setSecretCommand = new Command("set", "Set a secret value");
        var setSecretKeyOption = new Option<string>("--key", "The secret key");
        var setSecretValueOption = new Option<string?>("--value", "The secret value (if not provided, will prompt for input)");

        setSecretCommand.AddOption(setSecretKeyOption);
        setSecretCommand.AddOption(setSecretValueOption);

        setSecretCommand.SetHandler(async (string key, string? value) =>
        {
            WriteHeader("TARS Secrets - Set");

            var secretsService = _serviceProvider!.GetRequiredService<SecretsService>();
            var userInteractionService = _serviceProvider!.GetRequiredService<UserInteractionService>();

            if (string.IsNullOrEmpty(value))
            {
                value = userInteractionService.AskForSecret($"Enter value for secret '{key}'");
            }

            var success = await secretsService.SetSecretAsync(key, value);

            if (success)
            {
                WriteColorLine($"Secret '{key}' set successfully.", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine($"Failed to set secret '{key}'.", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, setSecretKeyOption, setSecretValueOption);

        // Remove secret command
        var removeSecretCommand = new Command("remove", "Remove a secret");
        var removeSecretKeyOption = new Option<string>("--key", "The secret key");

        removeSecretCommand.AddOption(removeSecretKeyOption);

        removeSecretCommand.SetHandler(async (string key) =>
        {
            WriteHeader("TARS Secrets - Remove");

            var secretsService = _serviceProvider!.GetRequiredService<SecretsService>();
            var userInteractionService = _serviceProvider!.GetRequiredService<UserInteractionService>();

            var confirm = userInteractionService.AskForConfirmation($"Are you sure you want to remove secret '{key}'?", false);

            if (!confirm)
            {
                WriteColorLine("Operation cancelled.", ConsoleColor.Yellow);
                return;
            }

            var success = await secretsService.RemoveSecretAsync(key);

            if (success)
            {
                WriteColorLine($"Secret '{key}' removed successfully.", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine($"Failed to remove secret '{key}'. It may not exist.", ConsoleColor.Red);
            }
        }, removeSecretKeyOption);

        // Clear secrets command
        var clearSecretsCommand = new Command("clear", "Clear all secrets");

        clearSecretsCommand.SetHandler(async () =>
        {
            WriteHeader("TARS Secrets - Clear");

            var secretsService = _serviceProvider!.GetRequiredService<SecretsService>();
            var userInteractionService = _serviceProvider!.GetRequiredService<UserInteractionService>();

            var confirm = userInteractionService.AskForConfirmation("Are you sure you want to clear all secrets? This cannot be undone.", false);

            if (!confirm)
            {
                WriteColorLine("Operation cancelled.", ConsoleColor.Yellow);
                return;
            }

            var success = await secretsService.ClearSecretsAsync();

            if (success)
            {
                WriteColorLine("All secrets cleared successfully.", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Failed to clear secrets.", ConsoleColor.Red);
                Environment.Exit(1);
            }
        });

        // Add subcommands to secrets command
        secretsCommand.AddCommand(listSecretsCommand);
        secretsCommand.AddCommand(setSecretCommand);
        secretsCommand.AddCommand(removeSecretCommand);
        secretsCommand.AddCommand(clearSecretsCommand);

        // Create auto-improve command
        var autoImproveCommand = new Command("auto-improve", "Run autonomous self-improvement");
        var autoImproveTimeLimitOption = new Option<int>("--time-limit", () => 60, "Time limit in minutes (default: 60)");
        var autoImproveModelOption = new Option<string>("--model", () => "llama3", "Model to use for improvements");
        var statusOption = new Option<bool>("--status", "Show status of autonomous improvement");
        var stopOption = new Option<bool>("--stop", "Stop autonomous improvement");

        autoImproveCommand.AddOption(autoImproveTimeLimitOption);
        autoImproveCommand.AddOption(autoImproveModelOption);
        autoImproveCommand.AddOption(statusOption);
        autoImproveCommand.AddOption(stopOption);

        autoImproveCommand.SetHandler(async (int timeLimit, string model, bool status, bool stop) =>
        {
            var autoImprovementService = _serviceProvider!.GetRequiredService<AutoImprovementService>();

            if (status)
            {
                // Show status
                WriteHeader("TARS Autonomous Improvement - Status");

                var improvementStatus = autoImprovementService.GetStatus();

                if (improvementStatus.IsRunning)
                {
                    WriteColorLine("Status: Running", ConsoleColor.Green);
                    WriteColorLine($"Started: {improvementStatus.StartTime}", ConsoleColor.Cyan);
                    WriteColorLine($"Time Limit: {improvementStatus.TimeLimit.TotalMinutes} minutes", ConsoleColor.Cyan);
                    WriteColorLine($"Elapsed Time: {improvementStatus.ElapsedTime.TotalMinutes:F2} minutes", ConsoleColor.Cyan);
                    WriteColorLine($"Remaining Time: {improvementStatus.RemainingTime.TotalMinutes:F2} minutes", ConsoleColor.Cyan);
                    WriteColorLine($"Files Processed: {improvementStatus.FilesProcessed}", ConsoleColor.Cyan);
                    WriteColorLine($"Files Remaining: {improvementStatus.FilesRemaining}", ConsoleColor.Cyan);
                    WriteColorLine($"Current File: {improvementStatus.CurrentFile ?? "None"}", ConsoleColor.Cyan);
                    WriteColorLine($"Last Improved File: {improvementStatus.LastImprovedFile ?? "None"}", ConsoleColor.Cyan);
                    WriteColorLine($"Total Improvements: {improvementStatus.TotalImprovements}", ConsoleColor.Cyan);
                }
                else
                {
                    WriteColorLine("Status: Not Running", ConsoleColor.Yellow);

                    if (improvementStatus.FilesProcessed > 0)
                    {
                        WriteColorLine($"Last Run Statistics:", ConsoleColor.Cyan);
                        WriteColorLine($"Files Processed: {improvementStatus.FilesProcessed}", ConsoleColor.Cyan);
                        WriteColorLine($"Files Remaining: {improvementStatus.FilesRemaining}", ConsoleColor.Cyan);
                        WriteColorLine($"Last Improved File: {improvementStatus.LastImprovedFile ?? "None"}", ConsoleColor.Cyan);
                        WriteColorLine($"Total Improvements: {improvementStatus.TotalImprovements}", ConsoleColor.Cyan);
                    }
                }

                return;
            }

            if (stop)
            {
                // Stop improvement
                WriteHeader("TARS Autonomous Improvement - Stop");

                var success = autoImprovementService.Stop();

                if (success)
                {
                    WriteColorLine("Autonomous improvement stopped successfully.", ConsoleColor.Green);
                    WriteColorLine("The process will complete the current file and then exit.", ConsoleColor.Yellow);
                }
                else
                {
                    WriteColorLine("Failed to stop autonomous improvement. It may not be running.", ConsoleColor.Red);
                }

                return;
            }

            // Start improvement
            WriteHeader("TARS Autonomous Improvement - Start");

            WriteColorLine($"Starting autonomous improvement with time limit of {timeLimit} minutes", ConsoleColor.Cyan);
            WriteColorLine($"Using model: {model}", ConsoleColor.Cyan);
            Console.WriteLine();

            var startSuccess = await autoImprovementService.StartAsync(timeLimit, model);

            if (startSuccess)
            {
                WriteColorLine("Autonomous improvement started successfully.", ConsoleColor.Green);
                WriteColorLine("The process will run in the background until the time limit is reached.", ConsoleColor.Yellow);
                WriteColorLine("You can check the status with 'tarscli auto-improve --status'.", ConsoleColor.Yellow);
                WriteColorLine("You can stop the process with 'tarscli auto-improve --stop'.", ConsoleColor.Yellow);
            }
            else
            {
                WriteColorLine("Failed to start autonomous improvement.", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, autoImproveTimeLimitOption, autoImproveModelOption, statusOption, stopOption);

        // Create slack command
        var slackCommand = new Command("slack", "Manage Slack integration");

        // Set webhook command
        var setWebhookCommand = new Command("set-webhook", "Set the Slack webhook URL");
        var webhookUrlOption = new Option<string>("--url", "The webhook URL");

        setWebhookCommand.AddOption(webhookUrlOption);

        setWebhookCommand.SetHandler(async (string url) =>
        {
            WriteHeader("TARS Slack Integration - Set Webhook");

            var slackService = _serviceProvider!.GetRequiredService<SlackIntegrationService>();
            var success = await slackService.SetWebhookUrlAsync(url);

            if (success)
            {
                WriteColorLine("Slack webhook URL set successfully.", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Failed to set Slack webhook URL.", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, webhookUrlOption);

        // Test command
        var testCommand = new Command("test", "Test the Slack integration");
        var messageOption = new Option<string>("--message", () => "Test message from TARS", "The message to send");
        var channelOption = new Option<string?>("--channel", "The channel to send to (optional)");

        testCommand.AddOption(messageOption);
        testCommand.AddOption(channelOption);

        testCommand.SetHandler(async (string message, string? channel) =>
        {
            WriteHeader("TARS Slack Integration - Test");

            var slackService = _serviceProvider!.GetRequiredService<SlackIntegrationService>();

            if (!slackService.IsEnabled())
            {
                WriteColorLine("Slack integration is not enabled. Use 'tarscli slack set-webhook' to set the webhook URL.", ConsoleColor.Red);
                Environment.Exit(1);
                return;
            }

            WriteColorLine($"Sending message to Slack{(channel != null ? $" channel {channel}" : "")}...", ConsoleColor.Cyan);
            var success = await slackService.PostMessageAsync(message, channel);

            if (success)
            {
                WriteColorLine("Message sent successfully.", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Failed to send message.", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, messageOption, channelOption);

        // Announce command
        var announceCommand = new Command("announce", "Post an announcement to Slack");
        var titleOption = new Option<string>("--title", "The announcement title");
        var announcementMessageOption = new Option<string>("--message", "The announcement message");
        var announceChannelOption = new Option<string?>("--channel", "The channel to send to (optional)");

        announceCommand.AddOption(titleOption);
        announceCommand.AddOption(announcementMessageOption);
        announceCommand.AddOption(announceChannelOption);

        announceCommand.SetHandler(async (string title, string message, string? channel) =>
        {
            WriteHeader("TARS Slack Integration - Announce");

            var slackService = _serviceProvider!.GetRequiredService<SlackIntegrationService>();

            if (!slackService.IsEnabled())
            {
                WriteColorLine("Slack integration is not enabled. Use 'tarscli slack set-webhook' to set the webhook URL.", ConsoleColor.Red);
                Environment.Exit(1);
                return;
            }

            WriteColorLine($"Posting announcement to Slack{(channel != null ? $" channel {channel}" : "")}...", ConsoleColor.Cyan);
            var success = await slackService.PostAnnouncementAsync(title, message, channel);

            if (success)
            {
                WriteColorLine("Announcement posted successfully.", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Failed to post announcement.", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, titleOption, announcementMessageOption, announceChannelOption);

        // Feature command
        var featureCommand = new Command("feature", "Post a feature update to Slack");
        var featureNameOption = new Option<string>("--name", "The feature name");
        var featureDescriptionOption = new Option<string>("--description", "The feature description");
        var featureChannelOption = new Option<string?>("--channel", "The channel to send to (optional)");

        featureCommand.AddOption(featureNameOption);
        featureCommand.AddOption(featureDescriptionOption);
        featureCommand.AddOption(featureChannelOption);

        featureCommand.SetHandler(async (string name, string description, string? channel) =>
        {
            WriteHeader("TARS Slack Integration - Feature Update");

            var slackService = _serviceProvider!.GetRequiredService<SlackIntegrationService>();

            if (!slackService.IsEnabled())
            {
                WriteColorLine("Slack integration is not enabled. Use 'tarscli slack set-webhook' to set the webhook URL.", ConsoleColor.Red);
                Environment.Exit(1);
                return;
            }

            WriteColorLine($"Posting feature update to Slack{(channel != null ? $" channel {channel}" : "")}...", ConsoleColor.Cyan);
            var success = await slackService.PostFeatureUpdateAsync(name, description, channel);

            if (success)
            {
                WriteColorLine("Feature update posted successfully.", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Failed to post feature update.", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, featureNameOption, featureDescriptionOption, featureChannelOption);

        // Milestone command
        var milestoneCommand = new Command("milestone", "Post a milestone to Slack");
        var milestoneNameOption = new Option<string>("--name", "The milestone name");
        var milestoneDescriptionOption = new Option<string>("--description", "The milestone description");
        var milestoneChannelOption = new Option<string?>("--channel", "The channel to send to (optional)");

        milestoneCommand.AddOption(milestoneNameOption);
        milestoneCommand.AddOption(milestoneDescriptionOption);
        milestoneCommand.AddOption(milestoneChannelOption);

        milestoneCommand.SetHandler(async (string name, string description, string? channel) =>
        {
            WriteHeader("TARS Slack Integration - Milestone");

            var slackService = _serviceProvider!.GetRequiredService<SlackIntegrationService>();

            if (!slackService.IsEnabled())
            {
                WriteColorLine("Slack integration is not enabled. Use 'tarscli slack set-webhook' to set the webhook URL.", ConsoleColor.Red);
                Environment.Exit(1);
                return;
            }

            WriteColorLine($"Posting milestone to Slack{(channel != null ? $" channel {channel}" : "")}...", ConsoleColor.Cyan);
            var success = await slackService.PostMilestoneAsync(name, description, channel);

            if (success)
            {
                WriteColorLine("Milestone posted successfully.", ConsoleColor.Green);
            }
            else
            {
                WriteColorLine("Failed to post milestone.", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, milestoneNameOption, milestoneDescriptionOption, milestoneChannelOption);

        // Add subcommands to slack command
        slackCommand.AddCommand(setWebhookCommand);
        slackCommand.AddCommand(testCommand);
        slackCommand.AddCommand(announceCommand);
        slackCommand.AddCommand(featureCommand);
        slackCommand.AddCommand(milestoneCommand);

        // Create speech command
        var speechCommand = new Command("speech", "Text-to-speech functionality");

        // Speak command
        var speakCommand = new Command("speak", "Speak text using text-to-speech");
        var textOption = new Option<string>("--text", "The text to speak");
        var voiceOption = new Option<string?>("--voice", "The voice model to use");
        var languageOption = new Option<string?>("--language", "The language code (e.g., en, fr, es)");
        var speakerWavOption = new Option<string?>("--speaker-wav", "Path to speaker reference audio for voice cloning");
        var agentIdOption = new Option<string?>("--agent-id", "Agent identifier");

        speakCommand.AddOption(textOption);
        speakCommand.AddOption(voiceOption);
        speakCommand.AddOption(languageOption);
        speakCommand.AddOption(speakerWavOption);
        speakCommand.AddOption(agentIdOption);

        speakCommand.SetHandler((string text, string? voice, string? language, string? speakerWav, string? agentId) =>
        {
            WriteHeader("TARS Speech - Speak");

            var speechService = _serviceProvider!.GetRequiredService<TarsSpeechService>();
            speechService.Speak(text, voice, language, speakerWav, agentId);

            WriteColorLine("Speech completed.", ConsoleColor.Green);
        }, textOption, voiceOption, languageOption, speakerWavOption, agentIdOption);

        // List voices command
        var listVoicesCommand = new Command("list-voices", "List available voice models");

        listVoicesCommand.SetHandler(() =>
        {
            WriteHeader("TARS Speech - Available Voices");

            var speechService = _serviceProvider!.GetRequiredService<TarsSpeechService>();
            speechService.Initialize();

            WriteColorLine("Standard voices:", ConsoleColor.Cyan);
            WriteColorLine("  English: tts_models/en/ljspeech/tacotron2-DDC", ConsoleColor.White);
            WriteColorLine("  French: tts_models/fr/mai/tacotron2-DDC", ConsoleColor.White);
            WriteColorLine("  Spanish: tts_models/es/mai/tacotron2-DDC", ConsoleColor.White);
            WriteColorLine("  German: tts_models/de/thorsten/tacotron2-DDC", ConsoleColor.White);
            WriteColorLine("  Italian: tts_models/it/mai_female/glow-tts", ConsoleColor.White);
            WriteColorLine("  Dutch: tts_models/nl/mai/tacotron2-DDC", ConsoleColor.White);
            WriteColorLine("  Russian: tts_models/ru/multi-dataset/vits", ConsoleColor.White);

            WriteColorLine("\nVoice cloning:", ConsoleColor.Cyan);
            WriteColorLine("  Multi-speaker: tts_models/multilingual/multi-dataset/your_tts", ConsoleColor.White);

            WriteColorLine("\nTo use voice cloning, provide a speaker reference audio file with --speaker-wav", ConsoleColor.Yellow);
        });

        // Configure command
        var configureCommand = new Command("configure", "Configure speech settings");
        var enabledOption = new Option<bool>("--enabled", () => true, "Whether speech is enabled");
        var defaultVoiceOption = new Option<string?>("--default-voice", "Default voice model");
        var defaultLanguageOption = new Option<string?>("--default-language", "Default language code");
        var preloadVoicesOption = new Option<bool?>("--preload-voices", "Whether to preload voice models");
        var maxConcurrencyOption = new Option<int?>("--max-concurrency", "Maximum concurrent speech operations");

        configureCommand.AddOption(enabledOption);
        configureCommand.AddOption(defaultVoiceOption);
        configureCommand.AddOption(defaultLanguageOption);
        configureCommand.AddOption(preloadVoicesOption);
        configureCommand.AddOption(maxConcurrencyOption);

        configureCommand.SetHandler((bool enabled, string? defaultVoice, string? defaultLanguage, bool? preloadVoices, int? maxConcurrency) =>
        {
            WriteHeader("TARS Speech - Configure");

            var speechService = _serviceProvider!.GetRequiredService<TarsSpeechService>();
            speechService.Configure(enabled, defaultVoice, defaultLanguage, preloadVoices, maxConcurrency);

            WriteColorLine("Speech configuration updated.", ConsoleColor.Green);
        }, enabledOption, defaultVoiceOption, defaultLanguageOption, preloadVoicesOption, maxConcurrencyOption);

        // Add subcommands to speech command
        speechCommand.AddCommand(speakCommand);
        speechCommand.AddCommand(listVoicesCommand);
        speechCommand.AddCommand(configureCommand);

        rootCommand.AddCommand(selfAnalyzeCommand);
        rootCommand.AddCommand(selfProposeCommand);
        rootCommand.AddCommand(selfRewriteCommand);
        rootCommand.AddCommand(learningCommand);
        rootCommand.AddCommand(templateCommand);
        rootCommand.AddCommand(workflowCommand);
        rootCommand.AddCommand(huggingFaceCommand);
        rootCommand.AddCommand(languageCommand);
        rootCommand.AddCommand(docsExploreCommand);
        rootCommand.AddCommand(demoCommand);
        rootCommand.AddCommand(secretsCommand);
        rootCommand.AddCommand(autoImproveCommand);
        rootCommand.AddCommand(slackCommand);
        rootCommand.AddCommand(speechCommand);

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