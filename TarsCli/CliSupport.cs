using System.CommandLine.Invocation;
using System.Diagnostics;
using System.Text.Json;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using TarsCli.Services;
using TarsCli.Commands;
using TarsCli.Controllers;

// For DifficultyLevel enum

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
        var helpCommand = new TarsCommand("help", "Display detailed help information about TARS CLI");
        helpCommand.SetHandler(() =>
        {
            WriteHeader("=== TARS CLI Help ===");
            Console.WriteLine("TARS (Transformative Autonomous Reasoning System) is a tool for processing files through AI models.");

            WriteHeader("Autonomous Improvement Status");
            Console.WriteLine("Current Status: TARS is in Phase 1 of development toward full autonomy.");
            Console.WriteLine("We are approximately 1-2 months away from enabling TARS to auto-improve itself in full autonomy using metascripts.");
            Console.WriteLine("\nSpecifically for improving documentation in docs/Explorations/v1/Chats and docs/Explorations/Reflections:");
            Console.WriteLine(" - The basic framework for autonomous improvement is implemented");
            Console.WriteLine(" - The workflow engine and state management are in place");
            Console.WriteLine(" - We need to implement real functionality for knowledge extraction from exploration chats");
            Console.WriteLine(" - We need to enhance code analysis to use the extracted knowledge");
            Console.WriteLine(" - We need to integrate the metascript system with the self-improvement system");
            Console.WriteLine("\nSee docs/STATUS.md and docs/CURRENT_STATE.md for more details on our progress toward full autonomy.");

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
            WriteCommand("self-diagnose", "Run comprehensive self-diagnostics and generate a report");
            WriteCommand("learning", "View and manage learning data");
            WriteCommand("template", "Manage TARS templates");
            WriteCommand("workflow", "Run a multi-agent workflow for a task");
            WriteCommand("huggingface", "Interact with Hugging Face models");
            WriteCommand("language", "Generate and manage language specifications");
            WriteCommand("docs-explore", "Explore TARS documentation");
            WriteCommand("demo", "Run a demonstration of TARS capabilities");
            WriteCommand("secrets", "Manage API keys and other secrets");
            WriteCommand("auto-improve", "Run autonomous self-improvement");
            WriteCommand("improve-explorations", "Improve TARS Explorations documentation using metascripts");
            WriteCommand("knowledge-apply", "Apply knowledge from the knowledge base to improve files");
            WriteCommand("knowledge-integrate", "Integrate knowledge with other TARS systems");
            WriteCommand("autonomous", "Autonomous improvement of TARS");
            WriteCommand("slack", "Manage Slack integration");
            WriteCommand("speech", "Text-to-speech functionality");
            WriteCommand("chat", "Interactive chat bot");
            WriteCommand("doc-extract", "Extract knowledge from documentation and integrate it with the RetroactionLoop");
            WriteCommand("knowledge-viz", "Visualize the knowledge base");
            WriteCommand("knowledge-test", "Generate tests from the knowledge base");
            WriteCommand("auto-improve-workflow", "Run autonomous improvement workflow");
            WriteCommand("deep-thinking", "Generate deep thinking explorations");
            WriteCommand("console-capture", "Capture console output and improve code");

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
            WriteExample("tarscli learning plan generate --name \"C# Basics\" --topic \"C# Programming\" --skill-level Beginner --goals \"Learn syntax\" \"Build applications\" --hours 20");
            WriteExample("tarscli learning plan list");
            WriteExample("tarscli learning plan view --id plan-id");
            WriteExample("tarscli learning course generate --title \"Introduction to C#\" --description \"Learn C# programming\" --topic \"C#\" --difficulty Beginner --hours 20 --audience \"Beginners\" \"Students\"");
            WriteExample("tarscli learning course list");
            WriteExample("tarscli learning course view --id course-id");
            WriteExample("tarscli learning tutorial add --title \"Getting Started with C#\" --description \"A beginner's guide\" --content \"# Tutorial content\" --category \"Programming\" --difficulty Beginner --tags \"C#\" \"Beginner\"");
            WriteExample("tarscli learning tutorial list");
            WriteExample("tarscli learning tutorial list --category \"Programming\" --difficulty Beginner --tag \"C#\"");
            WriteExample("tarscli learning tutorial view --id tutorial-id");
            WriteExample("tarscli learning tutorial categorize --ids tutorial-id1 tutorial-id2 --category \"New Category\"");
            WriteExample("tarscli template list");
            WriteExample("tarscli template create --name my_template.json --file path/to/template.json");
            WriteExample("tarscli workflow --task \"Create a simple web API in C#\"");
            WriteExample("tarscli mcp start");
            WriteExample("tarscli mcp status");
            WriteExample("tarscli mcp configure --port 8999 --auto-execute --auto-code");
            WriteExample("tarscli mcp augment");
            WriteExample("tarscli mcp execute \"echo Hello, World!\"");
            WriteExample("tarscli mcp code path/to/file.cs \"public class MyClass { }\"");
            WriteExample("tarscli mcp conversations --source augment --count 5");
            WriteExample("tarscli mcp conversations --open");
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
            WriteExample("tarscli demo --type chatbot");
            WriteExample("tarscli demo --type deep-thinking");
            WriteExample("tarscli demo --type learning-plan");
            WriteExample("tarscli demo --type course-generator");
            WriteExample("tarscli demo --type tutorial-organizer");
            WriteExample("tarscli demo --type speech");
            WriteExample("tarscli demo --type mcp");
            WriteExample("tarscli demo --type all");
            WriteExample("tarscli secrets list");
            WriteExample("tarscli secrets set --key HuggingFace:ApiKey");
            WriteExample("tarscli secrets remove --key OpenAI:ApiKey");
            WriteExample("tarscli auto-improve --time-limit 60 --model llama3");
            WriteExample("tarscli auto-improve --status");
            WriteExample("tarscli auto-improve --stop");
            WriteExample("tarscli auto-improve-workflow --start --directories docs/Explorations/v1/Chats docs/Explorations/Reflections --max-duration 60");
            WriteExample("tarscli auto-improve-workflow --status");
            WriteExample("tarscli auto-improve-workflow --report");
            WriteExample("tarscli auto-improve-workflow --stop");
            WriteExample("tarscli improve-explorations --time 60 --model llama3");
            WriteExample("tarscli improve-explorations --chats-only");
            WriteExample("tarscli improve-explorations --file docs/Explorations/v1/Chats/ChatGPT-TARS-Project-Implications.md");
            WriteExample("tarscli knowledge-apply --extract docs/Explorations/v1/Chats/ChatGPT-TARS-Project-Implications.md");
            WriteExample("tarscli knowledge-apply --file TarsCli/Services/DslService.cs");
            WriteExample("tarscli knowledge-apply --directory TarsCli/Services --pattern *.cs --recursive");
            WriteExample("tarscli knowledge-apply --report");
            WriteExample("tarscli knowledge-integrate --metascript --target TarsCli/Services --pattern *.cs");
            WriteExample("tarscli knowledge-integrate --cycle --exploration docs/Explorations/v1/Chats --target TarsCli/Services");
            WriteExample("tarscli knowledge-integrate --retroaction --exploration docs/Explorations/v1/Chats --target TarsCli/Services");
            WriteExample("tarscli autonomous start --exploration docs/Explorations/v1/Chats docs/Explorations/Reflections --target TarsCli/Services TarsCli/Commands --duration 60");
            WriteExample("tarscli autonomous status");
            WriteExample("tarscli autonomous stop");
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
            WriteExample("tarscli console-capture --start");
            WriteExample("tarscli console-capture --stop");
            WriteExample("tarscli console-capture --analyze path/to/file.cs --apply");

            Console.WriteLine("\nFor more information, visit: https://github.com/yourusername/tars");
        });

        // Create process command
        var processCommand = new TarsCommand("process", "Process a file through the TARS retroaction loop")
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
        var docsCommand = new TarsCommand("docs", "Process documentation files in the docs directory")
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
        var diagnosticsCommand = new TarsCommand("diagnostics", "Run system diagnostics and check environment setup");

        // Create GPU subcommand
        var gpuCommand = new TarsCommand("gpu", "Check GPU capabilities and Ollama GPU acceleration");
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
        var setupCommand = new TarsCommand("setup", "Run the prerequisites setup script");
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
        var modelsCommand = new TarsCommand("models", "Manage Ollama models");
        var installCommand = new TarsCommand("install", "Install required models");

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
        var mcpCommand = new TarsCommand("mcp", "Model Context Protocol - interact with AI models and tools");

        // Add subcommands for MCP
        var startCommand = new TarsCommand("start", "Start the MCP server");
        var stopCommand = new TarsCommand("stop", "Stop the MCP server");
        var statusCommand = new TarsCommand("status", "Show MCP server status");

        // Set handlers for MCP subcommands
        startCommand.SetHandler(async () =>
        {
            WriteHeader("MCP - Start Server");
            var mcpService = _serviceProvider!.GetRequiredService<TarsMcpService>();
            await mcpService.StartAsync();
            WriteColorLine("MCP server started successfully", ConsoleColor.Green);
            Console.WriteLine("The MCP server is now running and ready to accept requests.");
            Console.WriteLine("Other applications like Augment Code can now connect to TARS.");
        });

        stopCommand.SetHandler(() =>
        {
            WriteHeader("MCP - Stop Server");
            var mcpService = _serviceProvider!.GetRequiredService<TarsMcpService>();
            mcpService.Stop();
            WriteColorLine("MCP server stopped", ConsoleColor.Yellow);
        });

        statusCommand.SetHandler(() =>
        {
            WriteHeader("MCP - Server Status");
            var mcpService = _serviceProvider!.GetRequiredService<McpService>();
            var configuration = _serviceProvider!.GetRequiredService<IConfiguration>();
            var port = configuration.GetValue<int>("Tars:Mcp:Port", 8999);
            var autoExecuteEnabled = configuration.GetValue<bool>("Tars:Mcp:AutoExecuteEnabled", false);
            var autoCodeEnabled = configuration.GetValue<bool>("Tars:Mcp:AutoCodeEnabled", false);

            Console.WriteLine($"MCP Server URL: http://localhost:{port}/");
            Console.Write("Auto-execute commands: ");
            WriteColorLine(autoExecuteEnabled ? "Enabled" : "Disabled", autoExecuteEnabled ? ConsoleColor.Green : ConsoleColor.Yellow);
            Console.Write("Auto-code generation: ");
            WriteColorLine(autoCodeEnabled ? "Enabled" : "Disabled", autoCodeEnabled ? ConsoleColor.Green : ConsoleColor.Yellow);

            Console.WriteLine("\nAvailable MCP Actions:");
            WriteColorLine("  execute - Execute terminal commands", ConsoleColor.Cyan);
            WriteColorLine("  code - Generate and save code", ConsoleColor.Cyan);
            WriteColorLine("  status - Get system status", ConsoleColor.Cyan);
            WriteColorLine("  tars - Execute TARS-specific operations", ConsoleColor.Cyan);
            WriteColorLine("  ollama - Execute Ollama operations", ConsoleColor.Cyan);
            WriteColorLine("  self-improve - Execute self-improvement operations", ConsoleColor.Cyan);
            WriteColorLine("  slack - Execute Slack operations", ConsoleColor.Cyan);
            WriteColorLine("  speech - Execute speech operations", ConsoleColor.Cyan);
        });

        // Add execute command for backward compatibility
        var executeCommand = new TarsCommand("execute", "Execute a terminal command without asking for permission")
        {
            new Argument<string>("command", "The command to execute")
        };

        executeCommand.SetHandler(async (command) =>
        {
            WriteHeader("MCP - Execute Command");
            Console.WriteLine($"Command: {command}");

            var mcpService = _serviceProvider!.GetRequiredService<McpService>();
            var configuration = _serviceProvider!.GetRequiredService<IConfiguration>();
            var result = await mcpService.SendRequestAsync("http://localhost:" + configuration.GetValue<int>("Tars:Mcp:Port", 8999) + "/", "execute", new { command });
            Console.WriteLine(result.GetProperty("output").GetString());
        }, new Argument<string>("command"));

        // Add code command for backward compatibility
        var codeCommand = new TarsCommand("code", "Generate and save code without asking for permission");
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

            var mcpService = _serviceProvider!.GetRequiredService<McpService>();
            var configuration = _serviceProvider!.GetRequiredService<IConfiguration>();
            var result = await mcpService.SendRequestAsync("http://localhost:" + configuration.GetValue<int>("Tars:Mcp:Port", 8999) + "/", "code", new { filePath = file, content });
            Console.WriteLine(result.GetProperty("message").GetString());
        }, fileArgument, contentArgument);

        // Add configure command for Augment integration
        var configureCommand = new TarsCommand("configure", "Configure MCP settings");
        var portOption = new Option<int>("--port", () => 8999, "Port for the MCP server");
        var autoExecuteOption = new Option<bool>("--auto-execute", "Enable auto-execution of commands");
        var autoCodeOption = new Option<bool>("--auto-code", "Enable auto-code generation");

        configureCommand.AddOption(portOption);
        configureCommand.AddOption(autoExecuteOption);
        configureCommand.AddOption(autoCodeOption);

        configureCommand.SetHandler((int port, bool autoExecute, bool autoCode) =>
        {
            WriteHeader("MCP - Configure");

            // Update appsettings.json
            var appSettingsPath = Path.Combine(AppContext.BaseDirectory, "appsettings.json");
            var json = File.ReadAllText(appSettingsPath);
            var settings = JsonDocument.Parse(json);
            var root = new Dictionary<string, object>();

            // Copy existing settings
            foreach (var property in settings.RootElement.EnumerateObject())
            {
                if (property.Name == "Tars")
                {
                    var tarsSettings = JsonSerializer.Deserialize<Dictionary<string, object>>(property.Value.GetRawText());

                    if (tarsSettings.ContainsKey("Mcp"))
                    {
                        var mcpSettings = JsonSerializer.Deserialize<Dictionary<string, object>>(tarsSettings["Mcp"].ToString());
                        mcpSettings["Port"] = port;
                        mcpSettings["AutoExecuteEnabled"] = autoExecute;
                        mcpSettings["AutoCodeEnabled"] = autoCode;
                        tarsSettings["Mcp"] = mcpSettings;
                    }
                    else
                    {
                        tarsSettings["Mcp"] = new Dictionary<string, object>
                        {
                            ["Port"] = port,
                            ["AutoExecuteEnabled"] = autoExecute,
                            ["AutoCodeEnabled"] = autoCode
                        };
                    }

                    root["Tars"] = tarsSettings;
                }
                else
                {
                    root[property.Name] = JsonSerializer.Deserialize<object>(property.Value.GetRawText());
                }
            }

            // If Tars section doesn't exist, create it
            if (!root.ContainsKey("Tars"))
            {
                root["Tars"] = new Dictionary<string, object>
                {
                    ["Mcp"] = new Dictionary<string, object>
                    {
                        ["Port"] = port,
                        ["AutoExecuteEnabled"] = autoExecute,
                        ["AutoCodeEnabled"] = autoCode
                    }
                };
            }

            // Save the updated settings
            var newJson = JsonSerializer.Serialize(root, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(appSettingsPath, newJson);

            WriteColorLine("MCP configuration updated successfully", ConsoleColor.Green);
            Console.WriteLine($"Port: {port}");
            Console.WriteLine($"Auto-execute commands: {(autoExecute ? "Enabled" : "Disabled")}");
            Console.WriteLine($"Auto-code generation: {(autoCode ? "Enabled" : "Disabled")}");
            Console.WriteLine("\nRestart the application for changes to take effect.");
        }, portOption, autoExecuteOption, autoCodeOption);

        // Add augment command for configuring Augment Code integration
        var augmentCommand = new TarsCommand("augment", "Configure Augment Code integration");

        augmentCommand.SetHandler(() =>
        {
            WriteHeader("MCP - Configure Augment Integration");

            var configuration = _serviceProvider!.GetRequiredService<IConfiguration>();
            var port = configuration.GetValue<int>("Tars:Mcp:Port", 8999);
            var vscodeSettingsPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".vscode", "settings.json");

            if (!File.Exists(vscodeSettingsPath))
            {
                WriteColorLine("VS Code settings file not found. Creating a new one.", ConsoleColor.Yellow);
                Directory.CreateDirectory(Path.GetDirectoryName(vscodeSettingsPath));
                File.WriteAllText(vscodeSettingsPath, "{}");
            }

            var json = File.ReadAllText(vscodeSettingsPath);
            var settings = JsonDocument.Parse(json);
            var root = new Dictionary<string, object>();

            // Copy existing settings
            foreach (var property in settings.RootElement.EnumerateObject())
            {
                root[property.Name] = JsonSerializer.Deserialize<object>(property.Value.GetRawText());
            }

            // Add or update augment.advanced settings
            if (root.ContainsKey("augment.advanced"))
            {
                var advancedSettings = JsonSerializer.Deserialize<Dictionary<string, object>>(root["augment.advanced"].ToString());

                if (advancedSettings.ContainsKey("mcpServers"))
                {
                    var mcpServers = JsonSerializer.Deserialize<List<object>>(advancedSettings["mcpServers"].ToString());

                    // Check if TARS MCP server already exists
                    bool found = false;
                    for (int i = 0; i < mcpServers.Count; i++)
                    {
                        var server = JsonSerializer.Deserialize<Dictionary<string, object>>(mcpServers[i].ToString());
                        if (server.ContainsKey("name") && server["name"].ToString() == "tars")
                        {
                            // Update existing server
                            mcpServers[i] = new Dictionary<string, object>
                            {
                                ["name"] = "tars",
                                ["url"] = $"http://localhost:{port}/"
                            };
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        // Add new server
                        mcpServers.Add(new Dictionary<string, object>
                        {
                            ["name"] = "tars",
                            ["url"] = $"http://localhost:{port}/"
                        });
                    }

                    advancedSettings["mcpServers"] = mcpServers;
                }
                else
                {
                    // Create new mcpServers array
                    advancedSettings["mcpServers"] = new List<object>
                    {
                        new Dictionary<string, object>
                        {
                            ["name"] = "tars",
                            ["url"] = $"http://localhost:{port}/"
                        }
                    };
                }

                root["augment.advanced"] = advancedSettings;
            }
            else
            {
                // Create new augment.advanced section
                root["augment.advanced"] = new Dictionary<string, object>
                {
                    ["mcpServers"] = new List<object>
                    {
                        new Dictionary<string, object>
                        {
                            ["name"] = "tars",
                            ["url"] = $"http://localhost:{port}/"
                        }
                    }
                };
            }

            // Save the updated settings
            var newJson = JsonSerializer.Serialize(root, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(vscodeSettingsPath, newJson);

            WriteColorLine("Augment Code integration configured successfully", ConsoleColor.Green);
            Console.WriteLine($"TARS MCP server added to Augment Code with URL: http://localhost:{port}/");
            Console.WriteLine("\nTo use TARS from Augment Code:");
            Console.WriteLine("1. Start the MCP server with 'tarscli mcp start'");
            Console.WriteLine("2. In VS Code, use the command '@tars' to interact with TARS");
        });

        // Add subcommands to MCP command
        mcpCommand.AddCommand(startCommand);
        mcpCommand.AddCommand(stopCommand);
        mcpCommand.AddCommand(statusCommand);
        // Add conversations command to view conversation history
        var conversationsCommand = new TarsCommand("conversations", "View conversation history");
        var sourceOption = new Option<string?>("--source", "Filter conversations by source (e.g., 'augment')");
        var countOption = new Option<int>("--count", () => 10, "Number of conversations to show");
        var openOption = new Option<bool>("--open", "Open the conversation log in the default browser");

        conversationsCommand.AddOption(sourceOption);
        conversationsCommand.AddOption(countOption);
        conversationsCommand.AddOption(openOption);

        conversationsCommand.SetHandler(async (string? source, int count, bool open) =>
        {
            WriteHeader("MCP - Conversation History");

            var loggingService = _serviceProvider!.GetRequiredService<ConversationLoggingService>();

            if (open)
            {
                var logPath = loggingService.GetAugmentLogPath();
                if (File.Exists(logPath))
                {
                    WriteColorLine($"Opening conversation log: {logPath}", ConsoleColor.Green);
                    Process.Start(new ProcessStartInfo
                    {
                        FileName = logPath,
                        UseShellExecute = true
                    });
                }
                else
                {
                    WriteColorLine($"Conversation log file not found: {logPath}", ConsoleColor.Red);
                }
                return;
            }

            var conversations = await loggingService.GetRecentConversationsAsync(count, source);

            if (conversations.Count == 0)
            {
                WriteColorLine("No conversations found.", ConsoleColor.Yellow);
                return;
            }

            WriteColorLine($"Found {conversations.Count} conversations" + (source != null ? $" from {source}" : ""), ConsoleColor.Green);
            Console.WriteLine();

            foreach (var conversation in conversations)
            {
                var json = JsonSerializer.Serialize(conversation, new JsonSerializerOptions { WriteIndented = true });
                var conversationDict = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json);

                var timestamp = conversationDict["Timestamp"].GetString();
                var conversationSource = conversationDict["Source"].GetString();
                var action = conversationDict["Action"].GetString();

                WriteColorLine($"[{timestamp}] Source: {conversationSource}, Action: {action}", ConsoleColor.Cyan);

                // Show a condensed version of the request and response
                var request = conversationDict["Request"].ToString();
                var response = conversationDict["Response"].ToString();

                WriteColorLine("Request: " + (request.Length > 100 ? request.Substring(0, 100) + "..." : request), ConsoleColor.White);
                WriteColorLine("Response: " + (response.Length > 100 ? response.Substring(0, 100) + "..." : response), ConsoleColor.White);
                Console.WriteLine();
            }

            WriteColorLine("To view the full conversation log, use the --open option.", ConsoleColor.Yellow);
        }, sourceOption, countOption, openOption);

        mcpCommand.AddCommand(executeCommand);
        mcpCommand.AddCommand(codeCommand);
        mcpCommand.AddCommand(configureCommand);
        mcpCommand.AddCommand(augmentCommand);
        mcpCommand.AddCommand(conversationsCommand);

        // Add collaborate command for TARS-Augment-VSCode collaboration
        var collaborateCommand = new CollaborateCommand(_serviceProvider);
        mcpCommand.AddCommand(collaborateCommand);

        // Add test-collaboration command
        var testCollaborationCommand = new TarsCommand("test-collaboration", "Test the collaboration between TARS, Augment Code, and VS Code");

        var workflowOption = new Option<string>(
            "--workflow",
            () => "code_improvement",
            "The workflow to test (code_improvement, knowledge_extraction, self_improvement)");
        testCollaborationCommand.AddOption(workflowOption);

        testCollaborationCommand.SetHandler(async (string workflow) =>
        {
            WriteHeader("Testing TARS-Augment-VSCode Collaboration");

            var tarsMcpService = _serviceProvider!.GetRequiredService<TarsMcpService>();

            // Create a request to initiate the workflow
            var request = JsonSerializer.SerializeToElement(new
            {
                operation = "initiate_workflow",
                workflow_name = workflow,
                parameters = new Dictionary<string, string>
                {
                    ["target_directory"] = "TarsCli",
                    ["file_pattern"] = "*.cs"
                }
            });

            // Execute the collaboration operation
            var result = await tarsMcpService.ExecuteCollaborationOperationAsync("initiate_workflow", request);

            // Display the result
            var resultObj = JsonSerializer.Deserialize<Dictionary<string, object>>(result.GetRawText());

            if (resultObj != null && resultObj.TryGetValue("success", out var successObj) && successObj.ToString() == "True")
            {
                WriteColorLine($"Successfully initiated workflow: {workflow}", ConsoleColor.Green);

                if (resultObj.TryGetValue("workflow_id", out var workflowIdObj))
                {
                    Console.WriteLine($"Workflow ID: {workflowIdObj}");
                }

                if (resultObj.TryGetValue("message", out var messageObj))
                {
                    Console.WriteLine($"Message: {messageObj}");
                }
            }
            else
            {
                WriteColorLine("Failed to initiate workflow", ConsoleColor.Red);

                if (resultObj != null && resultObj.TryGetValue("error", out var errorObj))
                {
                    Console.WriteLine($"Error: {errorObj}");
                }
            }
        }, workflowOption);

        mcpCommand.AddCommand(testCollaborationCommand);

        // Add MCP command to root command
        rootCommand.AddCommand(mcpCommand);

        // Add Testing Framework command
        var testingFrameworkCommand = new TestingFrameworkCommand(_serviceProvider);
        rootCommand.AddCommand(testingFrameworkCommand);

        // Create init command
        var initCommand = new TarsCommand("init", "Initialize a new TARS session");
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
        var runPlanCommand = new TarsCommand("run", "Run a defined agent workflow from DSL script");
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
        var traceCommand = new TarsCommand("trace", "View trace logs for a completed run");
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
        var selfAnalyzeCommand = new TarsCommand("self-analyze", "Analyze a file for potential improvements");
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
        var selfProposeCommand = new TarsCommand("self-propose", "Propose improvements for a file");
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
        var selfRewriteCommand = new TarsCommand("self-rewrite", "Analyze, propose, and apply improvements to a file");
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
        var templateCommand = new TarsCommand("template", "Manage TARS templates");

        // Create template list command
        var templateListCommand = new TarsCommand("list", "List available templates");
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
        var templateCreateCommand = new TarsCommand("create", "Create a new template");
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
        var workflowCommand = new TarsCommand("workflow", "Run a multi-agent workflow for a task");
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
        var learningCommand = new TarsCommand("learning", "View and manage learning data");

        // Create learning stats command
        var learningStatsCommand = new TarsCommand("stats", "View learning statistics");
        learningStatsCommand.SetHandler(async () =>
        {
            WriteHeader("TARS Learning Statistics");

            var selfImprovementService = _serviceProvider!.GetRequiredService<SelfImprovementService>();
            var statistics = await selfImprovementService.GetLearningStatistics();

            Console.WriteLine(statistics);
        });

        // Create learning events command
        var learningEventsCommand = new TarsCommand("events", "View recent learning events");
        var eventsCountOption = new Option<int>("--count", () => 10, "Number of events to show");
        learningEventsCommand.AddOption(eventsCountOption);

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
        }, eventsCountOption);

        // Create learning clear command
        var learningClearCommand = new TarsCommand("clear", "Clear learning database");
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

        // Create learning plan commands
        var learningPlanCommand = new TarsCommand("plan", "Manage learning plans");

        // Create generate-plan command
        var generatePlanCommand = new TarsCommand("generate", "Generate a new learning plan");
        var planNameOption = new Option<string>("--name", "Name of the learning plan") { IsRequired = true };
        var topicOption = new Option<string>("--topic", "Topic of the learning plan") { IsRequired = true };
        var skillLevelOption = new Option<string>("--skill-level", () => "Intermediate", "Skill level (Beginner, Intermediate, Advanced)");
        var goalsOption = new Option<string[]>("--goals", "Learning goals") { AllowMultipleArgumentsPerToken = true };
        var preferencesOption = new Option<string[]>("--preferences", "Learning preferences") { AllowMultipleArgumentsPerToken = true };
        var hoursOption = new Option<int>("--hours", () => 20, "Estimated hours to complete");

        generatePlanCommand.AddOption(planNameOption);
        generatePlanCommand.AddOption(topicOption);
        generatePlanCommand.AddOption(skillLevelOption);
        generatePlanCommand.AddOption(goalsOption);
        generatePlanCommand.AddOption(preferencesOption);
        generatePlanCommand.AddOption(hoursOption);
        generatePlanCommand.AddOption(modelOption);

        generatePlanCommand.SetHandler(async (string name, string topic, string skillLevel, string[] goals, string[] preferences, int hours, string model) =>
        {
            WriteHeader("Generate Learning Plan");
            Console.WriteLine($"Name: {name}");
            Console.WriteLine($"Topic: {topic}");
            Console.WriteLine($"Skill Level: {skillLevel}");
            Console.WriteLine($"Goals: {string.Join(", ", goals)}");
            Console.WriteLine($"Preferences: {string.Join(", ", preferences)}");
            Console.WriteLine($"Estimated Hours: {hours}");
            Console.WriteLine($"Model: {model}");

            try
            {
                var learningPlanService = _serviceProvider!.GetRequiredService<LearningPlanService>();
                var parsedSkillLevel = Enum.Parse<SkillLevel>(skillLevel, true);
                var learningPlan = await learningPlanService.GenerateLearningPlan(name, topic, parsedSkillLevel, goals.ToList(), preferences.ToList(), hours, model);

                WriteColorLine("\nLearning plan generated successfully!", ConsoleColor.Green);
                Console.WriteLine($"ID: {learningPlan.Id}");
                Console.WriteLine($"Name: {learningPlan.Name}");
                Console.WriteLine($"Topic: {learningPlan.Topic}");
                Console.WriteLine($"Created: {learningPlan.CreatedDate:yyyy-MM-dd HH:mm:ss}");

                // Display a summary of the learning plan
                WriteColorLine("\nSummary:", ConsoleColor.Cyan);
                Console.WriteLine(learningPlan.Content.Introduction);

                WriteColorLine("\nModules:", ConsoleColor.Cyan);
                foreach (var module in learningPlan.Content.Modules)
                {
                    Console.WriteLine($"- {module.Title} ({module.EstimatedHours} hours)");
                }
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error generating learning plan: {ex.Message}", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, planNameOption, topicOption, skillLevelOption, goalsOption, preferencesOption, hoursOption, modelOption);

        // Create list-plans command
        var listPlansCommand = new TarsCommand("list", "List available learning plans");

        listPlansCommand.SetHandler(async () =>
        {
            WriteHeader("Learning Plans");

            try
            {
                var learningPlanService = _serviceProvider!.GetRequiredService<LearningPlanService>();
                var plans = await learningPlanService.GetLearningPlans();

                if (plans.Count == 0)
                {
                    WriteColorLine("No learning plans found", ConsoleColor.Yellow);
                    return;
                }

                foreach (var plan in plans)
                {
                    WriteColorLine($"ID: {plan.Id}", ConsoleColor.Cyan);
                    Console.WriteLine($"Name: {plan.Name}");
                    Console.WriteLine($"Topic: {plan.Topic}");
                    Console.WriteLine($"Skill Level: {plan.SkillLevel}");
                    Console.WriteLine($"Created: {plan.CreatedDate:yyyy-MM-dd HH:mm:ss}");
                    Console.WriteLine();
                }
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error listing learning plans: {ex.Message}", ConsoleColor.Red);
                Environment.Exit(1);
            }
        });

        // Create view-plan command
        var viewPlanCommand = new TarsCommand("view", "View details of a specific learning plan");
        var planIdOption = new Option<string>("--id", "ID of the learning plan") { IsRequired = true };
        viewPlanCommand.AddOption(planIdOption);

        viewPlanCommand.SetHandler(async (string id) =>
        {
            WriteHeader("View Learning Plan");

            try
            {
                var learningPlanService = _serviceProvider!.GetRequiredService<LearningPlanService>();
                var plan = await learningPlanService.GetLearningPlan(id);

                if (plan == null)
                {
                    WriteColorLine($"Learning plan with ID {id} not found", ConsoleColor.Red);
                    Environment.Exit(1);
                    return;
                }

                WriteColorLine($"ID: {plan.Id}", ConsoleColor.Cyan);
                Console.WriteLine($"Name: {plan.Name}");
                Console.WriteLine($"Topic: {plan.Topic}");
                Console.WriteLine($"Skill Level: {plan.SkillLevel}");
                Console.WriteLine($"Created: {plan.CreatedDate:yyyy-MM-dd HH:mm:ss}");
                Console.WriteLine($"Last Modified: {plan.LastModifiedDate:yyyy-MM-dd HH:mm:ss}");

                WriteColorLine("\nIntroduction:", ConsoleColor.Cyan);
                Console.WriteLine(plan.Content.Introduction);

                WriteColorLine("\nPrerequisites:", ConsoleColor.Cyan);
                foreach (var prereq in plan.Content.Prerequisites)
                {
                    Console.WriteLine($"- {prereq}");
                }

                WriteColorLine("\nModules:", ConsoleColor.Cyan);
                foreach (var module in plan.Content.Modules)
                {
                    WriteColorLine($"\n{module.Title} ({module.EstimatedHours} hours)", ConsoleColor.Green);
                    Console.WriteLine("Objectives:");
                    foreach (var objective in module.Objectives)
                    {
                        Console.WriteLine($"- {objective}");
                    }

                    Console.WriteLine("\nResources:");
                    foreach (var resource in module.Resources)
                    {
                        Console.WriteLine($"- {resource.Title} ({resource.Type}): {resource.Url}");
                    }

                    Console.WriteLine($"\nAssessment: {module.Assessment}");
                }

                WriteColorLine("\nTimeline:", ConsoleColor.Cyan);
                foreach (var item in plan.Content.Timeline)
                {
                    Console.WriteLine($"Week {item.Week}:");
                    foreach (var activity in item.Activities)
                    {
                        Console.WriteLine($"  - {activity}");
                    }
                }

                WriteColorLine("\nMilestones:", ConsoleColor.Cyan);
                foreach (var milestone in plan.Content.Milestones)
                {
                    Console.WriteLine($"- {milestone.Title}: {milestone.Description}");
                }

                WriteColorLine("\nPractice Projects:", ConsoleColor.Cyan);
                foreach (var project in plan.Content.PracticeProjects)
                {
                    Console.WriteLine($"- {project.Title}: {project.Description}");
                }
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error viewing learning plan: {ex.Message}", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, planIdOption);

        // Add subcommands to learning plan command
        learningPlanCommand.AddCommand(generatePlanCommand);
        learningPlanCommand.AddCommand(listPlansCommand);
        learningPlanCommand.AddCommand(viewPlanCommand);

        // Create course commands
        var courseCommand = new TarsCommand("course", "Manage courses");

        // Create generate-course command
        var generateCourseCommand = new TarsCommand("generate", "Generate a new course");
        var courseTitleOption = new Option<string>("--title", "Title of the course") { IsRequired = true };
        var courseDescriptionOption = new Option<string>("--description", "Description of the course") { IsRequired = true };
        var courseTopicOption = new Option<string>("--topic", "Topic of the course") { IsRequired = true };
        var courseDifficultyOption = new Option<string>("--difficulty", () => "Intermediate", "Difficulty level (Beginner, Intermediate, Advanced)");
        var courseHoursOption = new Option<int>("--hours", () => 20, "Estimated hours to complete");
        var targetAudienceOption = new Option<string[]>("--audience", "Target audience") { AllowMultipleArgumentsPerToken = true };

        generateCourseCommand.AddOption(courseTitleOption);
        generateCourseCommand.AddOption(courseDescriptionOption);
        generateCourseCommand.AddOption(courseTopicOption);
        generateCourseCommand.AddOption(courseDifficultyOption);
        generateCourseCommand.AddOption(courseHoursOption);
        generateCourseCommand.AddOption(targetAudienceOption);
        generateCourseCommand.AddOption(modelOption);

        generateCourseCommand.SetHandler(async (string title, string description, string topic, string difficulty, int hours, string[] audience, string model) =>
        {
            WriteHeader("Generate Course");
            Console.WriteLine($"Title: {title}");
            Console.WriteLine($"Description: {description}");
            Console.WriteLine($"Topic: {topic}");
            Console.WriteLine($"Difficulty: {difficulty}");
            Console.WriteLine($"Estimated Hours: {hours}");
            Console.WriteLine($"Target Audience: {string.Join(", ", audience)}");
            Console.WriteLine($"Model: {model}");

            try
            {
                var courseGeneratorService = _serviceProvider!.GetRequiredService<CourseGeneratorService>();
                var parsedDifficulty = Enum.Parse<DifficultyLevel>(difficulty, true);
                var course = await courseGeneratorService.GenerateCourse(title, description, topic, parsedDifficulty, hours, audience.ToList(), model);

                WriteColorLine("\nCourse generated successfully!", ConsoleColor.Green);
                Console.WriteLine($"ID: {course.Id}");
                Console.WriteLine($"Title: {course.Title}");
                Console.WriteLine($"Topic: {course.Topic}");
                Console.WriteLine($"Created: {course.CreatedDate:yyyy-MM-dd HH:mm:ss}");

                // Display a summary of the course
                WriteColorLine("\nOverview:", ConsoleColor.Cyan);
                Console.WriteLine(course.Content.Overview);

                WriteColorLine("\nLessons:", ConsoleColor.Cyan);
                foreach (var lesson in course.Content.Lessons)
                {
                    Console.WriteLine($"- {lesson.Title} ({lesson.EstimatedMinutes} minutes)");
                }
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error generating course: {ex.Message}", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, courseTitleOption, courseDescriptionOption, courseTopicOption, courseDifficultyOption, courseHoursOption, targetAudienceOption, modelOption);

        // Create list-courses command
        var listCoursesCommand = new TarsCommand("list", "List available courses");

        listCoursesCommand.SetHandler(async () =>
        {
            WriteHeader("Courses");

            try
            {
                var courseGeneratorService = _serviceProvider!.GetRequiredService<CourseGeneratorService>();
                var courses = await courseGeneratorService.GetCourses();

                if (courses.Count == 0)
                {
                    WriteColorLine("No courses found", ConsoleColor.Yellow);
                    return;
                }

                foreach (var course in courses)
                {
                    WriteColorLine($"ID: {course.Id}", ConsoleColor.Cyan);
                    Console.WriteLine($"Title: {course.Title}");
                    Console.WriteLine($"Topic: {course.Topic}");
                    Console.WriteLine($"Difficulty: {course.DifficultyLevel}");
                    Console.WriteLine($"Created: {course.CreatedDate:yyyy-MM-dd HH:mm:ss}");
                    Console.WriteLine();
                }
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error listing courses: {ex.Message}", ConsoleColor.Red);
                Environment.Exit(1);
            }
        });

        // Create view-course command
        var viewCourseCommand = new TarsCommand("view", "View details of a specific course");
        var courseIdOption = new Option<string>("--id", "ID of the course") { IsRequired = true };
        viewCourseCommand.AddOption(courseIdOption);

        viewCourseCommand.SetHandler(async (string id) =>
        {
            WriteHeader("View Course");

            try
            {
                var courseGeneratorService = _serviceProvider!.GetRequiredService<CourseGeneratorService>();
                var course = await courseGeneratorService.GetCourse(id);

                if (course == null)
                {
                    WriteColorLine($"Course with ID {id} not found", ConsoleColor.Red);
                    Environment.Exit(1);
                    return;
                }

                WriteColorLine($"ID: {course.Id}", ConsoleColor.Cyan);
                Console.WriteLine($"Title: {course.Title}");
                Console.WriteLine($"Description: {course.Description}");
                Console.WriteLine($"Topic: {course.Topic}");
                Console.WriteLine($"Difficulty: {course.DifficultyLevel}");
                Console.WriteLine($"Created: {course.CreatedDate:yyyy-MM-dd HH:mm:ss}");
                Console.WriteLine($"Last Modified: {course.LastModifiedDate:yyyy-MM-dd HH:mm:ss}");

                WriteColorLine("\nOverview:", ConsoleColor.Cyan);
                Console.WriteLine(course.Content.Overview);

                WriteColorLine("\nLearning Objectives:", ConsoleColor.Cyan);
                foreach (var objective in course.Content.LearningObjectives)
                {
                    Console.WriteLine($"- {objective}");
                }

                WriteColorLine("\nLessons:", ConsoleColor.Cyan);
                foreach (var lesson in course.Content.Lessons)
                {
                    WriteColorLine($"\n{lesson.Title} ({lesson.EstimatedMinutes} minutes)", ConsoleColor.Green);
                    Console.WriteLine("Objectives:");
                    foreach (var objective in lesson.Objectives)
                    {
                        Console.WriteLine($"- {objective}");
                    }

                    Console.WriteLine("\nContent:");
                    Console.WriteLine(lesson.Content);

                    Console.WriteLine("\nExercises:");
                    foreach (var exercise in lesson.Exercises)
                    {
                        Console.WriteLine($"- {exercise.Title}: {exercise.Description} (Difficulty: {exercise.Difficulty})");
                    }

                    Console.WriteLine("\nQuiz Questions:");
                    foreach (var quiz in lesson.QuizQuestions)
                    {
                        Console.WriteLine($"Q: {quiz.Question}");
                        for (int i = 0; i < quiz.Options.Count; i++)
                        {
                            if (i == quiz.CorrectAnswerIndex)
                            {
                                WriteColorLine($"   {i + 1}. {quiz.Options[i]} (CORRECT)", ConsoleColor.Green);
                            }
                            else
                            {
                                Console.WriteLine($"   {i + 1}. {quiz.Options[i]}");
                            }
                        }
                        Console.WriteLine($"   Explanation: {quiz.Explanation}");
                        Console.WriteLine();
                    }
                }

                WriteColorLine("\nFinal Assessment:", ConsoleColor.Cyan);
                Console.WriteLine($"Title: {course.Content.FinalAssessment.Title}");
                Console.WriteLine($"Description: {course.Content.FinalAssessment.Description}");
                Console.WriteLine($"Estimated Hours: {course.Content.FinalAssessment.EstimatedHours}");
                Console.WriteLine("Criteria:");
                foreach (var criterion in course.Content.FinalAssessment.Criteria)
                {
                    Console.WriteLine($"- {criterion}");
                }

                WriteColorLine("\nAdditional Resources:", ConsoleColor.Cyan);
                foreach (var resource in course.Content.AdditionalResources)
                {
                    Console.WriteLine($"- {resource.Title} ({resource.Type}): {resource.Url}");
                    Console.WriteLine($"  {resource.Description}");
                }
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error viewing course: {ex.Message}", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, courseIdOption);

        // Add subcommands to course command
        courseCommand.AddCommand(generateCourseCommand);
        courseCommand.AddCommand(listCoursesCommand);
        courseCommand.AddCommand(viewCourseCommand);

        // Create tutorial commands
        var tutorialCommand = new TarsCommand("tutorial", "Manage tutorials");

        // Create add-tutorial command
        var addTutorialCommand = new TarsCommand("add", "Add a new tutorial");
        var tutorialTitleOption = new Option<string>("--title", "Title of the tutorial") { IsRequired = true };
        var tutorialDescriptionOption = new Option<string>("--description", "Description of the tutorial") { IsRequired = true };
        var tutorialContentOption = new Option<string>("--content", "Content of the tutorial") { IsRequired = true };
        var tutorialCategoryOption = new Option<string>("--category", "Category of the tutorial") { IsRequired = true };
        var tutorialDifficultyOption = new Option<string>("--difficulty", () => "Intermediate", "Difficulty level (Beginner, Intermediate, Advanced)");
        var tutorialTagsOption = new Option<string[]>("--tags", "Tags for the tutorial") { AllowMultipleArgumentsPerToken = true };
        var tutorialPrerequisitesOption = new Option<string[]>("--prerequisites", "Prerequisites for the tutorial") { AllowMultipleArgumentsPerToken = true };

        addTutorialCommand.AddOption(tutorialTitleOption);
        addTutorialCommand.AddOption(tutorialDescriptionOption);
        addTutorialCommand.AddOption(tutorialContentOption);
        addTutorialCommand.AddOption(tutorialCategoryOption);
        addTutorialCommand.AddOption(tutorialDifficultyOption);
        addTutorialCommand.AddOption(tutorialTagsOption);
        addTutorialCommand.AddOption(tutorialPrerequisitesOption);

        addTutorialCommand.SetHandler(async (string title, string description, string content, string category, string difficulty, string[] tags, string[] prerequisites) =>
        {
            WriteHeader("Add Tutorial");
            Console.WriteLine($"Title: {title}");
            Console.WriteLine($"Description: {description}");
            Console.WriteLine($"Category: {category}");
            Console.WriteLine($"Difficulty: {difficulty}");
            Console.WriteLine($"Tags: {string.Join(", ", tags)}");
            Console.WriteLine($"Prerequisites: {string.Join(", ", prerequisites)}");

            try
            {
                var tutorialOrganizerService = _serviceProvider!.GetRequiredService<TutorialOrganizerService>();
                var parsedDifficulty = Enum.Parse<DifficultyLevel>(difficulty, true);
                var tutorial = await tutorialOrganizerService.AddTutorial(title, description, content, category, parsedDifficulty, tags.ToList(), prerequisites.ToList());

                WriteColorLine("\nTutorial added successfully!", ConsoleColor.Green);
                Console.WriteLine($"ID: {tutorial.Id}");
                Console.WriteLine($"Title: {tutorial.Title}");
                Console.WriteLine($"Category: {tutorial.Category}");
                Console.WriteLine($"Created: {tutorial.CreatedDate:yyyy-MM-dd HH:mm:ss}");
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error adding tutorial: {ex.Message}", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, tutorialTitleOption, tutorialDescriptionOption, tutorialContentOption, tutorialCategoryOption, tutorialDifficultyOption, tutorialTagsOption, tutorialPrerequisitesOption);

        // Create list-tutorials command
        var listTutorialsCommand = new TarsCommand("list", "List available tutorials");
        var categoryFilterOption = new Option<string>("--category", "Filter by category");
        var difficultyFilterOption = new Option<string>("--difficulty", "Filter by difficulty level");
        var tagFilterOption = new Option<string>("--tag", "Filter by tag");

        listTutorialsCommand.AddOption(categoryFilterOption);
        listTutorialsCommand.AddOption(difficultyFilterOption);
        listTutorialsCommand.AddOption(tagFilterOption);

        listTutorialsCommand.SetHandler(async (string category, string difficulty, string tag) =>
        {
            WriteHeader("Tutorials");

            try
            {
                var tutorialOrganizerService = _serviceProvider!.GetRequiredService<TutorialOrganizerService>();
                var tutorials = await tutorialOrganizerService.GetTutorials();

                // Filter tutorials based on criteria
                if (!string.IsNullOrEmpty(category))
                {
                    tutorials = tutorials.Where(t => string.Equals(t.Category, category, StringComparison.OrdinalIgnoreCase)).ToList();
                }

                if (!string.IsNullOrEmpty(difficulty) && Enum.TryParse<DifficultyLevel>(difficulty, true, out var difficultyLevel))
                {
                    tutorials = tutorials.Where(t => t.DifficultyLevel == difficultyLevel).ToList();
                }

                if (!string.IsNullOrEmpty(tag))
                {
                    tutorials = tutorials.Where(t => t.Tags.Any(tagItem => string.Equals(tagItem, tag, StringComparison.OrdinalIgnoreCase))).ToList();
                }

                if (tutorials.Count == 0)
                {
                    WriteColorLine("No tutorials found matching the criteria", ConsoleColor.Yellow);
                    return;
                }

                foreach (var tutorial in tutorials)
                {
                    WriteColorLine($"ID: {tutorial.Id}", ConsoleColor.Cyan);
                    Console.WriteLine($"Title: {tutorial.Title}");
                    Console.WriteLine($"Category: {tutorial.Category}");
                    Console.WriteLine($"Difficulty: {tutorial.DifficultyLevel}");
                    Console.WriteLine($"Tags: {string.Join(", ", tutorial.Tags)}");
                    Console.WriteLine($"Created: {tutorial.CreatedDate:yyyy-MM-dd HH:mm:ss}");
                    Console.WriteLine();
                }
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error listing tutorials: {ex.Message}", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, categoryFilterOption, difficultyFilterOption, tagFilterOption);

        // Create view-tutorial command
        var viewTutorialCommand = new TarsCommand("view", "View details of a specific tutorial");
        var tutorialIdOption = new Option<string>("--id", "ID of the tutorial") { IsRequired = true };
        viewTutorialCommand.AddOption(tutorialIdOption);

        viewTutorialCommand.SetHandler(async (string id) =>
        {
            WriteHeader("View Tutorial");

            try
            {
                var tutorialOrganizerService = _serviceProvider!.GetRequiredService<TutorialOrganizerService>();
                var tutorialWithContent = await tutorialOrganizerService.GetTutorial(id);

                // Use pattern matching to check if result is not null
                if (tutorialWithContent is not { } result)
                {
                    WriteColorLine($"Tutorial with ID {id} not found", ConsoleColor.Red);
                    Environment.Exit(1);
                    return;
                }

                // Deconstruct the record for easier access
                var (tutorial, content) = result;

                WriteColorLine($"ID: {tutorial.Id}", ConsoleColor.Cyan);
                Console.WriteLine($"Title: {tutorial.Title}");
                Console.WriteLine($"Description: {tutorial.Description}");
                Console.WriteLine($"Category: {tutorial.Category}");
                Console.WriteLine($"Difficulty: {tutorial.DifficultyLevel}");
                Console.WriteLine($"Tags: {string.Join(", ", tutorial.Tags)}");
                Console.WriteLine($"Prerequisites: {string.Join(", ", tutorial.Prerequisites)}");
                Console.WriteLine($"Created: {tutorial.CreatedDate:yyyy-MM-dd HH:mm:ss}");
                Console.WriteLine($"Last Modified: {tutorial.LastModifiedDate:yyyy-MM-dd HH:mm:ss}");

                WriteColorLine("\nContent:", ConsoleColor.Cyan);
                Console.WriteLine(content);
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error viewing tutorial: {ex.Message}", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, tutorialIdOption);

        // Create categorize command
        var categorizeTutorialsCommand = new TarsCommand("categorize", "Organize tutorials into categories");
        var tutorialIdsOption = new Option<string[]>("--ids", "IDs of the tutorials to categorize") { IsRequired = true, AllowMultipleArgumentsPerToken = true };
        var newCategoryOption = new Option<string>("--category", "New category for the tutorials") { IsRequired = true };

        categorizeTutorialsCommand.AddOption(tutorialIdsOption);
        categorizeTutorialsCommand.AddOption(newCategoryOption);

        categorizeTutorialsCommand.SetHandler(async (string[] ids, string category) =>
        {
            WriteHeader("Categorize Tutorials");
            Console.WriteLine($"Tutorial IDs: {string.Join(", ", ids)}");
            Console.WriteLine($"New Category: {category}");

            try
            {
                var tutorialOrganizerService = _serviceProvider!.GetRequiredService<TutorialOrganizerService>();
                var success = await tutorialOrganizerService.CategorizeTutorials(ids.ToList(), category);

                if (success)
                {
                    WriteColorLine("\nTutorials categorized successfully!", ConsoleColor.Green);
                }
                else
                {
                    WriteColorLine("\nFailed to categorize tutorials", ConsoleColor.Red);
                    Environment.Exit(1);
                }
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error categorizing tutorials: {ex.Message}", ConsoleColor.Red);
                Environment.Exit(1);
            }
        }, tutorialIdsOption, newCategoryOption);

        // Add subcommands to tutorial command
        tutorialCommand.AddCommand(addTutorialCommand);
        tutorialCommand.AddCommand(listTutorialsCommand);
        tutorialCommand.AddCommand(viewTutorialCommand);
        tutorialCommand.AddCommand(categorizeTutorialsCommand);

        // Add learning subcommands
        learningCommand.AddCommand(learningStatsCommand);
        learningCommand.AddCommand(learningEventsCommand);
        learningCommand.AddCommand(learningClearCommand);
        learningCommand.AddCommand(learningPlanCommand);
        learningCommand.AddCommand(courseCommand);
        learningCommand.AddCommand(tutorialCommand);

        // Create huggingface command
        var huggingFaceCommand = new TarsCommand("huggingface", "Interact with Hugging Face models");

        // Create huggingface search command
        var hfSearchCommand = new TarsCommand("search", "Search for models on Hugging Face");
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
        var hfBestCommand = new TarsCommand("best", "Get the best coding models from Hugging Face");
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
        var hfDetailsCommand = new TarsCommand("details", "Get detailed information about a model");
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
        var hfDownloadCommand = new TarsCommand("download", "Download a model from Hugging Face");
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
        var hfInstallCommand = new TarsCommand("install", "Install a model from Hugging Face to Ollama");
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
        var hfListCommand = new TarsCommand("list", "List installed models");

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
        var languageCommand = new TarsCommand("language", "Generate and manage language specifications");

        // Create language ebnf command
        var languageEbnfCommand = new TarsCommand("ebnf", "Generate EBNF specification for TARS DSL");
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
        var languageBnfCommand = new TarsCommand("bnf", "Generate BNF specification for TARS DSL");
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
        var languageJsonSchemaCommand = new TarsCommand("json-schema", "Generate JSON schema for TARS DSL");
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
        var languageDocsCommand = new TarsCommand("docs", "Generate markdown documentation for TARS DSL");
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
        var docsExploreCommand = new TarsCommand("docs-explore", "Explore TARS documentation");
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
        var demoCommand = new TarsCommand("demo", "Run a demonstration of TARS capabilities");
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
        var secretsCommand = new TarsCommand("secrets", "Manage API keys and other secrets");

        // List secrets command
        var listSecretsCommand = new TarsCommand("list", "List all secret keys");
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
        var setSecretCommand = new TarsCommand("set", "Set a secret value");
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
        var removeSecretCommand = new TarsCommand("remove", "Remove a secret");
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
        var clearSecretsCommand = new TarsCommand("clear", "Clear all secrets");

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
        var autoImproveCommand = new TarsCommand("auto-improve", "Run autonomous self-improvement");

        // Create console-capture command
        var consoleCaptureCommand = new TarsCommand("console-capture", "Capture console output and improve code");
        var startOption = new Option<bool>("--start", "Start capturing console output");
        var stopCaptureOption = new Option<bool>("--stop", "Stop capturing console output");
        var analyzeOption = new Option<string>("--analyze", "Analyze captured output and suggest improvements for a file");
        var applyOption = new Option<bool>("--apply", "Apply suggested improvements");
        var autoOption = new Option<string>("--auto", "Automatically improve a file based on captured output");

        consoleCaptureCommand.AddOption(startOption);
        consoleCaptureCommand.AddOption(stopCaptureOption);
        consoleCaptureCommand.AddOption(analyzeOption);
        consoleCaptureCommand.AddOption(applyOption);
        consoleCaptureCommand.AddOption(autoOption);

        consoleCaptureCommand.SetHandler(async (bool start, bool stop, string? analyze, bool apply, string? auto) =>
        {
            WriteHeader("TARS Console Capture");

            var consoleCaptureService = _serviceProvider!.GetRequiredService<ConsoleCaptureService>();

            if (start)
            {
                consoleCaptureService.StartCapture();
                WriteColorLine("Started capturing console output.", ConsoleColor.Green);
                WriteColorLine("Run your commands and then use '--stop' to stop capturing.", ConsoleColor.Yellow);
                return;
            }

            if (stop)
            {
                var capturedOutput = consoleCaptureService.StopCapture();
                WriteColorLine("Stopped capturing console output.", ConsoleColor.Green);
                WriteColorLine($"Captured {capturedOutput.Length} characters.", ConsoleColor.Yellow);
                WriteColorLine("Use '--analyze <file-path>' to analyze the captured output and suggest improvements.", ConsoleColor.Yellow);
                return;
            }

            if (!string.IsNullOrEmpty(analyze))
            {
                if (!File.Exists(analyze))
                {
                    WriteColorLine($"File not found: {analyze}", ConsoleColor.Red);
                    return;
                }

                WriteColorLine($"Analyzing captured output for file: {analyze}", ConsoleColor.Yellow);
                var capturedOutput = string.Join("\n", consoleCaptureService.GetCapturedOutput());

                if (string.IsNullOrEmpty(capturedOutput))
                {
                    WriteColorLine("No captured output to analyze. Use '--start' to start capturing first.", ConsoleColor.Red);
                    return;
                }

                var suggestions = await consoleCaptureService.AnalyzeAndSuggestImprovements(capturedOutput, analyze);
                WriteColorLine("Analysis complete. Suggested improvements:", ConsoleColor.Green);
                Console.WriteLine(suggestions);

                if (apply)
                {
                    WriteColorLine("\nApplying suggested improvements...", ConsoleColor.Yellow);
                    var result = await consoleCaptureService.ApplyImprovements(analyze, suggestions);
                    WriteColorLine(result, ConsoleColor.Green);
                }
                else
                {
                    WriteColorLine("\nUse '--apply' to apply these suggestions.", ConsoleColor.Yellow);
                }

                return;
            }

            if (!string.IsNullOrEmpty(auto))
            {
                if (!File.Exists(auto))
                {
                    WriteColorLine($"File not found: {auto}", ConsoleColor.Red);
                    return;
                }

                WriteColorLine($"Auto-improving code for file: {auto}", ConsoleColor.Yellow);
                var capturedOutput = string.Join("\n", consoleCaptureService.GetCapturedOutput());

                if (string.IsNullOrEmpty(capturedOutput))
                {
                    WriteColorLine("No captured output to analyze. Use '--start' to start capturing first.", ConsoleColor.Red);
                    return;
                }

                var result = await consoleCaptureService.AutoImproveCode(capturedOutput, auto);
                WriteColorLine(result, ConsoleColor.Green);
                return;
            }

            // If no options provided, show help
            WriteColorLine("Console Capture Commands:", ConsoleColor.Cyan);
            WriteColorLine("  --start: Start capturing console output", ConsoleColor.White);
            WriteColorLine("  --stop: Stop capturing console output", ConsoleColor.White);
            WriteColorLine("  --analyze <file-path>: Analyze captured output and suggest improvements", ConsoleColor.White);
            WriteColorLine("  --apply: Apply suggested improvements (use with --analyze)", ConsoleColor.White);
            WriteColorLine("  --auto <file-path>: Automatically improve code based on captured output", ConsoleColor.White);
        }, startOption, stopCaptureOption, analyzeOption, applyOption, autoOption);
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
        var slackCommand = new TarsCommand("slack", "Manage Slack integration");

        // Set webhook command
        var setWebhookCommand = new TarsCommand("set-webhook", "Set the Slack webhook URL");
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
        var testCommand = new TarsCommand("test", "Test the Slack integration");
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
        var announceCommand = new TarsCommand("announce", "Post an announcement to Slack");
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
        var featureCommand = new TarsCommand("feature", "Post a feature update to Slack");
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
        var milestoneCommand = new TarsCommand("milestone", "Post a milestone to Slack");
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
        var speechCommand = new TarsCommand("speech", "Text-to-speech functionality");

        // Speak command
        var speakCommand = new TarsCommand("speak", "Speak text using text-to-speech");
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
        var listVoicesCommand = new TarsCommand("list-voices", "List available voice models");

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

        // Configure command for speech
        var speechConfigureCommand = new TarsCommand("configure", "Configure speech settings");
        var enabledOption = new Option<bool>("--enabled", () => true, "Whether speech is enabled");
        var defaultVoiceOption = new Option<string?>("--default-voice", "Default voice model");
        var defaultLanguageOption = new Option<string?>("--default-language", "Default language code");
        var preloadVoicesOption = new Option<bool?>("--preload-voices", "Whether to preload voice models");
        var maxConcurrencyOption = new Option<int?>("--max-concurrency", "Maximum concurrent speech operations");

        speechConfigureCommand.AddOption(enabledOption);
        speechConfigureCommand.AddOption(defaultVoiceOption);
        speechConfigureCommand.AddOption(defaultLanguageOption);
        speechConfigureCommand.AddOption(preloadVoicesOption);
        speechConfigureCommand.AddOption(maxConcurrencyOption);

        speechConfigureCommand.SetHandler((bool enabled, string? defaultVoice, string? defaultLanguage, bool? preloadVoices, int? maxConcurrency) =>
        {
            WriteHeader("TARS Speech - Configure");

            var speechService = _serviceProvider!.GetRequiredService<TarsSpeechService>();
            speechService.Configure(enabled, defaultVoice, defaultLanguage);

            // Note: preloadVoices and maxConcurrency parameters are not currently supported

            WriteColorLine("Speech configuration updated.", ConsoleColor.Green);
        }, enabledOption, defaultVoiceOption, defaultLanguageOption, preloadVoicesOption, maxConcurrencyOption);

        // Add subcommands to speech command
        speechCommand.AddCommand(speakCommand);
        speechCommand.AddCommand(listVoicesCommand);
        speechCommand.AddCommand(speechConfigureCommand);

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

        // Create reflection command
        var reflectionCommand = new TarsCommand("reflect", "Reflect on TARS explorations");

        // Add list subcommand
        var reflectionListCommand = new TarsCommand("list", "List exploration files");
        var directoryOption = new Option<string>("--directory", () => "v1/Chats", "The subdirectory to list explorations from");

        reflectionListCommand.AddOption(directoryOption);

        reflectionListCommand.SetHandler((string directory) =>
        {
            WriteHeader("TARS Exploration Files");

            var reflectionService = _serviceProvider!.GetRequiredService<ExplorationReflectionService>();
            var files = reflectionService.GetExplorationFiles(directory);

            if (files.Count == 0)
            {
                WriteColorLine("No exploration files found.", ConsoleColor.Yellow);
                return;
            }

            WriteColorLine($"Found {files.Count} exploration files:", ConsoleColor.Green);
            Console.WriteLine();

            foreach (var file in files)
            {
                WriteColorLine($"{file.Title}", ConsoleColor.Cyan);
                Console.WriteLine($"  Path: {file.FilePath}");
                Console.WriteLine($"  Modified: {file.LastModified}");
                Console.WriteLine($"  Size: {file.SizeInBytes / 1024} KB");
                Console.WriteLine();
            }
        }, directoryOption);

        // Add generate subcommand
        var reflectionGenerateCommand = new TarsCommand("generate", "Generate a reflection on an exploration file");
        var reflectionFileOption = new Option<string>("--file", "The path to the exploration file");
        var reflectionModelOption = new Option<string>("--model", () => "llama3", "The model to use for reflection");

        reflectionGenerateCommand.AddOption(reflectionFileOption);
        reflectionGenerateCommand.AddOption(reflectionModelOption);

        reflectionGenerateCommand.SetHandler(async (string file, string model) =>
        {
            WriteHeader("TARS Exploration Reflection");

            var reflectionService = _serviceProvider!.GetRequiredService<ExplorationReflectionService>();

            WriteColorLine($"Generating reflection for: {file}", ConsoleColor.Green);
            WriteColorLine($"Using model: {model}", ConsoleColor.Green);
            Console.WriteLine();

            var reflection = await reflectionService.GenerateReflectionAsync(file, model);

            Console.WriteLine(reflection);
        }, reflectionFileOption, reflectionModelOption);

        // Add report subcommand
        var reflectionReportCommand = new TarsCommand("report", "Generate a comprehensive reflection report");
        var reportDirectoryOption = new Option<string>("--directory", () => "v1/Chats", "The subdirectory to generate reflections from");
        var reportModelOption = new Option<string>("--model", () => "llama3", "The model to use for reflection");
        var saveOption = new Option<bool>("--save", "Save the report to a file");

        reflectionReportCommand.AddOption(reportDirectoryOption);
        reflectionReportCommand.AddOption(reportModelOption);
        reflectionReportCommand.AddOption(saveOption);

        reflectionReportCommand.SetHandler(async (string directory, string model, bool save) =>
        {
            WriteHeader("TARS Exploration Reflection Report");

            var reflectionService = _serviceProvider!.GetRequiredService<ExplorationReflectionService>();

            WriteColorLine($"Generating reflection report for directory: {directory}", ConsoleColor.Green);
            WriteColorLine($"Using model: {model}", ConsoleColor.Green);
            Console.WriteLine();

            var report = await reflectionService.GenerateReflectionReportAsync(directory, model);

            if (save)
            {
                var filePath = await reflectionService.SaveReflectionReportAsync(report);
                WriteColorLine($"Report saved to: {filePath}", ConsoleColor.Green);
                Console.WriteLine();
            }

            Console.WriteLine(report);
        }, reportDirectoryOption, reportModelOption, saveOption);

        // Add subcommands to reflection command
        reflectionCommand.AddCommand(reflectionListCommand);
        reflectionCommand.AddCommand(reflectionGenerateCommand);
        reflectionCommand.AddCommand(reflectionReportCommand);

        rootCommand.AddCommand(reflectionCommand);

        // Create chat command
        var chatCommand = new TarsCommand("chat", "Interactive chat bot");

        // Add start subcommand
        var chatStartCommand = new TarsCommand("start", "Start an interactive chat session");
        var chatModelOption = new Option<string>("--model", () => "llama3", "The model to use for chat");
        var chatSpeechOption = new Option<bool>("--speech", "Enable speech output");

        chatStartCommand.AddOption(chatModelOption);
        chatStartCommand.AddOption(chatSpeechOption);

        chatStartCommand.SetHandler((string model, bool speech) =>
        {
            WriteHeader("TARS Chat Bot");

            var chatBotService = _serviceProvider!.GetRequiredService<ChatBotService>();
            chatBotService.StartNewConversation(model);
            chatBotService.EnableSpeech(speech);

            WriteColorLine($"Chat model: {model}", ConsoleColor.Cyan);
            WriteColorLine($"Speech: {(speech ? "Enabled" : "Disabled")}", ConsoleColor.Cyan);
            Console.WriteLine();
            WriteColorLine("Type 'exit', 'quit', or press Ctrl+C to end the chat session.", ConsoleColor.Yellow);
            WriteColorLine("Type 'examples' to see example prompts.", ConsoleColor.Yellow);
            WriteColorLine("Type 'model <name>' to change the model.", ConsoleColor.Yellow);
            WriteColorLine("Type 'speech on/off' to enable/disable speech.", ConsoleColor.Yellow);
            WriteColorLine("Type 'clear' to start a new conversation.", ConsoleColor.Yellow);
            Console.WriteLine();

            // Start the chat loop
            RunChatLoop(chatBotService).Wait();
        }, chatModelOption, chatSpeechOption);

        // Add websocket subcommand
        var chatWebSocketCommand = new TarsCommand("websocket", "Start a WebSocket server for chat");
        var chatPortOption = new Option<int>("--port", () => 8998, "The port to use for the WebSocket server");

        chatWebSocketCommand.AddOption(chatPortOption);

        chatWebSocketCommand.SetHandler((int port) =>
        {
            WriteHeader("TARS Chat WebSocket Server");

            var chatWebSocketService = _serviceProvider!.GetRequiredService<ChatWebSocketService>();

            // Start the WebSocket server
            chatWebSocketService.StartAsync().Wait();

            WriteColorLine($"WebSocket server started on {chatWebSocketService.GetServerUrl()}", ConsoleColor.Green);
            WriteColorLine("Press Ctrl+C to stop the server.", ConsoleColor.Yellow);

            // Keep the server running until Ctrl+C is pressed
            var cancellationTokenSource = new CancellationTokenSource();
            Console.CancelKeyPress += (sender, e) =>
            {
                e.Cancel = true;
                cancellationTokenSource.Cancel();
            };

            try
            {
                Task.Delay(-1, cancellationTokenSource.Token).Wait();
            }
            catch (TaskCanceledException)
            {
                // Expected when cancellation token is triggered
            }

            // Stop the WebSocket server
            chatWebSocketService.StopAsync().Wait();
            WriteColorLine("WebSocket server stopped.", ConsoleColor.Yellow);
        }, chatPortOption);

        // Add examples subcommand
        var chatExamplesCommand = new TarsCommand("examples", "Show example chat prompts");

        chatExamplesCommand.SetHandler(() =>
        {
            WriteHeader("TARS Chat Examples");

            var chatBotService = _serviceProvider!.GetRequiredService<ChatBotService>();
            var examples = chatBotService.GetExamplePrompts();

            for (int i = 0; i < examples.Count; i++)
            {
                WriteColorLine($"{i + 1}. {examples[i]}", ConsoleColor.Cyan);
            }
        });

        // Add history subcommand
        var chatHistoryCommand = new TarsCommand("history", "Show chat history");
        var chatCountOption = new Option<int>("--count", () => 5, "Number of history files to show");

        chatHistoryCommand.AddOption(chatCountOption);

        chatHistoryCommand.SetHandler((int count) =>
        {
            WriteHeader("TARS Chat History");

            var chatBotService = _serviceProvider!.GetRequiredService<ChatBotService>();
            var historyFiles = chatBotService.GetConversationHistoryFiles();

            if (historyFiles.Count == 0)
            {
                WriteColorLine("No chat history found.", ConsoleColor.Yellow);
                return;
            }

            WriteColorLine($"Found {historyFiles.Count} chat history files:", ConsoleColor.Green);
            Console.WriteLine();

            var filesToShow = historyFiles.Take(count).ToList();

            for (int i = 0; i < filesToShow.Count; i++)
            {
                var file = filesToShow[i];
                var fileName = Path.GetFileName(file);
                var creationTime = File.GetCreationTime(file);

                WriteColorLine($"{i + 1}. {fileName} - {creationTime}", ConsoleColor.Cyan);
            }
        }, chatCountOption);

        // Add subcommands to chat command
        chatCommand.AddCommand(chatStartCommand);
        chatCommand.AddCommand(chatWebSocketCommand);
        chatCommand.AddCommand(chatExamplesCommand);
        chatCommand.AddCommand(chatHistoryCommand);

        // Create deep thinking command
        var deepThinkingCommand = new TarsCommand("think", "Deep thinking about explorations");

        // Add generate subcommand
        var thinkGenerateCommand = new TarsCommand("generate", "Generate a new deep thinking exploration");
        var thinkTopicOption = new Option<string>("--topic", "The topic for deep thinking") { IsRequired = true };
        var baseFileOption = new Option<string>("--base-file", "The base exploration file to build upon");
        var thinkModelOption = new Option<string>("--model", () => "llama3", "The model to use for deep thinking");

        thinkGenerateCommand.AddOption(thinkTopicOption);
        thinkGenerateCommand.AddOption(baseFileOption);
        thinkGenerateCommand.AddOption(thinkModelOption);

        thinkGenerateCommand.SetHandler(async (string topic, string baseFile, string model) =>
        {
            WriteHeader("TARS Deep Thinking");

            // Use the new reflection service
            var reflectionService = _serviceProvider!.GetRequiredService<ExplorationReflectionService2>();
            var ollamaService = _serviceProvider!.GetRequiredService<OllamaService>();
            var configuration = _serviceProvider!.GetRequiredService<IConfiguration>();
            var logger = _serviceProvider!.GetRequiredService<ILogger<DeepThinkingService>>();

            // Create the deep thinking service manually
            var deepThinkingService = new DeepThinkingService(logger, configuration, ollamaService, reflectionService);

            WriteColorLine($"Generating deep thinking exploration on topic: {topic}", ConsoleColor.Cyan);
            WriteColorLine($"Using model: {model}", ConsoleColor.Cyan);

            if (!string.IsNullOrEmpty(baseFile))
            {
                WriteColorLine($"Building upon base file: {baseFile}", ConsoleColor.Cyan);
            }

            Console.WriteLine();
            WriteColorLine("Thinking deeply...", ConsoleColor.Yellow);

            try
            {
                // Generate the deep thinking exploration
                var result = await deepThinkingService.GenerateDeepThinkingExplorationAsync(topic, baseFile, model);

                // Save the exploration
                var filePath = await deepThinkingService.SaveDeepThinkingExplorationAsync(result);

                Console.WriteLine();
                WriteColorLine($"Deep thinking exploration generated and saved to: {filePath}", ConsoleColor.Green);
                WriteColorLine($"Version: {result.Version}", ConsoleColor.Green);
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error generating deep thinking exploration: {ex.Message}", ConsoleColor.Red);
            }
        }, thinkTopicOption, baseFileOption, thinkModelOption);

        // Add evolve subcommand
        var thinkEvolveCommand = new TarsCommand("evolve", "Evolve an existing exploration with deep thinking");
        var evolveFileOption = new Option<string>("--file", "The exploration file to evolve") { IsRequired = true };
        var evolveModelOption = new Option<string>("--model", () => "llama3", "The model to use for deep thinking");

        thinkEvolveCommand.AddOption(evolveFileOption);
        thinkEvolveCommand.AddOption(evolveModelOption);

        thinkEvolveCommand.SetHandler(async (string file, string model) =>
        {
            WriteHeader("TARS Deep Thinking Evolution");

            // Use the new reflection service
            var reflectionService = _serviceProvider!.GetRequiredService<ExplorationReflectionService2>();
            var ollamaService = _serviceProvider!.GetRequiredService<OllamaService>();
            var configuration = _serviceProvider!.GetRequiredService<IConfiguration>();
            var logger = _serviceProvider!.GetRequiredService<ILogger<DeepThinkingService>>();

            // Create the deep thinking service manually
            var deepThinkingService = new DeepThinkingService(logger, configuration, ollamaService, reflectionService);

            WriteColorLine($"Evolving exploration: {file}", ConsoleColor.Cyan);
            WriteColorLine($"Using model: {model}", ConsoleColor.Cyan);

            Console.WriteLine();
            WriteColorLine("Thinking deeply...", ConsoleColor.Yellow);

            try
            {
                // Generate the deep thinking evolution
                var filePath = await deepThinkingService.GenerateDeepThinkingEvolutionAsync(file, model);

                Console.WriteLine();
                WriteColorLine($"Deep thinking evolution generated and saved to: {filePath}", ConsoleColor.Green);
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error evolving exploration: {ex.Message}", ConsoleColor.Red);
            }
        }, evolveFileOption, evolveModelOption);

        // Add series subcommand
        var thinkSeriesCommand = new TarsCommand("series", "Generate a series of related deep thinking explorations");
        var baseTopicOption = new Option<string>("--base-topic", "The base topic for the series") { IsRequired = true };
        var seriesCountOption = new Option<int>("--count", () => 3, "The number of explorations to generate");
        var seriesModelOption = new Option<string>("--model", () => "llama3", "The model to use for deep thinking");

        thinkSeriesCommand.AddOption(baseTopicOption);
        thinkSeriesCommand.AddOption(seriesCountOption);
        thinkSeriesCommand.AddOption(seriesModelOption);

        thinkSeriesCommand.SetHandler(async (string baseTopic, int count, string model) =>
        {
            WriteHeader("TARS Deep Thinking Series");

            // Use the new reflection service
            var reflectionService = _serviceProvider!.GetRequiredService<ExplorationReflectionService2>();
            var ollamaService = _serviceProvider!.GetRequiredService<OllamaService>();
            var configuration = _serviceProvider!.GetRequiredService<IConfiguration>();
            var logger = _serviceProvider!.GetRequiredService<ILogger<DeepThinkingService>>();

            // Create the deep thinking service manually
            var deepThinkingService = new DeepThinkingService(logger, configuration, ollamaService, reflectionService);

            WriteColorLine($"Generating deep thinking series on base topic: {baseTopic}", ConsoleColor.Cyan);
            WriteColorLine($"Number of explorations: {count}", ConsoleColor.Cyan);
            WriteColorLine($"Using model: {model}", ConsoleColor.Cyan);

            Console.WriteLine();
            WriteColorLine("Thinking deeply...", ConsoleColor.Yellow);

            try
            {
                // Generate the deep thinking series
                var filePaths = await deepThinkingService.GenerateDeepThinkingSeriesAsync(baseTopic, count, model);

                Console.WriteLine();
                WriteColorLine($"Deep thinking series generated with {filePaths.Count} explorations:", ConsoleColor.Green);

                foreach (var filePath in filePaths)
                {
                    WriteColorLine($"- {filePath}", ConsoleColor.Green);
                }
            }
            catch (Exception ex)
            {
                WriteColorLine($"Error generating deep thinking series: {ex.Message}", ConsoleColor.Red);
            }
        }, baseTopicOption, seriesCountOption, seriesModelOption);

        // Add versions subcommand
        var thinkVersionsCommand = new TarsCommand("versions", "List exploration versions");

        thinkVersionsCommand.SetHandler(() =>
        {
            WriteHeader("TARS Exploration Versions");

            // Use the new reflection service
            var reflectionService = _serviceProvider!.GetRequiredService<ExplorationReflectionService2>();
            var ollamaService = _serviceProvider!.GetRequiredService<OllamaService>();
            var configuration = _serviceProvider!.GetRequiredService<IConfiguration>();
            var logger = _serviceProvider!.GetRequiredService<ILogger<DeepThinkingService>>();

            // Create the deep thinking service manually
            var deepThinkingService = new DeepThinkingService(logger, configuration, ollamaService, reflectionService);

            var versions = deepThinkingService.GetExplorationVersions();

            if (versions.Count == 0)
            {
                WriteColorLine("No exploration versions found.", ConsoleColor.Yellow);
                return;
            }

            WriteColorLine($"Found {versions.Count} exploration versions:", ConsoleColor.Green);
            Console.WriteLine();

            foreach (var version in versions)
            {
                WriteColorLine($"- {version}", ConsoleColor.Cyan);
            }

            Console.WriteLine();
            WriteColorLine($"Latest version: {deepThinkingService.GetLatestExplorationVersion()}", ConsoleColor.Green);
            WriteColorLine($"Next version: {deepThinkingService.GetNextExplorationVersion()}", ConsoleColor.Green);
        });

        // Add subcommands to deep thinking command
        deepThinkingCommand.AddCommand(thinkGenerateCommand);
        deepThinkingCommand.AddCommand(thinkEvolveCommand);
        deepThinkingCommand.AddCommand(thinkSeriesCommand);
        deepThinkingCommand.AddCommand(thinkVersionsCommand);

        // Create DSL command
        var dslCommand = new DslCommand(_serviceProvider!.GetRequiredService<DslService>());

        // Create DSL debug command
        var dslDebugCommand = new DslDebugCommand(_serviceProvider!);

        rootCommand.AddCommand(chatCommand);
        rootCommand.AddCommand(deepThinkingCommand);
        rootCommand.AddCommand(consoleCaptureCommand);
        rootCommand.AddCommand(dslCommand);
        rootCommand.AddCommand(dslDebugCommand);

        // Add Metascript command
        var metascriptCommand = new MetascriptCommand(_serviceProvider);
        rootCommand.AddCommand(metascriptCommand);

        // Add Docker Model Runner command
        var dockerModelRunnerCommand = new DockerModelRunnerCommand();
        rootCommand.AddCommand(dockerModelRunnerCommand);

        // Add Auto-Improve command using metascripts
        var autoImproveMetascriptCommand = new AutoImproveCommand(
            _serviceProvider.GetRequiredService<ILogger<AutoImproveCommand>>(),
            _serviceProvider.GetRequiredService<DslService>(),
            _serviceProvider.GetRequiredService<ConsoleService>());
        rootCommand.AddCommand(autoImproveMetascriptCommand);

        // Add Self-Diagnose command
        var selfDiagnoseCommand = new Commands.SelfDiagnoseCommand();
        rootCommand.AddCommand(selfDiagnoseCommand);

        // Add Explorations Improve command
        var explorationsImproveCommand = new Commands.ExplorationsImproveCommand(
            _serviceProvider.GetRequiredService<ILogger<Commands.ExplorationsImproveCommand>>(),
            _serviceProvider.GetRequiredService<DslService>(),
            _serviceProvider.GetRequiredService<ConsoleService>());
        rootCommand.AddCommand(explorationsImproveCommand);

        // Add Knowledge Apply command
        var knowledgeApplyCommand = new Commands.KnowledgeApplyCommand();
        knowledgeApplyCommand.SetServices(
            _serviceProvider.GetRequiredService<ILogger<Commands.KnowledgeApplyCommand>>(),
            _serviceProvider.GetRequiredService<KnowledgeApplicationService>(),
            _serviceProvider.GetRequiredService<ConsoleService>());
        rootCommand.AddCommand(knowledgeApplyCommand);

        // Add Knowledge Integrate command
        var knowledgeIntegrateCommand = new Commands.KnowledgeIntegrateCommand(
            _serviceProvider.GetRequiredService<ILogger<Commands.KnowledgeIntegrateCommand>>(),
            _serviceProvider.GetRequiredService<KnowledgeIntegrationService>(),
            _serviceProvider.GetRequiredService<ConsoleService>());
        rootCommand.AddCommand(knowledgeIntegrateCommand);

        // Add Autonomous command
        var autonomousCommand = new Commands.AutonomousCommand(
            _serviceProvider.GetRequiredService<ILogger<Commands.AutonomousCommand>>(),
            _serviceProvider.GetRequiredService<AutonomousImprovementService>(),
            _serviceProvider.GetRequiredService<ConsoleService>());
        rootCommand.AddCommand(autonomousCommand);

        // Add Self-Improvement command
        var selfImprovementController = _serviceProvider.GetRequiredService<SelfImprovementController>();
        selfImprovementController.RegisterCommands(rootCommand!);

        // Add Retroaction command
        var retroactionCommand = _serviceProvider.GetRequiredService<RetroactionCommand>();
        rootCommand.AddCommand(retroactionCommand.RegisterCommands());

        // Add RetroactionLoop command
        var retroactionLoopCommand = new RetroactionLoopCommand(
            _serviceProvider.GetRequiredService<ILogger<RetroactionLoopCommand>>(),
            _serviceProvider.GetRequiredService<RetroactionLoopService>(),
            _serviceProvider.GetRequiredService<ConsoleService>());
        rootCommand.AddCommand(retroactionLoopCommand);

        // Add DocumentationKnowledge command
        var docExtractCommand = new DocumentationKnowledgeCommand(
            _serviceProvider.GetRequiredService<ILogger<DocumentationKnowledgeCommand>>(),
            _serviceProvider.GetRequiredService<DocumentationKnowledgeService>(),
            _serviceProvider.GetRequiredService<ConsoleService>());
        rootCommand.AddCommand(docExtractCommand);

        // Add KnowledgeVisualization command
        var knowledgeVizCommand = new KnowledgeVisualizationCommand(
            _serviceProvider.GetRequiredService<ILogger<KnowledgeVisualizationCommand>>(),
            _serviceProvider.GetRequiredService<KnowledgeVisualizationService>(),
            _serviceProvider.GetRequiredService<ConsoleService>());
        rootCommand.AddCommand(knowledgeVizCommand);

        // Add KnowledgeTestGeneration command
        var knowledgeTestCommand = new KnowledgeTestGenerationCommand(
            _serviceProvider.GetRequiredService<ILogger<KnowledgeTestGenerationCommand>>(),
            _serviceProvider.GetRequiredService<KnowledgeTestGenerationService>(),
            _serviceProvider.GetRequiredService<ConsoleService>());
        rootCommand.AddCommand(knowledgeTestCommand);

        // Add AutonomousImprovement command
        var autonomousImprovementCommand = new AutonomousImprovementCommand(
            _serviceProvider.GetRequiredService<ILogger<AutonomousImprovementCommand>>(),
            _serviceProvider.GetRequiredService<AutonomousImprovementService>(),
            _serviceProvider.GetRequiredService<ConsoleService>());
        rootCommand.AddCommand(autonomousImprovementCommand);

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

    /// <summary>
    /// Run the chat loop for interactive chat
    /// </summary>
    private static async Task RunChatLoop(ChatBotService chatBotService)
    {
        try
        {
            while (true)
            {
                // Display prompt
                WriteColorLine("\nYou: ", ConsoleColor.Green);

                // Read user input
                var input = Console.ReadLine();

                // Check for exit commands
                if (string.IsNullOrWhiteSpace(input) || input.ToLower() == "exit" || input.ToLower() == "quit")
                {
                    WriteColorLine("\nChat session ended.", ConsoleColor.Yellow);
                    break;
                }

                // Check for special commands
                if (input.ToLower() == "examples")
                {
                    WriteColorLine("\nExample prompts:", ConsoleColor.Cyan);
                    var examples = chatBotService.GetExamplePrompts();
                    for (int i = 0; i < examples.Count; i++)
                    {
                        WriteColorLine($"{i + 1}. {examples[i]}", ConsoleColor.Cyan);
                    }
                    continue;
                }

                if (input.ToLower() == "clear")
                {
                    chatBotService.StartNewConversation();
                    WriteColorLine("\nStarted a new conversation.", ConsoleColor.Yellow);
                    continue;
                }

                if (input.ToLower().StartsWith("model "))
                {
                    var model = input.Substring(6).Trim();
                    chatBotService.SetModel(model);
                    WriteColorLine($"\nChanged model to: {model}", ConsoleColor.Yellow);
                    continue;
                }

                if (input.ToLower() == "speech on")
                {
                    chatBotService.EnableSpeech(true);
                    WriteColorLine("\nSpeech enabled.", ConsoleColor.Yellow);
                    continue;
                }

                if (input.ToLower() == "speech off")
                {
                    chatBotService.EnableSpeech(false);
                    WriteColorLine("\nSpeech disabled.", ConsoleColor.Yellow);
                    continue;
                }

                // Send message to chat bot
                WriteColorLine("\nTARS: ", ConsoleColor.Blue);

                // Create cancellation token for typing indicator
                var typingCts = new CancellationTokenSource();

                // Show typing indicator
                var typingTask = Task.Run(async () =>
                {
                    var typingChars = new[] { '|', '/', '-', '\\' };
                    var typingIndex = 0;

                    try
                    {
                        while (!typingCts.Token.IsCancellationRequested)
                        {
                            Console.Write(typingChars[typingIndex]);
                            await Task.Delay(100, typingCts.Token);
                            Console.Write("\b");
                            typingIndex = (typingIndex + 1) % typingChars.Length;
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        // Expected when cancellation is requested
                    }
                }, typingCts.Token);

                // Get response from chat bot
                var response = await chatBotService.SendMessageAsync(input);

                // Stop typing indicator
                typingCts.Cancel();
                try
                {
                    await typingTask; // Wait for the task to complete
                }
                catch (OperationCanceledException)
                {
                    // Expected when cancellation is requested
                }

                Console.Write(" ");
                Console.Write("\b");

                // Display response
                Console.WriteLine(response);
            }
        }
        catch (Exception ex)
        {
            WriteColorLine($"\nError in chat loop: {ex.Message}", ConsoleColor.Red);
            if (ex.InnerException != null)
            {
                WriteColorLine($"Inner exception: {ex.InnerException.Message}", ConsoleColor.Red);
            }
            WriteColorLine($"Stack trace: {ex.StackTrace}", ConsoleColor.DarkRed);
        }
    }
}
