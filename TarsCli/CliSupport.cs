using System.CommandLine;
using System.CommandLine.Invocation;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TarsCli.Services;

namespace TarsCli;

public static class CliSupport
{
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
        DiagnosticsService diagnosticsService,
        RetroactionService retroactionService)
    {
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
            string? filePath = result.GetValueForOption(fileOption);
            if (filePath != null && !File.Exists(filePath))
            {
                result.ErrorMessage = $"File does not exist: {filePath}";
            }
        });

        // Add validation for model option
        modelOption.AddValidator(result =>
        {
            string? model = result.GetValueForOption(modelOption);
            if (string.IsNullOrWhiteSpace(model))
            {
                result.ErrorMessage = "Model name cannot be empty";
            }
        });

        // Create root command
        var rootCommand = new RootCommand("TARS CLI - Transformative Autonomous Reasoning System");

        // Add help option
        var helpOption = new Option<bool>(
            aliases: new[] { "--help", "-h" },
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
            
            WriteHeader("Global Options");
            WriteCommand("--help, -h", "Display help information");
            
            WriteHeader("Examples");
            WriteExample("tarscli process --file path/to/file.cs --task \"Refactor this code\"");
            WriteExample("tarscli docs --task \"Improve documentation clarity\"");
            WriteExample("tarscli diagnostics");
            
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

            bool success = await retroactionService.ProcessFile(file, task, model);
            
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
            string docsPath = Path.Combine(configuration["Tars:ProjectRoot"] ?? "", "docs");
            
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
            int successCount = 0;
            
            foreach (var file in files)
            {
                Console.WriteLine($"Processing file: {file}");
                bool success = await retroactionService.ProcessFile(file, task, model);
                
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
            Console.WriteLine("\n=== Running TARS Diagnostics ===");
            Console.WriteLine("Checking system configuration, Ollama setup, and required models...");
            
            var diagnosticsResult = await diagnosticsService.RunInitialDiagnosticsAsync(verbose: true);
            
            WriteHeader("=== TARS Diagnostics Report ===");
            Console.WriteLine($"System: {diagnosticsResult.SystemInfo.OperatingSystem}");
            Console.WriteLine($"CPU Cores: {diagnosticsResult.SystemInfo.ProcessorCores}");
            Console.WriteLine($"Available Memory: {diagnosticsResult.SystemInfo.AvailableMemoryGB:F2} GB");
            Console.WriteLine();
            
            WriteHeader("Ollama Configuration");
            Console.WriteLine($"  Base URL: {diagnosticsResult.OllamaConfig.BaseUrl}");
            Console.WriteLine($"  Default Model: {diagnosticsResult.OllamaConfig.DefaultModel}");
            Console.WriteLine();
            
            WriteHeader("Required Models");
            foreach (var model in diagnosticsResult.ModelStatus)
            {
                var statusColor = model.Value ? ConsoleColor.Green : ConsoleColor.Red;
                var statusText = model.Value ? "Available ✓" : "Not Available ✗";
                Console.Write($"  {model.Key}: ");
                WriteColorLine(statusText, statusColor);
            }
            Console.WriteLine();
            
            WriteHeader("Project Configuration");
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
                    WriteColorLine("  - Missing required models. Run the Install-Prerequisites.ps1 script:", ConsoleColor.Yellow);
                    WriteColorLine("    .\\TarsCli\\Scripts\\Install-Prerequisites.ps1", ConsoleColor.Cyan);
                }
            }
            
            WriteColorLine("===========================", ConsoleColor.Cyan);
        });

        // Add commands to root command
        rootCommand.AddCommand(helpCommand);
        rootCommand.AddCommand(processCommand);
        rootCommand.AddCommand(docsCommand);
        rootCommand.AddCommand(diagnosticsCommand);

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