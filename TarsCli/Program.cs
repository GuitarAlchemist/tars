using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using System.CommandLine;
using TarsCli.Services;

// Setup configuration
var configuration = new ConfigurationBuilder()
    .SetBasePath(Directory.GetCurrentDirectory())
    .AddJsonFile("appsettings.json", optional: false)
    .AddEnvironmentVariables()
    .AddCommandLine(args)
    .Build();

// Setup DI
var serviceProvider = new ServiceCollection()
    .AddLogging(builder => 
    {
        builder.AddConsole();
        builder.SetMinimumLevel(LogLevel.Information);
    })
    .AddSingleton<IConfiguration>(configuration)
    .AddSingleton<OllamaService>()
    .AddSingleton<OllamaSetupService>()
    .AddSingleton<RetroactionService>()
    .BuildServiceProvider();

var logger = serviceProvider.GetRequiredService<ILogger<Program>>();
var setupService = serviceProvider.GetRequiredService<OllamaSetupService>();
var retroactionService = serviceProvider.GetRequiredService<RetroactionService>();

// Check Ollama setup
if (!await setupService.CheckOllamaSetupAsync())
{
    logger.LogError("Failed to set up Ollama. Please run the Install-Prerequisites.ps1 script or set up Ollama manually.");
    logger.LogInformation("You can find the script in the Scripts directory.");
    return 1;
}

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

// Create root command
var rootCommand = new RootCommand("TARS CLI - Transformative Autonomous Reasoning System");

// Create process command
var processCommand = new Command("process", "Process a file through the TARS retroaction loop")
{
    fileOption,
    taskOption,
    modelOption
};

processCommand.SetHandler(async (file, task, model) =>
{
    logger.LogInformation($"Processing file: {file}");
    logger.LogInformation($"Task: {task}");
    logger.LogInformation($"Model: {model}");

    bool success = await retroactionService.ProcessFile(file, task, model);
    
    if (success)
    {
        logger.LogInformation("Processing completed successfully");
    }
    else
    {
        logger.LogError("Processing failed");
        Environment.Exit(1);
    }
}, fileOption, taskOption, modelOption);

// Create docs command to process documentation files
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
        logger.LogError($"Directory not found: {docsPath}");
        Environment.Exit(1);
        return;
    }
    
    logger.LogInformation($"Processing docs in: {docsPath}");
    logger.LogInformation($"Task: {task}");
    logger.LogInformation($"Model: {model}");
    
    // Process all markdown files in the directory
    var files = Directory.GetFiles(docsPath, "*.md", SearchOption.AllDirectories);
    int successCount = 0;
    
    foreach (var file in files)
    {
        logger.LogInformation($"Processing file: {file}");
        bool success = await retroactionService.ProcessFile(file, task, model);
        
        if (success)
        {
            successCount++;
        }
    }
    
    logger.LogInformation($"Processing completed. {successCount}/{files.Length} files processed successfully.");
    
}, taskOption, modelOption, docsPathOption);

rootCommand.AddCommand(processCommand);
rootCommand.AddCommand(docsCommand);

// Run the command
return await rootCommand.InvokeAsync(args);