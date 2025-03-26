using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using NLog;
using NLog.Extensions.Logging;
using System.CommandLine;
using TarsCli.Services;

namespace TarsCli;

internal static class Program
{
    // Main method
    public static async Task<int> Main(string[] args)
    {
        // Setup NLog
        LogManager.Setup()
            .LoadConfigurationFromFile("nlog.config")
            .GetCurrentClassLogger();

        try
        {
            // Setup configuration
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: false)
                .AddEnvironmentVariables()
                .AddCommandLine(args)
                .Build();

            // Create logs directory if it doesn't exist
            Directory.CreateDirectory(Path.Combine(Directory.GetCurrentDirectory(), "logs"));
            Directory.CreateDirectory(Path.Combine(Directory.GetCurrentDirectory(), "logs", "archives"));

            // Setup DI
            var serviceProvider = new ServiceCollection()
                .AddLogging(builder => 
                {
                    builder.ClearProviders();
                    builder.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Trace);
                    builder.AddNLog(configuration);
                })
                .AddSingleton<IConfiguration>(configuration)
                .AddSingleton<OllamaService>()
                .AddSingleton<OllamaSetupService>()
                .AddSingleton<RetroactionService>()
                .AddSingleton<DiagnosticsService>()
                .AddSingleton<TarsCli.Mcp.McpController>()
                .BuildServiceProvider();

            // Get services
            var logger = serviceProvider.GetRequiredService<ILoggerFactory>().CreateLogger("TarsCli");
            var retroactionService = serviceProvider.GetRequiredService<RetroactionService>();
            var diagnosticsService = serviceProvider.GetRequiredService<DiagnosticsService>();
            var setupService = serviceProvider.GetRequiredService<OllamaSetupService>();

            // Run initial diagnostics
            logger.LogInformation("Running initial diagnostics...");
            var diagnosticsResult = await diagnosticsService.RunInitialDiagnosticsAsync();

            // Skip confirmation prompt for diagnostics command
            var skipConfirmation = args.Length > 0 && 
                                   (args[0].Equals("diagnostics", StringComparison.OrdinalIgnoreCase) || 
                                    args[0].Equals("setup", StringComparison.OrdinalIgnoreCase));

            if (!diagnosticsResult.IsReady && !skipConfirmation)
            {
                logger.LogWarning("System is not fully ready for TARS operations");
                logger.LogInformation("You can still proceed, but some features may not work as expected");
                
                // Prompt user to continue
                Console.Write("Do you want to continue anyway? (y/n): ");
                var response = Console.ReadLine()?.ToLower();
                if (response != "y" && response != "yes")
                {
                    logger.LogInformation("Operation cancelled by user");
                    return 1;
                }
            }
            else if (!diagnosticsResult.IsReady)
            {
                logger.LogWarning("System is not fully ready for TARS operations");
                logger.LogInformation("Proceeding with command execution anyway...");
            }
            else
            {
                logger.LogInformation("System is ready for TARS operations");
            }

            // Check Ollama setup
            Console.WriteLine("Checking Ollama setup...");
            if (!await setupService.CheckOllamaSetupAsync())
            {
                logger.LogError("Failed to set up Ollama. Please run the Install-Prerequisites.ps1 script or set up Ollama manually.");
                logger.LogInformation("You can find the script in the Scripts directory.");
                Console.WriteLine("Ollama setup failed!");
                return 1;
            }
            Console.WriteLine("Ollama setup successful!");

            // Setup and run command line
            Console.WriteLine("Setting up command line...");
            var rootCommand = CliSupport.SetupCommandLine(
                configuration, 
                logger,
                serviceProvider.GetRequiredService<ILoggerFactory>(), 
                diagnosticsService, 
                retroactionService,
                setupService);  // setupService is the OllamaSetupService instance
            
            Console.WriteLine($"Invoking command: {string.Join(" ", args)}");
            var result = await rootCommand.InvokeAsync(args);
            Console.WriteLine($"Command completed with result: {result}");
            return result;
        }
        catch (Exception ex)
        {
            // Log any startup exceptions
            Console.WriteLine($"CRITICAL ERROR: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
            LogManager.GetCurrentClassLogger().Error(ex, "Stopped program because of exception");
            return 1;
        }
        finally
        {
            // Ensure to flush and stop internal timers/threads before application-exit
            LogManager.Shutdown();
        }
    }
}