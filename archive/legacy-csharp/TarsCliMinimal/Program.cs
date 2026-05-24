using System;
using System.CommandLine;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using NLog.Extensions.Logging;

namespace TarsCliMinimal
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("TARS CLI Minimal Implementation");
            Console.WriteLine("This is a minimal implementation of the TARS CLI.");
            Console.WriteLine("It's designed to work around assembly attribute conflicts.");
            Console.WriteLine();

            // Setup configuration
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
                .AddEnvironmentVariables()
                .AddCommandLine(args)
                .Build();

            // Setup logging
            var serviceProvider = new ServiceCollection()
                .AddLogging(builder =>
                {
                    builder.AddConfiguration(configuration.GetSection("Logging"));
                    builder.AddConsole();
                    builder.AddNLog();
                })
                .BuildServiceProvider();

            var logger = serviceProvider.GetRequiredService<ILoggerFactory>()
                .CreateLogger<Program>();

            logger.LogInformation("TARS CLI Minimal started");

            // Create a root command
            var rootCommand = new RootCommand("TARS CLI - A command-line interface for TARS");

            // Add the test generator command
            var testGeneratorLogger = logger.GetLogger<TestGeneratorCommand>();
            var testGeneratorCommand = new TestGeneratorCommand(testGeneratorLogger);
            rootCommand.AddCommand(testGeneratorCommand);

            // Execute the command
            await rootCommand.InvokeAsync(args);
        }
    }
}
