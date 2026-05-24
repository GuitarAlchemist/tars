using System;
using System.CommandLine;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsCli.Services;
using TarsCli.Services.Testing;

namespace TarsCli.Commands;

/// <summary>
/// Command for testing the test generator
/// </summary>
public class TestGeneratorCommand : Command
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<TestGeneratorCommand> _logger;

    /// <summary>
    /// Initializes a new instance of the TestGeneratorCommand class
    /// </summary>
    /// <param name="serviceProvider">Service provider</param>
    /// <param name="logger">Logger instance</param>
    public TestGeneratorCommand(IServiceProvider serviceProvider, ILogger<TestGeneratorCommand> logger)
        : base("test-generator", "Test the test generator")
    {
        _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        // Add options
        var fileOption = new Option<string>(
            name: "--file",
            description: "Path to the file to generate tests for")
        {
            IsRequired = true
        };

        var generatorOption = new Option<string>(
            name: "--generator",
            description: "Generator to use (csharp, improved-csharp, fsharp)")
        {
            IsRequired = false
        };
        generatorOption.SetDefaultValue("improved-csharp");

        AddOption(fileOption);
        AddOption(generatorOption);

        this.SetHandler(HandleCommand, fileOption, generatorOption);
    }

    private async Task HandleCommand(string filePath, string generatorName)
    {
        try
        {
            _logger.LogInformation($"Testing test generator with file: {filePath}");

            // Validate file path
            if (!File.Exists(filePath))
            {
                _logger.LogError($"File not found: {filePath}");
                return;
            }

            // Get the file content
            var fileContent = await File.ReadAllTextAsync(filePath);

            // Get the appropriate test generator
            ITestGenerator generator = generatorName.ToLowerInvariant() switch
            {
                "csharp" => _serviceProvider.GetServices<ITestGenerator>().FirstOrDefault(g => g is CSharpTestGenerator),
                "improved-csharp" => _serviceProvider.GetServices<ITestGenerator>().FirstOrDefault(g => g is ImprovedCSharpTestGenerator),
                "fsharp" => _serviceProvider.GetServices<ITestGenerator>().FirstOrDefault(g => g is FSharpTestGenerator),
                _ => throw new ArgumentException($"Unknown generator: {generatorName}")
            };

            if (generator == null)
            {
                _logger.LogError($"Generator not found: {generatorName}");
                return;
            }

            // Generate tests
            var result = await generator.GenerateTestsAsync(filePath, fileContent);

            // Display the results
            Console.WriteLine($"Test generation {(result.Success ? "succeeded" : "failed")}");
            if (!result.Success)
            {
                Console.WriteLine($"Error: {result.ErrorMessage}");
                return;
            }

            Console.WriteLine($"Generated {result.Tests.Count} tests");
            Console.WriteLine($"Test file path: {result.TestFilePath}");
            Console.WriteLine();
            Console.WriteLine("Test file content:");
            Console.WriteLine(result.TestFileContent);

            // Save the test file
            var directory = Path.GetDirectoryName(result.TestFilePath);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }
            await File.WriteAllTextAsync(result.TestFilePath, result.TestFileContent);
            Console.WriteLine($"Test file saved to: {result.TestFilePath}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error testing test generator");
            Console.WriteLine($"Error: {ex.Message}");
        }
    }
}
