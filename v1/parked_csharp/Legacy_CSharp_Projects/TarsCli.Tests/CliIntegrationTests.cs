using System;
using System.CommandLine;
using System.CommandLine.Builder;
using System.CommandLine.Parsing;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using TarsCli.Mcp;
using TarsCli.Parsing;
using TarsCli.Services;
using Xunit;

namespace TarsCli.Tests;

public class CliIntegrationTests
{
    private readonly ServiceProvider _serviceProvider;
    private readonly RootCommand _rootCommand;

    public CliIntegrationTests()
    {
        // Set up configuration
        var configuration = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string>
            {
                { "Tars:Mcp:AutoExecuteEnabled", "true" }
            })
            .Build();

        // Set up services
        var services = new ServiceCollection();
        services.AddSingleton<IConfiguration>(configuration);
        services.AddLogging(builder => builder.AddConsole());
        services.AddSingleton<McpController>();
        services.AddSingleton<EnhancedMcpService>();

        _serviceProvider = services.BuildServiceProvider();

        // Set up command line
        _rootCommand = new RootCommand("TARS CLI");
        var mcpCommand = new Command("mcp", "Master Control Program commands");
        var codeCommand = new Command("code", "Generate and save code without asking for permission");
        var fileArgument = new Argument<string>("file", "Path to the file to create or update");
        var contentArgument = new Argument<string>("content", "The content to write to the file");

        codeCommand.AddArgument(fileArgument);
        codeCommand.AddArgument(contentArgument);

        codeCommand.SetHandler(async (string file, string content) =>
        {
            // Check if the content is triple-quoted
            if (content.StartsWith("\"\"\"") && content.EndsWith("\"\"\""))
            {
                // Remove the triple quotes
                content = content.Substring(3, content.Length - 6);
            }

            // Create a direct parameter string with file path and content
            var codeSpec = $"{file}:::{content}";

            var mcpController = _serviceProvider.GetRequiredService<McpController>();
            await mcpController.ExecuteCommand("code", codeSpec);
        }, fileArgument, contentArgument);

        mcpCommand.AddCommand(codeCommand);
        _rootCommand.AddCommand(mcpCommand);
    }

    [Fact]
    public async Task ParseAndExecute_WithTripleQuotedCode_GeneratesFile()
    {
        // Arrange
        var testFilePath = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.cs");
        var codeContent = "\"\"\"public class Test {}\"\"\"";
        var args = new[] { "mcp", "code", testFilePath, codeContent };

        try
        {
            // Create the parser
            var parser = new CommandLineBuilder(_rootCommand)
                .UseDefaults()
                .Build();

            // Act
            var parseResult = TripleQuotedArgumentParser.ParseCommandLine(parser, args);
            await parseResult.InvokeAsync();

            // Assert
            Assert.True(File.Exists(testFilePath));
            var fileContent = await File.ReadAllTextAsync(testFilePath);
            Assert.Equal("public class Test {}", fileContent);
        }
        finally
        {
            // Cleanup
            if (File.Exists(testFilePath))
            {
                File.Delete(testFilePath);
            }
        }
    }

    [Fact]
    public async Task ParseAndExecute_WithMultilineTripleQuotedCode_GeneratesFile()
    {
        // Arrange
        var testFilePath = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.cs");
        var codeContent = @"""""""using System;

public class Test
{
    public static void Main()
    {
        Console.WriteLine(""Hello from MCP!"");
    }
}""""""";

        var args = new[] { "mcp", "code", testFilePath, codeContent };

        try
        {
            // Create the parser
            var parser = new CommandLineBuilder(_rootCommand)
                .UseDefaults()
                .Build();

            // Act
            var parseResult = TripleQuotedArgumentParser.ParseCommandLine(parser, args);
            await parseResult.InvokeAsync();

            // Assert
            Assert.True(File.Exists(testFilePath));
            var fileContent = await File.ReadAllTextAsync(testFilePath);
            Assert.Contains("using System;", fileContent);
            Assert.Contains("Console.WriteLine", fileContent);
        }
        finally
        {
            // Cleanup
            if (File.Exists(testFilePath))
            {
                File.Delete(testFilePath);
            }
        }
    }
}
