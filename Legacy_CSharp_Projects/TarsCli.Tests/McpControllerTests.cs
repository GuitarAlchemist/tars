using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Moq;
using TarsCli.Mcp;
using Xunit;

namespace TarsCli.Tests;

public class McpControllerTests
{
    private readonly Mock<ILogger<McpController>> _loggerMock;
    private readonly Mock<IConfiguration> _configMock;
    private readonly McpController _controller;

    public McpControllerTests()
    {
        _loggerMock = new Mock<ILogger<McpController>>();
        _configMock = new Mock<IConfiguration>();

        // Set up configuration to enable auto-execute
        _configMock.Setup(c => c["Tars:Mcp:AutoExecuteEnabled"]).Returns("true");

        _controller = new McpController(_loggerMock.Object, _configMock.Object);
    }

    [Fact]
    public async Task ExecuteCommand_WithCodeCommand_GeneratesFile()
    {
        // Arrange
        var testFilePath = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.cs");
        var codeContent = "public class Test {}";
        var codeSpec = $"{testFilePath}:::{codeContent}";

        try
        {
            // Act
            var result = await _controller.ExecuteCommand("code", codeSpec);

            // Assert
            Assert.Contains($"Code generated and saved to: {testFilePath}", result);
            Assert.True(File.Exists(testFilePath));
            var fileContent = await File.ReadAllTextAsync(testFilePath);
            Assert.Equal(codeContent, fileContent);
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
    public async Task ExecuteCommand_WithTripleQuotedCode_GeneratesFile()
    {
        // Arrange
        var testFilePath = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid()}.cs");
        var codeContent = @"using System;

public class Test
{
    public static void Main()
    {
        Console.WriteLine(""Hello from MCP!"");
    }
}";
        var codeSpec = $"{testFilePath}:::{codeContent}";

        try
        {
            // Act
            var result = await _controller.ExecuteCommand("code", codeSpec);

            // Assert
            Assert.Contains($"Code generated and saved to: {testFilePath}", result);
            Assert.True(File.Exists(testFilePath));
            var fileContent = await File.ReadAllTextAsync(testFilePath);
            Assert.Equal(codeContent, fileContent);
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
