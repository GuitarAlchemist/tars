using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using TarsCli.Services;
using Xunit;

namespace TarsCli.Tests.Services
{
    public class DslServiceTests
    {
        private readonly Mock<ILogger<DslService>> _loggerMock;
        private readonly Mock<OllamaService> _ollamaServiceMock;
        private readonly Mock<TemplateService> _templateServiceMock;
        private readonly DslService _dslService;
        private readonly string _tempDirectory;

        public DslServiceTests()
        {
            _loggerMock = new Mock<ILogger<DslService>>();
            _ollamaServiceMock = new Mock<OllamaService>(MockBehavior.Loose, null);
            _templateServiceMock = new Mock<TemplateService>(MockBehavior.Loose, null);
            _dslService = new DslService(_loggerMock.Object, _ollamaServiceMock.Object, _templateServiceMock.Object);
            
            // Create a temporary directory for test files
            _tempDirectory = Path.Combine(Path.GetTempPath(), "TarsCli.Tests", Guid.NewGuid().ToString());
            Directory.CreateDirectory(_tempDirectory);
        }

        [Fact]
        public async Task GenerateEbnfAsync_CreatesFileWithEbnfGrammar()
        {
            // Arrange
            string outputPath = Path.Combine(_tempDirectory, "tars_grammar.ebnf");

            // Act
            int result = await _dslService.GenerateEbnfAsync(outputPath);

            // Assert
            Assert.Equal(0, result);
            Assert.True(File.Exists(outputPath));
            
            string content = await File.ReadAllTextAsync(outputPath);
            Assert.NotEmpty(content);
            Assert.Contains("(* TARS DSL EBNF Grammar Specification *)", content);
            Assert.Contains("Program = Block, {Block};", content);
            Assert.Contains("ConfigBlock = 'CONFIG', '{', [Properties], '}';", content);
            Assert.Contains("PropertyValue = StringValue | NumberValue | BoolValue | ListValue | ObjectValue;", content);
        }

        [Fact]
        public async Task GenerateEbnfAsync_CreatesDirectoryIfNotExists()
        {
            // Arrange
            string nestedDir = Path.Combine(_tempDirectory, "nested", "dir");
            string outputPath = Path.Combine(nestedDir, "tars_grammar.ebnf");

            // Act
            int result = await _dslService.GenerateEbnfAsync(outputPath);

            // Assert
            Assert.Equal(0, result);
            Assert.True(Directory.Exists(nestedDir));
            Assert.True(File.Exists(outputPath));
        }

        [Fact]
        public async Task GenerateDslTemplateAsync_CreatesFileWithTemplate()
        {
            // Arrange
            string outputPath = Path.Combine(_tempDirectory, "template.tars");
            string templateName = "basic";

            // Act
            int result = await _dslService.GenerateDslTemplateAsync(outputPath, templateName);

            // Assert
            Assert.Equal(0, result);
            Assert.True(File.Exists(outputPath));
            
            string content = await File.ReadAllTextAsync(outputPath);
            Assert.NotEmpty(content);
            Assert.Contains("CONFIG {", content);
        }

        [Fact]
        public async Task GenerateDslTemplateAsync_SupportsMultipleTemplates()
        {
            // Arrange
            string basicPath = Path.Combine(_tempDirectory, "basic.tars");
            string chatPath = Path.Combine(_tempDirectory, "chat.tars");
            string agentPath = Path.Combine(_tempDirectory, "agent.tars");

            // Act
            int basicResult = await _dslService.GenerateDslTemplateAsync(basicPath, "basic");
            int chatResult = await _dslService.GenerateDslTemplateAsync(chatPath, "chat");
            int agentResult = await _dslService.GenerateDslTemplateAsync(agentPath, "agent");

            // Assert
            Assert.Equal(0, basicResult);
            Assert.Equal(0, chatResult);
            Assert.Equal(0, agentResult);
            
            Assert.True(File.Exists(basicPath));
            Assert.True(File.Exists(chatPath));
            Assert.True(File.Exists(agentPath));
            
            string basicContent = await File.ReadAllTextAsync(basicPath);
            string chatContent = await File.ReadAllTextAsync(chatPath);
            string agentContent = await File.ReadAllTextAsync(agentPath);
            
            Assert.Contains("PROMPT {", basicContent);
            Assert.Contains("role: \"system\"", chatContent);
            Assert.Contains("AGENT {", agentContent);
        }

        public void Dispose()
        {
            // Clean up temporary directory
            if (Directory.Exists(_tempDirectory))
            {
                try
                {
                    Directory.Delete(_tempDirectory, true);
                }
                catch
                {
                    // Ignore cleanup errors
                }
            }
        }
    }
}
