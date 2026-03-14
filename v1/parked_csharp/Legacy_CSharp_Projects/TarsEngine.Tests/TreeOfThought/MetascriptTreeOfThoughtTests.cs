using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using TarsEngine.Services.Compilation;
using TarsEngine.Services.TreeOfThought;
using Xunit;

namespace TarsEngine.Tests.TreeOfThought
{
    public class MetascriptTreeOfThoughtTests
    {
        private readonly Mock<ILogger<MetascriptTreeOfThoughtService>> _loggerMock;
        private readonly Mock<FSharpScriptExecutor> _scriptExecutorMock;
        private readonly MetascriptTreeOfThoughtService _service;

        public MetascriptTreeOfThoughtTests()
        {
            _loggerMock = new Mock<ILogger<MetascriptTreeOfThoughtService>>();
            _scriptExecutorMock = new Mock<FSharpScriptExecutor>(MockBehavior.Loose, new object[] { Mock.Of<ILogger<FSharpScriptExecutor>>() });
            _service = new MetascriptTreeOfThoughtService(_loggerMock.Object, _scriptExecutorMock.Object);
        }

        [Fact]
        public async Task GenerateMetascriptAsync_ShouldReturnGeneratedMetascript()
        {
            // Arrange
            var templateContent = "This is a template with ${placeholder}";
            var templateValues = new Dictionary<string, string>
            {
                { "placeholder", "value" }
            };
            
            var expectedMetascript = "This is a template with value";
            var expectedThoughtTreeJson = "{ \"thought\": \"Generate Metascript from Template\", \"evaluation\": null, \"pruned\": false, \"metadata\": {}, \"children\": [] }";
            
            _scriptExecutorMock
                .Setup(x => x.ExecuteScriptAsync(It.IsAny<string>()))
                .ReturnsAsync(new ScriptExecutionResult
                {
                    Success = true,
                    Output = $"(\"{expectedMetascript}\", {expectedThoughtTreeJson})",
                    Errors = new List<string>()
                });

            // Act
            var (metascript, thoughtTreeJson) = await _service.GenerateMetascriptAsync(templateContent, templateValues);

            // Assert
            Assert.Equal(expectedMetascript, metascript);
            Assert.Equal(expectedThoughtTreeJson, thoughtTreeJson);
            _scriptExecutorMock.Verify(x => x.ExecuteScriptAsync(It.IsAny<string>()), Times.Once);
        }

        [Fact]
        public async Task ValidateMetascriptAsync_ShouldReturnValidationResults()
        {
            // Arrange
            var metascript = "VARIABLE test { value: \"test\" }";
            
            var expectedIsValid = true;
            var expectedErrors = new List<string>();
            var expectedWarnings = new List<string>();
            var expectedThoughtTreeJson = "{ \"thought\": \"Validate Metascript\", \"evaluation\": null, \"pruned\": false, \"metadata\": {}, \"children\": [] }";
            
            _scriptExecutorMock
                .Setup(x => x.ExecuteScriptAsync(It.IsAny<string>()))
                .ReturnsAsync(new ScriptExecutionResult
                {
                    Success = true,
                    Output = $"({expectedIsValid.ToString().ToLower()}, [], [], {expectedThoughtTreeJson})",
                    Errors = new List<string>()
                });

            // Act
            var (isValid, errors, warnings, thoughtTreeJson) = await _service.ValidateMetascriptAsync(metascript);

            // Assert
            Assert.Equal(expectedIsValid, isValid);
            Assert.Equal(expectedErrors, errors);
            Assert.Equal(expectedWarnings, warnings);
            Assert.Equal(expectedThoughtTreeJson, thoughtTreeJson);
            _scriptExecutorMock.Verify(x => x.ExecuteScriptAsync(It.IsAny<string>()), Times.Once);
        }

        [Fact]
        public async Task ExecuteMetascriptAsync_ShouldReturnExecutionResults()
        {
            // Arrange
            var metascript = "VARIABLE test { value: \"test\" }";
            
            var expectedOutput = "Metascript executed successfully";
            var expectedSuccess = true;
            var expectedReport = "Execution report";
            var expectedThoughtTreeJson = "{ \"thought\": \"Plan and Execute Metascript\", \"evaluation\": null, \"pruned\": false, \"metadata\": {}, \"children\": [] }";
            
            _scriptExecutorMock
                .Setup(x => x.ExecuteScriptAsync(It.IsAny<string>()))
                .ReturnsAsync(new ScriptExecutionResult
                {
                    Success = true,
                    Output = $"(\"{expectedOutput}\", {expectedSuccess.ToString().ToLower()}, \"{expectedReport}\", {expectedThoughtTreeJson})",
                    Errors = new List<string>()
                });

            // Act
            var (output, success, report, thoughtTreeJson) = await _service.ExecuteMetascriptAsync(metascript);

            // Assert
            Assert.Equal(expectedOutput, output);
            Assert.Equal(expectedSuccess, success);
            Assert.Equal(expectedReport, report);
            Assert.Equal(expectedThoughtTreeJson, thoughtTreeJson);
            _scriptExecutorMock.Verify(x => x.ExecuteScriptAsync(It.IsAny<string>()), Times.Once);
        }

        [Fact]
        public async Task AnalyzeResultsAsync_ShouldReturnAnalysisResults()
        {
            // Arrange
            var output = "Metascript executed successfully";
            var executionTime = 1000;
            var peakMemoryUsage = 100;
            var errorCount = 0;
            
            var expectedSuccess = true;
            var expectedErrors = new List<string>();
            var expectedWarnings = new List<string>();
            var expectedImpact = "The metascript execution had a positive impact with excellent performance";
            var expectedRecommendations = new List<string>();
            var expectedThoughtTreeJson = "{ \"thought\": \"Analyze Metascript Results\", \"evaluation\": null, \"pruned\": false, \"metadata\": {}, \"children\": [] }";
            
            _scriptExecutorMock
                .Setup(x => x.ExecuteScriptAsync(It.IsAny<string>()))
                .ReturnsAsync(new ScriptExecutionResult
                {
                    Success = true,
                    Output = $"({expectedSuccess.ToString().ToLower()}, [], [], \"{expectedImpact}\", [], {expectedThoughtTreeJson})",
                    Errors = new List<string>()
                });

            // Act
            var (success, errors, warnings, impact, recommendations, thoughtTreeJson) = 
                await _service.AnalyzeResultsAsync(output, executionTime, peakMemoryUsage, errorCount);

            // Assert
            Assert.Equal(expectedSuccess, success);
            Assert.Equal(expectedErrors, errors);
            Assert.Equal(expectedWarnings, warnings);
            Assert.Equal(expectedImpact, impact);
            Assert.Equal(expectedRecommendations, recommendations);
            Assert.Equal(expectedThoughtTreeJson, thoughtTreeJson);
            _scriptExecutorMock.Verify(x => x.ExecuteScriptAsync(It.IsAny<string>()), Times.Once);
        }

        [Fact]
        public async Task IntegrateResultsAsync_ShouldReturnIntegrationResults()
        {
            // Arrange
            var output = "Metascript executed successfully";
            var success = true;
            var recommendations = new List<string>();
            
            var expectedIntegrated = true;
            var expectedMessage = "Results integrated successfully";
            var expectedThoughtTreeJson = "{ \"thought\": \"Integrate Metascript Results\", \"evaluation\": null, \"pruned\": false, \"metadata\": {}, \"children\": [] }";
            
            _scriptExecutorMock
                .Setup(x => x.ExecuteScriptAsync(It.IsAny<string>()))
                .ReturnsAsync(new ScriptExecutionResult
                {
                    Success = true,
                    Output = $"({expectedIntegrated.ToString().ToLower()}, \"{expectedMessage}\", {expectedThoughtTreeJson})",
                    Errors = new List<string>()
                });

            // Act
            var (integrated, message, thoughtTreeJson) = 
                await _service.IntegrateResultsAsync(output, success, recommendations);

            // Assert
            Assert.Equal(expectedIntegrated, integrated);
            Assert.Equal(expectedMessage, message);
            Assert.Equal(expectedThoughtTreeJson, thoughtTreeJson);
            _scriptExecutorMock.Verify(x => x.ExecuteScriptAsync(It.IsAny<string>()), Times.Once);
        }
    }
}
