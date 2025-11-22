using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using TarsEngine.Services.Compilation;
using TarsEngine.Services.TreeOfThought;
using Xunit;

namespace TarsEngine.Tests.TreeOfThought
{
    public class SimpleTreeOfThoughtTests
    {
        private readonly Mock<ILogger<SimpleTreeOfThoughtService>> _loggerMock;
        private readonly Mock<FSharpScriptExecutor> _scriptExecutorMock;
        private readonly SimpleTreeOfThoughtService _service;

        public SimpleTreeOfThoughtTests()
        {
            _loggerMock = new Mock<ILogger<SimpleTreeOfThoughtService>>();
            _scriptExecutorMock = new Mock<FSharpScriptExecutor>(MockBehavior.Loose, new object[] { Mock.Of<ILogger<FSharpScriptExecutor>>() });
            _service = new SimpleTreeOfThoughtService(_loggerMock.Object, _scriptExecutorMock.Object);
        }

        [Fact]
        public async Task AnalyzeCodeAsync_ShouldReturnAnalysisResult()
        {
            // Arrange
            var code = "public class Test { }";
            var expectedResult = "{ \"thought\": \"Code Analysis\", \"score\": 0.0, \"pruned\": false, \"children\": [] }";
            
            _scriptExecutorMock
                .Setup(x => x.ExecuteScriptAsync(It.IsAny<string>()))
                .ReturnsAsync(new ScriptExecutionResult
                {
                    Success = true,
                    Output = expectedResult,
                    Errors = new System.Collections.Generic.List<string>()
                });

            // Act
            var result = await _service.AnalyzeCodeAsync(code);

            // Assert
            Assert.Equal(expectedResult, result);
            _scriptExecutorMock.Verify(x => x.ExecuteScriptAsync(It.IsAny<string>()), Times.Once);
        }

        [Fact]
        public async Task GenerateFixesAsync_ShouldReturnFixGenerationResult()
        {
            // Arrange
            var issue = "Unused variable";
            var expectedResult = "{ \"thought\": \"Fix Generation\", \"score\": 0.0, \"pruned\": false, \"children\": [] }";
            
            _scriptExecutorMock
                .Setup(x => x.ExecuteScriptAsync(It.IsAny<string>()))
                .ReturnsAsync(new ScriptExecutionResult
                {
                    Success = true,
                    Output = expectedResult,
                    Errors = new System.Collections.Generic.List<string>()
                });

            // Act
            var result = await _service.GenerateFixesAsync(issue);

            // Assert
            Assert.Equal(expectedResult, result);
            _scriptExecutorMock.Verify(x => x.ExecuteScriptAsync(It.IsAny<string>()), Times.Once);
        }

        [Fact]
        public async Task ApplyFixAsync_ShouldReturnFixApplicationResult()
        {
            // Arrange
            var fix = "Remove unused variable";
            var expectedResult = "{ \"thought\": \"Fix Application\", \"score\": 0.0, \"pruned\": false, \"children\": [] }";
            
            _scriptExecutorMock
                .Setup(x => x.ExecuteScriptAsync(It.IsAny<string>()))
                .ReturnsAsync(new ScriptExecutionResult
                {
                    Success = true,
                    Output = expectedResult,
                    Errors = new System.Collections.Generic.List<string>()
                });

            // Act
            var result = await _service.ApplyFixAsync(fix);

            // Assert
            Assert.Equal(expectedResult, result);
            _scriptExecutorMock.Verify(x => x.ExecuteScriptAsync(It.IsAny<string>()), Times.Once);
        }

        [Fact]
        public async Task SelectBestApproachAsync_ShouldReturnBestApproach()
        {
            // Arrange
            var thoughtTreeJson = @"{
                ""thought"": ""Root"",
                ""score"": 0.5,
                ""pruned"": false,
                ""children"": [
                    {
                        ""thought"": ""Child 1"",
                        ""score"": 0.7,
                        ""pruned"": false,
                        ""children"": []
                    },
                    {
                        ""thought"": ""Child 2"",
                        ""score"": 0.9,
                        ""pruned"": false,
                        ""children"": []
                    },
                    {
                        ""thought"": ""Child 3"",
                        ""score"": 0.6,
                        ""pruned"": false,
                        ""children"": []
                    }
                ]
            }";
            
            var expectedResult = @"{
                ""thought"": ""Child 2"",
                ""score"": 0.9,
                ""pruned"": false,
                ""children"": []
            }";
            
            _scriptExecutorMock
                .Setup(x => x.ExecuteScriptAsync(It.IsAny<string>()))
                .ReturnsAsync(new ScriptExecutionResult
                {
                    Success = true,
                    Output = expectedResult,
                    Errors = new System.Collections.Generic.List<string>()
                });

            // Act
            var result = await _service.SelectBestApproachAsync(thoughtTreeJson);

            // Assert
            Assert.Equal(expectedResult, result);
            _scriptExecutorMock.Verify(x => x.ExecuteScriptAsync(It.IsAny<string>()), Times.Once);
        }
    }
}
