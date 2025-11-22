using Microsoft.Extensions.Logging;
using Moq;
using Moq.AutoMock;
using TarsEngine.Services;
using TarsEngine.Services.Interfaces;
using ProgrammingLanguage = TarsEngine.Services.ProgrammingLanguage;
using Xunit;
using ProjectAnalysisResult = TarsEngine.Models.ProjectAnalysisResult;

namespace TarsEngine.Tests.Services;

public class SelfImprovementServiceTests
{
    private readonly AutoMocker _mocker;
    private readonly ISelfImprovementService _service;

    public SelfImprovementServiceTests()
    {
        _mocker = new AutoMocker();
        _service = _mocker.CreateInstance<SelfImprovementService>();

        // Setup default mocks
        _mocker.GetMock<ILogger<SelfImprovementService>>();
    }

    [Fact]
    public async Task AnalyzeFileForImprovementsAsync_FileNotFound_ThrowsFileNotFoundException()
    {
        // Arrange
        var filePath = "nonexistent.cs";
        var projectPath = "project";

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _service.AnalyzeFileForImprovementsAsync(filePath, projectPath));
    }

    [Fact]
    public async Task AnalyzeFileForImprovementsAsync_ValidFile_ReturnsSuggestions()
    {
        // Arrange
        var filePath = Path.GetTempFileName();
        var projectPath = "project";
        var fileContent = "public class Test { }";

        try
        {
            File.WriteAllText(filePath, fileContent);

            var codeAnalysisResult = new CodeAnalysisResult
            {
                Success = true,
                FilePath = filePath,
                ErrorMessage = string.Empty,
                Language = ProgrammingLanguage.CSharp
            };

            var projectAnalysisResult = new ProjectAnalysisResult
            {
                ProjectPath = projectPath,
                ProjectName = "Test"
            };

            _mocker.GetMock<ICodeAnalysisService>()
                .Setup(x => x.AnalyzeFileAsync(filePath))
                .ReturnsAsync(codeAnalysisResult);

            _mocker.GetMock<IProjectAnalysisService>()
                .Setup(x => x.AnalyzeProjectAsync(projectPath))
                .ReturnsAsync(projectAnalysisResult);

            var llmResponse = @"
SUGGESTION:
Line: 1
Issue: Missing namespace declaration
Improvement: Add a namespace declaration to organize the code
Code: namespace Test { public class Test { } }
";

            _mocker.GetMock<ILlmService>()
                .Setup(x => x.GetCompletionAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<double>(), It.IsAny<int>()))
                .ReturnsAsync(llmResponse);

            // Act
            var result = await _service.AnalyzeFileForImprovementsAsync(filePath, projectPath);

            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            Assert.Equal(filePath, result[0].FilePath);
            Assert.Equal(1, result[0].LineNumber);
            Assert.Equal("Missing namespace declaration", result[0].Issue);
            Assert.Equal("Add a namespace declaration to organize the code", result[0].Improvement);
            Assert.Equal("namespace Test { public class Test { } }", result[0].ReplacementCode);
        }
        finally
        {
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }
        }
    }

    [Fact]
    public async Task ApplyImprovementsAsync_FileNotFound_ThrowsFileNotFoundException()
    {
        // Arrange
        var filePath = "nonexistent.cs";
        var suggestions = new List<ImprovementSuggestion>();

        // Act & Assert
        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            _service.ApplyImprovementsAsync(filePath, suggestions));
    }

    [Fact]
    public async Task ApplyImprovementsAsync_ValidFile_AppliesSuggestions()
    {
        // Arrange
        var filePath = Path.GetTempFileName();
        var originalContent = "public class Test { }";
        var expectedContent = "namespace Test { public class Test { } }";

        try
        {
            File.WriteAllText(filePath, originalContent);

            var suggestions = new List<ImprovementSuggestion>
            {
                new()
                {
                    FilePath = filePath,
                    LineNumber = 1,
                    Issue = "Missing namespace declaration",
                    Improvement = "Add a namespace declaration to organize the code",
                    ReplacementCode = "namespace Test { public class Test { } }"
                }
            };

            // Act
            var result = await _service.ApplyImprovementsAsync(filePath, suggestions);

            // Assert
            Assert.Equal(filePath, result);
            var actualContent = File.ReadAllText(filePath);
            Assert.Equal(expectedContent, actualContent);
        }
        finally
        {
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }
        }
    }
}