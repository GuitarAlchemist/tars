using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Moq;
using TarsCli.Models;
using TarsCli.Services;
using Xunit;


namespace TarsCli.Tests.Services
{
    public class AutoImprovementServiceTests
    {
        private readonly Mock<ILogger<AutoImprovementService>> _loggerMock;
        private readonly Mock<IConfiguration> _configMock;
        private readonly Mock<SelfImprovementService> _selfImprovementServiceMock;
        private readonly Mock<SlackIntegrationService> _slackServiceMock;
        private readonly Mock<TemplateService> _templateServiceMock;
        private readonly Mock<OllamaService> _ollamaServiceMock;
        private readonly string _testDir;

        public AutoImprovementServiceTests()
        {
            _loggerMock = new Mock<ILogger<AutoImprovementService>>();
            _configMock = new Mock<IConfiguration>();
            _ollamaServiceMock = new Mock<OllamaService>(
                Mock.Of<ILogger<OllamaService>>(),
                Mock.Of<IConfiguration>());

            _selfImprovementServiceMock = new Mock<SelfImprovementService>(
                Mock.Of<ILogger<SelfImprovementService>>(),
                _ollamaServiceMock.Object);
            _slackServiceMock = new Mock<SlackIntegrationService>(
                Mock.Of<ILogger<SlackIntegrationService>>(),
                Mock.Of<IConfiguration>(),
                Mock.Of<SecretsService>());
            _templateServiceMock = new Mock<TemplateService>(
                Mock.Of<ILogger<TemplateService>>());

            // Set up test directory
            _testDir = Path.Combine(Path.GetTempPath(), "TarsCliTests", Guid.NewGuid().ToString());
            Directory.CreateDirectory(_testDir);

            // Set up configuration
            _configMock.Setup(c => c["Tars:DocsDir"]).Returns(Path.Combine(_testDir, "docs"));
            _configMock.Setup(c => c["Tars:ChatsDir"]).Returns(Path.Combine(_testDir, "chats"));
            _configMock.Setup(c => c["Tars:ProjectRoot"]).Returns(_testDir);
        }

        [Fact]
        public void CalculateFilePriorityScore_ShouldPrioritizeCodeFiles()
        {
            // Arrange
            var service = CreateService();
            var testFiles = CreateTestFiles();

            // Get the private method using reflection
            var calculateScoreMethod = typeof(AutoImprovementService).GetMethod(
                "CalculateFilePriorityScore",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            // Act
            var scores = new Dictionary<string, FilePriorityScore>();
            foreach (var file in testFiles)
            {
                var score = (FilePriorityScore)calculateScoreMethod.Invoke(service, new object[] { file });
                scores[file] = score;
            }

            // Assert
            var csFile = testFiles.First(f => f.EndsWith(".cs"));
            var fsFile = testFiles.First(f => f.EndsWith(".fs"));
            var mdFile = testFiles.First(f => f.EndsWith(".md"));

            Assert.True(scores[fsFile].BaseScore > scores[csFile].BaseScore, "F# files should have higher base score than C# files");
            Assert.True(scores[csFile].BaseScore > scores[mdFile].BaseScore, "C# files should have higher base score than markdown files");
        }

        [Fact]
        public void CalculateFilePriorityScore_ShouldConsiderFileContent()
        {
            // Arrange
            var service = CreateService();

            // Create test files with different content
            var cleanFile = Path.Combine(_testDir, "clean.cs");
            File.WriteAllText(cleanFile, "public class CleanClass { public void CleanMethod() { } }");

            var todoFile = Path.Combine(_testDir, "todo.cs");
            File.WriteAllText(todoFile, "public class TodoClass { public void TodoMethod() { // TODO: Fix this } }");

            var complexFile = Path.Combine(_testDir, "complex.cs");
            File.WriteAllText(complexFile, @"
public class ComplexClass {
    public void ComplexMethod() {
        if (condition) {
            if (anotherCondition) {
                if (yetAnotherCondition) {
                    // Nested if statements
                }
            }
        }
    }
}");

            // Get the private method using reflection
            var calculateScoreMethod = typeof(AutoImprovementService).GetMethod(
                "CalculateFilePriorityScore",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            // Act
            var cleanScore = (FilePriorityScore)calculateScoreMethod.Invoke(service, new object[] { cleanFile });
            var todoScore = (FilePriorityScore)calculateScoreMethod.Invoke(service, new object[] { todoFile });
            var complexScore = (FilePriorityScore)calculateScoreMethod.Invoke(service, new object[] { complexFile });

            // Assert
            Assert.True(todoScore.ContentScore > cleanScore.ContentScore, "Files with TODOs should have higher content score");
            Assert.True(complexScore.ComplexityScore > cleanScore.ComplexityScore, "Complex files should have higher complexity score");
        }

        [Fact]
        public void PrioritizeFiles_ShouldOrderFilesByTotalScore()
        {
            // Arrange
            var service = CreateService();
            var testFiles = CreateTestFiles();

            // Create a state with file priority scores
            var state = new TarsCli.Models.AutoImprovementState();

            // Get the private method using reflection
            var calculateScoreMethod = typeof(AutoImprovementService).GetMethod(
                "CalculateFilePriorityScore",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            // Calculate scores for test files
            foreach (var file in testFiles)
            {
                var score = (FilePriorityScore)calculateScoreMethod.Invoke(service, new object[] { file });
                state.FilePriorityScores[file] = score;
                state.PendingFiles.Add(file);
            }

            // Set the state using reflection
            var stateField = typeof(AutoImprovementService).GetField(
                "_state",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            stateField.SetValue(service, state);

            // Get the private method using reflection
            var prioritizeFilesMethod = typeof(AutoImprovementService).GetMethod(
                "PrioritizeFiles",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            // Act
            prioritizeFilesMethod.Invoke(service, null);

            // Assert
            var pendingFiles = state.PendingFiles;

            // The files should be ordered by total score (descending)
            for (int i = 0; i < pendingFiles.Count - 1; i++)
            {
                var currentScore = state.FilePriorityScores[pendingFiles[i]].TotalScore;
                var nextScore = state.FilePriorityScores[pendingFiles[i + 1]].TotalScore;
                Assert.True(currentScore >= nextScore, "Files should be ordered by total score (descending)");
            }
        }

        private AutoImprovementService CreateService()
        {
            return new AutoImprovementService(
                _loggerMock.Object,
                _configMock.Object,
                _selfImprovementServiceMock.Object,
                _ollamaServiceMock.Object,
                _slackServiceMock.Object);
        }

        private List<string> CreateTestFiles()
        {
            // Create test directories
            Directory.CreateDirectory(Path.Combine(_testDir, "docs"));
            Directory.CreateDirectory(Path.Combine(_testDir, "chats"));
            Directory.CreateDirectory(Path.Combine(_testDir, "TarsCli"));
            Directory.CreateDirectory(Path.Combine(_testDir, "TarsEngine"));

            // Create test files
            var files = new List<string>();

            // Markdown files
            var mdFile1 = Path.Combine(_testDir, "docs", "test1.md");
            File.WriteAllText(mdFile1, "# Test Document\n\nThis is a test document.");
            files.Add(mdFile1);

            var mdFile2 = Path.Combine(_testDir, "chats", "test2.md");
            File.WriteAllText(mdFile2, "# Test Chat\n\nThis is a test chat.");
            files.Add(mdFile2);

            // C# files
            var csFile1 = Path.Combine(_testDir, "TarsCli", "Test1.cs");
            File.WriteAllText(csFile1, "public class Test1 { public void Method1() { } }");
            files.Add(csFile1);

            var csFile2 = Path.Combine(_testDir, "TarsCli", "Test2.cs");
            File.WriteAllText(csFile2, "public class Test2 { public void Method2() { } }");
            files.Add(csFile2);

            // F# files
            var fsFile1 = Path.Combine(_testDir, "TarsEngine", "Test1.fs");
            File.WriteAllText(fsFile1, "module Test1\n\nlet test1 = 1");
            files.Add(fsFile1);

            var fsFile2 = Path.Combine(_testDir, "TarsEngine", "Test2.fs");
            File.WriteAllText(fsFile2, "module Test2\n\nlet test2 = 2");
            files.Add(fsFile2);

            return files;
        }
    }
}
