using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Moq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using TarsCli.Services;
using Xunit;

namespace TarsCli.Tests.Services
{
    public class TutorialOrganizerServiceTests
    {
        private readonly Mock<ILogger<TutorialOrganizerService>> _loggerMock;
        private readonly Mock<IConfiguration> _configMock;
        private readonly string _testDirectory;
        private readonly string _testCatalogPath;
        private readonly TutorialOrganizerService _tutorialOrganizerService;

        public TutorialOrganizerServiceTests()
        {
            _loggerMock = new Mock<ILogger<TutorialOrganizerService>>();
            _configMock = new Mock<IConfiguration>();
            
            // Create a temporary directory for testing
            _testDirectory = Path.Combine(Path.GetTempPath(), "TarsCliTests", "Tutorials", Guid.NewGuid().ToString());
            Directory.CreateDirectory(_testDirectory);
            _testCatalogPath = Path.Combine(_testDirectory, "catalog.json");
            
            // Setup configuration
            _configMock.Setup(c => c["Paths:Tutorials"]).Returns(_testDirectory);
            _configMock.Setup(c => c["Paths:TutorialCatalog"]).Returns(_testCatalogPath);
            
            // Create the service
            _tutorialOrganizerService = new TutorialOrganizerService(
                _loggerMock.Object,
                _configMock.Object);
        }

        [Fact]
        public async Task AddTutorial_ShouldCreateAndSaveTutorial()
        {
            // Arrange
            var title = "Getting Started with C#";
            var description = "A beginner's guide to C# programming";
            var content = "# Getting Started with C#\n\nThis tutorial will guide you through the basics of C# programming.";
            var category = "Programming";
            var difficultyLevel = DifficultyLevel.Beginner;
            var tags = new List<string> { "C#", "Programming", "Beginner" };
            var prerequisites = new List<string> { "Basic programming knowledge" };
            
            // Act
            var result = await _tutorialOrganizerService.AddTutorial(title, description, content, category, difficultyLevel, tags, prerequisites);
            
            // Assert
            Assert.NotNull(result);
            Assert.Equal(title, result.Title);
            Assert.Equal(description, result.Description);
            Assert.Equal(category, result.Category);
            Assert.Equal(difficultyLevel, result.DifficultyLevel);
            Assert.Equal(tags, result.Tags);
            Assert.Equal(prerequisites, result.Prerequisites);
            
            // Verify the tutorial content was saved
            var tutorialFilePath = Path.Combine(_testDirectory, $"{result.Id}.md");
            Assert.True(File.Exists(tutorialFilePath));
            var savedContent = await File.ReadAllTextAsync(tutorialFilePath);
            Assert.Equal(content, savedContent);
            
            // Verify the catalog was updated
            Assert.True(File.Exists(_testCatalogPath));
            var catalogJson = await File.ReadAllTextAsync(_testCatalogPath);
            var catalog = JsonSerializer.Deserialize<TutorialCatalog>(catalogJson, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            Assert.NotNull(catalog);
            Assert.Single(catalog.Categories);
            Assert.Equal(category, catalog.Categories[0].Name);
            Assert.Single(catalog.Categories[0].Tutorials);
            Assert.Equal(result.Id, catalog.Categories[0].Tutorials[0].Id);
        }

        [Fact]
        public async Task GetTutorials_ShouldReturnAllTutorials()
        {
            // Arrange
            // Add a few test tutorials
            var tutorial1 = await _tutorialOrganizerService.AddTutorial(
                "Tutorial 1",
                "Description 1",
                "Content 1",
                "Category 1",
                DifficultyLevel.Beginner,
                new List<string> { "Tag1" },
                new List<string>());
            
            var tutorial2 = await _tutorialOrganizerService.AddTutorial(
                "Tutorial 2",
                "Description 2",
                "Content 2",
                "Category 2",
                DifficultyLevel.Advanced,
                new List<string> { "Tag2" },
                new List<string>());
            
            // Act
            var result = await _tutorialOrganizerService.GetTutorials();
            
            // Assert
            Assert.NotNull(result);
            Assert.Equal(2, result.Count);
            Assert.Contains(result, t => t.Id == tutorial1.Id);
            Assert.Contains(result, t => t.Id == tutorial2.Id);
        }

        [Fact]
        public async Task GetTutorials_WithFilters_ShouldReturnFilteredTutorials()
        {
            // Arrange
            // Add a few test tutorials
            var tutorial1 = await _tutorialOrganizerService.AddTutorial(
                "C# Basics",
                "Learn C# basics",
                "C# basics content",
                "Programming",
                DifficultyLevel.Beginner,
                new List<string> { "C#", "Beginner" },
                new List<string>());
            
            var tutorial2 = await _tutorialOrganizerService.AddTutorial(
                "Advanced C#",
                "Advanced C# concepts",
                "Advanced C# content",
                "Programming",
                DifficultyLevel.Advanced,
                new List<string> { "C#", "Advanced" },
                new List<string>());
            
            var tutorial3 = await _tutorialOrganizerService.AddTutorial(
                "Python Basics",
                "Learn Python basics",
                "Python basics content",
                "Programming",
                DifficultyLevel.Beginner,
                new List<string> { "Python", "Beginner" },
                new List<string>());
            
            // Act - Filter by category
            var categoryResult = await _tutorialOrganizerService.GetTutorials("Programming");
            
            // Assert
            Assert.NotNull(categoryResult);
            Assert.Equal(3, categoryResult.Count);
            
            // Act - Filter by difficulty
            var difficultyResult = await _tutorialOrganizerService.GetTutorials(null, "Advanced");
            
            // Assert
            Assert.NotNull(difficultyResult);
            Assert.Single(difficultyResult);
            Assert.Equal(tutorial2.Id, difficultyResult[0].Id);
            
            // Act - Filter by tag
            var tagResult = await _tutorialOrganizerService.GetTutorials(null, null, "Python");
            
            // Assert
            Assert.NotNull(tagResult);
            Assert.Single(tagResult);
            Assert.Equal(tutorial3.Id, tagResult[0].Id);
            
            // Act - Combined filters
            var combinedResult = await _tutorialOrganizerService.GetTutorials("Programming", "Beginner", "C#");
            
            // Assert
            Assert.NotNull(combinedResult);
            Assert.Single(combinedResult);
            Assert.Equal(tutorial1.Id, combinedResult[0].Id);
        }

        [Fact]
        public async Task GetTutorial_ShouldReturnSpecificTutorial()
        {
            // Arrange
            var tutorial = await _tutorialOrganizerService.AddTutorial(
                "Test Tutorial",
                "Test Description",
                "Test Content",
                "Test Category",
                DifficultyLevel.Intermediate,
                new List<string> { "Test" },
                new List<string> { "Test Prerequisite" });
            
            // Act
            var result = await _tutorialOrganizerService.GetTutorial(tutorial.Id);
            
            // Assert
            Assert.NotNull(result);
            Assert.Equal(tutorial.Id, result.Id);
            Assert.Equal("Test Tutorial", result.Title);
            Assert.Equal("Test Description", result.Description);
            Assert.Equal("Test Category", result.Category);
            Assert.Equal(DifficultyLevel.Intermediate, result.DifficultyLevel);
            Assert.Single(result.Tags);
            Assert.Equal("Test", result.Tags[0]);
            Assert.Single(result.Prerequisites);
            Assert.Equal("Test Prerequisite", result.Prerequisites[0]);
            
            // Verify content retrieval
            var content = await _tutorialOrganizerService.GetTutorialContent(tutorial.Id);
            Assert.Equal("Test Content", content);
        }

        [Fact]
        public async Task CategorizeTutorials_ShouldUpdateTutorialCategories()
        {
            // Arrange
            var tutorial1 = await _tutorialOrganizerService.AddTutorial(
                "Tutorial 1",
                "Description 1",
                "Content 1",
                "Old Category",
                DifficultyLevel.Beginner,
                new List<string>(),
                new List<string>());
            
            var tutorial2 = await _tutorialOrganizerService.AddTutorial(
                "Tutorial 2",
                "Description 2",
                "Content 2",
                "Old Category",
                DifficultyLevel.Intermediate,
                new List<string>(),
                new List<string>());
            
            // Act
            var success = await _tutorialOrganizerService.CategorizeTutorials(
                new List<string> { tutorial1.Id, tutorial2.Id },
                "New Category");
            
            // Assert
            Assert.True(success);
            
            // Verify the tutorials were recategorized
            var updatedTutorial1 = await _tutorialOrganizerService.GetTutorial(tutorial1.Id);
            var updatedTutorial2 = await _tutorialOrganizerService.GetTutorial(tutorial2.Id);
            
            Assert.Equal("New Category", updatedTutorial1.Category);
            Assert.Equal("New Category", updatedTutorial2.Category);
            
            // Verify the catalog was updated
            var catalogJson = await File.ReadAllTextAsync(_testCatalogPath);
            var catalog = JsonSerializer.Deserialize<TutorialCatalog>(catalogJson, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            
            Assert.NotNull(catalog);
            Assert.Equal(2, catalog.Categories.Count);
            
            var oldCategory = catalog.Categories.Find(c => c.Name == "Old Category");
            var newCategory = catalog.Categories.Find(c => c.Name == "New Category");
            
            Assert.NotNull(oldCategory);
            Assert.NotNull(newCategory);
            Assert.Empty(oldCategory.Tutorials);
            Assert.Equal(2, newCategory.Tutorials.Count);
        }

        public void Dispose()
        {
            // Clean up the test directory
            if (Directory.Exists(_testDirectory))
            {
                Directory.Delete(_testDirectory, true);
            }
        }
    }
}
