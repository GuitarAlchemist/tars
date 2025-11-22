using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Moq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using TarsCli.Services;
using Xunit;

namespace TarsCli.Tests.Services
{
    public class CourseGeneratorServiceTests
    {
        private readonly Mock<ILogger<CourseGeneratorService>> _loggerMock;
        private readonly Mock<IConfiguration> _configMock;
        private readonly Mock<OllamaService> _ollamaServiceMock;
        private readonly string _testDirectory;
        private readonly CourseGeneratorService _courseGeneratorService;

        public CourseGeneratorServiceTests()
        {
            _loggerMock = new Mock<ILogger<CourseGeneratorService>>();
            _configMock = new Mock<IConfiguration>();
            _ollamaServiceMock = new Mock<OllamaService>();
            
            // Create a temporary directory for testing
            _testDirectory = Path.Combine(Path.GetTempPath(), "TarsCliTests", "Courses", Guid.NewGuid().ToString());
            Directory.CreateDirectory(_testDirectory);
            
            // Setup configuration
            _configMock.Setup(c => c["Paths:Courses"]).Returns(_testDirectory);
            _configMock.Setup(c => c["Ollama:DefaultModel"]).Returns("llama3");
            
            // Create the service
            _courseGeneratorService = new CourseGeneratorService(
                _loggerMock.Object,
                _configMock.Object,
                _ollamaServiceMock.Object);
        }

        [Fact]
        public async Task GenerateCourse_ShouldCreateAndSaveCourse()
        {
            // Arrange
            var title = "Introduction to C#";
            var description = "A comprehensive course on C# programming";
            var topic = "C# Programming";
            var difficultyLevel = DifficultyLevel.Intermediate;
            var estimatedHours = 20;
            var targetAudience = new List<string> { "Beginners", "Students" };
            var model = "llama3";
            
            var mockResponse = @"{
                ""overview"": ""This course provides a comprehensive introduction to C# programming."",
                ""learningObjectives"": [""Understand C# syntax"", ""Build console applications""],
                ""lessons"": [
                    {
                        ""title"": ""C# Basics"",
                        ""objectives"": [""Learn C# syntax"", ""Understand variables and data types""],
                        ""estimatedMinutes"": 60,
                        ""content"": ""C# is a modern, object-oriented programming language..."",
                        ""exercises"": [
                            {
                                ""title"": ""Hello World"",
                                ""description"": ""Create a simple Hello World application"",
                                ""difficulty"": ""Easy""
                            }
                        ],
                        ""quizQuestions"": [
                            {
                                ""question"": ""What is C#?"",
                                ""options"": [""A programming language"", ""A database"", ""A web framework"", ""An operating system""],
                                ""correctAnswerIndex"": 0,
                                ""explanation"": ""C# is a modern, object-oriented programming language developed by Microsoft.""
                            }
                        ]
                    }
                ],
                ""finalAssessment"": {
                    ""title"": ""Build a Console Application"",
                    ""description"": ""Create a fully functional console application"",
                    ""criteria"": [""Proper syntax"", ""Error handling"", ""User input validation""],
                    ""estimatedHours"": 3
                },
                ""additionalResources"": [
                    {
                        ""title"": ""Microsoft C# Documentation"",
                        ""type"": ""Documentation"",
                        ""url"": ""https://docs.microsoft.com/en-us/dotnet/csharp/"",
                        ""description"": ""Official C# documentation from Microsoft""
                    }
                ]
            }";
            
            _ollamaServiceMock.Setup(o => o.GenerateCompletion(It.IsAny<string>(), model))
                .ReturnsAsync(mockResponse);
            
            // Act
            var result = await _courseGeneratorService.GenerateCourse(title, description, topic, difficultyLevel, estimatedHours, targetAudience, model);
            
            // Assert
            Assert.NotNull(result);
            Assert.Equal(title, result.Title);
            Assert.Equal(description, result.Description);
            Assert.Equal(topic, result.Topic);
            Assert.Equal(difficultyLevel, result.DifficultyLevel);
            Assert.NotNull(result.Content);
            Assert.Equal("This course provides a comprehensive introduction to C# programming.", result.Content.Overview);
            Assert.Equal(2, result.Content.LearningObjectives.Count);
            Assert.Single(result.Content.Lessons);
            Assert.NotNull(result.Content.FinalAssessment);
            Assert.Single(result.Content.AdditionalResources);
            
            // Verify the file was saved
            var filePath = Path.Combine(_testDirectory, $"{result.Id}.json");
            Assert.True(File.Exists(filePath));
        }

        [Fact]
        public async Task GetCourses_ShouldReturnAllCourses()
        {
            // Arrange
            // Create a few test courses
            var course1 = new Course
            {
                Id = Guid.NewGuid().ToString(),
                Title = "Course 1",
                Description = "Description 1",
                Topic = "Topic 1",
                DifficultyLevel = DifficultyLevel.Beginner,
                CreatedDate = DateTime.UtcNow,
                LastModifiedDate = DateTime.UtcNow,
                Content = new CourseContent
                {
                    Overview = "Overview 1",
                    LearningObjectives = new List<string>(),
                    Lessons = new List<Lesson>(),
                    FinalAssessment = new FinalAssessment(),
                    AdditionalResources = new List<AdditionalResource>()
                }
            };
            
            var course2 = new Course
            {
                Id = Guid.NewGuid().ToString(),
                Title = "Course 2",
                Description = "Description 2",
                Topic = "Topic 2",
                DifficultyLevel = DifficultyLevel.Advanced,
                CreatedDate = DateTime.UtcNow,
                LastModifiedDate = DateTime.UtcNow,
                Content = new CourseContent
                {
                    Overview = "Overview 2",
                    LearningObjectives = new List<string>(),
                    Lessons = new List<Lesson>(),
                    FinalAssessment = new FinalAssessment(),
                    AdditionalResources = new List<AdditionalResource>()
                }
            };
            
            // Save the courses
            await _courseGeneratorService.SaveCourse(course1);
            await _courseGeneratorService.SaveCourse(course2);
            
            // Act
            var result = await _courseGeneratorService.GetCourses();
            
            // Assert
            Assert.NotNull(result);
            Assert.Equal(2, result.Count);
            Assert.Contains(result, c => c.Id == course1.Id);
            Assert.Contains(result, c => c.Id == course2.Id);
        }

        [Fact]
        public async Task GetCourse_ShouldReturnSpecificCourse()
        {
            // Arrange
            var courseId = Guid.NewGuid().ToString();
            var course = new Course
            {
                Id = courseId,
                Title = "Test Course",
                Description = "Test Description",
                Topic = "Test Topic",
                DifficultyLevel = DifficultyLevel.Intermediate,
                CreatedDate = DateTime.UtcNow,
                LastModifiedDate = DateTime.UtcNow,
                Content = new CourseContent
                {
                    Overview = "Test Overview",
                    LearningObjectives = new List<string> { "Objective 1" },
                    Lessons = new List<Lesson>
                    {
                        new Lesson
                        {
                            Title = "Lesson 1",
                            Objectives = new List<string> { "Lesson Objective 1" },
                            EstimatedMinutes = 60,
                            Content = "Lesson Content",
                            Exercises = new List<Exercise>(),
                            QuizQuestions = new List<QuizQuestion>()
                        }
                    },
                    FinalAssessment = new FinalAssessment
                    {
                        Title = "Final Project",
                        Description = "Build a project",
                        Criteria = new List<string> { "Criterion 1" },
                        EstimatedHours = 3
                    },
                    AdditionalResources = new List<AdditionalResource>()
                }
            };
            
            // Save the course
            await _courseGeneratorService.SaveCourse(course);
            
            // Act
            var result = await _courseGeneratorService.GetCourse(courseId);
            
            // Assert
            Assert.NotNull(result);
            Assert.Equal(courseId, result.Id);
            Assert.Equal("Test Course", result.Title);
            Assert.Equal("Test Description", result.Description);
            Assert.Equal("Test Topic", result.Topic);
            Assert.Equal(DifficultyLevel.Intermediate, result.DifficultyLevel);
            Assert.Equal("Test Overview", result.Content.Overview);
            Assert.Single(result.Content.LearningObjectives);
            Assert.Single(result.Content.Lessons);
            Assert.Equal("Lesson 1", result.Content.Lessons[0].Title);
            Assert.Equal("Final Project", result.Content.FinalAssessment.Title);
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
