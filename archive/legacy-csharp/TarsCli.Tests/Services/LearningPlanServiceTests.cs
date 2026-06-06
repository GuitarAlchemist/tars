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
    public class LearningPlanServiceTests
    {
        private readonly Mock<ILogger<LearningPlanService>> _loggerMock;
        private readonly Mock<IConfiguration> _configMock;
        private readonly Mock<OllamaService> _ollamaServiceMock;
        private readonly string _testDirectory;
        private readonly LearningPlanService _learningPlanService;

        public LearningPlanServiceTests()
        {
            _loggerMock = new Mock<ILogger<LearningPlanService>>();
            _configMock = new Mock<IConfiguration>();
            _ollamaServiceMock = new Mock<OllamaService>();
            
            // Create a temporary directory for testing
            _testDirectory = Path.Combine(Path.GetTempPath(), "TarsCliTests", "LearningPlans", Guid.NewGuid().ToString());
            Directory.CreateDirectory(_testDirectory);
            
            // Setup configuration
            _configMock.Setup(c => c["Paths:LearningPlans"]).Returns(_testDirectory);
            _configMock.Setup(c => c["Ollama:DefaultModel"]).Returns("llama3");
            
            // Create the service
            _learningPlanService = new LearningPlanService(
                _loggerMock.Object,
                _configMock.Object,
                _ollamaServiceMock.Object);
        }

        [Fact]
        public async Task GenerateLearningPlan_ShouldCreateAndSavePlan()
        {
            // Arrange
            var name = "Test Learning Plan";
            var topic = "C# Programming";
            var skillLevel = SkillLevel.Intermediate;
            var goals = new List<string> { "Learn C# basics", "Build a console application" };
            var preferences = new List<string> { "Visual learning", "Hands-on exercises" };
            var estimatedHours = 20;
            var model = "llama3";
            
            var mockResponse = @"{
                ""introduction"": ""This learning plan will guide you through C# programming."",
                ""prerequisites"": [""Basic programming knowledge""],
                ""modules"": [
                    {
                        ""title"": ""C# Fundamentals"",
                        ""objectives"": [""Understand C# syntax"", ""Learn about variables and data types""],
                        ""estimatedHours"": 5,
                        ""resources"": [
                            {
                                ""title"": ""C# Documentation"",
                                ""type"": ""Documentation"",
                                ""url"": ""https://docs.microsoft.com/en-us/dotnet/csharp/""
                            }
                        ],
                        ""assessment"": ""Create a simple console application""
                    }
                ],
                ""timeline"": [
                    {
                        ""week"": 1,
                        ""description"": ""Complete C# Fundamentals module""
                    }
                ],
                ""milestones"": [
                    {
                        ""title"": ""First Console Application"",
                        ""description"": ""Build a working console application""
                    }
                ],
                ""practiceProjects"": [
                    {
                        ""title"": ""Calculator App"",
                        ""description"": ""Build a simple calculator application""
                    }
                ]
            }";
            
            _ollamaServiceMock.Setup(o => o.GenerateCompletion(It.IsAny<string>(), model))
                .ReturnsAsync(mockResponse);
            
            // Act
            var result = await _learningPlanService.GenerateLearningPlan(name, topic, skillLevel, goals, preferences, estimatedHours, model);
            
            // Assert
            Assert.NotNull(result);
            Assert.Equal(name, result.Name);
            Assert.Equal(topic, result.Topic);
            Assert.Equal(skillLevel, result.SkillLevel);
            Assert.NotNull(result.Content);
            Assert.Equal("This learning plan will guide you through C# programming.", result.Content.Introduction);
            Assert.Single(result.Content.Prerequisites);
            Assert.Single(result.Content.Modules);
            Assert.Single(result.Content.Timeline);
            Assert.Single(result.Content.Milestones);
            Assert.Single(result.Content.PracticeProjects);
            
            // Verify the file was saved
            var filePath = Path.Combine(_testDirectory, $"{result.Id}.json");
            Assert.True(File.Exists(filePath));
        }

        [Fact]
        public async Task GetLearningPlans_ShouldReturnAllPlans()
        {
            // Arrange
            // Create a few test plans
            var plan1 = new LearningPlan
            {
                Id = Guid.NewGuid().ToString(),
                Name = "Plan 1",
                Topic = "Topic 1",
                SkillLevel = SkillLevel.Beginner,
                CreatedDate = DateTime.UtcNow,
                LastModifiedDate = DateTime.UtcNow,
                Content = new LearningPlanContent
                {
                    Introduction = "Introduction 1",
                    Prerequisites = new List<string>(),
                    Modules = new List<Module>(),
                    Timeline = new List<TimelineItem>(),
                    Milestones = new List<Milestone>(),
                    PracticeProjects = new List<PracticeProject>()
                }
            };
            
            var plan2 = new LearningPlan
            {
                Id = Guid.NewGuid().ToString(),
                Name = "Plan 2",
                Topic = "Topic 2",
                SkillLevel = SkillLevel.Advanced,
                CreatedDate = DateTime.UtcNow,
                LastModifiedDate = DateTime.UtcNow,
                Content = new LearningPlanContent
                {
                    Introduction = "Introduction 2",
                    Prerequisites = new List<string>(),
                    Modules = new List<Module>(),
                    Timeline = new List<TimelineItem>(),
                    Milestones = new List<Milestone>(),
                    PracticeProjects = new List<PracticeProject>()
                }
            };
            
            // Save the plans
            await _learningPlanService.SaveLearningPlan(plan1);
            await _learningPlanService.SaveLearningPlan(plan2);
            
            // Act
            var result = await _learningPlanService.GetLearningPlans();
            
            // Assert
            Assert.NotNull(result);
            Assert.Equal(2, result.Count);
            Assert.Contains(result, p => p.Id == plan1.Id);
            Assert.Contains(result, p => p.Id == plan2.Id);
        }

        [Fact]
        public async Task GetLearningPlan_ShouldReturnSpecificPlan()
        {
            // Arrange
            var planId = Guid.NewGuid().ToString();
            var plan = new LearningPlan
            {
                Id = planId,
                Name = "Test Plan",
                Topic = "Test Topic",
                SkillLevel = SkillLevel.Intermediate,
                CreatedDate = DateTime.UtcNow,
                LastModifiedDate = DateTime.UtcNow,
                Content = new LearningPlanContent
                {
                    Introduction = "Test Introduction",
                    Prerequisites = new List<string> { "Prerequisite 1" },
                    Modules = new List<Module>
                    {
                        new Module
                        {
                            Title = "Module 1",
                            Objectives = new List<string> { "Objective 1" },
                            EstimatedHours = 5,
                            Resources = new List<Resource>(),
                            Assessment = "Assessment 1"
                        }
                    },
                    Timeline = new List<TimelineItem>(),
                    Milestones = new List<Milestone>(),
                    PracticeProjects = new List<PracticeProject>()
                }
            };
            
            // Save the plan
            await _learningPlanService.SaveLearningPlan(plan);
            
            // Act
            var result = await _learningPlanService.GetLearningPlan(planId);
            
            // Assert
            Assert.NotNull(result);
            Assert.Equal(planId, result.Id);
            Assert.Equal("Test Plan", result.Name);
            Assert.Equal("Test Topic", result.Topic);
            Assert.Equal(SkillLevel.Intermediate, result.SkillLevel);
            Assert.Equal("Test Introduction", result.Content.Introduction);
            Assert.Single(result.Content.Prerequisites);
            Assert.Single(result.Content.Modules);
            Assert.Equal("Module 1", result.Content.Modules[0].Title);
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
