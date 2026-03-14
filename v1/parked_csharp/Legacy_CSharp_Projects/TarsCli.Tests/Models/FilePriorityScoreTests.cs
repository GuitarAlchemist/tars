using System;
using System.Collections.Generic;
using TarsCli.Models;
using Xunit;

namespace TarsCli.Tests.Models
{
    public class FilePriorityScoreTests
    {
        [Fact]
        public void Constructor_ShouldInitializeProperties()
        {
            // Arrange & Act
            var filePath = "test/path/file.cs";
            var score = new FilePriorityScore(filePath);

            // Assert
            Assert.Equal(filePath, score.FilePath);
            Assert.Equal(0, score.BaseScore);
            Assert.Equal(0, score.ContentScore);
            Assert.Equal(0, score.RecencyScore);
            Assert.Equal(0, score.ComplexityScore);
            Assert.Equal(0, score.ImprovementPotentialScore);
            Assert.Empty(score.ScoreFactors);
        }

        [Fact]
        public void TotalScore_ShouldSumAllScores()
        {
            // Arrange
            var score = new FilePriorityScore("test.cs");
            score.BaseScore = 1.0;
            score.ContentScore = 2.0;
            score.RecencyScore = 3.0;
            score.ComplexityScore = 4.0;
            score.ImprovementPotentialScore = 5.0;

            // Act
            var totalScore = score.TotalScore;

            // Assert
            Assert.Equal(15.0, totalScore);
        }

        [Fact]
        public void AddFactor_ShouldAddFactorToScoreFactors()
        {
            // Arrange
            var score = new FilePriorityScore("test.cs");

            // Act
            score.AddFactor("TestFactor", 2.5);

            // Assert
            Assert.Single(score.ScoreFactors);
            Assert.Equal(2.5, score.ScoreFactors["TestFactor"]);
        }

        [Fact]
        public void AddFactor_ShouldUpdateExistingFactor()
        {
            // Arrange
            var score = new FilePriorityScore("test.cs");
            score.AddFactor("TestFactor", 2.5);

            // Act
            score.AddFactor("TestFactor", 3.5);

            // Assert
            Assert.Single(score.ScoreFactors);
            Assert.Equal(3.5, score.ScoreFactors["TestFactor"]);
        }

        [Fact]
        public void GetDescription_ShouldIncludeAllFactors()
        {
            // Arrange
            var score = new FilePriorityScore("test.cs");
            score.BaseScore = 1.0;
            score.ContentScore = 2.0;
            score.RecencyScore = 3.0;
            score.ComplexityScore = 4.0;
            score.ImprovementPotentialScore = 5.0;
            score.AddFactor("Factor1", 1.5);
            score.AddFactor("Factor2", 2.5);

            // Act
            var description = score.GetDescription();

            // Assert
            Assert.Contains("Total Score: 15.00", description);
            Assert.Contains("Factor1: 1.50", description);
            Assert.Contains("Factor2: 2.50", description);
        }
    }
}
