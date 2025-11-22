using System;
using Xunit;
using TarsEngine.FSharp.Core.TreeOfThought;

namespace TarsEngine.FSharp.Tests.TreeOfThought
{
    public class WeightedMetricsTests
    {
        [Fact]
        public void CreateWeightedMetrics_ShouldCreateMetricsWithWeightedAverage()
        {
            // Arrange
            double correctness = 0.8;
            double efficiency = 0.7;
            double robustness = 0.6;
            double maintainability = 0.5;
            
            // Create a tuple with the weights
            var weights = Tuple.Create(2.0, 1.0, 1.0, 1.0);
            
            // Act
            var metrics = ThoughtNode.createWeightedMetrics(correctness, efficiency, robustness, maintainability, weights);
            
            // Assert
            Assert.Equal(0.8, metrics.Correctness);
            Assert.Equal(0.7, metrics.Efficiency);
            Assert.Equal(0.6, metrics.Robustness);
            Assert.Equal(0.5, metrics.Maintainability);
            
            // Calculate the expected overall score
            double expectedOverall = (0.8 * 2.0 + 0.7 * 1.0 + 0.6 * 1.0 + 0.5 * 1.0) / 5.0;
            Assert.Equal(expectedOverall, metrics.Overall);
        }
    }
}
