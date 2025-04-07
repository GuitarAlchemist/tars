using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using TarsEngine.Models.Metrics;
using TarsEngine.Services;
using Xunit;

namespace TarsEngine.Tests.Services;

/// <summary>
/// Tests for the CSharpComplexityAnalyzer
/// </summary>
public class CSharpComplexityAnalyzerTests
{
    private readonly Mock<ILogger<CSharpComplexityAnalyzer>> _loggerMock;
    private readonly CSharpComplexityAnalyzer _analyzer;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="CSharpComplexityAnalyzerTests"/> class
    /// </summary>
    public CSharpComplexityAnalyzerTests()
    {
        _loggerMock = new Mock<ILogger<CSharpComplexityAnalyzer>>();
        _analyzer = new CSharpComplexityAnalyzer(_loggerMock.Object);
    }
    
    /// <summary>
    /// Tests that AnalyzeCyclomaticComplexityAsync returns metrics for a simple C# file
    /// </summary>
    [Fact]
    public async Task AnalyzeCyclomaticComplexityAsync_SimpleFile_ReturnsMetrics()
    {
        // Arrange
        var tempFile = Path.GetTempFileName();
        File.Move(tempFile, Path.ChangeExtension(tempFile, ".cs"));
        tempFile = Path.ChangeExtension(tempFile, ".cs");
        
        try
        {
            // Create a simple C# file with a method
            var code = @"
using System;

namespace TestNamespace
{
    public class TestClass
    {
        public int SimpleMethod(int a, int b)
        {
            if (a > b)
            {
                return a;
            }
            else
            {
                return b;
            }
        }
        
        public int ComplexMethod(int a, int b, int c)
        {
            if (a > b)
            {
                if (a > c)
                {
                    return a;
                }
                else
                {
                    return c;
                }
            }
            else
            {
                if (b > c)
                {
                    return b;
                }
                else
                {
                    return c;
                }
            }
        }
    }
}";
            
            await File.WriteAllTextAsync(tempFile, code);
            
            // Act
            var metrics = await _analyzer.AnalyzeCyclomaticComplexityAsync(tempFile, "C#");
            
            // Assert
            Assert.NotEmpty(metrics);
            Assert.Equal(3, metrics.Count); // 2 methods + 1 file
            
            // Check method metrics
            var simpleMethodMetric = metrics.FirstOrDefault(m => m.Target.Contains("SimpleMethod"));
            Assert.NotNull(simpleMethodMetric);
            Assert.Equal(ComplexityType.Cyclomatic, simpleMethodMetric.Type);
            Assert.Equal(2, simpleMethodMetric.Value); // Base 1 + 1 if statement
            
            var complexMethodMetric = metrics.FirstOrDefault(m => m.Target.Contains("ComplexMethod"));
            Assert.NotNull(complexMethodMetric);
            Assert.Equal(ComplexityType.Cyclomatic, complexMethodMetric.Type);
            Assert.Equal(4, complexMethodMetric.Value); // Base 1 + 3 if statements
            
            // Check file metric
            var fileMetric = metrics.FirstOrDefault(m => m.TargetType == TargetType.File);
            Assert.NotNull(fileMetric);
            Assert.Equal(ComplexityType.Cyclomatic, fileMetric.Type);
            Assert.Equal(6, fileMetric.Value); // Sum of method complexities
        }
        finally
        {
            // Clean up
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }
    
    /// <summary>
    /// Tests that GetComplexityThresholdsAsync returns thresholds for C#
    /// </summary>
    [Fact]
    public async Task GetComplexityThresholdsAsync_CSharp_ReturnsThresholds()
    {
        // Act
        var thresholds = await _analyzer.GetComplexityThresholdsAsync("C#", ComplexityType.Cyclomatic);
        
        // Assert
        Assert.NotEmpty(thresholds);
        Assert.Contains("Method", thresholds.Keys);
        Assert.Contains("Class", thresholds.Keys);
        Assert.Contains("File", thresholds.Keys);
    }
    
    /// <summary>
    /// Tests that SetComplexityThresholdAsync sets a threshold
    /// </summary>
    [Fact]
    public async Task SetComplexityThresholdAsync_ValidInput_SetsThreshold()
    {
        // Arrange
        const double newThreshold = 15.0;
        
        // Act
        var result = await _analyzer.SetComplexityThresholdAsync("C#", ComplexityType.Cyclomatic, "Method", newThreshold);
        var thresholds = await _analyzer.GetComplexityThresholdsAsync("C#", ComplexityType.Cyclomatic);
        
        // Assert
        Assert.True(result);
        Assert.Equal(newThreshold, thresholds["Method"]);
    }
}
