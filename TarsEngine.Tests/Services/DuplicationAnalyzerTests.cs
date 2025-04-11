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

public class DuplicationAnalyzerTests
{
    private readonly Mock<ILogger<CSharpDuplicationAnalyzer>> _mockLogger;
    private readonly CSharpDuplicationAnalyzer _analyzer;

    public DuplicationAnalyzerTests()
    {
        _mockLogger = new Mock<ILogger<CSharpDuplicationAnalyzer>>();
        _analyzer = new CSharpDuplicationAnalyzer(_mockLogger.Object);
    }

    [Fact]
    public async Task AnalyzeTokenBasedDuplication_ShouldReturnMetrics()
    {
        // Arrange
        var testCode = @"
using System;

namespace TestNamespace
{
    public class TestClass
    {
        public void Method1()
        {
            int a = 1;
            int b = 2;
            int c = a + b;
            Console.WriteLine(c);
        }

        public void Method2()
        {
            int a = 1;
            int b = 2;
            int c = a + b;
            Console.WriteLine(c);
        }
    }
}";
        var filePath = Path.GetTempFileName();
        File.WriteAllText(filePath, testCode);

        try
        {
            // Act
            var metrics = await _analyzer.AnalyzeTokenBasedDuplicationAsync(filePath, "C#");

            // Assert
            Assert.NotNull(metrics);
            Assert.NotEmpty(metrics);

            // File-level metric
            var fileMetric = metrics.FirstOrDefault(m => m.TargetType.ToString() == "File");
            Assert.NotNull(fileMetric);
            Assert.Equal(DuplicationType.TokenBased, fileMetric.Type);

            // Class-level metric
            var classMetric = metrics.FirstOrDefault(m => m.TargetType.ToString() == "Class");
            Assert.NotNull(classMetric);
            Assert.Equal(DuplicationType.TokenBased, classMetric.Type);

            // Method-level metrics
            var methodMetrics = metrics.Where(m => m.TargetType.ToString() == "Method").ToList();
            Assert.NotEmpty(methodMetrics);
            Assert.All(methodMetrics, m => Assert.Equal(DuplicationType.TokenBased, m.Type));
        }
        finally
        {
            // Cleanup
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }
        }
    }

    [Fact]
    public async Task AnalyzeSemanticDuplication_ShouldReturnMetrics()
    {
        // Arrange
        var testCode = @"
using System;

namespace TestNamespace
{
    public class TestClass
    {
        public void Method1()
        {
            int x = 1;
            int y = 2;
            int z = x + y;
            Console.WriteLine(z);
        }

        public void Method2()
        {
            int a = 10;
            int b = 20;
            int c = a + b;
            Console.WriteLine(c);
        }
    }
}";
        var filePath = Path.GetTempFileName();
        File.WriteAllText(filePath, testCode);

        try
        {
            // Act
            var metrics = await _analyzer.AnalyzeSemanticDuplicationAsync(filePath, "C#");

            // Assert
            Assert.NotNull(metrics);
            Assert.NotEmpty(metrics);

            // File-level metric
            var fileMetric = metrics.FirstOrDefault(m => m.TargetType.ToString() == "File");
            Assert.NotNull(fileMetric);
            Assert.Equal(DuplicationType.Semantic, fileMetric.Type);

            // Class-level metric
            var classMetric = metrics.FirstOrDefault(m => m.TargetType.ToString() == "Class");
            Assert.NotNull(classMetric);
            Assert.Equal(DuplicationType.Semantic, classMetric.Type);

            // Method-level metrics
            var methodMetrics = metrics.Where(m => m.TargetType.ToString() == "Method").ToList();
            Assert.NotEmpty(methodMetrics);
            Assert.All(methodMetrics, m => Assert.Equal(DuplicationType.Semantic, m.Type));
        }
        finally
        {
            // Cleanup
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }
        }
    }

    [Fact]
    public async Task AnalyzeAllDuplicationMetrics_ShouldReturnAllMetrics()
    {
        // Arrange
        var testCode = @"
using System;

namespace TestNamespace
{
    public class TestClass
    {
        public void Method1()
        {
            int a = 1;
            int b = 2;
            int c = a + b;
            Console.WriteLine(c);
        }

        public void Method2()
        {
            int a = 1;
            int b = 2;
            int c = a + b;
            Console.WriteLine(c);
        }
    }
}";
        var filePath = Path.GetTempFileName();
        File.WriteAllText(filePath, testCode);

        try
        {
            // Act
            var metrics = await _analyzer.AnalyzeAllDuplicationMetricsAsync(filePath, "C#");

            // Assert
            Assert.NotNull(metrics);
            Assert.NotEmpty(metrics);

            // Should have both token-based and semantic metrics
            var tokenBasedMetrics = metrics.Where(m => m.Type == DuplicationType.TokenBased).ToList();
            var semanticMetrics = metrics.Where(m => m.Type == DuplicationType.Semantic).ToList();

            Assert.NotEmpty(tokenBasedMetrics);
            Assert.NotEmpty(semanticMetrics);
        }
        finally
        {
            // Cleanup
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }
        }
    }

    [Fact]
    public async Task GetDuplicationThresholds_ShouldReturnThresholds()
    {
        // Act
        var thresholds = await _analyzer.GetDuplicationThresholdsAsync("C#", DuplicationType.TokenBased);

        // Assert
        Assert.NotNull(thresholds);
        Assert.NotEmpty(thresholds);
        Assert.Contains("Method", thresholds.Keys);
        Assert.Contains("Class", thresholds.Keys);
        Assert.Contains("File", thresholds.Keys);
    }

    [Fact]
    public async Task SetDuplicationThreshold_ShouldUpdateThreshold()
    {
        // Arrange
        var language = "C#";
        var duplicationType = DuplicationType.TokenBased;
        var targetType = "Method";
        var newThreshold = 10.0;

        // Act
        var result = await _analyzer.SetDuplicationThresholdAsync(language, duplicationType, targetType, newThreshold);
        var thresholds = await _analyzer.GetDuplicationThresholdsAsync(language, duplicationType);

        // Assert
        Assert.True(result);
        Assert.Equal(newThreshold, thresholds[targetType]);
    }
}
