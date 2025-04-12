using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using TarsEngine.Models;
using TarsEngine.Services;
using Xunit;

namespace TarsEngine.Tests.Services
{
    public class CSharpAnalyzerRefactoredTests
    {
        private readonly Mock<ILogger<CSharpAnalyzerRefactored>> _loggerMock;
        private readonly CSharpAnalyzerRefactored _analyzer;

        public CSharpAnalyzerRefactoredTests()
        {
            _loggerMock = new Mock<ILogger<CSharpAnalyzerRefactored>>();
            _analyzer = new CSharpAnalyzerRefactored(_loggerMock.Object);
        }

        [Fact]
        public async Task AnalyzeAsync_WithValidCode_ReturnsSuccessfulResult()
        {
            // Arrange
            var code = @"
namespace TestNamespace
{
    public class TestClass
    {
        public string TestProperty { get; set; }

        public void TestMethod()
        {
            if (TestProperty != null)
            {
                Console.WriteLine(TestProperty);
            }
        }
    }
}";

            // Act
            var result = await _analyzer.AnalyzeAsync(code);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IsSuccessful);
            Assert.Equal(ProgrammingLanguage.CSharp, result.Language);
            Assert.NotEmpty(result.Structures);
            Assert.NotEmpty(result.Metrics);
        }

        [Fact]
        public async Task AnalyzeAsync_WithNullContent_ReturnsFailureResult()
        {
            // Arrange
            string? code = null;

            // Act
            var result = await _analyzer.AnalyzeAsync(code!);

            // Assert
            Assert.NotNull(result);
            Assert.False(result.IsSuccessful);
            Assert.Equal(ProgrammingLanguage.CSharp, result.Language);
            Assert.Equal("Content is null", result.ErrorMessage);
            Assert.Contains("Content is null", result.Errors);
        }

        [Fact]
        public async Task AnalyzeAsync_WithEmptyContent_ReturnsSuccessfulResultWithNoStructures()
        {
            // Arrange
            var code = string.Empty;

            // Act
            var result = await _analyzer.AnalyzeAsync(code);

            // Assert
            Assert.NotNull(result);
            Assert.True(result.IsSuccessful);
            Assert.Equal(ProgrammingLanguage.CSharp, result.Language);
            Assert.Empty(result.Structures);
        }

        [Fact]
        public void ExtractStructures_WithValidCode_ReturnsCorrectStructures()
        {
            // Arrange
            var code = @"
namespace TestNamespace
{
    public class TestClass
    {
        public string TestProperty { get; set; }

        public void TestMethod()
        {
            Console.WriteLine(""Hello"");
        }
    }
}";

            // Act
            var structures = _analyzer.ExtractStructures(code);

            // Assert
            Assert.NotEmpty(structures);
            Assert.Contains(structures, s => s.Type == StructureType.Namespace && s.Name == "TestNamespace");
            Assert.Contains(structures, s => s.Type == StructureType.Class && s.Name == "TestClass");
            Assert.Contains(structures, s => s.Type == StructureType.Method && s.Name == "TestMethod");
            Assert.Contains(structures, s => s.Type == StructureType.Property && s.Name == "TestProperty");
        }

        [Fact]
        public void CalculateMetrics_WithValidCode_ReturnsCorrectMetrics()
        {
            // Arrange
            var code = @"
namespace TestNamespace
{
    public class TestClass
    {
        public string TestProperty { get; set; }

        public void TestMethod()
        {
            Console.WriteLine(""Hello"");
        }
    }
}";
            var structures = _analyzer.ExtractStructures(code);

            // Act
            var metrics = _analyzer.CalculateMetrics(code, structures, true, true);

            // Assert
            Assert.NotEmpty(metrics);
            Assert.Contains(metrics, m => m.Type == MetricType.Size && m.Name == "Lines of Code");
            Assert.Contains(metrics, m => m.Type == MetricType.Size && m.Name == "Class Count");
            Assert.Contains(metrics, m => m.Type == MetricType.Size && m.Name == "Method Count");
        }

        [Fact]
        public void DetectSecurityIssues_WithVulnerableCode_ReturnsIssues()
        {
            // Arrange
            var code = @"
namespace TestNamespace
{
    public class TestClass
    {
        public void TestMethod(string input)
        {
            var sql = ""SELECT * FROM Users WHERE Username = '"" + input + ""'"";
            var cmd = new SqlCommand(sql);

            Response.Write(input);

            var password = ""myPassword123"";
        }
    }
}";

            // Act
            var issues = _analyzer.DetectSecurityIssues(code);

            // Assert
            Assert.NotEmpty(issues);
            // Check that we have at least one security issue
            Assert.Contains(issues, i => i.Type == CodeIssueType.Security);
        }

        [Fact]
        public async Task GetAvailableOptionsAsync_ReturnsCorrectOptions()
        {
            // Act
            var options = await _analyzer.GetAvailableOptionsAsync();

            // Assert
            Assert.NotEmpty(options);
            Assert.Contains("IncludeMetrics", options.Keys);
            Assert.Contains("IncludeStructures", options.Keys);
            Assert.Contains("IncludeIssues", options.Keys);
            Assert.Contains("AnalyzePerformance", options.Keys);
            Assert.Contains("AnalyzeComplexity", options.Keys);
            Assert.Contains("AnalyzeMaintainability", options.Keys);
            Assert.Contains("AnalyzeSecurity", options.Keys);
            Assert.Contains("AnalyzeStyle", options.Keys);
        }

        [Fact]
        public async Task GetLanguageSpecificIssueTypesAsync_ReturnsCorrectIssueTypes()
        {
            // Act
            var issueTypes = await _analyzer.GetLanguageSpecificIssueTypesAsync();

            // Assert
            Assert.NotEmpty(issueTypes);
            Assert.Contains(CodeIssueType.CodeSmell, issueTypes.Keys);
            Assert.Contains(CodeIssueType.Security, issueTypes.Keys);
            Assert.Contains(CodeIssueType.Performance, issueTypes.Keys);
            Assert.Contains(CodeIssueType.Complexity, issueTypes.Keys);
            Assert.Contains(CodeIssueType.Style, issueTypes.Keys);
        }

        [Fact]
        public async Task GetLanguageSpecificMetricTypesAsync_ReturnsCorrectMetricTypes()
        {
            // Act
            var metricTypes = await _analyzer.GetLanguageSpecificMetricTypesAsync();

            // Assert
            Assert.NotEmpty(metricTypes);
            Assert.Contains(MetricType.Size, metricTypes.Keys);
            Assert.Contains(MetricType.Complexity, metricTypes.Keys);
            Assert.Contains(MetricType.Maintainability, metricTypes.Keys);
            Assert.Contains(MetricType.Performance, metricTypes.Keys);
        }
    }
}
