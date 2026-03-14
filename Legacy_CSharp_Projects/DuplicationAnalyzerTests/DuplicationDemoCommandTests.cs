using System.IO;
using System.Threading.Tasks;
using Xunit;

namespace DuplicationAnalyzerTests;

public class DuplicationDemoCommandTests
{
    [Fact]
    public async Task RunAsync_WithValidFile_ShouldReturnResults()
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
        
        var command = new DuplicationDemoCommand();
        
        try
        {
            // Act
            var result = await command.RunAsync(filePath, "C#", "all", "console");
            
            // Assert
            Assert.NotNull(result);
            Assert.Contains("Duplication Analysis Results", result);
            Assert.Contains("Token-Based Duplication", result);
            Assert.Contains("Semantic Duplication", result);
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
    public async Task RunAsync_WithInvalidFile_ShouldReturnError()
    {
        // Arrange
        var filePath = "nonexistent.cs";
        var command = new DuplicationDemoCommand();
        
        // Act
        var result = await command.RunAsync(filePath);
        
        // Assert
        Assert.NotNull(result);
        Assert.Contains("does not exist", result);
    }
    
    [Fact]
    public async Task RunAsync_WithJsonOutput_ShouldReturnJsonString()
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
    }
}";
        var filePath = Path.GetTempFileName();
        File.WriteAllText(filePath, testCode);
        
        var command = new DuplicationDemoCommand();
        
        try
        {
            // Act
            var result = await command.RunAsync(filePath, "C#", "token", "json");
            
            // Assert
            Assert.NotNull(result);
            Assert.StartsWith("[", result);
            Assert.EndsWith("]", result);
            Assert.Contains("\"Type\":", result);
            Assert.Contains("\"DuplicationPercentage\":", result);
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
    public async Task RunAsync_WithHtmlOutput_ShouldReturnHtmlString()
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
    }
}";
        var filePath = Path.GetTempFileName();
        File.WriteAllText(filePath, testCode);
        
        var command = new DuplicationDemoCommand();
        
        try
        {
            // Act
            var result = await command.RunAsync(filePath, "C#", "all", "html");
            
            // Assert
            Assert.NotNull(result);
            Assert.StartsWith("<!DOCTYPE html>", result);
            Assert.Contains("<html>", result);
            Assert.Contains("<body>", result);
            Assert.Contains("</html>", result);
            Assert.Contains("<table>", result);
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
}
