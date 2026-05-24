using System;
using System.Linq;
using TarsCli.Parsing;
using Xunit;

namespace TarsCli.Tests;

public class TripleQuotedArgumentParserTests
{
    [Fact]
    public void ParseTripleQuotedArguments_WithNoTripleQuotes_ReturnsOriginalArgs()
    {
        // Arrange
        var args = new[] { "mcp", "code", "test.cs", "public class Test {}" };
        
        // Act
        var result = TripleQuotedArgumentParser.ParseTripleQuotedArguments(args);
        
        // Assert
        Assert.Equal(args, result);
    }
    
    [Fact]
    public void ParseTripleQuotedArguments_WithTripleQuotesInSingleArg_RemovesTripleQuotes()
    {
        // Arrange
        var args = new[] { "mcp", "code", "test.cs", "\"\"\"public class Test {}\"\"\"" };
        
        // Act
        var result = TripleQuotedArgumentParser.ParseTripleQuotedArguments(args);
        
        // Assert
        Assert.Equal(4, result.Length);
        Assert.Equal("mcp", result[0]);
        Assert.Equal("code", result[1]);
        Assert.Equal("test.cs", result[2]);
        Assert.Equal("public class Test {}", result[3]);
    }
    
    [Fact]
    public void ParseTripleQuotedArguments_WithTripleQuotesAcrossMultipleArgs_CombinesArgs()
    {
        // Arrange
        var args = new[] 
        { 
            "mcp", "code", "test.cs", "\"\"\"public class Test", 
            "{", 
            "    public static void Main()", 
            "    {", 
            "        Console.WriteLine(\\\"Hello\\\");", 
            "    }", 
            "}\"\"\"" 
        };
        
        // Act
        var result = TripleQuotedArgumentParser.ParseTripleQuotedArguments(args);
        
        // Assert
        Assert.Equal(4, result.Length);
        Assert.Equal("mcp", result[0]);
        Assert.Equal("code", result[1]);
        Assert.Equal("test.cs", result[2]);
        Assert.Contains("public class Test", result[3]);
        Assert.Contains("Console.WriteLine", result[3]);
    }
    
    [Fact]
    public void ParseTripleQuotedArguments_WithTripleQuotedFlag_HandlesCorrectly()
    {
        // Arrange
        var args = new[] 
        { 
            "mcp", "code", "test.cs", "-triple-quoted", "\"\"\"public class Test {}", 
            "    public static void Main()", 
            "    {", 
            "        Console.WriteLine(\\\"Hello\\\");", 
            "    }", 
            "}\"\"\"" 
        };
        
        // Act
        var result = TripleQuotedArgumentParser.ParseTripleQuotedArguments(args);
        
        // Assert
        Assert.Equal(5, result.Length);
        Assert.Equal("mcp", result[0]);
        Assert.Equal("code", result[1]);
        Assert.Equal("test.cs", result[2]);
        Assert.Equal("-triple-quoted", result[3]);
        Assert.Contains("public class Test {}", result[4]);
        Assert.Contains("Console.WriteLine", result[4]);
    }
}
