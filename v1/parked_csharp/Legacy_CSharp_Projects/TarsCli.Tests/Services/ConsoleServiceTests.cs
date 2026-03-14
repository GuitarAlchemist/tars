using System.IO;
using TarsCli.Services;

namespace TarsCli.Tests.Services;

public class ConsoleServiceTests : IDisposable
{
    private readonly StringWriter _stringWriter;
    private readonly ConsoleColor _originalForegroundColor;

    public ConsoleServiceTests()
    {
        _stringWriter = new StringWriter();
        Console.SetOut(_stringWriter);
        _originalForegroundColor = Console.ForegroundColor;
    }

    public void Dispose()
    {
        Console.SetOut(new StreamWriter(Console.OpenStandardOutput()));
        Console.ForegroundColor = _originalForegroundColor;
        _stringWriter.Dispose();
    }

    [Fact]
    public void WriteHeader_ShouldWriteFormattedHeader()
    {
        // Arrange
        var service = new ConsoleService();

        // Act
        service.WriteHeader("Test Header");

        // Assert
        var output = _stringWriter.ToString();
        Assert.Contains("=", output);
        Assert.Contains("Test Header", output);
    }

    [Fact]
    public void WriteSubHeader_ShouldWriteFormattedSubHeader()
    {
        // Arrange
        var service = new ConsoleService();

        // Act
        service.WriteSubHeader("Test SubHeader");

        // Assert
        var output = _stringWriter.ToString();
        Assert.Contains("-", output);
        Assert.Contains("Test SubHeader", output);
    }

    [Fact]
    public void WriteInfo_ShouldWriteInfoMessage()
    {
        // Arrange
        var service = new ConsoleService();

        // Act
        service.WriteInfo("Test Info");

        // Assert
        var output = _stringWriter.ToString();
        Assert.Contains("Test Info", output);
    }

    [Fact]
    public void WriteSuccess_ShouldWriteSuccessMessage()
    {
        // Arrange
        var service = new ConsoleService();

        // Act
        service.WriteSuccess("Test Success");

        // Assert
        var output = _stringWriter.ToString();
        Assert.Contains("✓", output);
        Assert.Contains("Test Success", output);
    }

    [Fact]
    public void WriteWarning_ShouldWriteWarningMessage()
    {
        // Arrange
        var service = new ConsoleService();

        // Act
        service.WriteWarning("Test Warning");

        // Assert
        var output = _stringWriter.ToString();
        Assert.Contains("⚠", output);
        Assert.Contains("Test Warning", output);
    }

    [Fact]
    public void WriteError_ShouldWriteErrorMessage()
    {
        // Arrange
        var service = new ConsoleService();

        // Act
        service.WriteError("Test Error");

        // Assert
        var output = _stringWriter.ToString();
        Assert.Contains("✗", output);
        Assert.Contains("Test Error", output);
    }

    [Fact]
    public void WriteTable_ShouldWriteFormattedTable()
    {
        // Arrange
        var service = new ConsoleService();
        var headers = new[] { "Header1", "Header2" };
        var rows = new List<string[]>
        {
            new[] { "Value1", "Value2" },
            new[] { "Value3", "Value4" }
        };

        // Act
        service.WriteTable(headers, rows);

        // Assert
        var output = _stringWriter.ToString();
        Assert.Contains("Header1", output);
        Assert.Contains("Header2", output);
        Assert.Contains("Value1", output);
        Assert.Contains("Value2", output);
        Assert.Contains("Value3", output);
        Assert.Contains("Value4", output);
        Assert.Contains("-+-", output); // Separator
    }
}
