using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Console-based implementation of the progress reporter
/// </summary>
public class ConsoleProgressReporter : IProgressReporter
{
    private readonly ILogger<ConsoleProgressReporter> _logger;
    private readonly bool _verbose;
    private int _lastPercentComplete = -1;

    /// <summary>
    /// Initializes a new instance of the <see cref="ConsoleProgressReporter"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="verbose">Whether to show verbose output</param>
    public ConsoleProgressReporter(ILogger<ConsoleProgressReporter> logger, bool verbose = false)
    {
        _logger = logger;
        _verbose = verbose;
    }

    /// <inheritdoc/>
    public void ReportProgress(string message, int percentComplete)
    {
        // Ensure percent complete is in range
        percentComplete = Math.Max(0, Math.Min(100, percentComplete));

        // Log progress
        _logger.LogInformation("Progress: {PercentComplete}% - {Message}", percentComplete, message);

        // Only update console if percent complete has changed
        if (percentComplete != _lastPercentComplete || _verbose)
        {
            // Save cursor position
            var originalLeft = Console.CursorLeft;
            var originalTop = Console.CursorTop;

            // Clear current line
            Console.Write(new string(' ', Console.WindowWidth - 1));
            Console.SetCursorPosition(0, originalTop);

            // Write progress bar
            Console.Write("[");
            var progressChars = (int)Math.Round(percentComplete / 2.0);
            Console.Write(new string('=', progressChars));
            if (progressChars < 50)
            {
                Console.Write(">");
                Console.Write(new string(' ', 49 - progressChars));
            }
            Console.Write("] ");
            Console.Write($"{percentComplete}%");

            // Write message if verbose
            if (_verbose)
            {
                Console.WriteLine();
                Console.WriteLine(message);
            }

            // Restore cursor position
            Console.SetCursorPosition(originalLeft, originalTop);

            _lastPercentComplete = percentComplete;
        }
    }

    /// <inheritdoc/>
    public void ReportWarning(string message, Exception? exception)
    {
        _logger.LogWarning(exception, "Warning: {Message}", message);

        if (_verbose)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"Warning: {message}");
            if (exception != null)
            {
                Console.WriteLine(exception.Message);
            }
            Console.ResetColor();
        }
    }

    /// <inheritdoc/>
    public void ReportError(string message, Exception exception)
    {
        _logger.LogError(exception, "Error: {Message}", message);

        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"Error: {message}");
        if (_verbose)
        {
            Console.WriteLine(exception.ToString());
        }
        else
        {
            Console.WriteLine(exception.Message);
        }
        Console.ResetColor();
    }
}
