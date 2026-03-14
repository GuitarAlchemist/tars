namespace TarsCli.Services;

/// <summary>
/// Logger adapter to convert between logger types
/// </summary>
/// <typeparam name="T">The target logger type</typeparam>
public class LoggerAdapter<T> : ILogger<T>
{
    private readonly ILogger _logger;

    /// <summary>
    /// Initializes a new instance of the LoggerAdapter class
    /// </summary>
    /// <param name="logger">The underlying logger</param>
    public LoggerAdapter(ILogger logger)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    /// <inheritdoc/>
    IDisposable ILogger.BeginScope<TState>(TState state) => _logger.BeginScope(state);

    /// <inheritdoc/>
    public bool IsEnabled(LogLevel logLevel) => _logger.IsEnabled(logLevel);

    /// <inheritdoc/>
    void ILogger.Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        _logger.Log(logLevel, eventId, state, exception, formatter);
    }
}