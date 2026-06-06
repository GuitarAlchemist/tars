namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the Model Context Protocol service
/// </summary>
public interface IMcpService
{
    /// <summary>
    /// Execute a command via the Model Context Protocol
    /// </summary>
    /// <param name="command">The command to execute</param>
    /// <param name="parameters">The command parameters</param>
    /// <returns>The command result</returns>
    Task<object> ExecuteCommandAsync(string command, Dictionary<string, object> parameters);
}
