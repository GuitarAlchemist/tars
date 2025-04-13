namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the metascript service
/// </summary>
public interface IMetascriptService
{
    /// <summary>
    /// Executes a metascript
    /// </summary>
    /// <param name="metascript">Metascript to execute</param>
    /// <returns>Result of the metascript execution</returns>
    Task<object> ExecuteMetascriptAsync(string metascript);
}
