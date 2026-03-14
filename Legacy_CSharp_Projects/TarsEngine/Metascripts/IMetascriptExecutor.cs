using System.Threading.Tasks;

namespace TarsEngine.Metascripts
{
    /// <summary>
    /// Interface for executing metascripts.
    /// </summary>
    public interface IMetascriptExecutor
    {
        /// <summary>
        /// Executes a metascript asynchronously.
        /// </summary>
        /// <param name="metascriptPath">The path to the metascript file.</param>
        /// <param name="parameters">Optional parameters to pass to the metascript.</param>
        /// <returns>The result of the metascript execution.</returns>
        Task<MetascriptExecutionResult> ExecuteMetascriptAsync(string metascriptPath, object? parameters = null);
    }
}
