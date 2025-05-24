using System.Threading.Tasks;

namespace TarsEngine.Services.CodeAnalysis
{
    /// <summary>
    /// Interface for code analyzers.
    /// </summary>
    public interface ICodeAnalyzer
    {
        /// <summary>
        /// Analyzes code.
        /// </summary>
        /// <param name="code">The code to analyze.</param>
        /// <returns>The analysis result.</returns>
        Task<string> AnalyzeCodeAsync(string code);
    }
}
