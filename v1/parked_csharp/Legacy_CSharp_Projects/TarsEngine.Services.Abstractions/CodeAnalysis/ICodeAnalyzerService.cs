using TarsEngine.Services.Abstractions.Common;
using TarsEngine.Services.Abstractions.Models.CodeAnalysis;

namespace TarsEngine.Services.Abstractions.CodeAnalysis
{
    /// <summary>
    /// Interface for services that analyze code.
    /// </summary>
    public interface ICodeAnalyzerService : IService
    {
        /// <summary>
        /// Analyzes the code in the specified file.
        /// </summary>
        /// <param name="filePath">The path to the file to analyze.</param>
        /// <returns>The analysis result.</returns>
        Task<CodeAnalysisResult> AnalyzeFileAsync(string filePath);

        /// <summary>
        /// Analyzes the provided code snippet.
        /// </summary>
        /// <param name="code">The code snippet to analyze.</param>
        /// <param name="language">The programming language of the code.</param>
        /// <returns>The analysis result.</returns>
        Task<CodeAnalysisResult> AnalyzeCodeAsync(string code, string language);
    }
}
