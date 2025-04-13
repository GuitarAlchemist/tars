using System.Collections.Generic;
using System.Threading.Tasks;
using TarsCli.Services.CodeAnalysis;

namespace TarsCli.Services.CodeGeneration
{
    /// <summary>
    /// Interface for code generators
    /// </summary>
    public interface ICodeGenerator
    {
        /// <summary>
        /// Generates improved code based on analysis results
        /// </summary>
        /// <param name="filePath">Path to the file to improve</param>
        /// <param name="originalContent">Original content of the file</param>
        /// <param name="analysisResult">Analysis result</param>
        /// <returns>Code generation result</returns>
        Task<CodeGenerationResult> GenerateCodeAsync(string filePath, string originalContent, CodeAnalysisResult analysisResult);

        /// <summary>
        /// Gets the supported file extensions for this generator
        /// </summary>
        /// <returns>List of supported file extensions</returns>
        IEnumerable<string> GetSupportedFileExtensions();
    }

    /// <summary>
    /// Result of code generation
    /// </summary>
    public class CodeGenerationResult
    {
        /// <summary>
        /// Path to the file
        /// </summary>
        public string FilePath { get; set; }

        /// <summary>
        /// Original content of the file
        /// </summary>
        public string OriginalContent { get; set; }

        /// <summary>
        /// Generated content of the file
        /// </summary>
        public string GeneratedContent { get; set; }

        /// <summary>
        /// Whether the generation was successful
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Error message if the generation failed
        /// </summary>
        public string ErrorMessage { get; set; }

        /// <summary>
        /// List of changes made to the file
        /// </summary>
        public List<CodeChange> Changes { get; set; } = new List<CodeChange>();

        /// <summary>
        /// Additional information about the generation
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
    }

    /// <summary>
    /// Represents a change made to a file
    /// </summary>
    public class CodeChange
    {
        /// <summary>
        /// Type of the change
        /// </summary>
        public CodeChangeType Type { get; set; }

        /// <summary>
        /// Description of the change
        /// </summary>
        public string Description { get; set; }

        /// <summary>
        /// Line number where the change was made
        /// </summary>
        public int LineNumber { get; set; }

        /// <summary>
        /// Original code segment
        /// </summary>
        public string OriginalCode { get; set; }

        /// <summary>
        /// New code segment
        /// </summary>
        public string NewCode { get; set; }

        /// <summary>
        /// Issue that triggered the change
        /// </summary>
        public CodeIssue Issue { get; set; }
    }

    /// <summary>
    /// Type of code change
    /// </summary>
    public enum CodeChangeType
    {
        /// <summary>
        /// Added code
        /// </summary>
        Addition,

        /// <summary>
        /// Removed code
        /// </summary>
        Removal,

        /// <summary>
        /// Modified code
        /// </summary>
        Modification,

        /// <summary>
        /// Refactored code
        /// </summary>
        Refactoring,

        /// <summary>
        /// Optimized code
        /// </summary>
        Optimization,

        /// <summary>
        /// Fixed security vulnerability
        /// </summary>
        SecurityFix,

        /// <summary>
        /// Added documentation
        /// </summary>
        Documentation,

        /// <summary>
        /// Fixed style issue
        /// </summary>
        StyleFix
    }
}
