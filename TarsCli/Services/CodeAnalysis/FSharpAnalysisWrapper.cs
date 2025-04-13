using Microsoft.FSharp.Collections;

namespace TarsCli.Services.CodeAnalysis
{
    /// <summary>
    /// Wrapper for F# code analysis functions
    /// </summary>
    public static class FSharpAnalysisWrapper
    {
        /// <summary>
        /// Analyzes a file using the F# code analysis module
        /// </summary>
        /// <param name="filePath">Path to the file to analyze</param>
        /// <returns>Analysis result</returns>
        public static FSharpAnalysisResult analyzeFile(string filePath)
        {
            // This is a mock implementation until the F# module is implemented
            return new FSharpAnalysisResult
            {
                FilePath = filePath,
                Issues = FSharpList<CodeIssue>.Empty,
                SuggestedFixes = FSharpList<Tuple<string, string>>.Empty
            };
        }

        /// <summary>
        /// Analyzes a project using the F# code analysis module
        /// </summary>
        /// <param name="projectPath">Path to the project to analyze</param>
        /// <param name="maxFiles">Maximum number of files to analyze</param>
        /// <returns>List of analysis results</returns>
        public static List<FSharpAnalysisResult> analyzeProject(string projectPath, int maxFiles = 50)
        {
            // This is a mock implementation until the F# module is implemented
            return new List<FSharpAnalysisResult>
            {
                new FSharpAnalysisResult
                {
                    FilePath = Path.Combine(projectPath, "Program.cs"),
                    Issues = FSharpList<CodeIssue>.Empty,
                    SuggestedFixes = FSharpList<Tuple<string, string>>.Empty
                }
            };
        }
    }

    /// <summary>
    /// Result of F# code analysis
    /// </summary>
    public class FSharpAnalysisResult
    {
        /// <summary>
        /// Path to the analyzed file
        /// </summary>
        public string FilePath { get; set; }

        /// <summary>
        /// List of issues found in the file
        /// </summary>
        public FSharpList<CodeIssue> Issues { get; set; }

        /// <summary>
        /// List of suggested fixes for the issues
        /// </summary>
        public FSharpList<Tuple<string, string>> SuggestedFixes { get; set; }
    }
}
