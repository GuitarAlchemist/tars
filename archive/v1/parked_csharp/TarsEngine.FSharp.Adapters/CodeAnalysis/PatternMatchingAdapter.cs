using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.FSharp.Core.CodeAnalysis;

namespace TarsEngine.FSharp.Adapters.CodeAnalysis
{
    /// <summary>
    /// Adapter for F# pattern matching functionality.
    /// </summary>
    public class PatternMatchingAdapter
    {
        private readonly ILogger<PatternMatchingAdapter> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="PatternMatchingAdapter"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public PatternMatchingAdapter(ILogger<PatternMatchingAdapter> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Finds patterns in content.
        /// </summary>
        /// <param name="content">The content to analyze.</param>
        /// <param name="patterns">The patterns to look for.</param>
        /// <param name="language">The language of the content.</param>
        /// <param name="includeLineNumbers">Whether to include line numbers.</param>
        /// <param name="includeColumnNumbers">Whether to include column numbers.</param>
        /// <param name="includeContext">Whether to include context.</param>
        /// <param name="contextLines">The number of context lines to include.</param>
        /// <param name="minConfidence">The minimum confidence threshold.</param>
        /// <param name="maxMatches">The maximum number of matches to return.</param>
        /// <returns>The pattern matches.</returns>
        public IReadOnlyList<PatternMatchAdapter> FindPatterns(
            string content,
            IEnumerable<CodePatternAdapter> patterns,
            string language,
            bool includeLineNumbers = true,
            bool includeColumnNumbers = true,
            bool includeContext = true,
            int contextLines = 2,
            double minConfidence = 0.7,
            int? maxMatches = null)
        {
            try
            {
                _logger.LogInformation("Finding patterns in {Language} content", language);

                // Convert C# patterns to F# patterns
                var fsharpPatterns = patterns.Select(p => p.FSharpPattern).ToList();

                // Create options
                var options = new PatternMatching.PatternMatchOptions(
                    includeLineNumbers,
                    includeColumnNumbers,
                    includeContext,
                    contextLines,
                    minConfidence,
                    maxMatches.HasValue ? Microsoft.FSharp.Core.FSharpOption<int>.Some(maxMatches.Value) : Microsoft.FSharp.Core.FSharpOption<int>.None);

                // Find patterns
                var matches = PatternMatching.findPatterns(content, fsharpPatterns, language, options);

                // Convert F# matches to C# matches
                return matches.Select(m => new PatternMatchAdapter(m)).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error finding patterns in {Language} content", language);
                return Array.Empty<PatternMatchAdapter>();
            }
        }

        /// <summary>
        /// Finds patterns in a file.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <param name="patterns">The patterns to look for.</param>
        /// <param name="includeLineNumbers">Whether to include line numbers.</param>
        /// <param name="includeColumnNumbers">Whether to include column numbers.</param>
        /// <param name="includeContext">Whether to include context.</param>
        /// <param name="contextLines">The number of context lines to include.</param>
        /// <param name="minConfidence">The minimum confidence threshold.</param>
        /// <param name="maxMatches">The maximum number of matches to return.</param>
        /// <returns>The pattern matches.</returns>
        public IReadOnlyList<PatternMatchAdapter> FindPatternsInFile(
            string filePath,
            IEnumerable<CodePatternAdapter> patterns,
            bool includeLineNumbers = true,
            bool includeColumnNumbers = true,
            bool includeContext = true,
            int contextLines = 2,
            double minConfidence = 0.7,
            int? maxMatches = null)
        {
            try
            {
                _logger.LogInformation("Finding patterns in file {FilePath}", filePath);

                // Convert C# patterns to F# patterns
                var fsharpPatterns = patterns.Select(p => p.FSharpPattern).ToList();

                // Create options
                var options = new PatternMatching.PatternMatchOptions(
                    includeLineNumbers,
                    includeColumnNumbers,
                    includeContext,
                    contextLines,
                    minConfidence,
                    maxMatches.HasValue ? Microsoft.FSharp.Core.FSharpOption<int>.Some(maxMatches.Value) : Microsoft.FSharp.Core.FSharpOption<int>.None);

                // Find patterns
                var matches = PatternMatching.findPatternsInFile(filePath, fsharpPatterns, options);

                // Convert F# matches to C# matches
                return matches.Select(m => new PatternMatchAdapter(m)).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error finding patterns in file {FilePath}", filePath);
                return Array.Empty<PatternMatchAdapter>();
            }
        }

        /// <summary>
        /// Finds patterns in a directory.
        /// </summary>
        /// <param name="directoryPath">The directory path.</param>
        /// <param name="patterns">The patterns to look for.</param>
        /// <param name="fileExtensions">The file extensions to include.</param>
        /// <param name="excludeDirs">The directories to exclude.</param>
        /// <param name="includeLineNumbers">Whether to include line numbers.</param>
        /// <param name="includeColumnNumbers">Whether to include column numbers.</param>
        /// <param name="includeContext">Whether to include context.</param>
        /// <param name="contextLines">The number of context lines to include.</param>
        /// <param name="minConfidence">The minimum confidence threshold.</param>
        /// <param name="maxMatches">The maximum number of matches to return.</param>
        /// <returns>The pattern matches.</returns>
        public IReadOnlyList<PatternMatchAdapter> FindPatternsInDirectory(
            string directoryPath,
            IEnumerable<CodePatternAdapter> patterns,
            IEnumerable<string> fileExtensions,
            IEnumerable<string> excludeDirs,
            bool includeLineNumbers = true,
            bool includeColumnNumbers = true,
            bool includeContext = true,
            int contextLines = 2,
            double minConfidence = 0.7,
            int? maxMatches = null)
        {
            try
            {
                _logger.LogInformation("Finding patterns in directory {DirectoryPath}", directoryPath);

                // Convert C# patterns to F# patterns
                var fsharpPatterns = patterns.Select(p => p.FSharpPattern).ToList();

                // Create options
                var options = new PatternMatching.PatternMatchOptions(
                    includeLineNumbers,
                    includeColumnNumbers,
                    includeContext,
                    contextLines,
                    minConfidence,
                    maxMatches.HasValue ? Microsoft.FSharp.Core.FSharpOption<int>.Some(maxMatches.Value) : Microsoft.FSharp.Core.FSharpOption<int>.None);

                // Find patterns
                var matches = PatternMatching.findPatternsInDirectory(
                    directoryPath,
                    fsharpPatterns,
                    options,
                    fileExtensions.ToList(),
                    excludeDirs.ToList());

                // Convert F# matches to C# matches
                return matches.Select(m => new PatternMatchAdapter(m)).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error finding patterns in directory {DirectoryPath}", directoryPath);
                return Array.Empty<PatternMatchAdapter>();
            }
        }

        /// <summary>
        /// Calculates the similarity between two strings.
        /// </summary>
        /// <param name="source">The source string.</param>
        /// <param name="target">The target string.</param>
        /// <returns>The similarity score (0-1).</returns>
        public double CalculateSimilarity(string source, string target)
        {
            try
            {
                return PatternMatching.calculateSimilarity(source, target);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error calculating similarity");
                return 0.0;
            }
        }

        /// <summary>
        /// Finds similar patterns.
        /// </summary>
        /// <param name="content">The content to analyze.</param>
        /// <param name="patterns">The patterns to look for.</param>
        /// <param name="language">The language of the content.</param>
        /// <param name="minSimilarity">The minimum similarity threshold.</param>
        /// <param name="maxResults">The maximum number of results to return.</param>
        /// <returns>The similar patterns with their similarity scores.</returns>
        public IReadOnlyList<(CodePatternAdapter Pattern, double Similarity)> FindSimilarPatterns(
            string content,
            IEnumerable<CodePatternAdapter> patterns,
            string language,
            double minSimilarity = 0.7,
            int maxResults = 10)
        {
            try
            {
                _logger.LogInformation("Finding similar patterns in {Language} content", language);

                // Convert C# patterns to F# patterns
                var fsharpPatterns = patterns.Select(p => p.FSharpPattern).ToList();

                // Find similar patterns
                var similarPatterns = PatternMatching.findSimilarPatterns(content, fsharpPatterns, language, minSimilarity, maxResults);

                // Convert F# patterns to C# patterns
                return similarPatterns.Select(p => (new CodePatternAdapter(p.Item1), p.Item2)).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error finding similar patterns in {Language} content", language);
                return Array.Empty<(CodePatternAdapter, double)>();
            }
        }
    }
}
