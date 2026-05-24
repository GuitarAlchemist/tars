using System;
using TarsEngine.FSharp.Core.CodeAnalysis;

namespace TarsEngine.FSharp.Adapters.CodeAnalysis
{
    /// <summary>
    /// Adapter for F# PatternMatch type.
    /// </summary>
    public class PatternMatchAdapter
    {
        private readonly PatternMatching.PatternMatch _fsharpMatch;

        /// <summary>
        /// Initializes a new instance of the <see cref="PatternMatchAdapter"/> class.
        /// </summary>
        /// <param name="fsharpMatch">The F# PatternMatch.</param>
        public PatternMatchAdapter(PatternMatching.PatternMatch fsharpMatch)
        {
            _fsharpMatch = fsharpMatch ?? throw new ArgumentNullException(nameof(fsharpMatch));
        }

        /// <summary>
        /// Gets the pattern that matched.
        /// </summary>
        public CodePatternAdapter Pattern => new CodePatternAdapter(_fsharpMatch.Pattern);

        /// <summary>
        /// Gets the matched text.
        /// </summary>
        public string MatchedText => _fsharpMatch.MatchedText;

        /// <summary>
        /// Gets the line number where the match was found.
        /// </summary>
        public int LineNumber => _fsharpMatch.LineNumber;

        /// <summary>
        /// Gets the column number where the match was found.
        /// </summary>
        public int ColumnNumber => _fsharpMatch.ColumnNumber;

        /// <summary>
        /// Gets the file path where the match was found.
        /// </summary>
        public string FilePath => _fsharpMatch.FilePath;

        /// <summary>
        /// Gets the context around the match.
        /// </summary>
        public string Context => _fsharpMatch.Context;

        /// <summary>
        /// Gets the confidence of the match.
        /// </summary>
        public double Confidence => _fsharpMatch.Confidence;

        /// <summary>
        /// Gets the F# PatternMatch.
        /// </summary>
        public PatternMatching.PatternMatch FSharpMatch => _fsharpMatch;
    }
}
