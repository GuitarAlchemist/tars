using System;

namespace TarsEngine.FSharp.Adapters.CodeAnalysis
{
    /// <summary>
    /// Adapter for F# Match type.
    /// </summary>
    public class MatchAdapter
    {
        private readonly FSharp.Core.CodeAnalysis.Types.Match _fsharpMatch;

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchAdapter"/> class.
        /// </summary>
        /// <param name="fsharpMatch">The F# Match.</param>
        public MatchAdapter(FSharp.Core.CodeAnalysis.Types.Match fsharpMatch)
        {
            _fsharpMatch = fsharpMatch ?? throw new ArgumentNullException(nameof(fsharpMatch));
        }

        /// <summary>
        /// Gets the pattern that matched.
        /// </summary>
        public PatternAdapter Pattern => new PatternAdapter(_fsharpMatch.Pattern);

        /// <summary>
        /// Gets the matched text.
        /// </summary>
        public string Text => _fsharpMatch.Text;

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
        /// Gets the F# Match.
        /// </summary>
        public FSharp.Core.CodeAnalysis.Types.Match FSharpMatch => _fsharpMatch;
    }
}
