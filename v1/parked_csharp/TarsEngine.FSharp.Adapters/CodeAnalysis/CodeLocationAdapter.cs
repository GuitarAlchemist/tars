using System;
using TarsEngine.FSharp.Core.CodeAnalysis;

namespace TarsEngine.FSharp.Adapters.CodeAnalysis
{
    /// <summary>
    /// Adapter for F# CodeLocation type.
    /// </summary>
    public class CodeLocationAdapter
    {
        private readonly Types.CodeLocation _fsharpLocation;

        /// <summary>
        /// Initializes a new instance of the <see cref="CodeLocationAdapter"/> class.
        /// </summary>
        /// <param name="fsharpLocation">The F# CodeLocation.</param>
        public CodeLocationAdapter(Types.CodeLocation fsharpLocation)
        {
            _fsharpLocation = fsharpLocation;
        }

        /// <summary>
        /// Gets the start offset.
        /// </summary>
        public int StartOffset => _fsharpLocation.StartOffset;

        /// <summary>
        /// Gets the end offset.
        /// </summary>
        public int EndOffset => _fsharpLocation.EndOffset;

        /// <summary>
        /// Gets the start line.
        /// </summary>
        public int StartLine => _fsharpLocation.StartLine;

        /// <summary>
        /// Gets the end line.
        /// </summary>
        public int EndLine => _fsharpLocation.EndLine;

        /// <summary>
        /// Gets the F# CodeLocation.
        /// </summary>
        public Types.CodeLocation FSharpLocation => _fsharpLocation;
    }
}
