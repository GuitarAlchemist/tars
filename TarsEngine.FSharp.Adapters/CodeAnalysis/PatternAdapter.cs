using System;
using System.Collections.Generic;

namespace TarsEngine.FSharp.Adapters.CodeAnalysis
{
    /// <summary>
    /// Adapter for F# Pattern type.
    /// </summary>
    public class PatternAdapter
    {
        private readonly FSharp.Core.CodeAnalysis.Types.Pattern _fsharpPattern;

        /// <summary>
        /// Initializes a new instance of the <see cref="PatternAdapter"/> class.
        /// </summary>
        /// <param name="fsharpPattern">The F# Pattern.</param>
        public PatternAdapter(FSharp.Core.CodeAnalysis.Types.Pattern fsharpPattern)
        {
            _fsharpPattern = fsharpPattern ?? throw new ArgumentNullException(nameof(fsharpPattern));
        }

        /// <summary>
        /// Gets the name of the pattern.
        /// </summary>
        public string Name => _fsharpPattern.Name;

        /// <summary>
        /// Gets the description of the pattern.
        /// </summary>
        public string Description => _fsharpPattern.Description;

        /// <summary>
        /// Gets the regular expression pattern.
        /// </summary>
        public string Regex => _fsharpPattern.Regex;

        /// <summary>
        /// Gets the severity of the pattern (0-1).
        /// </summary>
        public double Severity => _fsharpPattern.Severity;

        /// <summary>
        /// Gets the category of the pattern.
        /// </summary>
        public string Category => _fsharpPattern.Category;

        /// <summary>
        /// Gets the language the pattern applies to.
        /// </summary>
        public string Language => _fsharpPattern.Language;

        /// <summary>
        /// Gets the F# Pattern.
        /// </summary>
        public FSharp.Core.CodeAnalysis.Types.Pattern FSharpPattern => _fsharpPattern;

        /// <summary>
        /// Creates a new pattern.
        /// </summary>
        /// <param name="name">The name of the pattern.</param>
        /// <param name="description">The description of the pattern.</param>
        /// <param name="regex">The regular expression pattern.</param>
        /// <param name="severity">The severity of the pattern (0-1).</param>
        /// <param name="category">The category of the pattern.</param>
        /// <param name="language">The language the pattern applies to.</param>
        /// <returns>The pattern adapter.</returns>
        public static PatternAdapter CreatePattern(string name, string description, string regex, double severity, string category, string language)
        {
            var fsharpPattern = FSharp.Core.CodeAnalysis.CodeAnalyzer.createPattern(name, description, regex, severity, category, language);
            return new PatternAdapter(fsharpPattern);
        }
    }
}
