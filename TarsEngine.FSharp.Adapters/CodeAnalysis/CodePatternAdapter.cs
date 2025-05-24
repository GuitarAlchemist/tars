using System;
using System.Collections.Generic;
using System.Linq;
using TarsEngine.FSharp.Core.CodeAnalysis;

namespace TarsEngine.FSharp.Adapters.CodeAnalysis
{
    /// <summary>
    /// Adapter for F# CodePattern type.
    /// </summary>
    public class CodePatternAdapter
    {
        private readonly PatternMatching.CodePattern _fsharpPattern;

        /// <summary>
        /// Initializes a new instance of the <see cref="CodePatternAdapter"/> class.
        /// </summary>
        /// <param name="fsharpPattern">The F# CodePattern.</param>
        public CodePatternAdapter(PatternMatching.CodePattern fsharpPattern)
        {
            _fsharpPattern = fsharpPattern ?? throw new ArgumentNullException(nameof(fsharpPattern));
        }

        /// <summary>
        /// Gets the ID of the pattern.
        /// </summary>
        public string Id => _fsharpPattern.Id;

        /// <summary>
        /// Gets the name of the pattern.
        /// </summary>
        public string Name => _fsharpPattern.Name;

        /// <summary>
        /// Gets the description of the pattern.
        /// </summary>
        public string Description => _fsharpPattern.Description;

        /// <summary>
        /// Gets the language the pattern applies to.
        /// </summary>
        public string Language => _fsharpPattern.Language;

        /// <summary>
        /// Gets the pattern to match.
        /// </summary>
        public string Pattern => _fsharpPattern.Pattern;

        /// <summary>
        /// Gets the pattern language (Regex, Literal, AST, etc.).
        /// </summary>
        public string PatternLanguage => _fsharpPattern.PatternLanguage;

        /// <summary>
        /// Gets the replacement pattern (if applicable).
        /// </summary>
        public string? Replacement => _fsharpPattern.Replacement.IsSome() ? _fsharpPattern.Replacement.Value : null;

        /// <summary>
        /// Gets the explanation for the replacement.
        /// </summary>
        public string? ReplacementExplanation => _fsharpPattern.ReplacementExplanation.IsSome() ? _fsharpPattern.ReplacementExplanation.Value : null;

        /// <summary>
        /// Gets the expected improvement from applying the pattern.
        /// </summary>
        public string? ExpectedImprovement => _fsharpPattern.ExpectedImprovement.IsSome() ? _fsharpPattern.ExpectedImprovement.Value : null;

        /// <summary>
        /// Gets the severity of the pattern (0-1).
        /// </summary>
        public double Severity => _fsharpPattern.Severity;

        /// <summary>
        /// Gets the confidence threshold for the pattern.
        /// </summary>
        public double ConfidenceThreshold => _fsharpPattern.ConfidenceThreshold;

        /// <summary>
        /// Gets the impact score of the pattern.
        /// </summary>
        public double ImpactScore => _fsharpPattern.ImpactScore;

        /// <summary>
        /// Gets the difficulty score of the pattern.
        /// </summary>
        public double DifficultyScore => _fsharpPattern.DifficultyScore;

        /// <summary>
        /// Gets the tags associated with the pattern.
        /// </summary>
        public IReadOnlyList<string> Tags => _fsharpPattern.Tags.ToList();

        /// <summary>
        /// Gets the F# CodePattern.
        /// </summary>
        public PatternMatching.CodePattern FSharpPattern => _fsharpPattern;

        /// <summary>
        /// Creates a new code pattern.
        /// </summary>
        /// <param name="id">The ID of the pattern.</param>
        /// <param name="name">The name of the pattern.</param>
        /// <param name="description">The description of the pattern.</param>
        /// <param name="language">The language the pattern applies to.</param>
        /// <param name="pattern">The pattern to match.</param>
        /// <param name="patternLanguage">The pattern language (Regex, Literal, AST, etc.).</param>
        /// <param name="replacement">The replacement pattern (if applicable).</param>
        /// <param name="replacementExplanation">The explanation for the replacement.</param>
        /// <param name="expectedImprovement">The expected improvement from applying the pattern.</param>
        /// <param name="severity">The severity of the pattern (0-1).</param>
        /// <param name="confidenceThreshold">The confidence threshold for the pattern.</param>
        /// <param name="impactScore">The impact score of the pattern.</param>
        /// <param name="difficultyScore">The difficulty score of the pattern.</param>
        /// <param name="tags">The tags associated with the pattern.</param>
        /// <returns>The code pattern adapter.</returns>
        public static CodePatternAdapter CreatePattern(
            string id,
            string name,
            string description,
            string language,
            string pattern,
            string patternLanguage,
            string? replacement = null,
            string? replacementExplanation = null,
            string? expectedImprovement = null,
            double severity = 0.5,
            double confidenceThreshold = 0.7,
            double impactScore = 0.5,
            double difficultyScore = 0.5,
            IEnumerable<string>? tags = null)
        {
            var fsharpPattern = new PatternMatching.CodePattern(
                id,
                name,
                description,
                language,
                pattern,
                patternLanguage,
                replacement != null ? Microsoft.FSharp.Core.FSharpOption<string>.Some(replacement) : Microsoft.FSharp.Core.FSharpOption<string>.None,
                replacementExplanation != null ? Microsoft.FSharp.Core.FSharpOption<string>.Some(replacementExplanation) : Microsoft.FSharp.Core.FSharpOption<string>.None,
                expectedImprovement != null ? Microsoft.FSharp.Core.FSharpOption<string>.Some(expectedImprovement) : Microsoft.FSharp.Core.FSharpOption<string>.None,
                severity,
                confidenceThreshold,
                impactScore,
                difficultyScore,
                tags?.ToList() ?? new List<string>());

            return new CodePatternAdapter(fsharpPattern);
        }
    }
}
