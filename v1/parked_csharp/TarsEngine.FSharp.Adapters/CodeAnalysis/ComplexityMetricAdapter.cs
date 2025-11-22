using System;
using TarsEngine.FSharp.Core.CodeAnalysis;

namespace TarsEngine.FSharp.Adapters.CodeAnalysis
{
    /// <summary>
    /// Adapter for F# ComplexityMetric type.
    /// </summary>
    public class ComplexityMetricAdapter
    {
        private readonly ComplexityAnalysis.ComplexityMetric _fsharpMetric;

        /// <summary>
        /// Initializes a new instance of the <see cref="ComplexityMetricAdapter"/> class.
        /// </summary>
        /// <param name="fsharpMetric">The F# ComplexityMetric.</param>
        public ComplexityMetricAdapter(ComplexityAnalysis.ComplexityMetric fsharpMetric)
        {
            _fsharpMetric = fsharpMetric ?? throw new ArgumentNullException(nameof(fsharpMetric));
        }

        /// <summary>
        /// Gets the name of the metric.
        /// </summary>
        public string Name => _fsharpMetric.Name;

        /// <summary>
        /// Gets the value of the metric.
        /// </summary>
        public double Value => _fsharpMetric.Value;

        /// <summary>
        /// Gets the file path.
        /// </summary>
        public string FilePath => _fsharpMetric.FilePath;

        /// <summary>
        /// Gets the structure name (method, class, etc.).
        /// </summary>
        public string StructureName => _fsharpMetric.StructureName;

        /// <summary>
        /// Gets the structure type (method, class, etc.).
        /// </summary>
        public string StructureType => _fsharpMetric.StructureType;

        /// <summary>
        /// Gets the location of the structure.
        /// </summary>
        public CodeLocationAdapter Location => new CodeLocationAdapter(_fsharpMetric.Location);

        /// <summary>
        /// Gets the threshold for the metric.
        /// </summary>
        public double? Threshold => _fsharpMetric.Threshold.IsSome() ? _fsharpMetric.Threshold.Value : null;

        /// <summary>
        /// Gets a value indicating whether the metric exceeds the threshold.
        /// </summary>
        public bool ExceedsThreshold => _fsharpMetric.ExceedsThreshold;

        /// <summary>
        /// Gets the F# ComplexityMetric.
        /// </summary>
        public ComplexityAnalysis.ComplexityMetric FSharpMetric => _fsharpMetric;
    }
}
