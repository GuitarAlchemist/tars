using System;
using TarsEngine.FSharp.Core.CodeAnalysis;

namespace TarsEngine.FSharp.Adapters.CodeAnalysis
{
    /// <summary>
    /// Adapter for F# MaintainabilityMetric type.
    /// </summary>
    public class MaintainabilityMetricAdapter
    {
        private readonly ComplexityAnalysis.MaintainabilityMetric _fsharpMetric;

        /// <summary>
        /// Initializes a new instance of the <see cref="MaintainabilityMetricAdapter"/> class.
        /// </summary>
        /// <param name="fsharpMetric">The F# MaintainabilityMetric.</param>
        public MaintainabilityMetricAdapter(ComplexityAnalysis.MaintainabilityMetric fsharpMetric)
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
        /// Gets the rating (Excellent, Good, Fair, Poor).
        /// </summary>
        public string Rating => _fsharpMetric.Rating;

        /// <summary>
        /// Gets the F# MaintainabilityMetric.
        /// </summary>
        public ComplexityAnalysis.MaintainabilityMetric FSharpMetric => _fsharpMetric;
    }
}
