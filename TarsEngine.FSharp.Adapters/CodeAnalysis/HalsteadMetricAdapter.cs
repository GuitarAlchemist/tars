using System;
using TarsEngine.FSharp.Core.CodeAnalysis;

namespace TarsEngine.FSharp.Adapters.CodeAnalysis
{
    /// <summary>
    /// Adapter for F# HalsteadMetric type.
    /// </summary>
    public class HalsteadMetricAdapter
    {
        private readonly ComplexityAnalysis.HalsteadMetric _fsharpMetric;

        /// <summary>
        /// Initializes a new instance of the <see cref="HalsteadMetricAdapter"/> class.
        /// </summary>
        /// <param name="fsharpMetric">The F# HalsteadMetric.</param>
        public HalsteadMetricAdapter(ComplexityAnalysis.HalsteadMetric fsharpMetric)
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
        /// Gets the number of unique operators.
        /// </summary>
        public int UniqueOperators => _fsharpMetric.UniqueOperators;

        /// <summary>
        /// Gets the number of unique operands.
        /// </summary>
        public int UniqueOperands => _fsharpMetric.UniqueOperands;

        /// <summary>
        /// Gets the total number of operators.
        /// </summary>
        public int TotalOperators => _fsharpMetric.TotalOperators;

        /// <summary>
        /// Gets the total number of operands.
        /// </summary>
        public int TotalOperands => _fsharpMetric.TotalOperands;

        /// <summary>
        /// Gets the F# HalsteadMetric.
        /// </summary>
        public ComplexityAnalysis.HalsteadMetric FSharpMetric => _fsharpMetric;
    }
}
