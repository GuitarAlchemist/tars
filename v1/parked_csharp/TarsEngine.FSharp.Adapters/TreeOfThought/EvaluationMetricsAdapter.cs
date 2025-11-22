using System;

namespace TarsEngine.FSharp.Adapters.TreeOfThought
{
    /// <summary>
    /// Adapter for F# EvaluationMetrics type.
    /// </summary>
    public class EvaluationMetricsAdapter
    {
        private readonly FSharp.Core.TreeOfThought.ThoughtNode.EvaluationMetrics _fsharpMetrics;

        /// <summary>
        /// Initializes a new instance of the <see cref="EvaluationMetricsAdapter"/> class.
        /// </summary>
        /// <param name="fsharpMetrics">The F# EvaluationMetrics.</param>
        public EvaluationMetricsAdapter(FSharp.Core.TreeOfThought.ThoughtNode.EvaluationMetrics fsharpMetrics)
        {
            _fsharpMetrics = fsharpMetrics;
        }

        /// <summary>
        /// Gets the correctness score.
        /// </summary>
        public double Correctness => _fsharpMetrics.Correctness;

        /// <summary>
        /// Gets the efficiency score.
        /// </summary>
        public double Efficiency => _fsharpMetrics.Efficiency;

        /// <summary>
        /// Gets the robustness score.
        /// </summary>
        public double Robustness => _fsharpMetrics.Robustness;

        /// <summary>
        /// Gets the maintainability score.
        /// </summary>
        public double Maintainability => _fsharpMetrics.Maintainability;

        /// <summary>
        /// Gets the overall score.
        /// </summary>
        public double Overall => _fsharpMetrics.Overall;

        /// <summary>
        /// Gets the F# EvaluationMetrics.
        /// </summary>
        public FSharp.Core.TreeOfThought.ThoughtNode.EvaluationMetrics FSharpMetrics => _fsharpMetrics;

        /// <summary>
        /// Creates evaluation metrics with equal weights.
        /// </summary>
        /// <param name="correctness">The correctness score.</param>
        /// <param name="efficiency">The efficiency score.</param>
        /// <param name="robustness">The robustness score.</param>
        /// <param name="maintainability">The maintainability score.</param>
        /// <returns>The evaluation metrics adapter.</returns>
        public static EvaluationMetricsAdapter CreateMetrics(double correctness, double efficiency, double robustness, double maintainability)
        {
            var fsharpMetrics = FSharp.Core.TreeOfThought.ThoughtNode.createMetrics(correctness, efficiency, robustness, maintainability);
            return new EvaluationMetricsAdapter(fsharpMetrics);
        }

        /// <summary>
        /// Creates evaluation metrics with weighted average.
        /// </summary>
        /// <param name="correctness">The correctness score.</param>
        /// <param name="efficiency">The efficiency score.</param>
        /// <param name="robustness">The robustness score.</param>
        /// <param name="maintainability">The maintainability score.</param>
        /// <param name="weights">The weights for each metric (correctness, efficiency, robustness, maintainability).</param>
        /// <returns>The evaluation metrics adapter.</returns>
        public static EvaluationMetricsAdapter CreateWeightedMetrics(double correctness, double efficiency, double robustness, double maintainability, (double, double, double, double) weights)
        {
            var fsharpMetrics = FSharp.Core.TreeOfThought.ThoughtNode.createWeightedMetrics(correctness, efficiency, robustness, maintainability, weights);
            return new EvaluationMetricsAdapter(fsharpMetrics);
        }
    }
}
