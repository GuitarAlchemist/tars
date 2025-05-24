using System;
using TarsEngine.FSharp.Core.TreeOfThought;

namespace TarsEngine.CSharp.Adapters.TreeOfThought
{
    /// <summary>
    /// Adapter for F# EvaluationMetrics.
    /// </summary>
    public class EvaluationMetricsAdapter
    {
        /// <summary>
        /// Gets the correctness score.
        /// </summary>
        public double Correctness { get; }

        /// <summary>
        /// Gets the efficiency score.
        /// </summary>
        public double Efficiency { get; }

        /// <summary>
        /// Gets the robustness score.
        /// </summary>
        public double Robustness { get; }

        /// <summary>
        /// Gets the maintainability score.
        /// </summary>
        public double Maintainability { get; }

        /// <summary>
        /// Gets the overall score.
        /// </summary>
        public double Overall { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="EvaluationMetricsAdapter"/> class.
        /// </summary>
        /// <param name="correctness">The correctness score.</param>
        /// <param name="efficiency">The efficiency score.</param>
        /// <param name="robustness">The robustness score.</param>
        /// <param name="maintainability">The maintainability score.</param>
        /// <param name="overall">The overall score.</param>
        public EvaluationMetricsAdapter(
            double correctness,
            double efficiency,
            double robustness,
            double maintainability,
            double? overall = null)
        {
            Correctness = correctness;
            Efficiency = efficiency;
            Robustness = robustness;
            Maintainability = maintainability;
            Overall = overall ?? (correctness + efficiency + robustness + maintainability) / 4.0;
        }

        /// <summary>
        /// Creates a new instance of the <see cref="EvaluationMetricsAdapter"/> class.
        /// </summary>
        /// <param name="correctness">The correctness score.</param>
        /// <param name="efficiency">The efficiency score.</param>
        /// <param name="robustness">The robustness score.</param>
        /// <param name="maintainability">The maintainability score.</param>
        /// <param name="overall">The overall score.</param>
        /// <returns>A new instance of the <see cref="EvaluationMetricsAdapter"/> class.</returns>
        public static EvaluationMetricsAdapter CreateMetrics(
            double correctness,
            double efficiency,
            double robustness,
            double maintainability,
            double? overall = null)
        {
            return new EvaluationMetricsAdapter(
                correctness,
                efficiency,
                robustness,
                maintainability,
                overall);
        }

        /// <summary>
        /// Converts an F# EvaluationMetrics to a C# EvaluationMetricsAdapter.
        /// </summary>
        /// <param name="metrics">The F# EvaluationMetrics.</param>
        /// <returns>A C# EvaluationMetricsAdapter.</returns>
        public static EvaluationMetricsAdapter FromFSharpMetrics(EvaluationMetrics metrics)
        {
            return new EvaluationMetricsAdapter(
                metrics.Correctness,
                metrics.Efficiency,
                metrics.Robustness,
                metrics.Maintainability,
                metrics.Overall);
        }

        /// <summary>
        /// Converts a C# EvaluationMetricsAdapter to an F# EvaluationMetrics.
        /// </summary>
        /// <returns>An F# EvaluationMetrics.</returns>
        public EvaluationMetrics ToFSharpMetrics()
        {
            return new EvaluationMetrics(
                Correctness,
                Efficiency,
                Robustness,
                Maintainability,
                Overall);
        }
    }
}
