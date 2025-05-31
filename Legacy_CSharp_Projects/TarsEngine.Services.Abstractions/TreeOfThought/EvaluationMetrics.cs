namespace TarsEngine.Services.Abstractions.TreeOfThought
{
    /// <summary>
    /// Represents evaluation metrics for thought nodes.
    /// </summary>
    public class EvaluationMetrics
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EvaluationMetrics"/> class.
        /// </summary>
        /// <param name="correctness">The correctness score.</param>
        /// <param name="efficiency">The efficiency score.</param>
        /// <param name="robustness">The robustness score.</param>
        /// <param name="maintainability">The maintainability score.</param>
        /// <param name="overall">The overall score.</param>
        public EvaluationMetrics(double correctness, double efficiency, double robustness, double maintainability, double overall)
        {
            Correctness = correctness;
            Efficiency = efficiency;
            Robustness = robustness;
            Maintainability = maintainability;
            Overall = overall;
        }

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
    }
}
