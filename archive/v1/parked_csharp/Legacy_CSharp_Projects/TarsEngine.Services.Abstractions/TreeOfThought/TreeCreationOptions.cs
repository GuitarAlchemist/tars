using System.Collections.Generic;

namespace TarsEngine.Services.Abstractions.TreeOfThought
{
    /// <summary>
    /// Options for creating a thought tree.
    /// </summary>
    public class TreeCreationOptions
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TreeCreationOptions"/> class.
        /// </summary>
        public TreeCreationOptions()
        {
            Approaches = new List<string>();
            ApproachEvaluations = new Dictionary<string, EvaluationMetrics>();
        }

        /// <summary>
        /// Gets or sets the approaches to consider.
        /// </summary>
        public List<string> Approaches { get; set; }

        /// <summary>
        /// Gets or sets the evaluations for each approach.
        /// </summary>
        public Dictionary<string, EvaluationMetrics> ApproachEvaluations { get; set; }

        /// <summary>
        /// Gets or sets the maximum depth of the tree.
        /// </summary>
        public int MaxDepth { get; set; } = 3;

        /// <summary>
        /// Gets or sets the maximum breadth of the tree.
        /// </summary>
        public int MaxBreadth { get; set; } = 3;

        /// <summary>
        /// Gets or sets the evaluation threshold.
        /// </summary>
        public double EvaluationThreshold { get; set; } = 0.5;

        /// <summary>
        /// Gets or sets a value indicating whether to use beam search.
        /// </summary>
        public bool UseBeamSearch { get; set; } = false;

        /// <summary>
        /// Gets or sets the beam width.
        /// </summary>
        public int BeamWidth { get; set; } = 2;
    }
}
