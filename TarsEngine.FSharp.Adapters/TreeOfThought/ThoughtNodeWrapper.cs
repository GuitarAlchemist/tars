using System;
using System.Collections.Generic;
using System.Linq;
using TarsEngine.Services.Abstractions.TreeOfThought;

namespace TarsEngine.FSharp.Adapters.TreeOfThought
{
    /// <summary>
    /// Wrapper for ThoughtNodeAdapter that implements IThoughtNode.
    /// </summary>
    public class ThoughtNodeWrapper : IThoughtNode
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ThoughtNodeWrapper"/> class.
        /// </summary>
        /// <param name="node">The thought node adapter.</param>
        public ThoughtNodeWrapper(ThoughtNodeAdapter node)
        {
            Node = node ?? throw new ArgumentNullException(nameof(node));
        }

        /// <summary>
        /// Gets the thought node adapter.
        /// </summary>
        public ThoughtNodeAdapter Node { get; }

        /// <summary>
        /// Gets the thought content.
        /// </summary>
        public string Thought => Node.Thought;

        /// <summary>
        /// Gets the child nodes.
        /// </summary>
        public IReadOnlyList<IThoughtNode> Children => 
            Node.Children.Select(child => new ThoughtNodeWrapper(child)).ToList();

        /// <summary>
        /// Gets the evaluation metrics.
        /// </summary>
        public EvaluationMetrics? Evaluation => 
            Node.Evaluation != null 
                ? new EvaluationMetrics(
                    Node.Evaluation.Correctness,
                    Node.Evaluation.Efficiency,
                    Node.Evaluation.Robustness,
                    Node.Evaluation.Maintainability,
                    Node.Evaluation.Overall)
                : null;

        /// <summary>
        /// Gets a value indicating whether the node has been pruned.
        /// </summary>
        public bool Pruned => Node.Pruned;

        /// <summary>
        /// Gets the metadata.
        /// </summary>
        public IReadOnlyDictionary<string, object> Metadata => Node.Metadata;

        /// <summary>
        /// Gets the score of the node.
        /// </summary>
        public double Score => Node.GetScore();
    }
}
