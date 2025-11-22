using System.Collections.Generic;

namespace TarsEngine.Services.Abstractions.TreeOfThought
{
    /// <summary>
    /// Interface for a node in a thought tree.
    /// </summary>
    public interface IThoughtNode
    {
        /// <summary>
        /// Gets the thought content.
        /// </summary>
        string Thought { get; }

        /// <summary>
        /// Gets the child nodes.
        /// </summary>
        IReadOnlyList<IThoughtNode> Children { get; }

        /// <summary>
        /// Gets the evaluation metrics.
        /// </summary>
        EvaluationMetrics? Evaluation { get; }

        /// <summary>
        /// Gets a value indicating whether the node has been pruned.
        /// </summary>
        bool Pruned { get; }

        /// <summary>
        /// Gets the metadata.
        /// </summary>
        IReadOnlyDictionary<string, object> Metadata { get; }

        /// <summary>
        /// Gets the score of the node.
        /// </summary>
        double Score { get; }
    }
}
