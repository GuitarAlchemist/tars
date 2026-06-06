using System;
using System.Collections.Generic;
using System.Linq;
using TarsEngine.FSharp.Core.TreeOfThought;
using Microsoft.FSharp.Core;

namespace TarsEngine.CSharp.Adapters.TreeOfThought
{
    /// <summary>
    /// Adapter for F# ThoughtNode.
    /// </summary>
    public class ThoughtNodeAdapter
    {
        /// <summary>
        /// Gets the thought content.
        /// </summary>
        public string Thought { get; }

        /// <summary>
        /// Gets the child nodes.
        /// </summary>
        public IReadOnlyList<ThoughtNodeAdapter> Children { get; }

        /// <summary>
        /// Gets the evaluation metrics.
        /// </summary>
        public EvaluationMetricsAdapter? Evaluation { get; }

        /// <summary>
        /// Gets a value indicating whether the node has been pruned.
        /// </summary>
        public bool Pruned { get; }

        /// <summary>
        /// Gets the metadata.
        /// </summary>
        public IReadOnlyDictionary<string, object> Metadata { get; }

        /// <summary>
        /// Gets the score of the node.
        /// </summary>
        public double Score { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ThoughtNodeAdapter"/> class.
        /// </summary>
        /// <param name="thought">The thought content.</param>
        /// <param name="children">The child nodes.</param>
        /// <param name="evaluation">The evaluation metrics.</param>
        /// <param name="pruned">A value indicating whether the node has been pruned.</param>
        /// <param name="metadata">The metadata.</param>
        /// <param name="score">The score of the node.</param>
        public ThoughtNodeAdapter(
            string thought,
            IReadOnlyList<ThoughtNodeAdapter> children,
            EvaluationMetricsAdapter? evaluation,
            bool pruned,
            IReadOnlyDictionary<string, object> metadata,
            double score)
        {
            Thought = thought ?? throw new ArgumentNullException(nameof(thought));
            Children = children ?? Array.Empty<ThoughtNodeAdapter>();
            Evaluation = evaluation;
            Pruned = pruned;
            Metadata = metadata ?? new Dictionary<string, object>();
            Score = score;
        }

        /// <summary>
        /// Creates a new node with the specified thought.
        /// </summary>
        /// <param name="thought">The thought content.</param>
        /// <returns>A new node.</returns>
        public static ThoughtNodeAdapter CreateNode(string thought)
        {
            return new ThoughtNodeAdapter(
                thought,
                Array.Empty<ThoughtNodeAdapter>(),
                null,
                false,
                new Dictionary<string, object>(),
                0.0);
        }

        /// <summary>
        /// Adds a child to the node.
        /// </summary>
        /// <param name="child">The child node.</param>
        /// <returns>The updated node.</returns>
        public ThoughtNodeAdapter AddChild(ThoughtNodeAdapter child)
        {
            if (child == null)
            {
                throw new ArgumentNullException(nameof(child));
            }

            var newChildren = new List<ThoughtNodeAdapter>(Children) { child };
            return new ThoughtNodeAdapter(
                Thought,
                newChildren,
                Evaluation,
                Pruned,
                Metadata,
                Score);
        }

        /// <summary>
        /// Adds multiple children to the node.
        /// </summary>
        /// <param name="children">The child nodes.</param>
        /// <returns>The updated node.</returns>
        public ThoughtNodeAdapter AddChildren(IEnumerable<ThoughtNodeAdapter> children)
        {
            if (children == null)
            {
                throw new ArgumentNullException(nameof(children));
            }

            var newChildren = new List<ThoughtNodeAdapter>(Children);
            newChildren.AddRange(children);
            return new ThoughtNodeAdapter(
                newChildren,
                Evaluation,
                Pruned,
                Metadata,
                Score);
        }

        /// <summary>
        /// Evaluates the node with the specified metrics.
        /// </summary>
        /// <param name="metrics">The evaluation metrics.</param>
        /// <returns>The evaluated node.</returns>
        public ThoughtNodeAdapter EvaluateNode(EvaluationMetricsAdapter metrics)
        {
            if (metrics == null)
            {
                throw new ArgumentNullException(nameof(metrics));
            }

            return new ThoughtNodeAdapter(
                Thought,
                Children,
                metrics,
                Pruned,
                Metadata,
                metrics.Overall);
        }

        /// <summary>
        /// Prunes the node.
        /// </summary>
        /// <returns>The pruned node.</returns>
        public ThoughtNodeAdapter PruneNode()
        {
            return new ThoughtNodeAdapter(
                Thought,
                Children,
                Evaluation,
                true,
                Metadata,
                Score);
        }

        /// <summary>
        /// Adds metadata to the node.
        /// </summary>
        /// <param name="key">The metadata key.</param>
        /// <param name="value">The metadata value.</param>
        /// <returns>The updated node.</returns>
        public ThoughtNodeAdapter AddMetadata(string key, object value)
        {
            if (string.IsNullOrEmpty(key))
            {
                throw new ArgumentException("Key cannot be null or empty.", nameof(key));
            }

            var newMetadata = new Dictionary<string, object>(Metadata)
            {
                [key] = value
            };

            return new ThoughtNodeAdapter(
                Thought,
                Children,
                Evaluation,
                Pruned,
                newMetadata,
                Score);
        }

        /// <summary>
        /// Converts an F# ThoughtNode to a C# ThoughtNodeAdapter.
        /// </summary>
        /// <param name="node">The F# ThoughtNode.</param>
        /// <returns>A C# ThoughtNodeAdapter.</returns>
        public static ThoughtNodeAdapter FromFSharpNode(ThoughtNode node)
        {
            var children = node.Children
                .Select(FromFSharpNode)
                .ToList();

            var evaluation = node.Evaluation.HasValue
                ? EvaluationMetricsAdapter.FromFSharpMetrics(node.Evaluation.Value)
                : null;

            var metadata = node.Metadata
                .ToDictionary(kv => kv.Key, kv => (object)kv.Value);

            return new ThoughtNodeAdapter(
                node.Thought,
                children,
                evaluation,
                node.Pruned,
                metadata,
                node.Score);
        }

        /// <summary>
        /// Converts a C# ThoughtNodeAdapter to an F# ThoughtNode.
        /// </summary>
        /// <returns>An F# ThoughtNode.</returns>
        public ThoughtNode ToFSharpNode()
        {
            var children = Children
                .Select(c => c.ToFSharpNode())
                .ToArray();

            var evaluation = Evaluation != null
                ? FSharpOption<EvaluationMetrics>.Some(Evaluation.ToFSharpMetrics())
                : FSharpOption<EvaluationMetrics>.None;

            var metadata = Metadata
                .ToDictionary(kv => kv.Key, kv => kv.Value);

            return new ThoughtNode(
                Thought,
                children,
                evaluation,
                Pruned,
                metadata,
                Score);
        }
    }
}
