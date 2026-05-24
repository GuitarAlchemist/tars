using System;
using System.Collections.Generic;
using Microsoft.FSharp.Collections;
using Microsoft.FSharp.Core;

namespace TarsEngine.FSharp.Adapters.TreeOfThought
{
    /// <summary>
    /// Adapter for F# ThoughtNode type.
    /// </summary>
    public class ThoughtNodeAdapter
    {
        private readonly FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode _fsharpNode;

        /// <summary>
        /// Initializes a new instance of the <see cref="ThoughtNodeAdapter"/> class.
        /// </summary>
        /// <param name="fsharpNode">The F# ThoughtNode.</param>
        public ThoughtNodeAdapter(FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode fsharpNode)
        {
            _fsharpNode = fsharpNode ?? throw new ArgumentNullException(nameof(fsharpNode));
        }

        /// <summary>
        /// Gets the thought content.
        /// </summary>
        public string Thought => _fsharpNode.Thought;

        /// <summary>
        /// Gets the child nodes.
        /// </summary>
        public IReadOnlyList<ThoughtNodeAdapter> Children
        {
            get
            {
                var children = new List<ThoughtNodeAdapter>();
                foreach (var child in _fsharpNode.Children)
                {
                    children.Add(new ThoughtNodeAdapter(child));
                }
                return children;
            }
        }

        /// <summary>
        /// Gets the evaluation metrics.
        /// </summary>
        public EvaluationMetricsAdapter? Evaluation
        {
            get
            {
                if (_fsharpNode.Evaluation.IsNone())
                {
                    return null;
                }

                var metrics = _fsharpNode.Evaluation.Value;
                return new EvaluationMetricsAdapter(metrics);
            }
        }

        /// <summary>
        /// Gets a value indicating whether the node has been pruned.
        /// </summary>
        public bool Pruned => _fsharpNode.Pruned;

        /// <summary>
        /// Gets the metadata.
        /// </summary>
        public IReadOnlyDictionary<string, object> Metadata
        {
            get
            {
                var metadata = new Dictionary<string, object>();
                foreach (var kvp in _fsharpNode.Metadata)
                {
                    metadata.Add(kvp.Key, kvp.Value);
                }
                return metadata;
            }
        }

        /// <summary>
        /// Gets the F# ThoughtNode.
        /// </summary>
        public FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode FSharpNode => _fsharpNode;

        /// <summary>
        /// Creates a new thought node.
        /// </summary>
        /// <param name="thought">The thought content.</param>
        /// <returns>The thought node adapter.</returns>
        public static ThoughtNodeAdapter CreateNode(string thought)
        {
            var fsharpNode = FSharp.Core.TreeOfThought.ThoughtNode.createNode(thought);
            return new ThoughtNodeAdapter(fsharpNode);
        }

        /// <summary>
        /// Adds a child to a node.
        /// </summary>
        /// <param name="child">The child node.</param>
        /// <returns>The updated node adapter.</returns>
        public ThoughtNodeAdapter AddChild(ThoughtNodeAdapter child)
        {
            var updatedNode = FSharp.Core.TreeOfThought.ThoughtNode.addChild(_fsharpNode, child.FSharpNode);
            return new ThoughtNodeAdapter(updatedNode);
        }

        /// <summary>
        /// Adds multiple children to a node.
        /// </summary>
        /// <param name="children">The child nodes.</param>
        /// <returns>The updated node adapter.</returns>
        public ThoughtNodeAdapter AddChildren(IEnumerable<ThoughtNodeAdapter> children)
        {
            var fsharpChildren = new List<FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode>();
            foreach (var child in children)
            {
                fsharpChildren.Add(child.FSharpNode);
            }

            var updatedNode = FSharp.Core.TreeOfThought.ThoughtNode.addChildren(
                _fsharpNode, 
                ListModule.OfSeq(fsharpChildren));
            
            return new ThoughtNodeAdapter(updatedNode);
        }

        /// <summary>
        /// Evaluates a node with metrics.
        /// </summary>
        /// <param name="metrics">The evaluation metrics.</param>
        /// <returns>The updated node adapter.</returns>
        public ThoughtNodeAdapter EvaluateNode(EvaluationMetricsAdapter metrics)
        {
            var updatedNode = FSharp.Core.TreeOfThought.ThoughtNode.evaluateNode(_fsharpNode, metrics.FSharpMetrics);
            return new ThoughtNodeAdapter(updatedNode);
        }

        /// <summary>
        /// Marks a node as pruned.
        /// </summary>
        /// <returns>The updated node adapter.</returns>
        public ThoughtNodeAdapter PruneNode()
        {
            var updatedNode = FSharp.Core.TreeOfThought.ThoughtNode.pruneNode(_fsharpNode);
            return new ThoughtNodeAdapter(updatedNode);
        }

        /// <summary>
        /// Adds metadata to a node.
        /// </summary>
        /// <param name="key">The metadata key.</param>
        /// <param name="value">The metadata value.</param>
        /// <returns>The updated node adapter.</returns>
        public ThoughtNodeAdapter AddMetadata(string key, object value)
        {
            var updatedNode = FSharp.Core.TreeOfThought.ThoughtNode.addMetadata(_fsharpNode, key, value);
            return new ThoughtNodeAdapter(updatedNode);
        }

        /// <summary>
        /// Gets metadata from a node.
        /// </summary>
        /// <typeparam name="T">The metadata type.</typeparam>
        /// <param name="key">The metadata key.</param>
        /// <returns>The metadata value, or default if not found.</returns>
        public T? GetMetadata<T>(string key)
        {
            var option = FSharp.Core.TreeOfThought.ThoughtNode.getMetadata<T>(_fsharpNode, key);
            if (option.IsNone())
            {
                return default;
            }

            return option.Value;
        }

        /// <summary>
        /// Gets the score of a node, or 0.0 if not evaluated.
        /// </summary>
        /// <returns>The node score.</returns>
        public double GetScore()
        {
            return FSharp.Core.TreeOfThought.ThoughtNode.getScore(_fsharpNode);
        }
    }
}
