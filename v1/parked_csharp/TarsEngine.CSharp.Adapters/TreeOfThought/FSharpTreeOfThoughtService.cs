using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.FSharp.Core.TreeOfThought;

namespace TarsEngine.CSharp.Adapters.TreeOfThought
{
    /// <summary>
    /// Implementation of the ITreeOfThoughtService interface that uses the F# implementation.
    /// </summary>
    public class FSharpTreeOfThoughtService : ITreeOfThoughtService
    {
        private readonly ILogger<FSharpTreeOfThoughtService> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="FSharpTreeOfThoughtService"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public FSharpTreeOfThoughtService(ILogger<FSharpTreeOfThoughtService> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <inheritdoc/>
        public Task<ThoughtNodeAdapter> CreateRootNodeAsync(string thought)
        {
            _logger.LogInformation("Creating root node with thought: {Thought}", thought);
            
            var node = ThoughtNodeAdapter.CreateNode(thought);
            return Task.FromResult(node);
        }

        /// <inheritdoc/>
        public Task<ThoughtNodeAdapter> AddChildNodeAsync(ThoughtNodeAdapter parent, string thought)
        {
            if (parent == null)
            {
                throw new ArgumentNullException(nameof(parent));
            }

            _logger.LogInformation("Adding child node to parent {ParentThought} with thought: {Thought}", parent.Thought, thought);
            
            var child = ThoughtNodeAdapter.CreateNode(thought);
            var updatedParent = parent.AddChild(child);
            return Task.FromResult(updatedParent);
        }

        /// <inheritdoc/>
        public Task<ThoughtNodeAdapter> AddChildNodesAsync(ThoughtNodeAdapter parent, IEnumerable<string> thoughts)
        {
            if (parent == null)
            {
                throw new ArgumentNullException(nameof(parent));
            }

            if (thoughts == null)
            {
                throw new ArgumentNullException(nameof(thoughts));
            }

            _logger.LogInformation("Adding {Count} child nodes to parent {ParentThought}", thoughts.Count(), parent.Thought);
            
            var children = thoughts.Select(ThoughtNodeAdapter.CreateNode).ToList();
            var updatedParent = parent.AddChildren(children);
            return Task.FromResult(updatedParent);
        }

        /// <inheritdoc/>
        public Task<ThoughtNodeAdapter> EvaluateNodeAsync(ThoughtNodeAdapter node, double correctness, double efficiency, double robustness, double maintainability)
        {
            if (node == null)
            {
                throw new ArgumentNullException(nameof(node));
            }

            _logger.LogInformation("Evaluating node {Thought} with scores: Correctness={Correctness}, Efficiency={Efficiency}, Robustness={Robustness}, Maintainability={Maintainability}",
                node.Thought, correctness, efficiency, robustness, maintainability);
            
            var metrics = EvaluationMetricsAdapter.CreateMetrics(correctness, efficiency, robustness, maintainability);
            var updatedNode = node.EvaluateNode(metrics);
            return Task.FromResult(updatedNode);
        }

        /// <inheritdoc/>
        public Task<ThoughtNodeAdapter> PruneNodeAsync(ThoughtNodeAdapter node)
        {
            if (node == null)
            {
                throw new ArgumentNullException(nameof(node));
            }

            _logger.LogInformation("Pruning node {Thought}", node.Thought);
            
            var updatedNode = node.PruneNode();
            return Task.FromResult(updatedNode);
        }

        /// <inheritdoc/>
        public Task<ThoughtNodeAdapter> AddMetadataAsync(ThoughtNodeAdapter node, string key, object value)
        {
            if (node == null)
            {
                throw new ArgumentNullException(nameof(node));
            }

            if (string.IsNullOrEmpty(key))
            {
                throw new ArgumentException("Key cannot be null or empty.", nameof(key));
            }

            _logger.LogInformation("Adding metadata {Key}={Value} to node {Thought}", key, value, node.Thought);
            
            var updatedNode = node.AddMetadata(key, value);
            return Task.FromResult(updatedNode);
        }

        /// <inheritdoc/>
        public Task<string> VisualizeTreeAsync(ThoughtNodeAdapter root, string format = "text")
        {
            if (root == null)
            {
                throw new ArgumentNullException(nameof(root));
            }

            _logger.LogInformation("Visualizing tree with root {Thought} in format {Format}", root.Thought, format);
            
            // Convert to F# node
            var fsharpNode = root.ToFSharpNode();
            
            // Use F# visualization
            var visualization = Visualization.visualizeTree(fsharpNode, format);
            
            return Task.FromResult(visualization);
        }
    }
}
