using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Abstractions.TreeOfThought;

namespace TarsEngine.FSharp.Adapters.TreeOfThought
{
    /// <summary>
    /// Tree-of-Thought service implementation using F# core.
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

        /// <summary>
        /// Creates a thought tree for the specified problem.
        /// </summary>
        /// <param name="problem">The problem to solve.</param>
        /// <param name="options">The options for tree creation.</param>
        /// <returns>The thought tree.</returns>
        public Task<IThoughtNode> CreateThoughtTreeAsync(string problem, TreeCreationOptions options)
        {
            _logger.LogInformation("Creating thought tree for problem: {Problem}", problem);

            try
            {
                // Create the root thought
                var root = ThoughtNodeAdapter.CreateNode(problem);

                // Create approaches based on options
                var approaches = new List<ThoughtNodeAdapter>();
                foreach (var approach in options.Approaches)
                {
                    var approachNode = ThoughtNodeAdapter.CreateNode(approach);
                    
                    // Add evaluation if provided
                    if (options.ApproachEvaluations.TryGetValue(approach, out var evaluation))
                    {
                        var metrics = EvaluationMetricsAdapter.CreateMetrics(
                            evaluation.Correctness,
                            evaluation.Efficiency,
                            evaluation.Robustness,
                            evaluation.Maintainability);
                        
                        approachNode = approachNode.EvaluateNode(metrics);
                    }
                    
                    approaches.Add(approachNode);
                }

                // Add approaches to root
                root = root.AddChildren(approaches);

                // Return the root node
                return Task.FromResult<IThoughtNode>(new ThoughtNodeWrapper(root));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating thought tree");
                throw;
            }
        }

        /// <summary>
        /// Evaluates a thought node.
        /// </summary>
        /// <param name="node">The node to evaluate.</param>
        /// <param name="metrics">The evaluation metrics.</param>
        /// <returns>The evaluated node.</returns>
        public Task<IThoughtNode> EvaluateNodeAsync(IThoughtNode node, EvaluationMetrics metrics)
        {
            _logger.LogInformation("Evaluating node: {Thought}", node.Thought);

            try
            {
                // Convert the node to a ThoughtNodeAdapter
                if (node is not ThoughtNodeWrapper wrapper)
                {
                    throw new ArgumentException("Node must be created by this service", nameof(node));
                }

                // Create evaluation metrics
                var metricsAdapter = EvaluationMetricsAdapter.CreateMetrics(
                    metrics.Correctness,
                    metrics.Efficiency,
                    metrics.Robustness,
                    metrics.Maintainability);

                // Evaluate the node
                var evaluatedNode = wrapper.Node.EvaluateNode(metricsAdapter);

                // Return the evaluated node
                return Task.FromResult<IThoughtNode>(new ThoughtNodeWrapper(evaluatedNode));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error evaluating node");
                throw;
            }
        }

        /// <summary>
        /// Adds a child to a node.
        /// </summary>
        /// <param name="parent">The parent node.</param>
        /// <param name="childThought">The child thought.</param>
        /// <returns>The updated parent node.</returns>
        public Task<IThoughtNode> AddChildAsync(IThoughtNode parent, string childThought)
        {
            _logger.LogInformation("Adding child to node: {Thought}", parent.Thought);

            try
            {
                // Convert the parent to a ThoughtNodeAdapter
                if (parent is not ThoughtNodeWrapper wrapper)
                {
                    throw new ArgumentException("Parent must be created by this service", nameof(parent));
                }

                // Create the child node
                var childNode = ThoughtNodeAdapter.CreateNode(childThought);

                // Add the child to the parent
                var updatedParent = wrapper.Node.AddChild(childNode);

                // Return the updated parent
                return Task.FromResult<IThoughtNode>(new ThoughtNodeWrapper(updatedParent));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error adding child");
                throw;
            }
        }

        /// <summary>
        /// Selects the best node from a thought tree.
        /// </summary>
        /// <param name="root">The root node.</param>
        /// <returns>The best node.</returns>
        public Task<IThoughtNode> SelectBestNodeAsync(IThoughtNode root)
        {
            _logger.LogInformation("Selecting best node from tree: {Thought}", root.Thought);

            try
            {
                // Convert the root to a ThoughtNodeAdapter
                if (root is not ThoughtNodeWrapper wrapper)
                {
                    throw new ArgumentException("Root must be created by this service", nameof(root));
                }

                // Select the best node
                var bestNode = ThoughtTreeAdapter.SelectBestNode(wrapper.Node);

                // Return the best node
                return Task.FromResult<IThoughtNode>(new ThoughtNodeWrapper(bestNode));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error selecting best node");
                throw;
            }
        }

        /// <summary>
        /// Prunes nodes that don't meet a threshold.
        /// </summary>
        /// <param name="root">The root node.</param>
        /// <param name="threshold">The threshold.</param>
        /// <returns>The pruned tree.</returns>
        public Task<IThoughtNode> PruneByThresholdAsync(IThoughtNode root, double threshold)
        {
            _logger.LogInformation("Pruning tree by threshold: {Threshold}", threshold);

            try
            {
                // Convert the root to a ThoughtNodeAdapter
                if (root is not ThoughtNodeWrapper wrapper)
                {
                    throw new ArgumentException("Root must be created by this service", nameof(root));
                }

                // Prune the tree
                var prunedTree = ThoughtTreeAdapter.PruneByThreshold(threshold, wrapper.Node);

                // Return the pruned tree
                return Task.FromResult<IThoughtNode>(new ThoughtNodeWrapper(prunedTree));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error pruning tree");
                throw;
            }
        }

        /// <summary>
        /// Generates a report for a thought tree.
        /// </summary>
        /// <param name="root">The root node.</param>
        /// <param name="title">The report title.</param>
        /// <returns>The report.</returns>
        public Task<string> GenerateReportAsync(IThoughtNode root, string title)
        {
            _logger.LogInformation("Generating report for tree: {Thought}", root.Thought);

            try
            {
                // Convert the root to a ThoughtNodeAdapter
                if (root is not ThoughtNodeWrapper wrapper)
                {
                    throw new ArgumentException("Root must be created by this service", nameof(root));
                }

                // Generate the report
                var report = VisualizationAdapter.ToMarkdownReport(wrapper.Node, title);

                // Return the report
                return Task.FromResult(report);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating report");
                throw;
            }
        }

        /// <summary>
        /// Saves a report for a thought tree.
        /// </summary>
        /// <param name="root">The root node.</param>
        /// <param name="title">The report title.</param>
        /// <param name="filePath">The file path.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public Task SaveReportAsync(IThoughtNode root, string title, string filePath)
        {
            _logger.LogInformation("Saving report for tree: {Thought}", root.Thought);

            try
            {
                // Convert the root to a ThoughtNodeAdapter
                if (root is not ThoughtNodeWrapper wrapper)
                {
                    throw new ArgumentException("Root must be created by this service", nameof(root));
                }

                // Save the report
                VisualizationAdapter.SaveMarkdownReport(wrapper.Node, title, filePath);

                // Return a completed task
                return Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving report");
                throw;
            }
        }
    }
}
