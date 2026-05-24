using System.Threading.Tasks;

namespace TarsEngine.Services.Abstractions.TreeOfThought
{
    /// <summary>
    /// Interface for Tree-of-Thought service.
    /// </summary>
    public interface ITreeOfThoughtService
    {
        /// <summary>
        /// Creates a thought tree for the specified problem.
        /// </summary>
        /// <param name="problem">The problem to solve.</param>
        /// <param name="options">The options for tree creation.</param>
        /// <returns>The thought tree.</returns>
        Task<IThoughtNode> CreateThoughtTreeAsync(string problem, TreeCreationOptions options);

        /// <summary>
        /// Evaluates a thought node.
        /// </summary>
        /// <param name="node">The node to evaluate.</param>
        /// <param name="metrics">The evaluation metrics.</param>
        /// <returns>The evaluated node.</returns>
        Task<IThoughtNode> EvaluateNodeAsync(IThoughtNode node, EvaluationMetrics metrics);

        /// <summary>
        /// Adds a child to a node.
        /// </summary>
        /// <param name="parent">The parent node.</param>
        /// <param name="childThought">The child thought.</param>
        /// <returns>The updated parent node.</returns>
        Task<IThoughtNode> AddChildAsync(IThoughtNode parent, string childThought);

        /// <summary>
        /// Selects the best node from a thought tree.
        /// </summary>
        /// <param name="root">The root node.</param>
        /// <returns>The best node.</returns>
        Task<IThoughtNode> SelectBestNodeAsync(IThoughtNode root);

        /// <summary>
        /// Prunes nodes that don't meet a threshold.
        /// </summary>
        /// <param name="root">The root node.</param>
        /// <param name="threshold">The threshold.</param>
        /// <returns>The pruned tree.</returns>
        Task<IThoughtNode> PruneByThresholdAsync(IThoughtNode root, double threshold);

        /// <summary>
        /// Generates a report for a thought tree.
        /// </summary>
        /// <param name="root">The root node.</param>
        /// <param name="title">The report title.</param>
        /// <returns>The report.</returns>
        Task<string> GenerateReportAsync(IThoughtNode root, string title);

        /// <summary>
        /// Saves a report for a thought tree.
        /// </summary>
        /// <param name="root">The root node.</param>
        /// <param name="title">The report title.</param>
        /// <param name="filePath">The file path.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task SaveReportAsync(IThoughtNode root, string title, string filePath);
    }
}
