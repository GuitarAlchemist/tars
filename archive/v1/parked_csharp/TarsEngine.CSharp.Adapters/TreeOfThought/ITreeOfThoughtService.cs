using System.Collections.Generic;
using System.Threading.Tasks;

namespace TarsEngine.CSharp.Adapters.TreeOfThought
{
    /// <summary>
    /// Interface for the Tree of Thought service.
    /// </summary>
    public interface ITreeOfThoughtService
    {
        /// <summary>
        /// Creates a root node with the specified thought.
        /// </summary>
        /// <param name="thought">The thought content.</param>
        /// <returns>The root node.</returns>
        Task<ThoughtNodeAdapter> CreateRootNodeAsync(string thought);

        /// <summary>
        /// Adds a child node to the specified parent node.
        /// </summary>
        /// <param name="parent">The parent node.</param>
        /// <param name="thought">The thought content.</param>
        /// <returns>The updated parent node.</returns>
        Task<ThoughtNodeAdapter> AddChildNodeAsync(ThoughtNodeAdapter parent, string thought);

        /// <summary>
        /// Adds multiple child nodes to the specified parent node.
        /// </summary>
        /// <param name="parent">The parent node.</param>
        /// <param name="thoughts">The thought contents.</param>
        /// <returns>The updated parent node.</returns>
        Task<ThoughtNodeAdapter> AddChildNodesAsync(ThoughtNodeAdapter parent, IEnumerable<string> thoughts);

        /// <summary>
        /// Evaluates the specified node with the specified metrics.
        /// </summary>
        /// <param name="node">The node to evaluate.</param>
        /// <param name="correctness">The correctness score.</param>
        /// <param name="efficiency">The efficiency score.</param>
        /// <param name="robustness">The robustness score.</param>
        /// <param name="maintainability">The maintainability score.</param>
        /// <returns>The evaluated node.</returns>
        Task<ThoughtNodeAdapter> EvaluateNodeAsync(ThoughtNodeAdapter node, double correctness, double efficiency, double robustness, double maintainability);

        /// <summary>
        /// Prunes the specified node.
        /// </summary>
        /// <param name="node">The node to prune.</param>
        /// <returns>The pruned node.</returns>
        Task<ThoughtNodeAdapter> PruneNodeAsync(ThoughtNodeAdapter node);

        /// <summary>
        /// Adds metadata to the specified node.
        /// </summary>
        /// <param name="node">The node to add metadata to.</param>
        /// <param name="key">The metadata key.</param>
        /// <param name="value">The metadata value.</param>
        /// <returns>The updated node.</returns>
        Task<ThoughtNodeAdapter> AddMetadataAsync(ThoughtNodeAdapter node, string key, object value);

        /// <summary>
        /// Visualizes the tree with the specified root node.
        /// </summary>
        /// <param name="root">The root node.</param>
        /// <param name="format">The format to visualize the tree in.</param>
        /// <returns>The visualization.</returns>
        Task<string> VisualizeTreeAsync(ThoughtNodeAdapter root, string format = "text");
    }
}
