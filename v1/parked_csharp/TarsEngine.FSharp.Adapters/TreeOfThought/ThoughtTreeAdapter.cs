using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.FSharp.Core;

namespace TarsEngine.FSharp.Adapters.TreeOfThought
{
    /// <summary>
    /// Adapter for F# ThoughtTree module.
    /// </summary>
    public static class ThoughtTreeAdapter
    {
        /// <summary>
        /// Gets the depth of a tree.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <returns>The tree depth.</returns>
        public static int Depth(ThoughtNodeAdapter node)
        {
            return FSharp.Core.TreeOfThought.ThoughtTree.depth(node.FSharpNode);
        }

        /// <summary>
        /// Gets the breadth of a tree at the given level.
        /// </summary>
        /// <param name="level">The level.</param>
        /// <param name="node">The root node.</param>
        /// <returns>The breadth at the given level.</returns>
        public static int BreadthAtLevel(int level, ThoughtNodeAdapter node)
        {
            return FSharp.Core.TreeOfThought.ThoughtTree.breadthAtLevel(level, node.FSharpNode);
        }

        /// <summary>
        /// Gets the maximum breadth of a tree.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <returns>The maximum breadth.</returns>
        public static int MaxBreadth(ThoughtNodeAdapter node)
        {
            return FSharp.Core.TreeOfThought.ThoughtTree.maxBreadth(node.FSharpNode);
        }

        /// <summary>
        /// Finds a node by its thought content.
        /// </summary>
        /// <param name="thought">The thought content.</param>
        /// <param name="node">The root node.</param>
        /// <returns>The found node, or null if not found.</returns>
        public static ThoughtNodeAdapter? FindNode(string thought, ThoughtNodeAdapter node)
        {
            var option = FSharp.Core.TreeOfThought.ThoughtTree.findNode(thought, node.FSharpNode);
            if (option.IsNone())
            {
                return null;
            }

            return new ThoughtNodeAdapter(option.Value);
        }

        /// <summary>
        /// Finds all nodes that match a predicate.
        /// </summary>
        /// <param name="predicate">The predicate.</param>
        /// <param name="node">The root node.</param>
        /// <returns>The matching nodes.</returns>
        public static IReadOnlyList<ThoughtNodeAdapter> FindNodes(Func<ThoughtNodeAdapter, bool> predicate, ThoughtNodeAdapter node)
        {
            // Convert C# predicate to F# predicate
            Func<FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode, bool> fsharpPredicate = 
                fsharpNode => predicate(new ThoughtNodeAdapter(fsharpNode));

            var fsharpNodes = FSharp.Core.TreeOfThought.ThoughtTree.findNodes(
                FuncConvert.ToFSharpFunc(fsharpPredicate), 
                node.FSharpNode);

            var nodes = new List<ThoughtNodeAdapter>();
            foreach (var fsharpNode in fsharpNodes)
            {
                nodes.Add(new ThoughtNodeAdapter(fsharpNode));
            }

            return nodes;
        }

        /// <summary>
        /// Selects the best node based on evaluation.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <returns>The best node.</returns>
        public static ThoughtNodeAdapter SelectBestNode(ThoughtNodeAdapter node)
        {
            var bestNode = FSharp.Core.TreeOfThought.ThoughtTree.selectBestNode(node.FSharpNode);
            return new ThoughtNodeAdapter(bestNode);
        }

        /// <summary>
        /// Prunes nodes that don't meet a threshold.
        /// </summary>
        /// <param name="threshold">The threshold.</param>
        /// <param name="node">The root node.</param>
        /// <returns>The pruned tree.</returns>
        public static ThoughtNodeAdapter PruneByThreshold(double threshold, ThoughtNodeAdapter node)
        {
            var prunedNode = FSharp.Core.TreeOfThought.ThoughtTree.pruneByThreshold(threshold, node.FSharpNode);
            return new ThoughtNodeAdapter(prunedNode);
        }

        /// <summary>
        /// Prunes all but the top k nodes at each level.
        /// </summary>
        /// <param name="k">The number of nodes to keep.</param>
        /// <param name="node">The root node.</param>
        /// <returns>The pruned tree.</returns>
        public static ThoughtNodeAdapter PruneBeamSearch(int k, ThoughtNodeAdapter node)
        {
            var prunedNode = FSharp.Core.TreeOfThought.ThoughtTree.pruneBeamSearch(k, node.FSharpNode);
            return new ThoughtNodeAdapter(prunedNode);
        }

        /// <summary>
        /// Maps a function over all nodes in a tree.
        /// </summary>
        /// <param name="func">The function to apply.</param>
        /// <param name="node">The root node.</param>
        /// <returns>The mapped tree.</returns>
        public static ThoughtNodeAdapter MapTree(Func<ThoughtNodeAdapter, ThoughtNodeAdapter> func, ThoughtNodeAdapter node)
        {
            // Convert C# function to F# function
            Func<FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode, FSharp.Core.TreeOfThought.ThoughtNode.ThoughtNode> fsharpFunc = 
                fsharpNode => func(new ThoughtNodeAdapter(fsharpNode)).FSharpNode;

            var mappedNode = FSharp.Core.TreeOfThought.ThoughtTree.mapTree(
                FuncConvert.ToFSharpFunc(fsharpFunc), 
                node.FSharpNode);

            return new ThoughtNodeAdapter(mappedNode);
        }

        /// <summary>
        /// Counts the number of nodes in a tree.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <returns>The number of nodes.</returns>
        public static int CountNodes(ThoughtNodeAdapter node)
        {
            return FSharp.Core.TreeOfThought.ThoughtTree.countNodes(node.FSharpNode);
        }

        /// <summary>
        /// Counts the number of evaluated nodes in a tree.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <returns>The number of evaluated nodes.</returns>
        public static int CountEvaluatedNodes(ThoughtNodeAdapter node)
        {
            return FSharp.Core.TreeOfThought.ThoughtTree.countEvaluatedNodes(node.FSharpNode);
        }

        /// <summary>
        /// Counts the number of pruned nodes in a tree.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <returns>The number of pruned nodes.</returns>
        public static int CountPrunedNodes(ThoughtNodeAdapter node)
        {
            return FSharp.Core.TreeOfThought.ThoughtTree.countPrunedNodes(node.FSharpNode);
        }
    }
}
