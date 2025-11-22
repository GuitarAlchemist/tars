using System;
using System.IO;

namespace TarsEngine.FSharp.Adapters.TreeOfThought
{
    /// <summary>
    /// Adapter for F# Visualization module.
    /// </summary>
    public static class VisualizationAdapter
    {
        /// <summary>
        /// Converts a tree to JSON.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <returns>The JSON representation.</returns>
        public static string ToJson(ThoughtNodeAdapter node)
        {
            return FSharp.Core.TreeOfThought.Visualization.toJson(node.FSharpNode);
        }

        /// <summary>
        /// Converts a tree to a formatted JSON string.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <returns>The formatted JSON string.</returns>
        public static string ToFormattedJson(ThoughtNodeAdapter node)
        {
            return FSharp.Core.TreeOfThought.Visualization.toFormattedJson(node.FSharpNode);
        }

        /// <summary>
        /// Converts a tree to a Markdown representation.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <param name="level">The indentation level.</param>
        /// <returns>The Markdown representation.</returns>
        public static string ToMarkdown(ThoughtNodeAdapter node, int level)
        {
            return FSharp.Core.TreeOfThought.Visualization.toMarkdown(node.FSharpNode, level);
        }

        /// <summary>
        /// Converts a tree to a Markdown report.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <param name="title">The report title.</param>
        /// <returns>The Markdown report.</returns>
        public static string ToMarkdownReport(ThoughtNodeAdapter node, string title)
        {
            return FSharp.Core.TreeOfThought.Visualization.toMarkdownReport(node.FSharpNode, title);
        }

        /// <summary>
        /// Converts a tree to a DOT graph representation for Graphviz.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <param name="title">The graph title.</param>
        /// <returns>The DOT graph representation.</returns>
        public static string ToDotGraph(ThoughtNodeAdapter node, string title)
        {
            return FSharp.Core.TreeOfThought.Visualization.toDotGraph(node.FSharpNode, title);
        }

        /// <summary>
        /// Saves a DOT graph to a file.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <param name="title">The graph title.</param>
        /// <param name="filePath">The file path.</param>
        public static void SaveDotGraph(ThoughtNodeAdapter node, string title, string filePath)
        {
            FSharp.Core.TreeOfThought.Visualization.saveDotGraph(node.FSharpNode, title, filePath);
        }

        /// <summary>
        /// Saves a Markdown report to a file.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <param name="title">The report title.</param>
        /// <param name="filePath">The file path.</param>
        public static void SaveMarkdownReport(ThoughtNodeAdapter node, string title, string filePath)
        {
            FSharp.Core.TreeOfThought.Visualization.saveMarkdownReport(node.FSharpNode, title, filePath);
        }

        /// <summary>
        /// Saves a JSON representation to a file.
        /// </summary>
        /// <param name="node">The root node.</param>
        /// <param name="filePath">The file path.</param>
        public static void SaveJsonReport(ThoughtNodeAdapter node, string filePath)
        {
            FSharp.Core.TreeOfThought.Visualization.saveJsonReport(node.FSharpNode, filePath);
        }
    }
}
