using System.Collections.Generic;
using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces
{
    /// <summary>
    /// Interface for extracting code structures from source code
    /// </summary>
    public interface ICodeStructureExtractor
    {
        /// <summary>
        /// Gets the language supported by this extractor
        /// </summary>
        string Language { get; }

        /// <summary>
        /// Extracts code structures from the provided content
        /// </summary>
        /// <param name="content">The source code content</param>
        /// <returns>A list of extracted code structures</returns>
        List<CodeStructure> ExtractStructures(string content);

        /// <summary>
        /// Gets the namespace for a position in the content
        /// </summary>
        /// <param name="structures">The list of already extracted structures</param>
        /// <param name="position">The position in the content</param>
        /// <param name="content">The source code content</param>
        /// <returns>The namespace name, or empty string if not found</returns>
        string GetNamespaceForPosition(List<CodeStructure> structures, int position, string content);

        /// <summary>
        /// Gets the class for a position in the content
        /// </summary>
        /// <param name="structures">The list of already extracted structures</param>
        /// <param name="position">The position in the content</param>
        /// <param name="content">The source code content</param>
        /// <returns>The class name, or empty string if not found</returns>
        string GetClassForPosition(List<CodeStructure> structures, int position, string content);

        /// <summary>
        /// Calculates the sizes of structures
        /// </summary>
        /// <param name="structures">The list of structures to update</param>
        /// <param name="content">The source code content</param>
        void CalculateStructureSizes(List<CodeStructure> structures, string content);
    }
}
