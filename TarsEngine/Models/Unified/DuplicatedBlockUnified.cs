using System.Collections.Generic;

namespace TarsEngine.Models.Unified
{
    /// <summary>
    /// Represents a unified duplicated block of code that combines all properties from different DuplicatedBlock classes
    /// </summary>
    public class DuplicatedBlockUnified
    {
        /// <summary>
        /// Gets or sets the source file path
        /// </summary>
        public string SourceFilePath { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the source start line
        /// </summary>
        public int SourceStartLine { get; set; }

        /// <summary>
        /// Gets or sets the source end line
        /// </summary>
        public int SourceEndLine { get; set; }

        /// <summary>
        /// Gets or sets the target file path
        /// </summary>
        public string TargetFilePath { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the target start line
        /// </summary>
        public int TargetStartLine { get; set; }

        /// <summary>
        /// Gets or sets the target end line
        /// </summary>
        public int TargetEndLine { get; set; }

        /// <summary>
        /// Gets or sets the number of duplicated lines
        /// </summary>
        public int DuplicatedLines { get; set; }

        /// <summary>
        /// Gets or sets the similarity percentage
        /// </summary>
        public double SimilarityPercentage { get; set; }

        /// <summary>
        /// Gets or sets the duplicated code
        /// </summary>
        public string DuplicatedCode { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the source method
        /// </summary>
        public string SourceMethod { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the target method
        /// </summary>
        public string TargetMethod { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the list of duplicate locations
        /// </summary>
        public List<DuplicateLocation> DuplicateLocations { get; set; } = new List<DuplicateLocation>();
    }

    /// <summary>
    /// Represents a duplicate location
    /// </summary>
    public class DuplicateLocation
    {
        /// <summary>
        /// Gets or sets the file path
        /// </summary>
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the start line
        /// </summary>
        public int StartLine { get; set; }

        /// <summary>
        /// Gets or sets the end line
        /// </summary>
        public int EndLine { get; set; }
    }
}
