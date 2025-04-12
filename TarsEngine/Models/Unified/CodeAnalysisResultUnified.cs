using System;
using System.Collections.Generic;
using TarsEngine.Models.Metrics;

namespace TarsEngine.Models.Unified
{
    /// <summary>
    /// Represents a unified code analysis result that combines all properties from different CodeAnalysisResult classes
    /// </summary>
    public class CodeAnalysisResultUnified
    {
        /// <summary>
        /// Gets or sets the file path
        /// </summary>
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the path (alternative property name)
        /// </summary>
        public string Path { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the language
        /// </summary>
        public string Language { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets a value indicating whether the analysis was successful
        /// </summary>
        public bool IsSuccessful { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether the analysis was successful (alternative property name)
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Gets or sets the error message
        /// </summary>
        public string ErrorMessage { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the errors
        /// </summary>
        public List<string> Errors { get; set; } = new();

        /// <summary>
        /// Gets or sets the namespaces
        /// </summary>
        public List<string> Namespaces { get; set; } = new();

        /// <summary>
        /// Gets or sets the classes
        /// </summary>
        public List<string> Classes { get; set; } = new();

        /// <summary>
        /// Gets or sets the interfaces
        /// </summary>
        public List<string> Interfaces { get; set; } = new();

        /// <summary>
        /// Gets or sets the methods
        /// </summary>
        public List<string> Methods { get; set; } = new();

        /// <summary>
        /// Gets or sets the properties
        /// </summary>
        public List<string> Properties { get; set; } = new();

        /// <summary>
        /// Gets or sets the metrics
        /// </summary>
        public List<CodeMetric> Metrics { get; set; } = new();

        /// <summary>
        /// Gets or sets the issues
        /// </summary>
        public List<CodeIssue> Issues { get; set; } = new();

        /// <summary>
        /// Gets or sets the structures
        /// </summary>
        public List<CodeStructure> Structures { get; set; } = new();

        /// <summary>
        /// Gets or sets the analyzed at timestamp
        /// </summary>
        public DateTime AnalyzedAt { get; set; } = DateTime.UtcNow;
    }
}
