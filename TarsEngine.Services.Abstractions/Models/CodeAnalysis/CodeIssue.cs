namespace TarsEngine.Services.Abstractions.Models.CodeAnalysis
{
    /// <summary>
    /// Represents an issue found during code analysis.
    /// </summary>
    public class CodeIssue
    {
        /// <summary>
        /// Gets or sets the unique identifier for the issue.
        /// </summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Gets or sets the title of the issue.
        /// </summary>
        public string Title { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the description of the issue.
        /// </summary>
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the severity of the issue.
        /// </summary>
        public IssueSeverity Severity { get; set; } = IssueSeverity.Info;

        /// <summary>
        /// Gets or sets the line number where the issue was found.
        /// </summary>
        public int LineNumber { get; set; }

        /// <summary>
        /// Gets or sets the column number where the issue was found.
        /// </summary>
        public int ColumnNumber { get; set; }

        /// <summary>
        /// Gets or sets the file path where the issue was found.
        /// </summary>
        public string? FilePath { get; set; }

        /// <summary>
        /// Gets or sets the code snippet where the issue was found.
        /// </summary>
        public string? CodeSnippet { get; set; }

        /// <summary>
        /// Gets or sets the suggested fix for the issue.
        /// </summary>
        public string? SuggestedFix { get; set; }

        /// <summary>
        /// Gets or sets the rule ID that triggered the issue.
        /// </summary>
        public string? RuleId { get; set; }
    }
}
