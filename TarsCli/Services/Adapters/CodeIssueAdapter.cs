using TarsCli.Services.CodeAnalysis;

namespace TarsCli.Services.Adapters
{
    /// <summary>
    /// Adapter for converting between different CodeIssue types
    /// </summary>
    public static class CodeIssueAdapter
    {
        /// <summary>
        /// Converts from TarsCli.Services.CodeAnalysis.CodeIssue to TarsCli.Services.CodeIssue
        /// </summary>
        /// <param name="issue">The CodeAnalysis issue to convert</param>
        /// <returns>The converted issue</returns>
        public static CodeIssue ToServiceCodeIssue(CodeAnalysis.CodeIssue issue)
        {
            return new CodeIssue
            {
                Type = ConvertIssueType(issue.Type),
                Location = $"{issue.LineNumber}:{issue.ColumnNumber}",
                Description = issue.Description,
                Suggestion = issue.SuggestedFix
            };
        }

        /// <summary>
        /// Converts from TarsCli.Services.CodeIssue to TarsCli.Services.CodeAnalysis.CodeIssue
        /// </summary>
        /// <param name="issue">The service issue to convert</param>
        /// <returns>The converted issue</returns>
        public static CodeAnalysis.CodeIssue ToCodeAnalysisCodeIssue(CodeIssue issue)
        {
            return new CodeAnalysis.CodeIssue
            {
                Type = ConvertIssueType(issue.Type),
                Severity = IssueSeverityAdapter.ToCodeAnalysisIssueSeverity(IssueSeverity.Warning), // Default severity
                Description = issue.Description,
                LineNumber = ParseLineNumber(issue.Location),
                ColumnNumber = ParseColumnNumber(issue.Location),
                Length = 0, // Default length
                CodeSegment = string.Empty, // Default code segment
                SuggestedFix = issue.Suggestion
            };
        }

        /// <summary>
        /// Converts from TarsCli.Services.CodeAnalysis.CodeIssueType to TarsCli.Services.IssueType
        /// </summary>
        /// <param name="type">The CodeAnalysis issue type to convert</param>
        /// <returns>The converted issue type</returns>
        public static IssueType ConvertIssueType(CodeAnalysis.CodeIssueType type)
        {
            return type switch
            {
                CodeAnalysis.CodeIssueType.Security => IssueType.MissingExceptionHandling,
                CodeAnalysis.CodeIssueType.Performance => IssueType.IneffectiveCode,
                CodeAnalysis.CodeIssueType.Style => IssueType.StyleViolation,
                CodeAnalysis.CodeIssueType.Documentation => IssueType.DocumentationIssue,
                _ => IssueType.IneffectiveCode // Default
            };
        }

        /// <summary>
        /// Converts from TarsCli.Services.IssueType to TarsCli.Services.CodeAnalysis.CodeIssueType
        /// </summary>
        /// <param name="type">The service issue type to convert</param>
        /// <returns>The converted issue type</returns>
        public static CodeAnalysis.CodeIssueType ConvertIssueType(IssueType type)
        {
            return type switch
            {
                IssueType.MissingExceptionHandling => CodeAnalysis.CodeIssueType.Security,
                IssueType.IneffectiveCode => CodeAnalysis.CodeIssueType.Performance,
                IssueType.StyleViolation => CodeAnalysis.CodeIssueType.Style,
                IssueType.DocumentationIssue => CodeAnalysis.CodeIssueType.Documentation,
                _ => CodeAnalysis.CodeIssueType.Performance // Default
            };
        }

        /// <summary>
        /// Parses the line number from a location string
        /// </summary>
        /// <param name="location">The location string (format: "line:column")</param>
        /// <returns>The line number</returns>
        private static int ParseLineNumber(string location)
        {
            if (string.IsNullOrEmpty(location))
                return 0;

            var parts = location.Split(':');
            if (parts.Length < 1)
                return 0;

            if (int.TryParse(parts[0], out var lineNumber))
                return lineNumber;

            return 0;
        }

        /// <summary>
        /// Parses the column number from a location string
        /// </summary>
        /// <param name="location">The location string (format: "line:column")</param>
        /// <returns>The column number</returns>
        private static int ParseColumnNumber(string location)
        {
            if (string.IsNullOrEmpty(location))
                return 0;

            var parts = location.Split(':');
            if (parts.Length < 2)
                return 0;

            if (int.TryParse(parts[1], out var columnNumber))
                return columnNumber;

            return 0;
        }
    }
}
