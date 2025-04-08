namespace TarsEngine.Models.Adapters
{
    /// <summary>
    /// Adapter class to resolve ambiguity between TarsEngine.Services.Interfaces.ValidationIssue and TarsEngine.Models.ValidationIssue
    /// </summary>
    public class ValidationIssueAdapter
    {
        public string RuleId { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public IssueSeverity Severity { get; set; } = IssueSeverity.Major;
        public string Location { get; set; } = string.Empty;

        // Convert from service interface to model
        public static ValidationIssue ToModel(Services.Interfaces.ValidationIssue issue)
        {
            return new ValidationIssue
            {
                RuleId = issue.Description,
                Message = issue.Description,
                RuleName = issue.Location,
                Severity = ValidationSeverity.Error
            };
        }

        // Convert from model to service interface
        public static Services.Interfaces.ValidationIssue ToService(ValidationIssue issue)
        {
            return new Services.Interfaces.ValidationIssue
            {
                Description = issue.Message,
                Severity = Services.Interfaces.IssueSeverity.Major,
                Location = issue.RuleId,
                SuggestedFix = null
            };
        }

        private static ValidationSeverity ConvertSeverity(IssueSeverity severity)
        {
            return severity switch
            {
                IssueSeverity.Blocker => ValidationSeverity.Critical,
                IssueSeverity.Critical => ValidationSeverity.Critical,
                IssueSeverity.Major => ValidationSeverity.Error,
                IssueSeverity.Minor => ValidationSeverity.Warning,
                IssueSeverity.Trivial => ValidationSeverity.Info,
                IssueSeverity.Info => ValidationSeverity.Info,
                _ => ValidationSeverity.Warning
            };
        }

        private static IssueSeverity ConvertSeverity(ValidationSeverity severity)
        {
            return severity switch
            {
                ValidationSeverity.Critical => IssueSeverity.Critical,
                ValidationSeverity.Error => IssueSeverity.Major,
                ValidationSeverity.Warning => IssueSeverity.Minor,
                ValidationSeverity.Info => IssueSeverity.Trivial,
                _ => IssueSeverity.Major
            };
        }
    }
}
