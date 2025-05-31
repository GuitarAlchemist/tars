namespace TarsCli.Services.Adapters
{
    /// <summary>
    /// Adapter for converting between different IssueSeverity enums
    /// </summary>
    public static class IssueSeverityAdapter
    {
        /// <summary>
        /// Converts from TarsCli.Services.IssueSeverity to TarsCli.Services.CodeAnalysis.IssueSeverity
        /// </summary>
        /// <param name="severity">The service severity to convert</param>
        /// <returns>The CodeAnalysis severity</returns>
        public static CodeAnalysis.IssueSeverity ToCodeAnalysisIssueSeverity(IssueSeverity severity)
        {
            return severity switch
            {
                IssueSeverity.Error => CodeAnalysis.IssueSeverity.Error,
                IssueSeverity.Warning => CodeAnalysis.IssueSeverity.Warning,
                IssueSeverity.Info => CodeAnalysis.IssueSeverity.Info,
                _ => CodeAnalysis.IssueSeverity.Warning // Default
            };
        }

        /// <summary>
        /// Converts from TarsCli.Services.CodeAnalysis.IssueSeverity to TarsCli.Services.IssueSeverity
        /// </summary>
        /// <param name="severity">The CodeAnalysis severity to convert</param>
        /// <returns>The service severity</returns>
        public static IssueSeverity ToServiceIssueSeverity(CodeAnalysis.IssueSeverity severity)
        {
            return severity switch
            {
                CodeAnalysis.IssueSeverity.Critical => IssueSeverity.Error,
                CodeAnalysis.IssueSeverity.Error => IssueSeverity.Error,
                CodeAnalysis.IssueSeverity.Warning => IssueSeverity.Warning,
                CodeAnalysis.IssueSeverity.Info => IssueSeverity.Info,
                _ => IssueSeverity.Warning // Default
            };
        }
    }
}
