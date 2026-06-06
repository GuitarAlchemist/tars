namespace TarsEngine.Services.Abstractions.Models.Knowledge
{
    /// <summary>
    /// Represents the type of a knowledge item.
    /// </summary>
    public enum KnowledgeType
    {
        /// <summary>
        /// General knowledge.
        /// </summary>
        General = 0,

        /// <summary>
        /// Programming concept.
        /// </summary>
        ProgrammingConcept = 1,

        /// <summary>
        /// Code pattern.
        /// </summary>
        CodePattern = 2,

        /// <summary>
        /// Best practice.
        /// </summary>
        BestPractice = 3,

        /// <summary>
        /// Algorithm.
        /// </summary>
        Algorithm = 4,

        /// <summary>
        /// Design pattern.
        /// </summary>
        DesignPattern = 5,

        /// <summary>
        /// API documentation.
        /// </summary>
        ApiDocumentation = 6,

        /// <summary>
        /// Framework documentation.
        /// </summary>
        FrameworkDocumentation = 7,

        /// <summary>
        /// Language feature.
        /// </summary>
        LanguageFeature = 8,

        /// <summary>
        /// Tool usage.
        /// </summary>
        ToolUsage = 9,

        /// <summary>
        /// Error solution.
        /// </summary>
        ErrorSolution = 10,

        /// <summary>
        /// Performance optimization.
        /// </summary>
        PerformanceOptimization = 11,

        /// <summary>
        /// Security guideline.
        /// </summary>
        SecurityGuideline = 12,

        /// <summary>
        /// Testing strategy.
        /// </summary>
        TestingStrategy = 13,

        /// <summary>
        /// Deployment practice.
        /// </summary>
        DeploymentPractice = 14,

        /// <summary>
        /// User preference.
        /// </summary>
        UserPreference = 15
    }
}
