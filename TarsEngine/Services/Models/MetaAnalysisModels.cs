namespace TarsEngine.Services.Models;

/// <summary>
/// Represents the result of a meta-analysis of the codebase
/// </summary>
public class MetaAnalysisResult
{
    /// <summary>
    /// Whether the analysis was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if the analysis failed
    /// </summary>
    public string ErrorMessage { get; set; }

    /// <summary>
    /// The root path of the codebase
    /// </summary>
    public string RootPath { get; set; }

    /// <summary>
    /// The components in the codebase
    /// </summary>
    public List<CodeComponent> Components { get; set; } = new List<CodeComponent>();

    /// <summary>
    /// The dependencies between components
    /// </summary>
    public List<ComponentDependency> Dependencies { get; set; } = new List<ComponentDependency>();

    /// <summary>
    /// Overall metrics for the codebase
    /// </summary>
    public CodebaseMetrics Metrics { get; set; } = new CodebaseMetrics();

    /// <summary>
    /// The time the analysis was performed
    /// </summary>
    public DateTime AnalysisTime { get; set; } = DateTime.Now;
}

/// <summary>
/// Represents a component in the codebase
/// </summary>
public class CodeComponent
{
    /// <summary>
    /// The name of the component
    /// </summary>
    public string Name { get; set; }

    /// <summary>
    /// The path to the component
    /// </summary>
    public string Path { get; set; }

    /// <summary>
    /// The type of the component (e.g., Service, Controller, Model)
    /// </summary>
    public string Type { get; set; }

    /// <summary>
    /// The programming language of the component
    /// </summary>
    public ProgrammingLanguage Language { get; set; }

    /// <summary>
    /// The files in the component
    /// </summary>
    public List<string> Files { get; set; } = new List<string>();

    /// <summary>
    /// Metrics for the component
    /// </summary>
    public ComponentMetrics Metrics { get; set; } = new ComponentMetrics();
}

/// <summary>
/// Represents a dependency between components
/// </summary>
public class ComponentDependency
{
    /// <summary>
    /// The source component
    /// </summary>
    public string SourceComponent { get; set; }

    /// <summary>
    /// The target component
    /// </summary>
    public string TargetComponent { get; set; }

    /// <summary>
    /// The type of dependency
    /// </summary>
    public string DependencyType { get; set; }

    /// <summary>
    /// The strength of the dependency (0.0 to 1.0)
    /// </summary>
    public double Strength { get; set; }
}

/// <summary>
/// Represents metrics for the entire codebase
/// </summary>
public class CodebaseMetrics
{
    /// <summary>
    /// Total number of files
    /// </summary>
    public int TotalFiles { get; set; }

    /// <summary>
    /// Total number of lines of code
    /// </summary>
    public int TotalLinesOfCode { get; set; }

    /// <summary>
    /// Total number of components
    /// </summary>
    public int TotalComponents { get; set; }

    /// <summary>
    /// Average complexity across all components
    /// </summary>
    public double AverageComplexity { get; set; }

    /// <summary>
    /// Average test coverage across all components
    /// </summary>
    public double AverageTestCoverage { get; set; }

    /// <summary>
    /// Number of components with low test coverage
    /// </summary>
    public int ComponentsWithLowTestCoverage { get; set; }

    /// <summary>
    /// Number of components with high complexity
    /// </summary>
    public int ComponentsWithHighComplexity { get; set; }

    /// <summary>
    /// Number of components with code smells
    /// </summary>
    public int ComponentsWithCodeSmells { get; set; }
}

/// <summary>
/// Represents metrics for a component
/// </summary>
public class ComponentMetrics
{
    /// <summary>
    /// Number of files in the component
    /// </summary>
    public int FileCount { get; set; }

    /// <summary>
    /// Number of lines of code in the component
    /// </summary>
    public int LinesOfCode { get; set; }

    /// <summary>
    /// Cyclomatic complexity of the component
    /// </summary>
    public double Complexity { get; set; }

    /// <summary>
    /// Test coverage of the component (0.0 to 1.0)
    /// </summary>
    public double TestCoverage { get; set; }

    /// <summary>
    /// Number of code smells in the component
    /// </summary>
    public int CodeSmells { get; set; }

    /// <summary>
    /// Number of bugs in the component
    /// </summary>
    public int Bugs { get; set; }

    /// <summary>
    /// Number of vulnerabilities in the component
    /// </summary>
    public int Vulnerabilities { get; set; }

    /// <summary>
    /// Number of duplications in the component
    /// </summary>
    public int Duplications { get; set; }

    /// <summary>
    /// Maintainability index of the component (0 to 100)
    /// </summary>
    public double MaintainabilityIndex { get; set; }

    /// <summary>
    /// Number of times the component has been modified
    /// </summary>
    public int ModificationCount { get; set; }

    /// <summary>
    /// Last time the component was modified
    /// </summary>
    public DateTime LastModified { get; set; }
}

/// <summary>
/// Represents a component that needs improvement
/// </summary>
public class ComponentToImprove
{
    /// <summary>
    /// The component to improve
    /// </summary>
    public CodeComponent Component { get; set; }

    /// <summary>
    /// The priority of the improvement (0.0 to 1.0)
    /// </summary>
    public double Priority { get; set; }

    /// <summary>
    /// The reasons why the component needs improvement
    /// </summary>
    public List<string> ImprovementReasons { get; set; } = new List<string>();

    /// <summary>
    /// The areas that need improvement
    /// </summary>
    public List<ImprovementArea> ImprovementAreas { get; set; } = new List<ImprovementArea>();
}

/// <summary>
/// Represents an area that needs improvement
/// </summary>
public class ImprovementArea
{
    /// <summary>
    /// The type of improvement area
    /// </summary>
    public ImprovementAreaType Type { get; set; }

    /// <summary>
    /// The severity of the issue (0.0 to 1.0)
    /// </summary>
    public double Severity { get; set; }

    /// <summary>
    /// Description of the issue
    /// </summary>
    public string Description { get; set; }

    /// <summary>
    /// Specific files affected by the issue
    /// </summary>
    public List<string> AffectedFiles { get; set; } = new List<string>();
}

/// <summary>
/// Represents the type of improvement area
/// </summary>
public enum ImprovementAreaType
{
    /// <summary>
    /// Code quality issues
    /// </summary>
    CodeQuality,

    /// <summary>
    /// Performance issues
    /// </summary>
    Performance,

    /// <summary>
    /// Security issues
    /// </summary>
    Security,

    /// <summary>
    /// Maintainability issues
    /// </summary>
    Maintainability,

    /// <summary>
    /// Test coverage issues
    /// </summary>
    TestCoverage,

    /// <summary>
    /// Documentation issues
    /// </summary>
    Documentation,

    /// <summary>
    /// Architecture issues
    /// </summary>
    Architecture
}

/// <summary>
/// Represents a strategy for improving a component
/// </summary>
public class ImprovementStrategy
{
    /// <summary>
    /// The component to improve
    /// </summary>
    public ComponentToImprove Component { get; set; }

    /// <summary>
    /// The type of strategy
    /// </summary>
    public StrategyType Type { get; set; }

    /// <summary>
    /// The steps to follow for the improvement
    /// </summary>
    public List<ImprovementStep> Steps { get; set; } = new List<ImprovementStep>();

    /// <summary>
    /// The expected impact of the improvement
    /// </summary>
    public ExpectedImpact ExpectedImpact { get; set; } = new ExpectedImpact();

    /// <summary>
    /// The estimated effort required for the improvement (in hours)
    /// </summary>
    public double EstimatedEffort { get; set; }

    /// <summary>
    /// The confidence level in the strategy (0.0 to 1.0)
    /// </summary>
    public double ConfidenceLevel { get; set; }
}

/// <summary>
/// Represents the type of improvement strategy
/// </summary>
public enum StrategyType
{
    /// <summary>
    /// Refactoring the code
    /// </summary>
    Refactoring,

    /// <summary>
    /// Adding tests
    /// </summary>
    AddTests,

    /// <summary>
    /// Optimizing performance
    /// </summary>
    OptimizePerformance,

    /// <summary>
    /// Fixing security issues
    /// </summary>
    FixSecurity,

    /// <summary>
    /// Improving documentation
    /// </summary>
    ImproveDocumentation,

    /// <summary>
    /// Restructuring architecture
    /// </summary>
    RestructureArchitecture
}

/// <summary>
/// Represents a step in an improvement strategy
/// </summary>
public class ImprovementStep
{
    /// <summary>
    /// The order of the step
    /// </summary>
    public int Order { get; set; }

    /// <summary>
    /// The description of the step
    /// </summary>
    public string Description { get; set; }

    /// <summary>
    /// The files affected by the step
    /// </summary>
    public List<string> AffectedFiles { get; set; } = new List<string>();

    /// <summary>
    /// Whether the step requires testing
    /// </summary>
    public bool RequiresTesting { get; set; }

    /// <summary>
    /// The estimated effort for the step (in hours)
    /// </summary>
    public double EstimatedEffort { get; set; }
}

/// <summary>
/// Represents the expected impact of an improvement
/// </summary>
public class ExpectedImpact
{
    /// <summary>
    /// The expected improvement in code quality (0.0 to 1.0)
    /// </summary>
    public double CodeQualityImprovement { get; set; }

    /// <summary>
    /// The expected improvement in performance (0.0 to 1.0)
    /// </summary>
    public double PerformanceImprovement { get; set; }

    /// <summary>
    /// The expected improvement in maintainability (0.0 to 1.0)
    /// </summary>
    public double MaintainabilityImprovement { get; set; }

    /// <summary>
    /// The expected improvement in test coverage (0.0 to 1.0)
    /// </summary>
    public double TestCoverageImprovement { get; set; }

    /// <summary>
    /// The expected reduction in bugs (0.0 to 1.0)
    /// </summary>
    public double BugReduction { get; set; }

    /// <summary>
    /// The expected reduction in vulnerabilities (0.0 to 1.0)
    /// </summary>
    public double VulnerabilityReduction { get; set; }
}

/// <summary>
/// Represents an assessment of the impact of improvements
/// </summary>
public class ImpactAssessment
{
    /// <summary>
    /// The component that was improved
    /// </summary>
    public ComponentToImprove Component { get; set; }

    /// <summary>
    /// The strategy that was used
    /// </summary>
    public ImprovementStrategy Strategy { get; set; }

    /// <summary>
    /// Whether the improvement was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// The metrics before the improvement
    /// </summary>
    public ComponentMetrics BeforeMetrics { get; set; }

    /// <summary>
    /// The metrics after the improvement
    /// </summary>
    public ComponentMetrics AfterMetrics { get; set; }

    /// <summary>
    /// The actual impact of the improvement
    /// </summary>
    public ActualImpact ActualImpact { get; set; } = new ActualImpact();

    /// <summary>
    /// The time taken for the improvement (in hours)
    /// </summary>
    public double TimeTaken { get; set; }

    /// <summary>
    /// Any issues encountered during the improvement
    /// </summary>
    public List<string> Issues { get; set; } = new List<string>();

    /// <summary>
    /// Lessons learned from the improvement
    /// </summary>
    public List<string> LessonsLearned { get; set; } = new List<string>();
}

/// <summary>
/// Represents the actual impact of an improvement
/// </summary>
public class ActualImpact
{
    /// <summary>
    /// The actual improvement in code quality (0.0 to 1.0)
    /// </summary>
    public double CodeQualityImprovement { get; set; }

    /// <summary>
    /// The actual improvement in performance (0.0 to 1.0)
    /// </summary>
    public double PerformanceImprovement { get; set; }

    /// <summary>
    /// The actual improvement in maintainability (0.0 to 1.0)
    /// </summary>
    public double MaintainabilityImprovement { get; set; }

    /// <summary>
    /// The actual improvement in test coverage (0.0 to 1.0)
    /// </summary>
    public double TestCoverageImprovement { get; set; }

    /// <summary>
    /// The actual reduction in bugs (0.0 to 1.0)
    /// </summary>
    public double BugReduction { get; set; }

    /// <summary>
    /// The actual reduction in vulnerabilities (0.0 to 1.0)
    /// </summary>
    public double VulnerabilityReduction { get; set; }

    /// <summary>
    /// Whether the impact met expectations
    /// </summary>
    public bool MetExpectations { get; set; }

    /// <summary>
    /// Reasons why the impact did or did not meet expectations
    /// </summary>
    public List<string> Reasons { get; set; } = new List<string>();
}