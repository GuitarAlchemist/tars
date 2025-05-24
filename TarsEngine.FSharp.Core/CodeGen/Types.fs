namespace TarsEngine.FSharp.Core.CodeGen

open System
open System.Collections.Generic
open System.Threading.Tasks

/// <summary>
/// Represents a code refactoring.
/// </summary>
type CodeRefactoring = {
    /// <summary>
    /// The name of the refactoring.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the refactoring.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The language of the refactoring.
    /// </summary>
    Language: string
    
    /// <summary>
    /// The category of the refactoring.
    /// </summary>
    Category: string
    
    /// <summary>
    /// The tags associated with the refactoring.
    /// </summary>
    Tags: string list
    
    /// <summary>
    /// The before code snippet.
    /// </summary>
    BeforeCode: string
    
    /// <summary>
    /// The after code snippet.
    /// </summary>
    AfterCode: string
    
    /// <summary>
    /// The explanation of the refactoring.
    /// </summary>
    Explanation: string
    
    /// <summary>
    /// Additional information about the refactoring.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a code generation template.
/// </summary>
type CodeGenerationTemplate = {
    /// <summary>
    /// The name of the template.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the template.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The language of the template.
    /// </summary>
    Language: string
    
    /// <summary>
    /// The category of the template.
    /// </summary>
    Category: string
    
    /// <summary>
    /// The tags associated with the template.
    /// </summary>
    Tags: string list
    
    /// <summary>
    /// The template content.
    /// </summary>
    Content: string
    
    /// <summary>
    /// The placeholders in the template.
    /// </summary>
    Placeholders: Map<string, string>
    
    /// <summary>
    /// Additional information about the template.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a code generation result.
/// </summary>
type CodeGenerationResult = {
    /// <summary>
    /// The generated code.
    /// </summary>
    GeneratedCode: string
    
    /// <summary>
    /// The template used for generation.
    /// </summary>
    Template: CodeGenerationTemplate
    
    /// <summary>
    /// The values used for placeholders.
    /// </summary>
    PlaceholderValues: Map<string, string>
    
    /// <summary>
    /// Additional information about the generation.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a code refactoring result.
/// </summary>
type CodeRefactoringResult = {
    /// <summary>
    /// The original code.
    /// </summary>
    OriginalCode: string
    
    /// <summary>
    /// The refactored code.
    /// </summary>
    RefactoredCode: string
    
    /// <summary>
    /// The refactoring applied.
    /// </summary>
    Refactoring: CodeRefactoring
    
    /// <summary>
    /// The explanation of the refactoring.
    /// </summary>
    Explanation: string
    
    /// <summary>
    /// Additional information about the refactoring.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a test generation result.
/// </summary>
type TestGenerationResult = {
    /// <summary>
    /// The generated test code.
    /// </summary>
    GeneratedTestCode: string
    
    /// <summary>
    /// The source code that was tested.
    /// </summary>
    SourceCode: string
    
    /// <summary>
    /// The test framework used.
    /// </summary>
    TestFramework: string
    
    /// <summary>
    /// The test coverage percentage.
    /// </summary>
    Coverage: float
    
    /// <summary>
    /// Additional information about the test generation.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a documentation generation result.
/// </summary>
type DocumentationGenerationResult = {
    /// <summary>
    /// The generated documentation.
    /// </summary>
    GeneratedDocumentation: string
    
    /// <summary>
    /// The source code that was documented.
    /// </summary>
    SourceCode: string
    
    /// <summary>
    /// The documentation format.
    /// </summary>
    Format: string
    
    /// <summary>
    /// Additional information about the documentation generation.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a test result.
/// </summary>
type TestResult = {
    /// <summary>
    /// The name of the test.
    /// </summary>
    TestName: string
    
    /// <summary>
    /// Whether the test passed.
    /// </summary>
    Passed: bool
    
    /// <summary>
    /// The error message, if any.
    /// </summary>
    ErrorMessage: string option
    
    /// <summary>
    /// The execution time in milliseconds.
    /// </summary>
    ExecutionTime: int64
    
    /// <summary>
    /// Additional information about the test.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a test run result.
/// </summary>
type TestRunResult = {
    /// <summary>
    /// The test results.
    /// </summary>
    TestResults: TestResult list
    
    /// <summary>
    /// The total number of tests.
    /// </summary>
    TotalTests: int
    
    /// <summary>
    /// The number of passed tests.
    /// </summary>
    PassedTests: int
    
    /// <summary>
    /// The number of failed tests.
    /// </summary>
    FailedTests: int
    
    /// <summary>
    /// The total execution time in milliseconds.
    /// </summary>
    TotalExecutionTime: int64
    
    /// <summary>
    /// Additional information about the test run.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a regression test result.
/// </summary>
type RegressionTestResult = {
    /// <summary>
    /// The test run result.
    /// </summary>
    TestRunResult: TestRunResult
    
    /// <summary>
    /// The original code.
    /// </summary>
    OriginalCode: string
    
    /// <summary>
    /// The improved code.
    /// </summary>
    ImprovedCode: string
    
    /// <summary>
    /// The regression issues found.
    /// </summary>
    RegressionIssues: RegressionIssue list
    
    /// <summary>
    /// Additional information about the regression test.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a regression issue.
/// </summary>
and RegressionIssue = {
    /// <summary>
    /// The description of the issue.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The severity of the issue.
    /// </summary>
    Severity: string
    
    /// <summary>
    /// The test that failed.
    /// </summary>
    FailedTest: TestResult option
    
    /// <summary>
    /// The code snippet where the issue was found.
    /// </summary>
    CodeSnippet: string option
    
    /// <summary>
    /// Additional information about the issue.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a test coverage result.
/// </summary>
type TestCoverageResult = {
    /// <summary>
    /// The line coverage percentage.
    /// </summary>
    LineCoverage: float
    
    /// <summary>
    /// The branch coverage percentage.
    /// </summary>
    BranchCoverage: float
    
    /// <summary>
    /// The method coverage percentage.
    /// </summary>
    MethodCoverage: float
    
    /// <summary>
    /// The class coverage percentage.
    /// </summary>
    ClassCoverage: float
    
    /// <summary>
    /// The overall coverage percentage.
    /// </summary>
    OverallCoverage: float
    
    /// <summary>
    /// The uncovered lines.
    /// </summary>
    UncoveredLines: (string * int) list
    
    /// <summary>
    /// The uncovered branches.
    /// </summary>
    UncoveredBranches: (string * int) list
    
    /// <summary>
    /// Additional information about the coverage.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a suggested test.
/// </summary>
type SuggestedTest = {
    /// <summary>
    /// The name of the test.
    /// </summary>
    TestName: string
    
    /// <summary>
    /// The description of the test.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The generated test code.
    /// </summary>
    TestCode: string
    
    /// <summary>
    /// The target method or class to test.
    /// </summary>
    Target: string
    
    /// <summary>
    /// The reason for suggesting this test.
    /// </summary>
    Reason: string
    
    /// <summary>
    /// Additional information about the suggested test.
    /// </summary>
    AdditionalInfo: Map<string, string>
}
