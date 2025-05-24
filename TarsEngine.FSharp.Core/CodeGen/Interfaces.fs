namespace TarsEngine.FSharp.Core.CodeGen

open System
open System.Collections.Generic
open System.Threading.Tasks

/// <summary>
/// Interface for code generator.
/// </summary>
type ICodeGenerator =
    /// <summary>
    /// Gets the language supported by this generator.
    /// </summary>
    abstract member Language : string
    
    /// <summary>
    /// Generates code from a template.
    /// </summary>
    /// <param name="template">The template to use.</param>
    /// <param name="placeholderValues">The values for placeholders.</param>
    /// <returns>The code generation result.</returns>
    abstract member GenerateCode : template:CodeGenerationTemplate * placeholderValues:Map<string, string> -> CodeGenerationResult
    
    /// <summary>
    /// Generates code from a template name.
    /// </summary>
    /// <param name="templateName">The name of the template to use.</param>
    /// <param name="placeholderValues">The values for placeholders.</param>
    /// <returns>The code generation result.</returns>
    abstract member GenerateCodeFromTemplate : templateName:string * placeholderValues:Map<string, string> -> Task<CodeGenerationResult>
    
    /// <summary>
    /// Gets all available templates.
    /// </summary>
    /// <returns>The list of available templates.</returns>
    abstract member GetAvailableTemplates : unit -> Task<CodeGenerationTemplate list>
    
    /// <summary>
    /// Gets a template by name.
    /// </summary>
    /// <param name="templateName">The name of the template to get.</param>
    /// <returns>The template, if found.</returns>
    abstract member GetTemplateByName : templateName:string -> Task<CodeGenerationTemplate option>
    
    /// <summary>
    /// Gets templates by category.
    /// </summary>
    /// <param name="category">The category of templates to get.</param>
    /// <returns>The list of templates in the category.</returns>
    abstract member GetTemplatesByCategory : category:string -> Task<CodeGenerationTemplate list>
    
    /// <summary>
    /// Gets templates by tag.
    /// </summary>
    /// <param name="tag">The tag of templates to get.</param>
    /// <returns>The list of templates with the tag.</returns>
    abstract member GetTemplatesByTag : tag:string -> Task<CodeGenerationTemplate list>

/// <summary>
/// Interface for code refactorer.
/// </summary>
type IRefactorer =
    /// <summary>
    /// Gets the language supported by this refactorer.
    /// </summary>
    abstract member Language : string
    
    /// <summary>
    /// Refactors code using a specific refactoring.
    /// </summary>
    /// <param name="code">The code to refactor.</param>
    /// <param name="refactoring">The refactoring to apply.</param>
    /// <returns>The code refactoring result.</returns>
    abstract member RefactorCode : code:string * refactoring:CodeRefactoring -> CodeRefactoringResult
    
    /// <summary>
    /// Refactors code using a refactoring name.
    /// </summary>
    /// <param name="code">The code to refactor.</param>
    /// <param name="refactoringName">The name of the refactoring to apply.</param>
    /// <returns>The code refactoring result.</returns>
    abstract member RefactorCodeWithName : code:string * refactoringName:string -> Task<CodeRefactoringResult>
    
    /// <summary>
    /// Refactors a file using a specific refactoring.
    /// </summary>
    /// <param name="filePath">The path to the file to refactor.</param>
    /// <param name="refactoring">The refactoring to apply.</param>
    /// <returns>The code refactoring result.</returns>
    abstract member RefactorFile : filePath:string * refactoring:CodeRefactoring -> Task<CodeRefactoringResult>
    
    /// <summary>
    /// Refactors a file using a refactoring name.
    /// </summary>
    /// <param name="filePath">The path to the file to refactor.</param>
    /// <param name="refactoringName">The name of the refactoring to apply.</param>
    /// <returns>The code refactoring result.</returns>
    abstract member RefactorFileWithName : filePath:string * refactoringName:string -> Task<CodeRefactoringResult>
    
    /// <summary>
    /// Gets all available refactorings.
    /// </summary>
    /// <returns>The list of available refactorings.</returns>
    abstract member GetAvailableRefactorings : unit -> Task<CodeRefactoring list>
    
    /// <summary>
    /// Gets a refactoring by name.
    /// </summary>
    /// <param name="refactoringName">The name of the refactoring to get.</param>
    /// <returns>The refactoring, if found.</returns>
    abstract member GetRefactoringByName : refactoringName:string -> Task<CodeRefactoring option>
    
    /// <summary>
    /// Gets refactorings by category.
    /// </summary>
    /// <param name="category">The category of refactorings to get.</param>
    /// <returns>The list of refactorings in the category.</returns>
    abstract member GetRefactoringsByCategory : category:string -> Task<CodeRefactoring list>
    
    /// <summary>
    /// Gets refactorings by tag.
    /// </summary>
    /// <param name="tag">The tag of refactorings to get.</param>
    /// <returns>The list of refactorings with the tag.</returns>
    abstract member GetRefactoringsByTag : tag:string -> Task<CodeRefactoring list>
    
    /// <summary>
    /// Suggests refactorings for code.
    /// </summary>
    /// <param name="code">The code to suggest refactorings for.</param>
    /// <returns>The list of suggested refactorings.</returns>
    abstract member SuggestRefactorings : code:string -> Task<CodeRefactoring list>
    
    /// <summary>
    /// Suggests refactorings for a file.
    /// </summary>
    /// <param name="filePath">The path to the file to suggest refactorings for.</param>
    /// <returns>The list of suggested refactorings.</returns>
    abstract member SuggestRefactoringsForFile : filePath:string -> Task<CodeRefactoring list>

/// <summary>
/// Interface for test generator.
/// </summary>
type ITestGenerator =
    /// <summary>
    /// Gets the language supported by this test generator.
    /// </summary>
    abstract member Language : string
    
    /// <summary>
    /// Generates tests for code.
    /// </summary>
    /// <param name="code">The code to generate tests for.</param>
    /// <param name="testFramework">The test framework to use.</param>
    /// <returns>The test generation result.</returns>
    abstract member GenerateTests : code:string * testFramework:string -> Task<TestGenerationResult>
    
    /// <summary>
    /// Generates tests for a file.
    /// </summary>
    /// <param name="filePath">The path to the file to generate tests for.</param>
    /// <param name="testFramework">The test framework to use.</param>
    /// <returns>The test generation result.</returns>
    abstract member GenerateTestsForFile : filePath:string * testFramework:string -> Task<TestGenerationResult>
    
    /// <summary>
    /// Generates tests for a project.
    /// </summary>
    /// <param name="projectPath">The path to the project to generate tests for.</param>
    /// <param name="testFramework">The test framework to use.</param>
    /// <returns>The list of test generation results.</returns>
    abstract member GenerateTestsForProject : projectPath:string * testFramework:string -> Task<TestGenerationResult list>
    
    /// <summary>
    /// Gets the supported test frameworks.
    /// </summary>
    /// <returns>The list of supported test frameworks.</returns>
    abstract member GetSupportedTestFrameworks : unit -> string list
    
    /// <summary>
    /// Suggests tests for code.
    /// </summary>
    /// <param name="code">The code to suggest tests for.</param>
    /// <param name="testFramework">The test framework to use.</param>
    /// <returns>The list of suggested tests.</returns>
    abstract member SuggestTests : code:string * testFramework:string -> Task<SuggestedTest list>
    
    /// <summary>
    /// Suggests tests for a file.
    /// </summary>
    /// <param name="filePath">The path to the file to suggest tests for.</param>
    /// <param name="testFramework">The test framework to use.</param>
    /// <returns>The list of suggested tests.</returns>
    abstract member SuggestTestsForFile : filePath:string * testFramework:string -> Task<SuggestedTest list>

/// <summary>
/// Interface for test runner.
/// </summary>
type ITestRunner =
    /// <summary>
    /// Runs tests in a file.
    /// </summary>
    /// <param name="testFilePath">The path to the test file to run.</param>
    /// <returns>The test run result.</returns>
    abstract member RunTests : testFilePath:string -> Task<TestRunResult>
    
    /// <summary>
    /// Runs tests in a project.
    /// </summary>
    /// <param name="testProjectPath">The path to the test project to run.</param>
    /// <returns>The test run result.</returns>
    abstract member RunTestsInProject : testProjectPath:string -> Task<TestRunResult>
    
    /// <summary>
    /// Runs tests in a solution.
    /// </summary>
    /// <param name="solutionPath">The path to the solution to run tests in.</param>
    /// <returns>The test run result.</returns>
    abstract member RunTestsInSolution : solutionPath:string -> Task<TestRunResult>
    
    /// <summary>
    /// Runs a specific test.
    /// </summary>
    /// <param name="testFilePath">The path to the test file.</param>
    /// <param name="testName">The name of the test to run.</param>
    /// <returns>The test result.</returns>
    abstract member RunTest : testFilePath:string * testName:string -> Task<TestResult>
    
    /// <summary>
    /// Gets the test coverage.
    /// </summary>
    /// <param name="testProjectPath">The path to the test project.</param>
    /// <returns>The test coverage result.</returns>
    abstract member GetTestCoverage : testProjectPath:string -> Task<TestCoverageResult>
    
    /// <summary>
    /// Gets the test coverage for a specific file.
    /// </summary>
    /// <param name="testProjectPath">The path to the test project.</param>
    /// <param name="filePath">The path to the file to get coverage for.</param>
    /// <returns>The test coverage result.</returns>
    abstract member GetTestCoverageForFile : testProjectPath:string * filePath:string -> Task<TestCoverageResult>

/// <summary>
/// Interface for regression testing service.
/// </summary>
type IRegressionTestingService =
    /// <summary>
    /// Runs regression tests for a specific file.
    /// </summary>
    /// <param name="filePath">Path to the file to test.</param>
    /// <param name="projectPath">Path to the project containing the file.</param>
    /// <returns>Regression test result.</returns>
    abstract member RunRegressionTestsAsync : filePath:string * projectPath:string -> Task<RegressionTestResult>
    
    /// <summary>
    /// Runs regression tests for a specific project.
    /// </summary>
    /// <param name="projectPath">Path to the project to test.</param>
    /// <returns>Regression test result.</returns>
    abstract member RunRegressionTestsForProjectAsync : projectPath:string -> Task<RegressionTestResult>
    
    /// <summary>
    /// Runs regression tests for a specific solution.
    /// </summary>
    /// <param name="solutionPath">Path to the solution to test.</param>
    /// <returns>Regression test result.</returns>
    abstract member RunRegressionTestsForSolutionAsync : solutionPath:string -> Task<RegressionTestResult>
    
    /// <summary>
    /// Identifies potential regression issues.
    /// </summary>
    /// <param name="originalCode">Original code.</param>
    /// <param name="improvedCode">Improved code.</param>
    /// <param name="language">Programming language.</param>
    /// <returns>List of potential regression issues.</returns>
    abstract member IdentifyPotentialRegressionIssuesAsync : originalCode:string * improvedCode:string * language:string -> Task<RegressionIssue list>
    
    /// <summary>
    /// Tracks test coverage.
    /// </summary>
    /// <param name="projectPath">Path to the project to track.</param>
    /// <returns>Test coverage result.</returns>
    abstract member TrackTestCoverageAsync : projectPath:string -> Task<TestCoverageResult>
    
    /// <summary>
    /// Suggests additional tests to improve coverage.
    /// </summary>
    /// <param name="coverageResult">Test coverage result.</param>
    /// <param name="projectPath">Path to the project.</param>
    /// <returns>List of suggested tests.</returns>
    abstract member SuggestAdditionalTestsAsync : coverageResult:TestCoverageResult * projectPath:string -> Task<SuggestedTest list>

/// <summary>
/// Interface for documentation generator.
/// </summary>
type IDocumentationGenerator =
    /// <summary>
    /// Gets the language supported by this documentation generator.
    /// </summary>
    abstract member Language : string
    
    /// <summary>
    /// Generates documentation for code.
    /// </summary>
    /// <param name="code">The code to generate documentation for.</param>
    /// <param name="format">The documentation format.</param>
    /// <returns>The documentation generation result.</returns>
    abstract member GenerateDocumentation : code:string * format:string -> Task<DocumentationGenerationResult>
    
    /// <summary>
    /// Generates documentation for a file.
    /// </summary>
    /// <param name="filePath">The path to the file to generate documentation for.</param>
    /// <param name="format">The documentation format.</param>
    /// <returns>The documentation generation result.</returns>
    abstract member GenerateDocumentationForFile : filePath:string * format:string -> Task<DocumentationGenerationResult>
    
    /// <summary>
    /// Generates documentation for a project.
    /// </summary>
    /// <param name="projectPath">The path to the project to generate documentation for.</param>
    /// <param name="format">The documentation format.</param>
    /// <returns>The list of documentation generation results.</returns>
    abstract member GenerateDocumentationForProject : projectPath:string * format:string -> Task<DocumentationGenerationResult list>
    
    /// <summary>
    /// Gets the supported documentation formats.
    /// </summary>
    /// <returns>The list of supported documentation formats.</returns>
    abstract member GetSupportedDocumentationFormats : unit -> string list
    
    /// <summary>
    /// Updates documentation for code.
    /// </summary>
    /// <param name="code">The code to update documentation for.</param>
    /// <param name="existingDocumentation">The existing documentation.</param>
    /// <param name="format">The documentation format.</param>
    /// <returns>The documentation generation result.</returns>
    abstract member UpdateDocumentation : code:string * existingDocumentation:string * format:string -> Task<DocumentationGenerationResult>
    
    /// <summary>
    /// Updates documentation for a file.
    /// </summary>
    /// <param name="filePath">The path to the file to update documentation for.</param>
    /// <param name="format">The documentation format.</param>
    /// <returns>The documentation generation result.</returns>
    abstract member UpdateDocumentationForFile : filePath:string * format:string -> Task<DocumentationGenerationResult>

/// <summary>
/// Interface for workflow coordinator.
/// </summary>
type IWorkflowCoordinator =
    /// <summary>
    /// Executes a workflow.
    /// </summary>
    /// <param name="workflowName">The name of the workflow to execute.</param>
    /// <param name="parameters">The parameters for the workflow.</param>
    /// <returns>The result of the workflow execution.</returns>
    abstract member ExecuteWorkflow : workflowName:string * parameters:Map<string, obj> -> Task<Map<string, obj>>
    
    /// <summary>
    /// Gets all available workflows.
    /// </summary>
    /// <returns>The list of available workflows.</returns>
    abstract member GetAvailableWorkflows : unit -> Task<string list>
    
    /// <summary>
    /// Gets the parameters for a workflow.
    /// </summary>
    /// <param name="workflowName">The name of the workflow.</param>
    /// <returns>The parameters for the workflow.</returns>
    abstract member GetWorkflowParameters : workflowName:string -> Task<Map<string, string>>
    
    /// <summary>
    /// Creates a new workflow.
    /// </summary>
    /// <param name="workflowName">The name of the workflow.</param>
    /// <param name="workflowDefinition">The definition of the workflow.</param>
    /// <returns>Whether the workflow was created successfully.</returns>
    abstract member CreateWorkflow : workflowName:string * workflowDefinition:string -> Task<bool>
    
    /// <summary>
    /// Updates a workflow.
    /// </summary>
    /// <param name="workflowName">The name of the workflow.</param>
    /// <param name="workflowDefinition">The new definition of the workflow.</param>
    /// <returns>Whether the workflow was updated successfully.</returns>
    abstract member UpdateWorkflow : workflowName:string * workflowDefinition:string -> Task<bool>
    
    /// <summary>
    /// Deletes a workflow.
    /// </summary>
    /// <param name="workflowName">The name of the workflow to delete.</param>
    /// <returns>Whether the workflow was deleted successfully.</returns>
    abstract member DeleteWorkflow : workflowName:string -> Task<bool>
    
    /// <summary>
    /// Gets the status of a workflow execution.
    /// </summary>
    /// <param name="executionId">The ID of the workflow execution.</param>
    /// <returns>The status of the workflow execution.</returns>
    abstract member GetWorkflowStatus : executionId:string -> Task<string>
    
    /// <summary>
    /// Cancels a workflow execution.
    /// </summary>
    /// <param name="executionId">The ID of the workflow execution to cancel.</param>
    /// <returns>Whether the workflow execution was cancelled successfully.</returns>
    abstract member CancelWorkflow : executionId:string -> Task<bool>
