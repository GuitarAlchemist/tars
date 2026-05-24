using TarsCli.Services.CodeAnalysis;
using TarsCli.Services.CodeGeneration;
using TarsCli.Services.Testing;
using TarsCli.Services.Adapters;
using TestingTestGenerationResult = TarsCli.Services.Testing.TestGenerationResult;

namespace TarsCli.Services.Workflow;

/// <summary>
/// Definition of the self-coding workflow
/// </summary>
public class SelfCodingWorkflowDefinition : IWorkflowDefinition
{
    private readonly ILogger<SelfCodingWorkflowDefinition> _logger;
    private readonly CodeAnalyzerService _codeAnalyzerService;
    private readonly CodeGeneratorService _codeGeneratorService;
    private readonly TestGeneratorService _testGeneratorService;
    private readonly Testing.TestRunnerService _testRunnerService;
    private readonly TestResultAnalyzer _testResultAnalyzer;

    // Define workflow states
    private const string STATE_INITIALIZE = "Initialize";
    private const string STATE_SELECT_FILES = "SelectFiles";
    private const string STATE_ANALYZE_CODE = "AnalyzeCode";
    private const string STATE_GENERATE_CODE = "GenerateCode";
    private const string STATE_GENERATE_TESTS = "GenerateTests";
    private const string STATE_RUN_TESTS = "RunTests";
    private const string STATE_APPLY_CHANGES = "ApplyChanges";
    private const string STATE_LEARN = "Learn";
    private const string STATE_COMPLETE = "Complete";
    private const string STATE_FAILED = "Failed";

    // Define valid transitions
    private readonly Dictionary<string, List<string>> _validTransitions = new()
    {
        { STATE_INITIALIZE, [STATE_SELECT_FILES, STATE_FAILED] },
        { STATE_SELECT_FILES, [STATE_ANALYZE_CODE, STATE_FAILED] },
        { STATE_ANALYZE_CODE, [STATE_GENERATE_CODE, STATE_COMPLETE, STATE_FAILED] },
        { STATE_GENERATE_CODE, [STATE_GENERATE_TESTS, STATE_FAILED] },
        { STATE_GENERATE_TESTS, [STATE_RUN_TESTS, STATE_FAILED] },
        { STATE_RUN_TESTS, [STATE_APPLY_CHANGES, STATE_GENERATE_CODE, STATE_FAILED] },
        { STATE_APPLY_CHANGES, [STATE_LEARN, STATE_FAILED] },
        { STATE_LEARN, [STATE_COMPLETE, STATE_FAILED] }
    };

    // Define final states
    private readonly HashSet<string> _finalStates =
    [
        STATE_COMPLETE,
        STATE_FAILED
    ];

    /// <inheritdoc/>
    public string Type => "SelfCoding";

    /// <inheritdoc/>
    public string Name => "Self-Coding Workflow";

    /// <inheritdoc/>
    public string Description => "Workflow for autonomous self-improvement of code";

    /// <summary>
    /// Initializes a new instance of the SelfCodingWorkflowDefinition class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="codeAnalyzerService">Code analyzer service</param>
    /// <param name="codeGeneratorService">Code generator service</param>
    /// <param name="testGeneratorService">Test generator service</param>
    /// <param name="testRunnerService">Test runner service</param>
    /// <param name="testResultAnalyzer">Test result analyzer</param>
    public SelfCodingWorkflowDefinition(
        ILogger<SelfCodingWorkflowDefinition> logger,
        CodeAnalyzerService codeAnalyzerService,
        CodeGeneratorService codeGeneratorService,
        TestGeneratorService testGeneratorService,
        Testing.TestRunnerService testRunnerService,
        TestResultAnalyzer testResultAnalyzer)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _codeAnalyzerService = codeAnalyzerService ?? throw new ArgumentNullException(nameof(codeAnalyzerService));
        _codeGeneratorService = codeGeneratorService ?? throw new ArgumentNullException(nameof(codeGeneratorService));
        _testGeneratorService = testGeneratorService ?? throw new ArgumentNullException(nameof(testGeneratorService));
        _testRunnerService = testRunnerService ?? throw new ArgumentNullException(nameof(testRunnerService));
        _testResultAnalyzer = testResultAnalyzer ?? throw new ArgumentNullException(nameof(testResultAnalyzer));
    }

    /// <inheritdoc/>
    public string GetInitialState()
    {
        return STATE_INITIALIZE;
    }

    /// <inheritdoc/>
    public string GetNextState(string currentState, object result)
    {
        switch (currentState)
        {
            case STATE_INITIALIZE:
                return STATE_SELECT_FILES;

            case STATE_SELECT_FILES:
                var selectedFiles = result as List<string>;
                return selectedFiles != null && selectedFiles.Count > 0 ? STATE_ANALYZE_CODE : STATE_COMPLETE;

            case STATE_ANALYZE_CODE:
                var analysisResults = result as List<CodeAnalysisResult>;
                return analysisResults != null && analysisResults.Any(r => r.NeedsImprovement) ? STATE_GENERATE_CODE : STATE_COMPLETE;

            case STATE_GENERATE_CODE:
                return STATE_GENERATE_TESTS;

            case STATE_GENERATE_TESTS:
                return STATE_RUN_TESTS;

            case STATE_RUN_TESTS:
                var testRunResult = result as TestRunResult;
                return testRunResult != null && testRunResult.Success ? STATE_APPLY_CHANGES : STATE_GENERATE_CODE;

            case STATE_APPLY_CHANGES:
                return STATE_LEARN;

            case STATE_LEARN:
                return STATE_COMPLETE;

            default:
                return null;
        }
    }

    /// <inheritdoc/>
    public bool IsFinalState(string state)
    {
        return _finalStates.Contains(state);
    }

    /// <inheritdoc/>
    public bool IsValidTransition(string fromState, string toState)
    {
        return _validTransitions.TryGetValue(fromState, out var validStates) && validStates.Contains(toState);
    }

    /// <inheritdoc/>
    public ParameterValidationResult ValidateParameters(Dictionary<string, object> parameters)
    {
        // Check if the target directory is specified
        if (!parameters.TryGetValue("TargetDirectory", out var targetDirectoryObj) || targetDirectoryObj == null)
        {
            return ParameterValidationResult.Failure("Target directory is required");
        }

        var targetDirectory = targetDirectoryObj.ToString();
        if (!Directory.Exists(targetDirectory))
        {
            return ParameterValidationResult.Failure($"Target directory {targetDirectory} does not exist");
        }

        // Check if the file patterns are specified
        if (!parameters.TryGetValue("FilePatterns", out var filePatternsObj) || filePatternsObj == null)
        {
            return ParameterValidationResult.Failure("File patterns are required");
        }

        // Check if the max files parameter is valid
        if (parameters.TryGetValue("MaxFiles", out var maxFilesObj) && maxFilesObj != null)
        {
            if (!int.TryParse(maxFilesObj.ToString(), out var maxFiles) || maxFiles <= 0)
            {
                return ParameterValidationResult.Failure("Max files must be a positive integer");
            }
        }

        return ParameterValidationResult.Success();
    }

    /// <inheritdoc/>
    public async Task<object> ExecuteStateAsync(WorkflowInstance workflow, string state)
    {
        _logger.LogInformation($"Executing state {state} for workflow {workflow.Id}");

        try
        {
            switch (state)
            {
                case STATE_INITIALIZE:
                    return await InitializeAsync(workflow);

                case STATE_SELECT_FILES:
                    return await SelectFilesAsync(workflow);

                case STATE_ANALYZE_CODE:
                    return await AnalyzeCodeAsync(workflow);

                case STATE_GENERATE_CODE:
                    return await GenerateCodeAsync(workflow);

                case STATE_GENERATE_TESTS:
                    return await GenerateTestsAsync(workflow);

                case STATE_RUN_TESTS:
                    return await RunTestsAsync(workflow);

                case STATE_APPLY_CHANGES:
                    return await ApplyChangesAsync(workflow);

                case STATE_LEARN:
                    return await LearnAsync(workflow);

                case STATE_COMPLETE:
                    return await CompleteAsync(workflow);

                case STATE_FAILED:
                    return await FailedAsync(workflow);

                default:
                    throw new ArgumentException($"Unknown state: {state}");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing state {state} for workflow {workflow.Id}");
            throw;
        }
    }

    /// <summary>
    /// Initializes the workflow
    /// </summary>
    /// <param name="workflow">Workflow instance</param>
    /// <returns>Initialization result</returns>
    private async Task<object> InitializeAsync(WorkflowInstance workflow)
    {
        _logger.LogInformation($"Initializing workflow {workflow.Id}");

        // Create a result object
        var result = new Dictionary<string, object>
        {
            ["Status"] = "Initialized"
        };

        return await Task.FromResult(result);
    }

    /// <summary>
    /// Selects files for improvement
    /// </summary>
    /// <param name="workflow">Workflow instance</param>
    /// <returns>List of selected files</returns>
    private async Task<object> SelectFilesAsync(WorkflowInstance workflow)
    {
        _logger.LogInformation($"Selecting files for workflow {workflow.Id}");

        // Get parameters
        var targetDirectory = workflow.Parameters["TargetDirectory"].ToString();
        var filePatterns = workflow.Parameters["FilePatterns"] as IEnumerable<string> ?? [workflow.Parameters["FilePatterns"].ToString()
        ];
        var maxFiles = workflow.Parameters.TryGetValue("MaxFiles", out var maxFilesObj) && maxFilesObj != null
            ? int.Parse(maxFilesObj.ToString())
            : 10;

        // Find files matching the patterns
        var selectedFiles = new List<string>();
        foreach (var pattern in filePatterns)
        {
            var files = Directory.GetFiles(targetDirectory, pattern, SearchOption.AllDirectories);
            selectedFiles.AddRange(files);
        }

        // Limit the number of files
        if (selectedFiles.Count > maxFiles)
        {
            selectedFiles = selectedFiles.Take(maxFiles).ToList();
        }

        _logger.LogInformation($"Selected {selectedFiles.Count} files for workflow {workflow.Id}");

        // Store the selected files in the workflow results
        workflow.Results["SelectedFiles"] = selectedFiles;

        return await Task.FromResult(selectedFiles);
    }

    /// <summary>
    /// Analyzes code for improvement opportunities
    /// </summary>
    /// <param name="workflow">Workflow instance</param>
    /// <returns>List of analysis results</returns>
    private async Task<object> AnalyzeCodeAsync(WorkflowInstance workflow)
    {
        _logger.LogInformation($"Analyzing code for workflow {workflow.Id}");

        // Get the selected files
        var selectedFiles = workflow.Results["SelectedFiles"] as List<string>;
        if (selectedFiles == null || selectedFiles.Count == 0)
        {
            _logger.LogWarning($"No files selected for workflow {workflow.Id}");
            return new List<CodeAnalysisResult>();
        }

        // Analyze each file
        var analysisResults = new List<CodeAnalysisResult>();
        foreach (var file in selectedFiles)
        {
            var result = await _codeAnalyzerService.AnalyzeFileAsync(file);
            if (result != null)
            {
                analysisResults.Add(result);
            }
        }

        _logger.LogInformation($"Analyzed {analysisResults.Count} files for workflow {workflow.Id}");

        // Store the analysis results in the workflow results
        workflow.Results["AnalysisResults"] = analysisResults;

        return analysisResults;
    }

    /// <summary>
    /// Generates improved code
    /// </summary>
    /// <param name="workflow">Workflow instance</param>
    /// <returns>List of code generation results</returns>
    private async Task<object> GenerateCodeAsync(WorkflowInstance workflow)
    {
        _logger.LogInformation($"Generating code for workflow {workflow.Id}");

        // Get the analysis results
        var analysisResults = workflow.Results["AnalysisResults"] as List<CodeAnalysisResult>;
        if (analysisResults == null || analysisResults.Count == 0)
        {
            _logger.LogWarning($"No analysis results for workflow {workflow.Id}");
            return new List<CodeGenerationResult>();
        }

        // Generate improved code for each file that needs improvement
        var generationResults = new List<CodeGenerationResult>();
        foreach (var analysisResult in analysisResults.Where(r => r.NeedsImprovement))
        {
            var result = await _codeGeneratorService.GenerateCodeAsync(analysisResult.FilePath);
            if (result != null)
            {
                generationResults.Add(result);
            }
        }

        _logger.LogInformation($"Generated code for {generationResults.Count} files for workflow {workflow.Id}");

        // Store the generation results in the workflow results
        workflow.Results["GenerationResults"] = generationResults;

        return generationResults;
    }

    /// <summary>
    /// Generates tests for the improved code
    /// </summary>
    /// <param name="workflow">Workflow instance</param>
    /// <returns>List of test generation results</returns>
    private async Task<object> GenerateTestsAsync(WorkflowInstance workflow)
    {
        _logger.LogInformation($"Generating tests for workflow {workflow.Id}");

        // Get the generation results
        var generationResults = workflow.Results["GenerationResults"] as List<CodeGenerationResult>;
        if (generationResults == null || generationResults.Count == 0)
        {
            _logger.LogWarning($"No generation results for workflow {workflow.Id}");
            return new List<TestGenerationResult>();
        }

        // Generate tests for each file
        var testGenerationResults = new List<TestGenerationResult>();
        foreach (var generationResult in generationResults)
        {
            var result = await _testGeneratorService.GenerateTestsAsync(generationResult.FilePath);
            if (result != null)
            {
                testGenerationResults.Add(TestGenerationResultAdapter.ToServiceTestGenerationResult(result));
                await _testGeneratorService.SaveTestsAsync(result);
            }
        }

        _logger.LogInformation($"Generated tests for {testGenerationResults.Count} files for workflow {workflow.Id}");

        // Store the test generation results in the workflow results
        workflow.Results["TestGenerationResults"] = testGenerationResults;

        return testGenerationResults;
    }

    /// <summary>
    /// Runs the generated tests
    /// </summary>
    /// <param name="workflow">Workflow instance</param>
    /// <returns>Test run result</returns>
    private async Task<object> RunTestsAsync(WorkflowInstance workflow)
    {
        _logger.LogInformation($"Running tests for workflow {workflow.Id}");

        // Get the test generation results
        var testGenerationResults = workflow.Results["TestGenerationResults"] as List<TestGenerationResult>;
        if (testGenerationResults == null || testGenerationResults.Count == 0)
        {
            _logger.LogWarning($"No test generation results for workflow {workflow.Id}");
            return null;
        }

        // Find the test project
        var testFilePath = TestGenerationResultAdapter.ToTestingTestGenerationResult(testGenerationResults.First()).TestFilePath;
        var projectPath = FindProjectFile(testFilePath);
        if (string.IsNullOrEmpty(projectPath))
        {
            _logger.LogError($"Could not find project file for test file: {testFilePath}");
            return null;
        }

        // Run the tests
        var testRunResult = await _testRunnerService.RunTestsAsync(projectPath);

        _logger.LogInformation($"Ran tests for workflow {workflow.Id}. Success: {testRunResult.Success}, Total tests: {testRunResult.TotalCount}");

        // Analyze the test results
        var testAnalysisResult = await _testResultAnalyzer.AnalyzeResultsAsync(testRunResult);

        // Store the test run and analysis results in the workflow results
        workflow.Results["TestRunResult"] = testRunResult;
        workflow.Results["TestAnalysisResult"] = testAnalysisResult;

        return testRunResult;
    }

    /// <summary>
    /// Applies the generated code changes
    /// </summary>
    /// <param name="workflow">Workflow instance</param>
    /// <returns>List of applied changes</returns>
    private async Task<object> ApplyChangesAsync(WorkflowInstance workflow)
    {
        _logger.LogInformation($"Applying changes for workflow {workflow.Id}");

        // Get the generation results
        var generationResults = workflow.Results["GenerationResults"] as List<CodeGenerationResult>;
        if (generationResults == null || generationResults.Count == 0)
        {
            _logger.LogWarning($"No generation results for workflow {workflow.Id}");
            return new List<string>();
        }

        // Apply the changes
        var appliedChanges = new List<string>();
        foreach (var generationResult in generationResults)
        {
            var success = await _codeGeneratorService.ApplyGeneratedCodeAsync(generationResult, true);
            if (success)
            {
                appliedChanges.Add(generationResult.FilePath);
            }
        }

        _logger.LogInformation($"Applied changes to {appliedChanges.Count} files for workflow {workflow.Id}");

        // Store the applied changes in the workflow results
        workflow.Results["AppliedChanges"] = appliedChanges;

        return appliedChanges;
    }

    /// <summary>
    /// Learns from the improvements
    /// </summary>
    /// <param name="workflow">Workflow instance</param>
    /// <returns>Learning result</returns>
    private async Task<object> LearnAsync(WorkflowInstance workflow)
    {
        _logger.LogInformation($"Learning from improvements for workflow {workflow.Id}");

        // Get the applied changes
        var appliedChanges = workflow.Results["AppliedChanges"] as List<string>;
        if (appliedChanges == null || appliedChanges.Count == 0)
        {
            _logger.LogWarning($"No applied changes for workflow {workflow.Id}");
            return new Dictionary<string, object>();
        }

        // Create a learning result
        var learningResult = new Dictionary<string, object>
        {
            ["ImprovedFiles"] = appliedChanges.Count,
            ["Patterns"] = new List<string>()
        };

        _logger.LogInformation($"Learned from improvements for workflow {workflow.Id}");

        return await Task.FromResult(learningResult);
    }

    /// <summary>
    /// Completes the workflow
    /// </summary>
    /// <param name="workflow">Workflow instance</param>
    /// <returns>Completion result</returns>
    private async Task<object> CompleteAsync(WorkflowInstance workflow)
    {
        _logger.LogInformation($"Completing workflow {workflow.Id}");

        // Create a completion result
        var completionResult = new Dictionary<string, object>
        {
            ["Status"] = "Completed",
            ["SelectedFiles"] = (workflow.Results.TryGetValue("SelectedFiles", out var selectedFiles) ? selectedFiles : null) ?? new List<string>(),
            ["AnalyzedFiles"] = (workflow.Results.TryGetValue("AnalysisResults", out var analysisResults) ? (analysisResults as List<CodeAnalysisResult>)?.Count : 0) ?? 0,
            ["GeneratedFiles"] = (workflow.Results.TryGetValue("GenerationResults", out var generationResults) ? (generationResults as List<CodeGenerationResult>)?.Count : 0) ?? 0,
            ["AppliedChanges"] = (workflow.Results.TryGetValue("AppliedChanges", out var appliedChanges) ? appliedChanges : null) ?? new List<string>()
        };

        return await Task.FromResult(completionResult);
    }

    /// <summary>
    /// Handles workflow failure
    /// </summary>
    /// <param name="workflow">Workflow instance</param>
    /// <returns>Failure result</returns>
    private async Task<object> FailedAsync(WorkflowInstance workflow)
    {
        _logger.LogInformation($"Handling failure for workflow {workflow.Id}");

        // Create a failure result
        var failureResult = new Dictionary<string, object>
        {
            ["Status"] = "Failed",
            ["ErrorMessage"] = workflow.ErrorMessage
        };

        return await Task.FromResult(failureResult);
    }

    /// <summary>
    /// Finds the project file for a test file
    /// </summary>
    /// <param name="testFilePath">Path to the test file</param>
    /// <returns>Path to the project file, or null if not found</returns>
    private string FindProjectFile(string testFilePath)
    {
        var directory = Path.GetDirectoryName(testFilePath);
        while (!string.IsNullOrEmpty(directory))
        {
            var projectFiles = Directory.GetFiles(directory, "*.csproj").Concat(Directory.GetFiles(directory, "*.fsproj"));
            if (projectFiles.Any())
            {
                return projectFiles.First();
            }

            directory = Path.GetDirectoryName(directory);
        }

        return null;
    }
}