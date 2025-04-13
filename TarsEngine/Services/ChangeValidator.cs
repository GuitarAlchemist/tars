using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Adapters;

namespace TarsEngine.Services;

/// <summary>
/// Validates changes to ensure quality and functionality
/// </summary>
public class ChangeValidator
{
    private readonly ILogger<ChangeValidator> _logger;
    private readonly SyntaxValidator _syntaxValidator;
    private readonly SemanticValidator _semanticValidator;
    private readonly TestExecutor _testExecutor;

    /// <summary>
    /// Initializes a new instance of the <see cref="ChangeValidator"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="syntaxValidator">The syntax validator</param>
    /// <param name="semanticValidator">The semantic validator</param>
    /// <param name="testExecutor">The test executor</param>
    public ChangeValidator(
        ILogger<ChangeValidator> logger,
        SyntaxValidator syntaxValidator,
        SemanticValidator semanticValidator,
        TestExecutor testExecutor)
    {
        _logger = logger;
        _syntaxValidator = syntaxValidator;
        _semanticValidator = semanticValidator;
        _testExecutor = testExecutor;
    }

    /// <summary>
    /// Validates a file
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="filePath">The file path</param>
    /// <param name="content">The file content</param>
    /// <param name="projectPath">The project path</param>
    /// <param name="options">Optional validation options</param>
    /// <returns>The validation result</returns>
    public async Task<ChangeValidationResult> ValidateFileAsync(
        string contextId,
        string filePath,
        string content,
        string projectPath,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Validating file: {FilePath}", filePath);

            // Create result
            var result = new ChangeValidationResult
            {
                FilePath = filePath,
                ProjectPath = projectPath,
                StartedAt = DateTime.UtcNow
            };

            // Parse options
            var validateSyntax = ParseOption(options, "ValidateSyntax", true);
            var validateSemantics = ParseOption(options, "ValidateSemantics", true);
            var runTests = ParseOption(options, "RunTests", true);
            var testFilter = options?.TryGetValue("TestFilter", out var tf) == true ? tf : null;

            // Validate syntax
            if (validateSyntax)
            {
                _logger.LogInformation("Validating syntax of file: {FilePath}", filePath);
                var syntaxResults = await _syntaxValidator.ValidateFileSyntaxAsync(filePath, content);
                result.ValidationResults.AddRange(syntaxResults);

                // Check if syntax validation failed
                if (syntaxResults.Any(r => r.Severity == ValidationRuleSeverity.Error && !r.IsPassed))
                {
                    result.IsSuccessful = false;
                    result.ErrorMessage = "Syntax validation failed";
                    result.CompletedAt = DateTime.UtcNow;
                    result.DurationMs = (long)(result.CompletedAt - result.StartedAt).TotalMilliseconds;
                    return result;
                }
            }

            // Validate semantics
            if (validateSemantics)
            {
                _logger.LogInformation("Validating semantics of file: {FilePath}", filePath);
                var semanticResults = await _semanticValidator.ValidateFileSemanticsAsync(filePath, content, projectPath);
                result.ValidationResults.AddRange(semanticResults);

                // Check if semantic validation failed
                if (semanticResults.Any(r => r.Severity == ValidationRuleSeverity.Error && !r.IsPassed))
                {
                    result.IsSuccessful = false;
                    result.ErrorMessage = "Semantic validation failed";
                    result.CompletedAt = DateTime.UtcNow;
                    result.DurationMs = (long)(result.CompletedAt - result.StartedAt).TotalMilliseconds;
                    return result;
                }
            }

            // Run tests
            if (runTests)
            {
                _logger.LogInformation("Running tests for file: {FilePath}", filePath);
                var testResult = await _testExecutor.ExecuteTestsAsync(contextId, projectPath, testFilter, options);
                var testValidationResults = _testExecutor.ConvertToValidationResults(testResult, filePath);
                result.ValidationResults.AddRange(testValidationResults);
                result.TestResult = TestExecutionResultAdapter.ToModel(testResult);

                // Check if tests failed
                if (!testResult.IsSuccessful)
                {
                    result.IsSuccessful = false;
                    result.ErrorMessage = "Tests failed";
                    result.CompletedAt = DateTime.UtcNow;
                    result.DurationMs = (long)(result.CompletedAt - result.StartedAt).TotalMilliseconds;
                    return result;
                }
            }

            // All validations passed
            result.IsSuccessful = true;
            result.CompletedAt = DateTime.UtcNow;
            result.DurationMs = (long)(result.CompletedAt - result.StartedAt).TotalMilliseconds;
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating file: {FilePath}", filePath);
            return new ChangeValidationResult
            {
                FilePath = filePath,
                ProjectPath = projectPath,
                IsSuccessful = false,
                ErrorMessage = ex.Message,
                StartedAt = DateTime.UtcNow,
                CompletedAt = DateTime.UtcNow,
                DurationMs = 0,
                ValidationResults =
                [
                    new ValidationResult
                    {
                        RuleName = "ChangeValidation",
                        IsPassed = false,
                        Severity = ValidationRuleSeverity.Error,
                        Message = $"Error validating file: {ex.Message}",
                        Target = filePath,
                        Timestamp = DateTime.UtcNow,
                        Details = ex.ToString(),
                        Exception = ex
                    }
                ]
            };
        }
    }

    /// <summary>
    /// Validates multiple files
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="files">The files to validate</param>
    /// <param name="projectPath">The project path</param>
    /// <param name="options">Optional validation options</param>
    /// <returns>The validation results</returns>
    public async Task<List<ChangeValidationResult>> ValidateFilesAsync(
        string contextId,
        Dictionary<string, string> files,
        string projectPath,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Validating {FileCount} files", files.Count);

            var results = new List<ChangeValidationResult>();

            // Validate each file
            foreach (var file in files)
            {
                var result = await ValidateFileAsync(contextId, file.Key, file.Value, projectPath, options);
                results.Add(result);

                // Stop if validation fails and stopOnFailure is true
                if (!result.IsSuccessful && ParseOption(options, "StopOnFailure", true))
                {
                    _logger.LogWarning("Validation failed for file: {FilePath}. Stopping validation.", file.Key);
                    break;
                }
            }

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating files");
            throw;
        }
    }

    /// <summary>
    /// Validates a project
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="projectPath">The project path</param>
    /// <param name="options">Optional validation options</param>
    /// <returns>The validation result</returns>
    public async Task<ChangeValidationResult> ValidateProjectAsync(
        string contextId,
        string projectPath,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Validating project: {ProjectPath}", projectPath);

            // Create result
            var result = new ChangeValidationResult
            {
                ProjectPath = projectPath,
                StartedAt = DateTime.UtcNow
            };

            // Parse options
            var runTests = ParseOption(options, "RunTests", true);
            var testFilter = options?.TryGetValue("TestFilter", out var tf) == true ? tf : null;

            // Run tests
            if (runTests)
            {
                _logger.LogInformation("Running tests for project: {ProjectPath}", projectPath);
                var testResult = await _testExecutor.ExecuteTestsAsync(contextId, projectPath, testFilter, options);
                var testValidationResults = _testExecutor.ConvertToValidationResults(testResult, projectPath);
                result.ValidationResults.AddRange(testValidationResults);
                result.TestResult = TestExecutionResultAdapter.ToModel(testResult);

                // Check if tests failed
                if (!testResult.IsSuccessful)
                {
                    result.IsSuccessful = false;
                    result.ErrorMessage = "Tests failed";
                    result.CompletedAt = DateTime.UtcNow;
                    result.DurationMs = (long)(result.CompletedAt - result.StartedAt).TotalMilliseconds;
                    return result;
                }
            }

            // All validations passed
            result.IsSuccessful = true;
            result.CompletedAt = DateTime.UtcNow;
            result.DurationMs = (long)(result.CompletedAt - result.StartedAt).TotalMilliseconds;
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating project: {ProjectPath}", projectPath);
            return new ChangeValidationResult
            {
                ProjectPath = projectPath,
                IsSuccessful = false,
                ErrorMessage = ex.Message,
                StartedAt = DateTime.UtcNow,
                CompletedAt = DateTime.UtcNow,
                DurationMs = 0,
                ValidationResults =
                [
                    new ValidationResult
                    {
                        RuleName = "ProjectValidation",
                        IsPassed = false,
                        Severity = ValidationRuleSeverity.Error,
                        Message = $"Error validating project: {ex.Message}",
                        Target = projectPath,
                        Timestamp = DateTime.UtcNow,
                        Details = ex.ToString(),
                        Exception = ex
                    }
                ]
            };
        }
    }

    /// <summary>
    /// Validates a solution
    /// </summary>
    /// <param name="contextId">The execution context ID</param>
    /// <param name="solutionPath">The solution path</param>
    /// <param name="options">Optional validation options</param>
    /// <returns>The validation result</returns>
    public async Task<ChangeValidationResult> ValidateSolutionAsync(
        string contextId,
        string solutionPath,
        Dictionary<string, string>? options = null)
    {
        try
        {
            _logger.LogInformation("Validating solution: {SolutionPath}", solutionPath);

            // Create result
            var result = new ChangeValidationResult
            {
                ProjectPath = solutionPath,
                StartedAt = DateTime.UtcNow
            };

            // Parse options
            var runTests = ParseOption(options, "RunTests", true);
            var testFilter = options?.TryGetValue("TestFilter", out var tf) == true ? tf : null;

            // Run tests
            if (runTests)
            {
                _logger.LogInformation("Running tests for solution: {SolutionPath}", solutionPath);
                var testResult = await _testExecutor.ExecuteTestsAsync(contextId, solutionPath, testFilter, options);
                var testValidationResults = _testExecutor.ConvertToValidationResults(testResult, solutionPath);
                result.ValidationResults.AddRange(testValidationResults);
                result.TestResult = TestExecutionResultAdapter.ToModel(testResult);

                // Check if tests failed
                if (!testResult.IsSuccessful)
                {
                    result.IsSuccessful = false;
                    result.ErrorMessage = "Tests failed";
                    result.CompletedAt = DateTime.UtcNow;
                    result.DurationMs = (long)(result.CompletedAt - result.StartedAt).TotalMilliseconds;
                    return result;
                }
            }

            // All validations passed
            result.IsSuccessful = true;
            result.CompletedAt = DateTime.UtcNow;
            result.DurationMs = (long)(result.CompletedAt - result.StartedAt).TotalMilliseconds;
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating solution: {SolutionPath}", solutionPath);
            return new ChangeValidationResult
            {
                ProjectPath = solutionPath,
                IsSuccessful = false,
                ErrorMessage = ex.Message,
                StartedAt = DateTime.UtcNow,
                CompletedAt = DateTime.UtcNow,
                DurationMs = 0,
                ValidationResults =
                [
                    new ValidationResult
                    {
                        RuleName = "SolutionValidation",
                        IsPassed = false,
                        Severity = ValidationRuleSeverity.Error,
                        Message = $"Error validating solution: {ex.Message}",
                        Target = solutionPath,
                        Timestamp = DateTime.UtcNow,
                        Details = ex.ToString(),
                        Exception = ex
                    }
                ]
            };
        }
    }

    /// <summary>
    /// Gets the available validation options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        var options = new Dictionary<string, string>
        {
            { "ValidateSyntax", "Whether to validate syntax (true, false)" },
            { "ValidateSemantics", "Whether to validate semantics (true, false)" },
            { "RunTests", "Whether to run tests (true, false)" },
            { "TestFilter", "The test filter to use" },
            { "StopOnFailure", "Whether to stop validation on failure (true, false)" }
        };

        // Add test executor options
        var testExecutorOptions = _testExecutor.GetAvailableOptions();
        foreach (var option in testExecutorOptions)
        {
            options[option.Key] = option.Value;
        }

        return options;
    }

    private T ParseOption<T>(Dictionary<string, string>? options, string key, T defaultValue)
    {
        if (options == null || !options.TryGetValue(key, out var value))
        {
            return defaultValue;
        }

        try
        {
            return (T)Convert.ChangeType(value, typeof(T));
        }
        catch
        {
            return defaultValue;
        }
    }
}
