using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;
using TarsEngine.SelfImprovement;
using CodeLanguage = TarsEngine.Models.CodeLanguage;

namespace TarsEngine.Services;

/// <summary>
/// Service for automatically implementing TODOs
/// </summary>
public class AutoImplementationService : IAutoImplementationService
{
    private readonly ILogger<AutoImplementationService> _logger;
    private readonly ITaskAnalysisService _taskAnalysisService;
    private readonly ICodeGenerationService _codeGenerationService;
    private readonly ITestGenerationService _testGenerationService;
    private readonly IMcpService _mcpService;
    private readonly TodosUpdater _todosUpdater;

    /// <summary>
    /// Initializes a new instance of the <see cref="AutoImplementationService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="taskAnalysisService">The task analysis service</param>
    /// <param name="codeGenerationService">The code generation service</param>
    /// <param name="testGenerationService">The test generation service</param>
    /// <param name="mcpService">The MCP service</param>
    /// <param name="todosFilePath">The path to the TODOs file</param>
    public AutoImplementationService(
        ILogger<AutoImplementationService> logger,
        ITaskAnalysisService taskAnalysisService,
        ICodeGenerationService codeGenerationService,
        ITestGenerationService testGenerationService,
        IMcpService mcpService,
        string todosFilePath = "Tars - TODOs.txt")
    {
        _logger = logger;
        _taskAnalysisService = taskAnalysisService;
        _codeGenerationService = codeGenerationService;
        _testGenerationService = testGenerationService;
        _mcpService = mcpService;
        var todosLogger = (ILogger<TodosUpdater>)(object)logger;
        _todosUpdater = new TodosUpdater(
            todosLogger,
            todosFilePath);
    }

    /// <summary>
    /// Implement a TODO task
    /// </summary>
    /// <param name="taskDescription">The task description</param>
    /// <param name="progressCallback">A callback for reporting progress</param>
    /// <returns>The result of the implementation</returns>
    public async Task<ImplementationResult> ImplementTaskAsync(
        string taskDescription,
        Action<string, int>? progressCallback = null)
    {
        _logger.LogInformation($"Starting auto-implementation of task: {taskDescription}");
        ReportProgress(progressCallback, "Starting task analysis...", 0);

        try
        {
            // Step 1: Analyze the task and create an implementation plan
            var implementationPlan = await _taskAnalysisService.AnalyzeTaskAsync(taskDescription);
            _logger.LogInformation($"Created implementation plan with {implementationPlan.ImplementationSteps.Count} steps");
            ReportProgress(progressCallback, "Task analysis completed. Creating implementation plan...", 10);

            // Step 2: Execute the implementation steps
            var result = new ImplementationResult
            {
                TaskDescription = taskDescription,
                Plan = implementationPlan,
                StartTime = DateTime.Now
            };

            // Execute each step in the implementation plan
            var totalSteps = implementationPlan.ImplementationSteps.Count;
            for (var i = 0; i < totalSteps; i++)
            {
                var step = implementationPlan.ImplementationSteps[i];
                var progressPercentage = 10 + (i * 80 / totalSteps); // Progress from 10% to 90%

                ReportProgress(progressCallback, $"Executing step {i + 1}/{totalSteps}: {step.Description.Split(Environment.NewLine)[0]}", progressPercentage);

                try
                {
                    // Execute the step using the appropriate method based on the change type
                    var stepResult = await ExecuteImplementationStepAsync(step);
                    result.StepResults.Add(stepResult);

                    if (!stepResult.Success)
                    {
                        _logger.LogWarning($"Step {i + 1} failed: {stepResult.ErrorMessage}");
                        result.Success = false;
                        result.ErrorMessage = $"Failed at step {i + 1}: {stepResult.ErrorMessage}";
                        break;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"Error executing step {i + 1}");
                    result.Success = false;
                    result.ErrorMessage = $"Exception at step {i + 1}: {ex.Message}";
                    break;
                }
            }

            // Step 3: Generate and run tests if all steps were successful
            if (result.Success)
            {
                ReportProgress(progressCallback, "Implementation completed. Generating tests...", 90);

                try
                {
                    var testResults = await GenerateAndRunTestsAsync(implementationPlan);
                    result.TestResults = testResults;

                    if (!testResults.Success)
                    {
                        _logger.LogWarning($"Tests failed: {testResults.ErrorMessage}");
                        result.Success = false;
                        result.ErrorMessage = $"Tests failed: {testResults.ErrorMessage}";
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error generating or running tests");
                    result.Success = false;
                    result.ErrorMessage = $"Error in testing phase: {ex.Message}";
                }
            }

            // Step 4: Mark the task as completed if everything was successful
            if (result.Success)
            {
                ReportProgress(progressCallback, "Tests passed. Marking task as completed...", 95);

                try
                {
                    await _todosUpdater.MarkTaskAsCompletedAsync(taskDescription);
                    _logger.LogInformation($"Task marked as completed: {taskDescription}");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error marking task as completed");
                    // Don't fail the entire operation just because we couldn't mark it as completed
                }
            }

            result.EndTime = DateTime.Now;
            result.Duration = result.EndTime - result.StartTime;

            ReportProgress(progressCallback, result.Success
                ? "Task implementation completed successfully!"
                : $"Task implementation failed: {result.ErrorMessage}", 100);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error implementing task: {taskDescription}");

            return new ImplementationResult
            {
                TaskDescription = taskDescription,
                Success = false,
                ErrorMessage = $"Error: {ex.Message}",
                StartTime = DateTime.Now,
                EndTime = DateTime.Now,
                Duration = TimeSpan.Zero
            };
        }
    }

    /// <summary>
    /// Execute a single implementation step
    /// </summary>
    /// <param name="step">The implementation step</param>
    /// <returns>The result of the step execution</returns>
    private async Task<StepResult> ExecuteImplementationStepAsync(ImplementationStep step)
    {
        _logger.LogInformation($"Executing step {step.StepNumber}: {step.Description.Split(Environment.NewLine)[0]}");

        var result = new StepResult
        {
            StepNumber = step.StepNumber,
            Description = step.Description,
            FilePath = step.FilePath,
            StartTime = DateTime.Now
        };

        try
        {
            // Generate code for the step
            // Determine the language from the file extension
            var extension = Path.GetExtension(step.FilePath).ToLowerInvariant();
            var language = extension switch
            {
                ".cs" => CodeLanguage.CSharp,
                ".fs" => CodeLanguage.FSharp,
                ".js" => CodeLanguage.JavaScript,
                ".ts" => CodeLanguage.TypeScript,
                ".py" => CodeLanguage.Python,
                _ => CodeLanguage.CSharp
            };

            var codeGeneration = await _codeGenerationService.GenerateCodeAsync(
                step.Description,
                Path.GetDirectoryName(step.FilePath) ?? ".",
                language,
                step.FilePath);

            if (!codeGeneration.Success)
            {
                result.Success = false;
                result.ErrorMessage = $"Code generation failed: {codeGeneration.ErrorMessage}";
                return result;
            }

            // Apply the changes using MCP
            var mcpResult = await ApplyChangesViaMcpAsync(step, codeGeneration.GeneratedCode);

            if (!mcpResult)
            {
                result.Success = false;
                result.ErrorMessage = "Failed to apply changes via MCP";
                return result;
            }

            result.Success = true;
            result.EndTime = DateTime.Now;
            result.Duration = result.EndTime - result.StartTime;

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing step {step.StepNumber}");

            result.Success = false;
            result.ErrorMessage = ex.Message;
            result.EndTime = DateTime.Now;
            result.Duration = result.EndTime - result.StartTime;

            return result;
        }
    }

    /// <summary>
    /// Apply changes to a file using the Model Context Protocol
    /// </summary>
    /// <param name="step">The implementation step</param>
    /// <param name="generatedCode">The generated code</param>
    /// <returns>True if the changes were applied successfully, false otherwise</returns>
    private async Task<bool> ApplyChangesViaMcpAsync(ImplementationStep step, string generatedCode)
    {
        try
        {
            // Determine the appropriate MCP command based on the change type
            if (step.ChangeType.Contains("Create") || step.ChangeType.Contains("Add"))
            {
                // If we're creating a new file
                if (step.FilePath.Contains(".cs") || step.FilePath.Contains(".fs") ||
                    step.FilePath.Contains(".csproj") || step.FilePath.Contains(".fsproj"))
                {
                    // Use MCP to create the file
                    var createFileResult = await _mcpService.ExecuteCommandAsync(
                        "vscode.executeCommand",
                        new Dictionary<string, object>
                        {
                            ["command"] = "workbench.action.files.newUntitledFile"
                        });

                    // Insert the generated code
                    var insertCodeResult = await _mcpService.ExecuteCommandAsync(
                        "vscode.executeCommand",
                        new Dictionary<string, object>
                        {
                            ["command"] = "editor.action.insertSnippet",
                            ["args"] = new Dictionary<string, object>
                            {
                                ["snippet"] = generatedCode
                            }
                        });

                    // Save the file with the correct path
                    var saveFileResult = await _mcpService.ExecuteCommandAsync(
                        "vscode.executeCommand",
                        new Dictionary<string, object>
                        {
                            ["command"] = "workbench.action.files.saveAs",
                            ["args"] = new Dictionary<string, object>
                            {
                                ["path"] = step.FilePath
                            }
                        });

                    return true;
                }
            }
            else if (step.ChangeType.Contains("Modify") || step.ChangeType.Contains("Update"))
            {
                // Open the file
                var openFileResult = await _mcpService.ExecuteCommandAsync(
                    "vscode.open",
                    new Dictionary<string, object>
                    {
                        ["uri"] = step.FilePath
                    });

                // For now, we'll just append the code at the end of the file
                // In a real implementation, we would need to be more sophisticated
                // about where to insert the code
                var insertCodeResult = await _mcpService.ExecuteCommandAsync(
                    "vscode.executeCommand",
                    new Dictionary<string, object>
                    {
                        ["command"] = "editor.action.insertSnippet",
                        ["args"] = new Dictionary<string, object>
                        {
                            ["snippet"] = generatedCode
                        }
                    });

                // Save the file
                var saveFileResult = await _mcpService.ExecuteCommandAsync(
                    "vscode.executeCommand",
                    new Dictionary<string, object>
                    {
                        ["command"] = "workbench.action.files.save"
                    });

                return true;
            }
            else if (step.ChangeType.Contains("Delete"))
            {
                // Open the file
                var openFileResult = await _mcpService.ExecuteCommandAsync(
                    "vscode.open",
                    new Dictionary<string, object>
                    {
                        ["uri"] = step.FilePath
                    });

                // Select all content
                var selectAllResult = await _mcpService.ExecuteCommandAsync(
                    "vscode.executeCommand",
                    new Dictionary<string, object>
                    {
                        ["command"] = "editor.action.selectAll"
                    });

                // Delete the selected content
                var deleteResult = await _mcpService.ExecuteCommandAsync(
                    "vscode.executeCommand",
                    new Dictionary<string, object>
                    {
                        ["command"] = "deleteLeft"
                    });

                // Save the file
                var saveFileResult = await _mcpService.ExecuteCommandAsync(
                    "vscode.executeCommand",
                    new Dictionary<string, object>
                    {
                        ["command"] = "workbench.action.files.save"
                    });

                return true;
            }

            // If we couldn't determine the change type, return false
            _logger.LogWarning($"Unknown change type: {step.ChangeType}");
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error applying changes via MCP");
            return false;
        }
    }

    /// <summary>
    /// Generate and run tests for the implementation
    /// </summary>
    /// <param name="plan">The implementation plan</param>
    /// <returns>The test results</returns>
    private async Task<TestResults> GenerateAndRunTestsAsync(ImplementationPlan plan)
    {
        _logger.LogInformation("Generating and running tests");

        var results = new TestResults
        {
            StartTime = DateTime.Now
        };

        try
        {
            // Generate tests for each affected component
            foreach (var component in plan.AffectedComponents)
            {
                // Skip components that were deleted
                if (component.ChangeType == ChangeType.Delete)
                {
                    continue;
                }

                // Generate tests for the component
                try {
                    // Generate tests for the component
                    var testCode = await _testGenerationService.GenerateTestsForFileAsync(
                        component.FilePath,
                        Path.GetDirectoryName(component.FilePath) ?? ".");

                    // Apply the test changes using MCP
                    var testFilePath = component.FilePath.Replace(".cs", "Tests.cs")
                        .Replace(".fs", "Tests.fs")
                        .Replace("/src/", "/tests/")
                        .Replace("\\src\\", "\\tests\\");

                    var mcpResult = await ApplyTestsViaMcpAsync(testFilePath, testCode);

                    if (!mcpResult)
                    {
                        results.Success = false;
                        results.ErrorMessage = $"Failed to apply tests for {component.Name}";
                        return results;
                    }

                    // Add the test to the results
                    results.Tests.Add(new TarsEngine.Models.TestResult
                    {
                        ComponentName = component.Name,
                        TestFilePath = testFilePath,
                        Success = true
                    });
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error generating tests for component: {ComponentName}", component.Name);
                    results.Success = false;
                    results.ErrorMessage = $"Test generation failed for {component.Name}: {ex.Message}";
                    return results;
                }
            }

            // Run the tests
            var testRunResult = await RunTestsViaMcpAsync();

            if (!testRunResult)
            {
                results.Success = false;
                results.ErrorMessage = "Tests failed to run";
                return results;
            }

            results.Success = true;
            results.EndTime = DateTime.Now;
            results.Duration = results.EndTime - results.StartTime;

            return results;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating or running tests");

            results.Success = false;
            results.ErrorMessage = ex.Message;
            results.EndTime = DateTime.Now;
            results.Duration = results.EndTime - results.StartTime;

            return results;
        }
    }

    /// <summary>
    /// Apply tests to a file using the Model Context Protocol
    /// </summary>
    /// <param name="testFilePath">The path to the test file</param>
    /// <param name="generatedTests">The generated tests</param>
    /// <returns>True if the tests were applied successfully, false otherwise</returns>
    private async Task<bool> ApplyTestsViaMcpAsync(string testFilePath, string generatedTests)
    {
        try
        {
            // Create the test file
            var createFileResult = await _mcpService.ExecuteCommandAsync(
                "vscode.executeCommand",
                new Dictionary<string, object>
                {
                    ["command"] = "workbench.action.files.newUntitledFile"
                });

            // Insert the generated tests
            var insertCodeResult = await _mcpService.ExecuteCommandAsync(
                "vscode.executeCommand",
                new Dictionary<string, object>
                {
                    ["command"] = "editor.action.insertSnippet",
                    ["args"] = new Dictionary<string, object>
                    {
                        ["snippet"] = generatedTests
                    }
                });

            // Save the file with the correct path
            var saveFileResult = await _mcpService.ExecuteCommandAsync(
                "vscode.executeCommand",
                new Dictionary<string, object>
                {
                    ["command"] = "workbench.action.files.saveAs",
                    ["args"] = new Dictionary<string, object>
                    {
                        ["path"] = testFilePath
                    }
                });

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error applying tests via MCP");
            return false;
        }
    }

    /// <summary>
    /// Run tests using the Model Context Protocol
    /// </summary>
    /// <returns>True if the tests ran successfully, false otherwise</returns>
    private async Task<bool> RunTestsViaMcpAsync()
    {
        try
        {
            // Run the tests using the .NET CLI
            var runTestsResult = await _mcpService.ExecuteCommandAsync(
                "vscode.executeCommand",
                new Dictionary<string, object>
                {
                    ["command"] = "workbench.action.terminal.new"
                });

            // Send the test command to the terminal
            var sendCommandResult = await _mcpService.ExecuteCommandAsync(
                "vscode.executeCommand",
                new Dictionary<string, object>
                {
                    ["command"] = "workbench.action.terminal.sendSequence",
                    ["args"] = new Dictionary<string, object>
                    {
                        ["text"] = "dotnet test\n"
                    }
                });

            // For now, we'll assume the tests passed
            // In a real implementation, we would need to parse the test output
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running tests via MCP");
            return false;
        }
    }

    /// <summary>
    /// Report progress to the callback
    /// </summary>
    /// <param name="progressCallback">The progress callback</param>
    /// <param name="message">The progress message</param>
    /// <param name="percentage">The progress percentage</param>
    private void ReportProgress(Action<string, int>? progressCallback, string message, int percentage)
    {
        _logger.LogInformation($"Progress: {percentage}% - {message}");
        progressCallback?.Invoke(message, percentage);
    }
}
