using System.CommandLine;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TarsCli.Services;
using TarsCli.Services.SelfCoding;

namespace TarsCli.Commands;

/// <summary>
/// Command for auto-coding a file
/// </summary>
public class AutoCodingCommand : Command
{
    private readonly IServiceProvider _serviceProvider;

    /// <summary>
    /// Initializes a new instance of the AutoCodingCommand class
    /// </summary>
    /// <param name="serviceProvider">Service provider</param>
    public AutoCodingCommand(IServiceProvider serviceProvider) : base("auto-code", "Auto-code a file")
    {
        _serviceProvider = serviceProvider;

        // Add file path argument
        var filePathArgument = new Argument<string>("file-path", "Path to the file to auto-code");
        AddArgument(filePathArgument);

        // Add model option
        var modelOption = new Option<string>("--model", () => "llama3", "The model to use for auto-coding");
        modelOption.AddAlias("-m");
        AddOption(modelOption);

        // Add auto-apply option
        var autoApplyOption = new Option<bool>("--auto-apply", () => false, "Automatically apply the improvements");
        autoApplyOption.AddAlias("-a");
        AddOption(autoApplyOption);

        // Set the handler
        this.SetHandler(async (string filePath, string model, bool autoApply) =>
        {
            var logger = _serviceProvider.GetRequiredService<ILogger<AutoCodingCommand>>();
            var consoleService = _serviceProvider.GetRequiredService<ConsoleService>();
            var fileProcessor = _serviceProvider.GetRequiredService<FileProcessor>();
            var analysisProcessor = _serviceProvider.GetRequiredService<AnalysisProcessor>();
            var codeGenerationProcessor = _serviceProvider.GetRequiredService<CodeGenerationProcessor>();
            var testProcessor = _serviceProvider.GetRequiredService<TestProcessor>();

            try
            {
                // Display header
                consoleService.WriteHeader($"Auto-Coding File: {Path.GetFileName(filePath)}");
                consoleService.WriteInfo($"Model: {model}");
                consoleService.WriteInfo($"Auto-Apply: {autoApply}");
                Console.WriteLine();

                // Step 1: Read the file
                consoleService.WriteSubHeader("Step 1: Reading File");
                var fileContent = await fileProcessor.ReadFileAsync(filePath);
                if (fileContent == null)
                {
                    consoleService.WriteError($"Failed to read file: {filePath}");
                    return;
                }
                consoleService.WriteSuccess($"File read successfully: {filePath}");
                Console.WriteLine();

                // Step 2: Analyze the file
                consoleService.WriteSubHeader("Step 2: Analyzing File");
                var analysisResult = await analysisProcessor.AnalyzeFileAsync(filePath, fileContent);
                if (analysisResult == null)
                {
                    consoleService.WriteError($"Failed to analyze file: {filePath}");
                    return;
                }
                consoleService.WriteSuccess($"File analyzed successfully: {filePath}");
                consoleService.WriteInfo($"Issues found: {analysisResult.Issues.Count}");
                foreach (var issue in analysisResult.Issues)
                {
                    consoleService.WriteInfo($"- {issue.Description}");
                }
                Console.WriteLine();

                // Step 3: Generate improved code
                consoleService.WriteSubHeader("Step 3: Generating Improved Code");
                var generationResult = await codeGenerationProcessor.GenerateCodeAsync(filePath, fileContent, analysisResult, model);
                if (generationResult == null)
                {
                    consoleService.WriteError($"Failed to generate improved code for file: {filePath}");
                    return;
                }
                consoleService.WriteSuccess($"Improved code generated successfully for file: {filePath}");
                Console.WriteLine();

                // Step 4: Apply the improvements if auto-apply is enabled
                if (autoApply)
                {
                    consoleService.WriteSubHeader("Step 4: Applying Improvements");
                    var applyResult = await fileProcessor.WriteFileAsync(filePath, generationResult.GeneratedContent);
                    if (!applyResult)
                    {
                        consoleService.WriteError($"Failed to apply improvements to file: {filePath}");
                        return;
                    }
                    consoleService.WriteSuccess($"Improvements applied successfully to file: {filePath}");
                    Console.WriteLine();
                }
                else
                {
                    consoleService.WriteSubHeader("Step 4: Improvements Not Applied");
                    consoleService.WriteInfo("Auto-apply is disabled. Improvements were not applied.");
                    consoleService.WriteInfo("To apply the improvements, run the command with the --auto-apply option.");
                    Console.WriteLine();
                }

                // Step 5: Generate tests if auto-apply is enabled
                if (autoApply)
                {
                    consoleService.WriteSubHeader("Step 5: Generating Tests");
                    var testGenerationResult = await testProcessor.GenerateTestsForFileAsync(filePath);
                    if (testGenerationResult == null)
                    {
                        consoleService.WriteWarning($"Failed to generate tests for file: {filePath}");
                    }
                    else
                    {
                        consoleService.WriteSuccess($"Tests generated successfully for file: {filePath}");
                        consoleService.WriteInfo($"Test file: {testGenerationResult.TestFilePath}");
                    }
                    Console.WriteLine();
                }

                // Display summary
                consoleService.WriteHeader("Auto-Coding Complete");
                consoleService.WriteSuccess($"File: {filePath}");
                consoleService.WriteInfo($"Model: {model}");
                consoleService.WriteInfo($"Auto-Apply: {autoApply}");
                if (!autoApply)
                {
                    consoleService.WriteInfo("To apply the improvements, run the command with the --auto-apply option.");
                }
            }
            catch (Exception ex)
            {
                logger.LogError(ex, $"Error auto-coding file: {filePath}");
                consoleService.WriteError($"Error auto-coding file: {ex.Message}");
            }
        }, filePathArgument, modelOption, autoApplyOption);
    }
}
