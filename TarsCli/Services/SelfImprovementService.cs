using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Text;
using TarsEngine.SelfImprovement;
using Microsoft.FSharp.Control;
using Microsoft.FSharp.Core;

namespace TarsCli.Services;

public class SelfImprovementService
{
    private readonly ILogger<SelfImprovementService> _logger;
    private readonly IConfiguration _configuration;
    private readonly OllamaService _ollamaService;

    public SelfImprovementService(
        ILogger<SelfImprovementService> logger,
        IConfiguration configuration,
        OllamaService ollamaService)
    {
        _logger = logger;
        _configuration = configuration;
        _ollamaService = ollamaService;
    }

    public async Task<bool> AnalyzeFile(string filePath, string model)
    {
        try
        {
            _logger.LogInformation($"Analyzing file: {filePath}");

            // Ensure file exists
            if (!File.Exists(filePath))
            {
                _logger.LogError($"File not found: {filePath}");
                return false;
            }

            // Get Ollama endpoint from configuration
            var ollamaEndpoint = _configuration["Ollama:BaseUrl"] ?? "http://localhost:11434";

            // Call the F# self-analyzer
            var analysisResult = await FSharpAsync.StartAsTask(
                SelfAnalyzer.analyzeFile(filePath, ollamaEndpoint, model),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            // Display the analysis results
            Console.WriteLine();
            CliSupport.WriteHeader($"Analysis Results for {Path.GetFileName(filePath)}");
            Console.WriteLine($"Score: {analysisResult.Score:F1}/10.0");

            Console.WriteLine();
            CliSupport.WriteColorLine("Issues:", ConsoleColor.Yellow);
            foreach (var issue in analysisResult.Issues)
            {
                Console.WriteLine($"- {issue}");
            }

            Console.WriteLine();
            CliSupport.WriteColorLine("Recommendations:", ConsoleColor.Cyan);
            foreach (var recommendation in analysisResult.Recommendations)
            {
                Console.WriteLine($"- {recommendation}");
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error analyzing file");
            return false;
        }
    }

    public async Task<bool> ProposeImprovement(string filePath, string model)
    {
        try
        {
            _logger.LogInformation($"Proposing improvements for file: {filePath}");

            // Ensure file exists
            if (!File.Exists(filePath))
            {
                _logger.LogError($"File not found: {filePath}");
                return false;
            }

            // Get Ollama endpoint from configuration
            var ollamaEndpoint = _configuration["Ollama:BaseUrl"] ?? "http://localhost:11434";

            // Call the F# self-improvement engine
            var result = await FSharpAsync.StartAsTask(
                SelfImprovement.analyzeAndImprove(filePath, ollamaEndpoint, model),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            // Display the results
            Console.WriteLine();
            CliSupport.WriteHeader($"Self-Improvement Results for {Path.GetFileName(filePath)}");
            CliSupport.WriteColorLine(result.Message, result.Success ? ConsoleColor.Green : ConsoleColor.Red);

            if (FSharpOption<ImprovementProposal>.get_IsSome(result.Proposal))
            {
                var proposal = result.Proposal.Value;

                // Create a diff-like display
                Console.WriteLine();
                CliSupport.WriteColorLine("Proposed Changes:", ConsoleColor.Cyan);
                Console.WriteLine();

                // Save the proposal to a temporary file
                var outputDir = Path.Combine(
                    _configuration["Tars:ProjectRoot"] ?? Directory.GetCurrentDirectory(),
                    "output",
                    $"v{DateTime.UtcNow:yyyyMMdd}");

                Directory.CreateDirectory(outputDir);

                var originalPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(filePath)}_original{Path.GetExtension(filePath)}");
                var improvedPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(filePath)}_improved{Path.GetExtension(filePath)}");
                var explanationPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(filePath)}_explanation.md");

                await File.WriteAllTextAsync(originalPath, proposal.OriginalContent);
                await File.WriteAllTextAsync(improvedPath, proposal.ImprovedContent);
                await File.WriteAllTextAsync(explanationPath, proposal.Explanation);

                Console.WriteLine($"Original file saved to: {originalPath}");
                Console.WriteLine($"Improved file saved to: {improvedPath}");
                Console.WriteLine($"Explanation saved to: {explanationPath}");

                Console.WriteLine();
                CliSupport.WriteColorLine("Explanation:", ConsoleColor.Yellow);
                Console.WriteLine(proposal.Explanation);

                // Ask user if they want to apply changes
                Console.WriteLine();
                Console.Write("Would you like to apply these changes? (y/n): ");
                var response = Console.ReadLine()?.ToLower();

                if (response == "y" || response == "yes")
                {
                    var applyResult = await FSharpAsync.StartAsTask(
                    SelfImprover.applyImprovement(proposal),
                    FSharpOption<TaskCreationOptions>.None,
                    FSharpOption<CancellationToken>.None);

                    if (applyResult)
                    {
                        CliSupport.WriteColorLine("Changes applied successfully", ConsoleColor.Green);
                        return true;
                    }
                    else
                    {
                        CliSupport.WriteColorLine("Failed to apply changes", ConsoleColor.Red);
                        return false;
                    }
                }
                else
                {
                    CliSupport.WriteColorLine("Changes not applied", ConsoleColor.Yellow);
                    return true; // Still consider this a success since the process completed
                }
            }
            else
            {
                Console.WriteLine("No improvements proposed.");
                return result.Success;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error proposing improvements");
            return false;
        }
    }

    public async Task<bool> SelfRewrite(string filePath, string model, bool autoApply = false)
    {
        try
        {
            _logger.LogInformation($"Self-rewriting file: {filePath}");

            // Ensure file exists
            if (!File.Exists(filePath))
            {
                _logger.LogError($"File not found: {filePath}");
                return false;
            }

            // Get Ollama endpoint from configuration
            var ollamaEndpoint = _configuration["Ollama:BaseUrl"] ?? "http://localhost:11434";

            // Call the F# self-improvement engine with auto-apply option
            var result = await FSharpAsync.StartAsTask(
                SelfImprovement.analyzeAndImproveWithApply(filePath, ollamaEndpoint, model, autoApply),
                FSharpOption<TaskCreationOptions>.None,
                FSharpOption<CancellationToken>.None);

            // Display the results
            Console.WriteLine();
            CliSupport.WriteHeader($"Self-Rewrite Results for {Path.GetFileName(filePath)}");
            CliSupport.WriteColorLine(result.Message, result.Success ? ConsoleColor.Green : ConsoleColor.Red);

            return result.Success;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error self-rewriting file");
            return false;
        }
    }
}
