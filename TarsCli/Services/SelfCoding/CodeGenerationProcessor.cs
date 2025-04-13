using TarsCli.Services.CodeAnalysis;
using TarsCli.Services.CodeGeneration;
using TarsCli.Services.Mcp;

namespace TarsCli.Services.SelfCoding;

/// <summary>
/// Service for processing code generation in the self-coding workflow
/// </summary>
public class CodeGenerationProcessor
{
    private readonly ILogger<CodeGenerationProcessor> _logger;
    private readonly CodeGeneratorService _codeGeneratorService;
    private readonly ReplicaCommunicationProtocol _replicaCommunicationProtocol;

    /// <summary>
    /// Initializes a new instance of the CodeGenerationProcessor class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="codeGeneratorService">Code generator service</param>
    /// <param name="replicaCommunicationProtocol">Replica communication protocol</param>
    public CodeGenerationProcessor(
        ILogger<CodeGenerationProcessor> logger,
        CodeGeneratorService codeGeneratorService,
        ReplicaCommunicationProtocol replicaCommunicationProtocol)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _codeGeneratorService = codeGeneratorService ?? throw new ArgumentNullException(nameof(codeGeneratorService));
        _replicaCommunicationProtocol = replicaCommunicationProtocol ?? throw new ArgumentNullException(nameof(replicaCommunicationProtocol));
    }

    /// <summary>
    /// Generates improved code for files
    /// </summary>
    /// <param name="analysisResults">Analysis results</param>
    /// <returns>List of code generation results</returns>
    public async Task<List<CodeGenerationResult>> GenerateCodeAsync(List<CodeAnalysisResult> analysisResults)
    {
        _logger.LogInformation($"Generating improved code for {analysisResults.Count} files");

        try
        {
            // Validate parameters
            if (analysisResults == null || !analysisResults.Any())
            {
                throw new ArgumentException("Analysis results are required", nameof(analysisResults));
            }

            // Generate improved code for each file that needs improvement
            var generationResults = new List<CodeGenerationResult>();
            foreach (var analysisResult in analysisResults.Where(r => r.NeedsImprovement))
            {
                var result = await GenerateCodeForFileAsync(analysisResult.FilePath);
                if (result != null)
                {
                    generationResults.Add(result);
                }
            }

            _logger.LogInformation($"Generated improved code for {generationResults.Count} files");
            return generationResults;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating improved code");
            throw;
        }
    }

    /// <summary>
    /// Generates improved code for a file
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <returns>Code generation result</returns>
    public async Task<CodeGenerationResult> GenerateCodeForFileAsync(string filePath)
    {
        _logger.LogInformation($"Generating improved code for file: {filePath}");

        try
        {
            // Validate parameters
            if (string.IsNullOrEmpty(filePath))
            {
                throw new ArgumentException("File path is required", nameof(filePath));
            }

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            // Generate improved code
            var generationResult = await _codeGeneratorService.GenerateCodeAsync(filePath);
            if (generationResult == null)
            {
                _logger.LogWarning($"Failed to generate improved code for file: {filePath}");
                return null;
            }

            _logger.LogInformation($"Generated improved code for file: {filePath}, with {generationResult.Changes.Count} changes");
            return generationResult;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating improved code for file: {filePath}");
            throw;
        }
    }

    /// <summary>
    /// Generates improved code for a file using a generator replica
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="analysisResult">Analysis result</param>
    /// <param name="replicaUrl">URL of the generator replica</param>
    /// <returns>Code generation result</returns>
    public async Task<CodeGenerationResult> GenerateCodeWithReplicaAsync(string filePath, CodeAnalysisResult analysisResult, string replicaUrl)
    {
        _logger.LogInformation($"Generating improved code for file with replica: {filePath}");

        try
        {
            // Validate parameters
            if (string.IsNullOrEmpty(filePath))
            {
                throw new ArgumentException("File path is required", nameof(filePath));
            }

            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"File not found: {filePath}");
            }

            if (string.IsNullOrEmpty(replicaUrl))
            {
                throw new ArgumentException("Replica URL is required", nameof(replicaUrl));
            }

            // Read the file content
            var fileContent = await File.ReadAllTextAsync(filePath);

            // Create the request
            var request = new
            {
                file_path = filePath,
                original_content = fileContent,
                analysis_result = analysisResult
            };

            // Send the request to the replica
            var response = await _replicaCommunicationProtocol.SendMessageAsync(replicaUrl, "generate_code", request);

            // Check if the response was successful
            var success = response.TryGetProperty("success", out var successElement) && successElement.GetBoolean();
            if (!success)
            {
                // Extract the error message
                var error = response.TryGetProperty("error", out var errorElement)
                    ? errorElement.GetString()
                    : "Unknown error";

                _logger.LogError($"Error generating improved code with replica: {error}");
                return null;
            }

            // Convert the response to a CodeGenerationResult
            var generationResult = ConvertResponseToGenerationResult(response, filePath, fileContent);

            _logger.LogInformation($"Generated improved code for file with replica: {filePath}, with {generationResult.Changes.Count} changes");
            return generationResult;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating improved code for file with replica: {filePath}");
            throw;
        }
    }

    /// <summary>
    /// Applies generated code changes
    /// </summary>
    /// <param name="generationResults">List of code generation results</param>
    /// <param name="createBackup">Whether to create a backup of the original file</param>
    /// <returns>List of files that were changed</returns>
    public async Task<List<string>> ApplyGeneratedCodeAsync(List<CodeGenerationResult> generationResults, bool createBackup = true)
    {
        _logger.LogInformation($"Applying generated code changes to {generationResults.Count} files");

        try
        {
            // Validate parameters
            if (generationResults == null || !generationResults.Any())
            {
                throw new ArgumentException("Generation results are required", nameof(generationResults));
            }

            // Apply the changes
            var changedFiles = new List<string>();
            foreach (var generationResult in generationResults)
            {
                var success = await _codeGeneratorService.ApplyGeneratedCodeAsync(generationResult, createBackup);
                if (success)
                {
                    changedFiles.Add(generationResult.FilePath);
                }
            }

            _logger.LogInformation($"Applied generated code changes to {changedFiles.Count} files");
            return changedFiles;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error applying generated code changes");
            throw;
        }
    }

    /// <summary>
    /// Converts a response from a generator replica to a CodeGenerationResult
    /// </summary>
    /// <param name="response">Response from the replica</param>
    /// <param name="filePath">Path to the file</param>
    /// <param name="originalContent">Original content of the file</param>
    /// <returns>CodeGenerationResult</returns>
    private CodeGenerationResult ConvertResponseToGenerationResult(System.Text.Json.JsonElement response, string filePath, string originalContent)
    {
        var result = new CodeGenerationResult
        {
            FilePath = filePath,
            OriginalContent = originalContent,
            Success = true
        };

        // Extract generated_content
        if (response.TryGetProperty("generated_content", out var generatedContentElement))
        {
            result.GeneratedContent = generatedContentElement.GetString();
        }
        else
        {
            result.GeneratedContent = originalContent;
        }

        // Extract changes
        if (response.TryGetProperty("changes", out var changesElement) && changesElement.ValueKind == System.Text.Json.JsonValueKind.Array)
        {
            foreach (var changeElement in changesElement.EnumerateArray())
            {
                var change = new CodeChange();

                // Extract type
                if (changeElement.TryGetProperty("type", out var typeElement))
                {
                    var typeString = typeElement.GetString();
                    if (Enum.TryParse<CodeChangeType>(typeString, true, out var type))
                    {
                        change.Type = type;
                    }
                }

                // Extract description
                if (changeElement.TryGetProperty("description", out var descriptionElement))
                {
                    change.Description = descriptionElement.GetString();
                }

                // Extract line_number
                if (changeElement.TryGetProperty("line_number", out var lineNumberElement))
                {
                    change.LineNumber = lineNumberElement.GetInt32();
                }

                // Extract original_code
                if (changeElement.TryGetProperty("original_code", out var originalCodeElement))
                {
                    change.OriginalCode = originalCodeElement.GetString();
                }

                // Extract new_code
                if (changeElement.TryGetProperty("new_code", out var newCodeElement))
                {
                    change.NewCode = newCodeElement.GetString();
                }

                result.Changes.Add(change);
            }
        }

        return result;
    }
}