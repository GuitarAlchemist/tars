using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsCli.Services.CodeAnalysis;

namespace TarsCli.Services.CodeGeneration
{
    /// <summary>
    /// Service for generating improved code based on analysis results
    /// </summary>
    public class CodeGeneratorService
    {
        private readonly ILogger<CodeGeneratorService> _logger;
        private readonly IEnumerable<ICodeGenerator> _generators;
        private readonly CodeAnalyzerService _analyzerService;

        /// <summary>
        /// Initializes a new instance of the CodeGeneratorService class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="generators">Collection of code generators</param>
        /// <param name="analyzerService">Code analyzer service</param>
        public CodeGeneratorService(
            ILogger<CodeGeneratorService> logger,
            IEnumerable<ICodeGenerator> generators,
            CodeAnalyzerService analyzerService)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _generators = generators ?? throw new ArgumentNullException(nameof(generators));
            _analyzerService = analyzerService ?? throw new ArgumentNullException(nameof(analyzerService));
        }

        /// <summary>
        /// Generates improved code for a file
        /// </summary>
        /// <param name="filePath">Path to the file to improve</param>
        /// <returns>Code generation result, or null if the file type is not supported</returns>
        public async Task<CodeGenerationResult> GenerateCodeAsync(string filePath)
        {
            try
            {
                _logger.LogInformation($"Generating improved code for {filePath}");

                // Check if the file exists
                if (!File.Exists(filePath))
                {
                    _logger.LogError($"File not found: {filePath}");
                    return null;
                }

                // Get the file extension
                var extension = Path.GetExtension(filePath).ToLowerInvariant();

                // Find a generator that supports this file type
                var generator = _generators.FirstOrDefault(g => g.GetSupportedFileExtensions().Contains(extension));
                if (generator == null)
                {
                    _logger.LogWarning($"No generator found for file type: {extension}");
                    return null;
                }

                // Read the file content
                var originalContent = await File.ReadAllTextAsync(filePath);

                // Analyze the file
                var analysisResult = await _analyzerService.AnalyzeFileAsync(filePath);
                if (analysisResult == null)
                {
                    _logger.LogError($"Failed to analyze file: {filePath}");
                    return null;
                }

                // Generate improved code
                var result = await generator.GenerateCodeAsync(filePath, originalContent, analysisResult);

                _logger.LogInformation($"Generated improved code for {filePath} with {result.Changes.Count} changes");
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error generating improved code for {filePath}");
                return new CodeGenerationResult
                {
                    FilePath = filePath,
                    OriginalContent = await File.ReadAllTextAsync(filePath),
                    GeneratedContent = await File.ReadAllTextAsync(filePath),
                    Success = false,
                    ErrorMessage = ex.Message
                };
            }
        }

        /// <summary>
        /// Gets all supported file extensions
        /// </summary>
        /// <returns>List of supported file extensions</returns>
        public IEnumerable<string> GetSupportedFileExtensions()
        {
            return _generators.SelectMany(g => g.GetSupportedFileExtensions()).Distinct();
        }

        /// <summary>
        /// Applies generated code to a file
        /// </summary>
        /// <param name="result">Code generation result</param>
        /// <param name="createBackup">Whether to create a backup of the original file</param>
        /// <returns>True if the code was applied successfully, false otherwise</returns>
        public async Task<bool> ApplyGeneratedCodeAsync(CodeGenerationResult result, bool createBackup = true)
        {
            try
            {
                _logger.LogInformation($"Applying generated code to {result.FilePath}");

                // Check if the result is valid
                if (result == null || !result.Success)
                {
                    _logger.LogError($"Invalid code generation result for {result?.FilePath ?? "unknown file"}");
                    return false;
                }

                // Create a backup if requested
                if (createBackup)
                {
                    var backupPath = $"{result.FilePath}.bak";
                    await File.WriteAllTextAsync(backupPath, result.OriginalContent);
                    _logger.LogInformation($"Created backup of {result.FilePath} at {backupPath}");
                }

                // Write the generated code to the file
                await File.WriteAllTextAsync(result.FilePath, result.GeneratedContent);

                _logger.LogInformation($"Applied generated code to {result.FilePath}");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error applying generated code to {result?.FilePath ?? "unknown file"}");
                return false;
            }
        }
    }
}
