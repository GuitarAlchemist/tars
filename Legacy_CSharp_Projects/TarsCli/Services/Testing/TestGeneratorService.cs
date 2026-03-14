namespace TarsCli.Services.Testing;

/// <summary>
/// Service for generating tests for code files
/// </summary>
public class TestGeneratorService
{
    private readonly ILogger<TestGeneratorService> _logger;
    private readonly IEnumerable<ITestGenerator> _generators;

    /// <summary>
    /// Initializes a new instance of the TestGeneratorService class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="generators">Collection of test generators</param>
    public TestGeneratorService(ILogger<TestGeneratorService> logger, IEnumerable<ITestGenerator> generators)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _generators = generators ?? throw new ArgumentNullException(nameof(generators));
    }

    /// <summary>
    /// Generates tests for a file
    /// </summary>
    /// <param name="filePath">Path to the file to test</param>
    /// <returns>Test generation result, or null if the file type is not supported</returns>
    public async Task<TestGenerationResult> GenerateTestsAsync(string filePath)
    {
        try
        {
            _logger.LogInformation($"Generating tests for {filePath}");

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
                _logger.LogWarning($"No test generator found for file type: {extension}");
                return null;
            }

            // Read the file content
            var fileContent = await File.ReadAllTextAsync(filePath);

            // Generate tests
            var result = await generator.GenerateTestsAsync(filePath, fileContent);

            _logger.LogInformation($"Generated {result.Tests.Count} tests for {filePath}");
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating tests for {filePath}");
            return new TestGenerationResult
            {
                SourceFilePath = filePath,
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
    /// Saves generated tests to a file
    /// </summary>
    /// <param name="result">Test generation result</param>
    /// <returns>True if the tests were saved successfully, false otherwise</returns>
    public async Task<bool> SaveTestsAsync(TestGenerationResult result)
    {
        try
        {
            _logger.LogInformation($"Saving tests for {result.SourceFilePath} to {result.TestFilePath}");

            // Check if the result is valid
            if (result == null || !result.Success)
            {
                _logger.LogError($"Invalid test generation result for {result?.SourceFilePath ?? "unknown file"}");
                return false;
            }

            // Create the directory if it doesn't exist
            var directory = Path.GetDirectoryName(result.TestFilePath);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Write the test file
            await File.WriteAllTextAsync(result.TestFilePath, result.TestFileContent);

            _logger.LogInformation($"Saved tests for {result.SourceFilePath} to {result.TestFilePath}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error saving tests for {result?.SourceFilePath ?? "unknown file"}");
            return false;
        }
    }
}