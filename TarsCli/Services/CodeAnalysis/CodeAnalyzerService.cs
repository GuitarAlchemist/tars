namespace TarsCli.Services.CodeAnalysis;

/// <summary>
/// Service for analyzing code files
/// </summary>
public class CodeAnalyzerService
{
    private readonly ILogger<CodeAnalyzerService> _logger;
    private readonly IEnumerable<ICodeAnalyzer> _analyzers;

    /// <summary>
    /// Initializes a new instance of the CodeAnalyzerService class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="analyzers">Collection of code analyzers</param>
    public CodeAnalyzerService(ILogger<CodeAnalyzerService> logger, IEnumerable<ICodeAnalyzer> analyzers)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _analyzers = analyzers ?? throw new ArgumentNullException(nameof(analyzers));
    }

    /// <summary>
    /// Analyzes a file for improvement opportunities
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <returns>Analysis result, or null if the file type is not supported</returns>
    public async Task<CodeAnalysisResult> AnalyzeFileAsync(string filePath)
    {
        try
        {
            _logger.LogInformation($"Analyzing file: {filePath}");

            // Check if the file exists
            if (!File.Exists(filePath))
            {
                _logger.LogError($"File not found: {filePath}");
                return null;
            }

            // Get the file extension
            var extension = Path.GetExtension(filePath).ToLowerInvariant();

            // Find an analyzer that supports this file type
            var analyzer = _analyzers.FirstOrDefault(a => a.GetSupportedFileExtensions().Contains(extension));
            if (analyzer == null)
            {
                _logger.LogWarning($"No analyzer found for file type: {extension}");
                return null;
            }

            // Read the file content
            var fileContent = await File.ReadAllTextAsync(filePath);

            // Analyze the file
            var result = await analyzer.AnalyzeFileAsync(filePath, fileContent);

            _logger.LogInformation($"Analysis completed for {filePath}. Found {result.Issues.Count} issues.");
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing file {filePath}");
            return new CodeAnalysisResult
            {
                FilePath = filePath,
                NeedsImprovement = false,
                Issues = new List<CodeIssue>
                {
                    new()
                    {
                        Type = CodeIssueType.Functional,
                        Severity = IssueSeverity.Error,
                        Description = $"Error analyzing file: {ex.Message}",
                        LineNumber = 1,
                        ColumnNumber = 1
                    }
                }
            };
        }
    }

    /// <summary>
    /// Gets all supported file extensions
    /// </summary>
    /// <returns>List of supported file extensions</returns>
    public IEnumerable<string> GetSupportedFileExtensions()
    {
        return _analyzers.SelectMany(a => a.GetSupportedFileExtensions()).Distinct();
    }
}