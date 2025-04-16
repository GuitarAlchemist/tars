using TarsCli.Services.CodeAnalysis;
using TarsCli.Services.Mcp;
using TarsCli.Services.Workflow;
using TarsCli.Services.Adapters;

namespace TarsCli.Services.SelfCoding;

/// <summary>
/// Service for processing code analysis in the self-coding workflow
/// </summary>
public class AnalysisProcessor
{
    private readonly ILogger<AnalysisProcessor> _logger;
    private readonly CodeAnalyzerService _codeAnalyzerService;
    private readonly ImprovementSuggestionGenerator _improvementSuggestionGenerator;
    private readonly TaskPrioritizer _taskPrioritizer;
    private readonly ReplicaCommunicationProtocol _replicaCommunicationProtocol;

    /// <summary>
    /// Initializes a new instance of the AnalysisProcessor class
    /// </summary>
    /// <param name="logger">Logger instance</param>
    /// <param name="codeAnalyzerService">Code analyzer service</param>
    /// <param name="improvementSuggestionGenerator">Improvement suggestion generator</param>
    /// <param name="taskPrioritizer">Task prioritizer</param>
    /// <param name="replicaCommunicationProtocol">Replica communication protocol</param>
    public AnalysisProcessor(
        ILogger<AnalysisProcessor> logger,
        CodeAnalyzerService codeAnalyzerService,
        ImprovementSuggestionGenerator improvementSuggestionGenerator,
        TaskPrioritizer taskPrioritizer,
        ReplicaCommunicationProtocol replicaCommunicationProtocol)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _codeAnalyzerService = codeAnalyzerService ?? throw new ArgumentNullException(nameof(codeAnalyzerService));
        _improvementSuggestionGenerator = improvementSuggestionGenerator ?? throw new ArgumentNullException(nameof(improvementSuggestionGenerator));
        _taskPrioritizer = taskPrioritizer ?? throw new ArgumentNullException(nameof(taskPrioritizer));
        _replicaCommunicationProtocol = replicaCommunicationProtocol ?? throw new ArgumentNullException(nameof(replicaCommunicationProtocol));
    }

    /// <summary>
    /// Analyzes files for improvement opportunities
    /// </summary>
    /// <param name="files">List of files to analyze</param>
    /// <returns>List of analysis results</returns>
    public async Task<List<CodeAnalysisResult>> AnalyzeFilesAsync(List<string> files)
    {
        _logger.LogInformation($"Analyzing {files.Count} files for improvement opportunities");

        try
        {
            // Validate parameters
            if (files == null || !files.Any())
            {
                throw new ArgumentException("Files are required", nameof(files));
            }

            // Analyze each file
            var analysisResults = new List<CodeAnalysisResult>();
            foreach (var file in files)
            {
                var result = await AnalyzeFileAsync(file);
                if (result != null)
                {
                    analysisResults.Add(result);
                }
            }

            _logger.LogInformation($"Analyzed {analysisResults.Count} files");
            return analysisResults;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing files for improvement opportunities");
            throw;
        }
    }

    /// <summary>
    /// Analyzes a file for improvement opportunities
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <returns>Analysis result</returns>
    public async Task<CodeAnalysisResult> AnalyzeFileAsync(string filePath)
    {
        _logger.LogInformation($"Analyzing file: {filePath}");

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

            // Analyze the file
            var analysisResult = await _codeAnalyzerService.AnalyzeFileAsync(filePath);
            if (analysisResult == null)
            {
                _logger.LogWarning($"Failed to analyze file: {filePath}");
                return null;
            }

            _logger.LogInformation($"Analyzed file: {filePath}, Found {analysisResult.Issues.Count} issues");
            return analysisResult;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing file: {filePath}");
            throw;
        }
    }

    /// <summary>
    /// Analyzes a file for improvement opportunities
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="fileContent">Content of the file</param>
    /// <returns>Analysis result</returns>
    public async Task<CodeAnalysisResult> AnalyzeFileAsync(string filePath, string fileContent)
    {
        _logger.LogInformation($"Analyzing file content: {filePath}");

        try
        {
            // Validate parameters
            if (string.IsNullOrEmpty(filePath))
            {
                throw new ArgumentException("File path is required", nameof(filePath));
            }

            if (string.IsNullOrEmpty(fileContent))
            {
                throw new ArgumentException("File content is required", nameof(fileContent));
            }

            // Create a temporary file with the content and proper extension
            var fileExtension = Path.GetExtension(filePath);
            var tempFilePath = Path.Combine(Path.GetTempPath(), $"{Path.GetFileNameWithoutExtension(Path.GetTempFileName())}{fileExtension}");
            await File.WriteAllTextAsync(tempFilePath, fileContent);

            try
            {
                // Analyze the temporary file
                var analysisResult = await _codeAnalyzerService.AnalyzeFileAsync(tempFilePath);
                if (analysisResult == null)
                {
                    _logger.LogWarning($"Failed to analyze file content: {filePath}");
                    return null;
                }

                // Update the file path in the result
                analysisResult.FilePath = filePath;

                _logger.LogInformation($"Analyzed file content: {filePath}, Found {analysisResult.Issues.Count} issues");
                return analysisResult;
            }
            finally
            {
                // Delete the temporary file
                if (File.Exists(tempFilePath))
                {
                    File.Delete(tempFilePath);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing file content: {filePath}");
            throw;
        }
    }

    /// <summary>
    /// Analyzes a file using an analyzer replica
    /// </summary>
    /// <param name="filePath">Path to the file to analyze</param>
    /// <param name="replicaUrl">URL of the analyzer replica</param>
    /// <returns>Analysis result</returns>
    public async Task<CodeAnalysisResult> AnalyzeFileWithReplicaAsync(string filePath, string replicaUrl)
    {
        _logger.LogInformation($"Analyzing file with replica: {filePath}");

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
                file_content = fileContent
            };

            // Send the request to the replica
            var response = await _replicaCommunicationProtocol.SendMessageAsync(replicaUrl, "analyze_code", request);

            // Check if the response was successful
            var success = response.TryGetProperty("success", out var successElement) && successElement.GetBoolean();
            if (!success)
            {
                // Extract the error message
                var error = response.TryGetProperty("error", out var errorElement)
                    ? errorElement.GetString()
                    : "Unknown error";

                _logger.LogError($"Error analyzing file with replica: {error}");
                return null;
            }

            // Convert the response to a CodeAnalysisResult
            var analysisResult = ConvertResponseToAnalysisResult(response, filePath);

            _logger.LogInformation($"Analyzed file with replica: {filePath}, Found {analysisResult.Issues.Count} issues");
            return analysisResult;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing file with replica: {filePath}");
            throw;
        }
    }

    /// <summary>
    /// Generates improvement suggestions for a file
    /// </summary>
    /// <param name="analysisResult">Analysis result</param>
    /// <returns>Improvement suggestion</returns>
    public async Task<ImprovementSuggestion> GenerateImprovementSuggestionAsync(CodeAnalysisResult analysisResult)
    {
        _logger.LogInformation($"Generating improvement suggestion for {analysisResult.FilePath}");

        try
        {
            // Validate parameters
            if (analysisResult == null)
            {
                throw new ArgumentNullException(nameof(analysisResult));
            }

            // Generate improvement suggestion
            var improvementSuggestion = await _improvementSuggestionGenerator.GenerateSuggestionAsync(analysisResult);

            _logger.LogInformation($"Generated improvement suggestion for {analysisResult.FilePath}");
            return improvementSuggestion;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating improvement suggestion for {analysisResult.FilePath}");
            throw;
        }
    }

    /// <summary>
    /// Prioritizes issues for fixing
    /// </summary>
    /// <param name="issues">List of issues</param>
    /// <param name="maxIssues">Maximum number of issues to prioritize</param>
    /// <returns>Prioritized list of issues</returns>
    public async Task<List<CodeIssue>> PrioritizeIssuesAsync(List<CodeIssue> issues, int maxIssues = 100)
    {
        _logger.LogInformation($"Prioritizing {issues.Count} issues for fixing");

        try
        {
            // Validate parameters
            if (issues == null || !issues.Any())
            {
                throw new ArgumentException("Issues are required", nameof(issues));
            }

            // Prioritize issues
            var prioritizedIssues = _taskPrioritizer.PrioritizeIssues(issues, maxIssues);

            _logger.LogInformation($"Prioritized {prioritizedIssues.Count} issues for fixing");
            return await Task.FromResult(prioritizedIssues);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error prioritizing issues for fixing");
            throw;
        }
    }

    /// <summary>
    /// Converts a response from an analyzer replica to a CodeAnalysisResult
    /// </summary>
    /// <param name="response">Response from the replica</param>
    /// <param name="filePath">Path to the file</param>
    /// <returns>CodeAnalysisResult</returns>
    private CodeAnalysisResult ConvertResponseToAnalysisResult(System.Text.Json.JsonElement response, string filePath)
    {
        var result = new CodeAnalysisResult
        {
            FilePath = filePath
        };

        // Extract needs_improvement
        if (response.TryGetProperty("needs_improvement", out var needsImprovementElement))
        {
            result.NeedsImprovement = needsImprovementElement.GetBoolean();
        }

        // Extract issues
        if (response.TryGetProperty("issues", out var issuesElement) && issuesElement.ValueKind == System.Text.Json.JsonValueKind.Array)
        {
            foreach (var issueElement in issuesElement.EnumerateArray())
            {
                var issue = new CodeAnalysis.CodeIssue();

                // Extract type
                if (issueElement.TryGetProperty("type", out var typeElement))
                {
                    var typeString = typeElement.GetString();
                    if (Enum.TryParse<CodeIssueType>(typeString, true, out var type))
                    {
                        issue.Type = type;
                    }
                }

                // Extract severity
                if (issueElement.TryGetProperty("severity", out var severityElement))
                {
                    var severityString = severityElement.GetString();
                    if (Enum.TryParse<IssueSeverity>(severityString, true, out var severity))
                    {
                        issue.Severity = IssueSeverityAdapter.ToCodeAnalysisIssueSeverity(severity);
                    }
                }

                // Extract description
                if (issueElement.TryGetProperty("description", out var descriptionElement))
                {
                    issue.Description = descriptionElement.GetString();
                }

                // Extract line_number
                if (issueElement.TryGetProperty("line_number", out var lineNumberElement))
                {
                    issue.LineNumber = lineNumberElement.GetInt32();
                }

                // Extract column_number
                if (issueElement.TryGetProperty("column_number", out var columnNumberElement))
                {
                    issue.ColumnNumber = columnNumberElement.GetInt32();
                }

                // Extract length
                if (issueElement.TryGetProperty("length", out var lengthElement))
                {
                    issue.Length = lengthElement.GetInt32();
                }

                // Extract code_segment
                if (issueElement.TryGetProperty("code_segment", out var codeSegmentElement))
                {
                    issue.CodeSegment = codeSegmentElement.GetString();
                }

                // Extract suggested_fix
                if (issueElement.TryGetProperty("suggested_fix", out var suggestedFixElement))
                {
                    issue.SuggestedFix = suggestedFixElement.GetString();
                }

                result.Issues.Add(issue);
            }
        }

        // Extract metrics
        if (response.TryGetProperty("metrics", out var metricsElement) && metricsElement.ValueKind == System.Text.Json.JsonValueKind.Object)
        {
            foreach (var metric in metricsElement.EnumerateObject())
            {
                if (metric.Value.ValueKind == System.Text.Json.JsonValueKind.Number)
                {
                    result.Metrics[metric.Name] = metric.Value.GetDouble();
                }
            }
        }

        return result;
    }
}