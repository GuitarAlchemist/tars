using System.Text.Json;

namespace TarsCli.Services;

/// <summary>
/// Service for extracting knowledge from documentation and integrating it with the RetroactionLoop
/// </summary>
public class DocumentationKnowledgeService
{
    private readonly ILogger<DocumentationKnowledgeService> _logger;
    private readonly ConsoleService _consoleService;
    private readonly DslService _dslService;
    private readonly RetroactionLoopService _retroactionLoopService;
    private readonly string _knowledgeBaseFile = "knowledge_base.json";
    private readonly string _retroactionPatternsFile = "retroaction_patterns.json";

    /// <summary>
    /// Initializes a new instance of the <see cref="DocumentationKnowledgeService"/> class.
    /// </summary>
    public DocumentationKnowledgeService(
        ILogger<DocumentationKnowledgeService> logger,
        ConsoleService consoleService,
        DslService dslService,
        RetroactionLoopService retroactionLoopService)
    {
        _logger = logger;
        _consoleService = consoleService;
        _dslService = dslService;
        _retroactionLoopService = retroactionLoopService;
    }

    /// <summary>
    /// Extracts knowledge from documentation using the metascript
    /// </summary>
    /// <param name="maxFiles">Maximum number of files to process</param>
    /// <returns>True if the extraction was successful</returns>
    public async Task<bool> ExtractKnowledgeAsync(int maxFiles = 5)
    {
        try
        {
            _consoleService.WriteHeader("TARS Documentation Knowledge Extraction");
            _consoleService.WriteInfo("Extracting knowledge from documentation...");

            // Set environment variables for the metascript
            Environment.SetEnvironmentVariable("TARS_MAX_FILES_TO_PROCESS", maxFiles.ToString());

            // Run the metascript
            string metascriptPath = Path.Combine("Examples", "metascripts", "documentation_knowledge_extraction.tars");
            if (!File.Exists(metascriptPath))
            {
                _consoleService.WriteError($"Metascript not found: {metascriptPath}");
                return false;
            }

            _consoleService.WriteInfo($"Running metascript: {metascriptPath}");
            int result = await _dslService.RunDslFileAsync(metascriptPath, true);

            if (result == 0)
            {
                _consoleService.WriteSuccess("Knowledge extraction completed successfully");
                return true;
            }
            else
            {
                _consoleService.WriteError($"Knowledge extraction failed with exit code {result}");
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error extracting knowledge from documentation");
            _consoleService.WriteError($"Error: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Imports patterns from the knowledge extraction process into the RetroactionLoop
    /// </summary>
    /// <returns>Number of patterns imported</returns>
    public async Task<int> ImportPatternsToRetroactionLoopAsync()
    {
        try
        {
            _consoleService.WriteHeader("TARS Documentation Knowledge Integration");
            _consoleService.WriteInfo("Importing patterns to RetroactionLoop...");

            if (!File.Exists(_retroactionPatternsFile))
            {
                _consoleService.WriteError($"Retroaction patterns file not found: {_retroactionPatternsFile}");
                return 0;
            }

            // Read the retroaction patterns
            string patternsJson = await File.ReadAllTextAsync(_retroactionPatternsFile);
            var patterns = JsonSerializer.Deserialize<List<JsonElement>>(patternsJson);

            if (patterns == null || patterns.Count == 0)
            {
                _consoleService.WriteInfo("No patterns found to import");
                return 0;
            }

            int importedCount = 0;

            // Import each pattern
            foreach (var patternElement in patterns)
            {
                try
                {
                    string name = patternElement.GetProperty("name").GetString() ?? "Unnamed Pattern";
                    string description = patternElement.GetProperty("description").GetString() ?? "";
                    string pattern = patternElement.GetProperty("pattern").GetString() ?? "";
                    string replacement = patternElement.GetProperty("replacement").GetString() ?? "";
                    string context = patternElement.GetProperty("context").GetString() ?? "CSharp";

                    // Skip invalid patterns
                    if (string.IsNullOrWhiteSpace(pattern) || string.IsNullOrWhiteSpace(replacement))
                    {
                        continue;
                    }

                    // Create the pattern in the RetroactionLoop
                    bool success = await _retroactionLoopService.CreatePatternAsync(name, description, pattern, replacement, context);
                    if (success)
                    {
                        importedCount++;
                        _consoleService.WriteInfo($"Imported pattern: {name}");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error importing pattern");
                }
            }

            _consoleService.WriteSuccess($"Imported {importedCount} patterns to RetroactionLoop");
            return importedCount;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error importing patterns to RetroactionLoop");
            _consoleService.WriteError($"Error: {ex.Message}");
            return 0;
        }
    }

    /// <summary>
    /// Gets statistics about the knowledge base
    /// </summary>
    /// <returns>Statistics about the knowledge base</returns>
    public async Task<dynamic> GetKnowledgeBaseStatisticsAsync()
    {
        try
        {
            if (!File.Exists(_knowledgeBaseFile))
            {
                return new
                {
                    Exists = false,
                    TotalEntries = 0,
                    Patterns = 0,
                    BestPractices = 0,
                    CodeExamples = 0,
                    ImprovementStrategies = 0,
                    ArchitectureInsights = 0,
                    LastUpdated = DateTime.MinValue
                };
            }

            // Read the knowledge base
            string kbJson = await File.ReadAllTextAsync(_knowledgeBaseFile);
            var kb = JsonSerializer.Deserialize<JsonElement>(kbJson);

            // Extract statistics
            int patterns = kb.GetProperty("patterns").GetArrayLength();
            int bestPractices = kb.GetProperty("best_practices").GetArrayLength();
            int codeExamples = kb.GetProperty("code_examples").GetArrayLength();
            int improvementStrategies = kb.GetProperty("improvement_strategies").GetArrayLength();
            int architectureInsights = kb.GetProperty("architecture_insights").GetArrayLength();
            string lastUpdated = kb.GetProperty("last_updated").GetString() ?? DateTime.MinValue.ToString();

            return new
            {
                Exists = true,
                TotalEntries = patterns + bestPractices + codeExamples + improvementStrategies + architectureInsights,
                Patterns = patterns,
                BestPractices = bestPractices,
                CodeExamples = codeExamples,
                ImprovementStrategies = improvementStrategies,
                ArchitectureInsights = architectureInsights,
                LastUpdated = DateTime.Parse(lastUpdated)
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting knowledge base statistics");
            return new
            {
                Exists = false,
                Error = ex.Message
            };
        }
    }

    /// <summary>
    /// Runs the full knowledge extraction and integration process
    /// </summary>
    /// <param name="maxFiles">Maximum number of files to process</param>
    /// <returns>True if the process was successful</returns>
    public async Task<bool> RunFullKnowledgeIntegrationAsync(int maxFiles = 5)
    {
        try
        {
            _consoleService.WriteHeader("TARS Documentation Knowledge Integration");
            _consoleService.WriteInfo("Running full knowledge integration process...");

            // Step 1: Extract knowledge from documentation
            bool extractionSuccess = await ExtractKnowledgeAsync(maxFiles);
            if (!extractionSuccess)
            {
                _consoleService.WriteError("Knowledge extraction failed");
                return false;
            }

            // Step 2: Import patterns to RetroactionLoop
            int importedCount = await ImportPatternsToRetroactionLoopAsync();
            if (importedCount == 0)
            {
                _consoleService.WriteWarning("No patterns were imported to RetroactionLoop");
            }

            // Step 3: Get statistics
            var stats = await GetKnowledgeBaseStatisticsAsync();
            _consoleService.WriteInfo($"Knowledge base statistics:");
            _consoleService.WriteInfo($"- Total entries: {stats.TotalEntries}");
            _consoleService.WriteInfo($"- Patterns: {stats.Patterns}");
            _consoleService.WriteInfo($"- Best practices: {stats.BestPractices}");
            _consoleService.WriteInfo($"- Code examples: {stats.CodeExamples}");
            _consoleService.WriteInfo($"- Improvement strategies: {stats.ImprovementStrategies}");
            _consoleService.WriteInfo($"- Architecture insights: {stats.ArchitectureInsights}");
            _consoleService.WriteInfo($"- Last updated: {stats.LastUpdated}");

            _consoleService.WriteSuccess("Knowledge integration process completed successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running knowledge integration process");
            _consoleService.WriteError($"Error: {ex.Message}");
            return false;
        }
    }
}
