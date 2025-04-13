using Microsoft.Extensions.Logging;
using TarsEngine.Models;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Services;

/// <summary>
/// Service for generating metascripts from pattern matches
/// </summary>
public class MetascriptGeneratorService : IMetascriptGeneratorService
{
    private readonly ILogger<MetascriptGeneratorService> _logger;
    private readonly MetascriptTemplateService _templateService;
    private readonly TemplateFiller _templateFiller;
    private readonly ParameterOptimizer _parameterOptimizer;
    private readonly MetascriptSandbox _metascriptSandbox;
    private readonly string _metascriptDirectory;
    private readonly Dictionary<string, GeneratedMetascript> _metascripts = new();
    private bool _isInitialized = false;

    /// <summary>
    /// Initializes a new instance of the <see cref="MetascriptGeneratorService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="templateService">The template service</param>
    /// <param name="templateFiller">The template filler</param>
    /// <param name="parameterOptimizer">The parameter optimizer</param>
    /// <param name="metascriptSandbox">The metascript sandbox</param>
    /// <param name="metascriptDirectory">The directory for storing metascripts</param>
    public MetascriptGeneratorService(
        ILogger<MetascriptGeneratorService> logger,
        MetascriptTemplateService templateService,
        TemplateFiller templateFiller,
        ParameterOptimizer parameterOptimizer,
        MetascriptSandbox metascriptSandbox,
        string metascriptDirectory)
    {
        _logger = logger;
        _templateService = templateService;
        _templateFiller = templateFiller;
        _parameterOptimizer = parameterOptimizer;
        _metascriptSandbox = metascriptSandbox;
        _metascriptDirectory = metascriptDirectory;
    }

    /// <summary>
    /// Initializes the metascript generator service
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing metascript generator service from directory: {MetascriptDirectory}", _metascriptDirectory);

            if (!Directory.Exists(_metascriptDirectory))
            {
                _logger.LogWarning("Metascript directory not found: {MetascriptDirectory}", _metascriptDirectory);
                Directory.CreateDirectory(_metascriptDirectory);
            }

            // Clear existing metascripts
            _metascripts.Clear();

            // Load metascripts from files
            var metascriptFiles = Directory.GetFiles(_metascriptDirectory, "*.json", SearchOption.AllDirectories);
            _logger.LogInformation("Found {MetascriptFileCount} metascript files", metascriptFiles.Length);

            foreach (var file in metascriptFiles)
            {
                try
                {
                    var json = await File.ReadAllTextAsync(file);
                    var metascript = System.Text.Json.JsonSerializer.Deserialize<GeneratedMetascript>(json, new System.Text.Json.JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });

                    if (metascript != null)
                    {
                        _metascripts[metascript.Id] = metascript;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error loading metascript file: {FilePath}", file);
                }
            }

            _logger.LogInformation("Initialized metascript generator service with {MetascriptCount} metascripts", _metascripts.Count);
            _isInitialized = true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing metascript generator service");
        }
    }

    /// <inheritdoc/>
    public async Task<GeneratedMetascript> GenerateMetascriptAsync(PatternMatch patternMatch, Dictionary<string, string>? options = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Generating metascript for pattern match: {PatternName} ({PatternId})", patternMatch.PatternName, patternMatch.PatternId);

            // Find suitable template
            var template = await FindTemplateForPatternMatchAsync(patternMatch, options);
            if (template == null)
            {
                throw new InvalidOperationException($"No suitable template found for pattern: {patternMatch.PatternName} ({patternMatch.PatternId})");
            }

            // Extract parameters from pattern match
            var parameters = _templateFiller.ExtractParametersFromPatternMatch(patternMatch, template);

            // Optimize parameters
            var context = new Dictionary<string, object>
            {
                ["PatternMatch"] = patternMatch
            };
            parameters = _parameterOptimizer.OptimizeParameters(parameters, template, context);

            // Validate parameters
            var (isValid, errors) = _templateFiller.ValidateParameters(parameters, template);
            if (!isValid)
            {
                throw new InvalidOperationException($"Invalid parameters for template: {template.Name} ({template.Id}). Errors: {string.Join(", ", errors)}");
            }

            // Fill template
            var metascriptCode = _templateFiller.FillTemplate(template, parameters);

            // Create metascript
            var metascript = new GeneratedMetascript
            {
                Name = $"{patternMatch.PatternName} Metascript",
                Description = $"Metascript generated from pattern match: {patternMatch.PatternName} ({patternMatch.PatternId})",
                Code = metascriptCode,
                Language = template.Language,
                TemplateId = template.Id,
                PatternId = patternMatch.PatternId,
                Parameters = parameters,
                ExpectedImprovement = patternMatch.ExpectedImprovement,
                ImpactScore = patternMatch.ImpactScore,
                DifficultyScore = patternMatch.DifficultyScore,
                Tags = [..patternMatch.Tags],
                AffectedFiles = [patternMatch.FilePath]
            };

            // Validate metascript
            var validationResult = await ValidateMetascriptAsync(metascript, options);
            metascript.ValidationStatus = validationResult.Status;
            metascript.ValidationMessages = validationResult.Messages;

            // Save metascript
            await SaveMetascriptAsync(metascript);

            _logger.LogInformation("Generated metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);
            return metascript;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating metascript for pattern match: {PatternName} ({PatternId})", patternMatch.PatternName, patternMatch.PatternId);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<List<GeneratedMetascript>> GenerateMetascriptsAsync(List<PatternMatch> patternMatches, Dictionary<string, string>? options = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Generating metascripts for {PatternMatchCount} pattern matches", patternMatches.Count);

            var metascripts = new List<GeneratedMetascript>();

            foreach (var patternMatch in patternMatches)
            {
                try
                {
                    var metascript = await GenerateMetascriptAsync(patternMatch, options);
                    metascripts.Add(metascript);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error generating metascript for pattern match: {PatternName} ({PatternId})", patternMatch.PatternName, patternMatch.PatternId);
                }
            }

            _logger.LogInformation("Generated {MetascriptCount} metascripts", metascripts.Count);
            return metascripts;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating metascripts");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<MetascriptValidationResult> ValidateMetascriptAsync(GeneratedMetascript metascript, Dictionary<string, string>? options = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Validating metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);

            // Validate metascript using sandbox
            var validationResult = await _metascriptSandbox.ValidateMetascriptAsync(metascript, options);

            _logger.LogInformation("Validated metascript: {MetascriptName} ({MetascriptId}), Status: {Status}", metascript.Name, metascript.Id, validationResult.Status);
            return validationResult;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<MetascriptExecutionResult> ExecuteMetascriptAsync(GeneratedMetascript metascript, Dictionary<string, string>? options = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Executing metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);

            // Create execution context
            var context = new Dictionary<string, object>();
            foreach (var parameter in metascript.Parameters)
            {
                context[parameter.Key] = parameter.Value;
            }

            // Execute metascript using sandbox
            var executionResult = await _metascriptSandbox.ExecuteMetascriptAsync(metascript, context, options);

            // Update metascript with execution result
            metascript.ExecutionStatus = executionResult.Status;
            metascript.ExecutionResult = executionResult.Output;
            metascript.LastExecutedAt = executionResult.CompletedAt;
            metascript.ExecutionTimeMs = executionResult.ExecutionTimeMs;

            // Save metascript
            await SaveMetascriptAsync(metascript);

            _logger.LogInformation("Executed metascript: {MetascriptName} ({MetascriptId}), Status: {Status}", metascript.Name, metascript.Id, executionResult.Status);
            return executionResult;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<List<MetascriptTemplate>> GetTemplatesAsync(string? language = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting templates with language filter: {Language}", language ?? "all");
            return await _templateService.GetTemplatesAsync(language);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting templates");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<MetascriptTemplate?> GetTemplateAsync(string templateId)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting template by ID: {TemplateId}", templateId);
            return await _templateService.GetTemplateAsync(templateId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting template: {TemplateId}", templateId);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> AddTemplateAsync(MetascriptTemplate template)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Adding template: {TemplateName} ({TemplateId})", template.Name, template.Id);
            return await _templateService.AddTemplateAsync(template);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding template: {TemplateName} ({TemplateId})", template.Name, template.Id);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> UpdateTemplateAsync(MetascriptTemplate template)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Updating template: {TemplateName} ({TemplateId})", template.Name, template.Id);
            return await _templateService.UpdateTemplateAsync(template);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating template: {TemplateName} ({TemplateId})", template.Name, template.Id);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> RemoveTemplateAsync(string templateId)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Removing template: {TemplateId}", templateId);
            return await _templateService.RemoveTemplateAsync(templateId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing template: {TemplateId}", templateId);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<List<GeneratedMetascript>> GetMetascriptsAsync(Dictionary<string, string>? options = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting metascripts");

            // Apply filters
            var metascripts = _metascripts.Values.ToList();

            if (options != null)
            {
                // Filter by language
                if (options.TryGetValue("Language", out var language) && !string.IsNullOrEmpty(language))
                {
                    metascripts = metascripts.Where(m => m.Language.Equals(language, StringComparison.OrdinalIgnoreCase)).ToList();
                }

                // Filter by tag
                if (options.TryGetValue("Tag", out var tag) && !string.IsNullOrEmpty(tag))
                {
                    metascripts = metascripts.Where(m => m.Tags.Contains(tag, StringComparer.OrdinalIgnoreCase)).ToList();
                }

                // Filter by status
                if (options.TryGetValue("ExecutionStatus", out var statusStr) && Enum.TryParse<MetascriptExecutionStatus>(statusStr, true, out var status))
                {
                    metascripts = metascripts.Where(m => m.ExecutionStatus == status).ToList();
                }

                // Filter by validation status
                if (options.TryGetValue("ValidationStatus", out var validationStatusStr) && Enum.TryParse<MetascriptValidationStatus>(validationStatusStr, true, out var validationStatus))
                {
                    metascripts = metascripts.Where(m => m.ValidationStatus == validationStatus).ToList();
                }

                // Filter by pattern ID
                if (options.TryGetValue("PatternId", out var patternId) && !string.IsNullOrEmpty(patternId))
                {
                    metascripts = metascripts.Where(m => m.PatternId == patternId).ToList();
                }

                // Filter by template ID
                if (options.TryGetValue("TemplateId", out var templateId) && !string.IsNullOrEmpty(templateId))
                {
                    metascripts = metascripts.Where(m => m.TemplateId == templateId).ToList();
                }

                // Filter by affected file
                if (options.TryGetValue("AffectedFile", out var affectedFile) && !string.IsNullOrEmpty(affectedFile))
                {
                    metascripts = metascripts.Where(m => m.AffectedFiles.Contains(affectedFile)).ToList();
                }

                // Sort by priority
                if (options.TryGetValue("SortBy", out var sortBy) && !string.IsNullOrEmpty(sortBy))
                {
                    switch (sortBy.ToLowerInvariant())
                    {
                        case "priority":
                            metascripts = metascripts.OrderByDescending(m => m.PriorityScore).ToList();
                            break;
                        case "impact":
                            metascripts = metascripts.OrderByDescending(m => m.ImpactScore).ToList();
                            break;
                        case "difficulty":
                            metascripts = metascripts.OrderBy(m => m.DifficultyScore).ToList();
                            break;
                        case "date":
                            metascripts = metascripts.OrderByDescending(m => m.GeneratedAt).ToList();
                            break;
                        case "name":
                            metascripts = metascripts.OrderBy(m => m.Name).ToList();
                            break;
                    }
                }

                // Limit results
                if (options.TryGetValue("Limit", out var limitStr) && int.TryParse(limitStr, out var limit) && limit > 0)
                {
                    metascripts = metascripts.Take(limit).ToList();
                }
            }

            _logger.LogInformation("Got {MetascriptCount} metascripts", metascripts.Count);
            return metascripts;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting metascripts");
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<GeneratedMetascript?> GetMetascriptAsync(string metascriptId)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting metascript by ID: {MetascriptId}", metascriptId);

            if (_metascripts.TryGetValue(metascriptId, out var metascript))
            {
                return metascript;
            }

            _logger.LogWarning("Metascript not found: {MetascriptId}", metascriptId);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting metascript: {MetascriptId}", metascriptId);
            throw;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> SaveMetascriptAsync(GeneratedMetascript metascript)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Saving metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);

            // Add or update metascript in memory
            _metascripts[metascript.Id] = metascript;

            // Save metascript to file
            var filePath = GetMetascriptFilePath(metascript);
            var directory = Path.GetDirectoryName(filePath);
            if (!Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory!);
            }

            var json = System.Text.Json.JsonSerializer.Serialize(metascript, new System.Text.Json.JsonSerializerOptions
            {
                WriteIndented = true
            });

            await File.WriteAllTextAsync(filePath, json);

            _logger.LogInformation("Saved metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving metascript: {MetascriptName} ({MetascriptId})", metascript.Name, metascript.Id);
            return false;
        }
    }

    /// <inheritdoc/>
    public async Task<bool> RemoveMetascriptAsync(string metascriptId)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Removing metascript: {MetascriptId}", metascriptId);

            if (!_metascripts.TryGetValue(metascriptId, out var metascript))
            {
                _logger.LogWarning("Metascript not found: {MetascriptId}", metascriptId);
                return false;
            }

            // Remove metascript from memory
            _metascripts.Remove(metascriptId);

            // Remove metascript file
            var filePath = GetMetascriptFilePath(metascript);
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }

            _logger.LogInformation("Removed metascript: {MetascriptId}", metascriptId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing metascript: {MetascriptId}", metascriptId);
            return false;
        }
    }

    /// <inheritdoc/>
    public async Task<Dictionary<string, string>> GetAvailableOptionsAsync()
    {
        var options = new Dictionary<string, string>
        {
            { "Language", "Filter metascripts by language" },
            { "Tag", "Filter metascripts by tag" },
            { "ExecutionStatus", "Filter metascripts by execution status" },
            { "ValidationStatus", "Filter metascripts by validation status" },
            { "PatternId", "Filter metascripts by pattern ID" },
            { "TemplateId", "Filter metascripts by template ID" },
            { "AffectedFile", "Filter metascripts by affected file" },
            { "SortBy", "Sort metascripts by priority, impact, difficulty, date, or name" },
            { "Limit", "Limit the number of metascripts returned" },
            { "TimeoutMs", "Execution timeout in milliseconds (default: 30000)" },
            { "MemoryLimitMb", "Memory limit in megabytes (default: 100)" },
            { "CaptureOutput", "Whether to capture output (default: true)" },
            { "ValidateResult", "Whether to validate the execution result (default: true)" },
            { "TrackPerformance", "Whether to track performance metrics (default: true)" }
        };

        // Add sandbox options
        var sandboxOptions = _metascriptSandbox.GetAvailableOptions();
        foreach (var option in sandboxOptions)
        {
            if (!options.ContainsKey(option.Key))
            {
                options[option.Key] = option.Value;
            }
        }

        return options;
    }

    /// <inheritdoc/>
    public async Task<List<string>> GetSupportedLanguagesAsync()
    {
        return _metascriptSandbox.GetSupportedLanguages();
    }

    /// <summary>
    /// Finds a suitable template for a pattern match
    /// </summary>
    /// <param name="patternMatch">The pattern match</param>
    /// <param name="options">Optional options</param>
    /// <returns>The suitable template, or null if not found</returns>
    private async Task<MetascriptTemplate?> FindTemplateForPatternMatchAsync(PatternMatch patternMatch, Dictionary<string, string>? options)
    {
        try
        {
            // Get all templates for the pattern's language
            var templates = await _templateService.GetTemplatesAsync(patternMatch.Language);

            // Find templates that are applicable to the pattern
            var applicableTemplates = templates
                .Where(t => t.ApplicablePatterns.Contains(patternMatch.PatternId))
                .ToList();

            if (applicableTemplates.Count == 0)
            {
                // If no specific templates found, try to find templates by tag
                applicableTemplates = templates
                    .Where(t => patternMatch.Tags.Any(tag => t.Tags.Contains(tag, StringComparer.OrdinalIgnoreCase)))
                    .ToList();
            }

            if (applicableTemplates.Count == 0)
            {
                // If still no templates found, use any template for the language
                applicableTemplates = templates;
            }

            // Sort templates by usage count (descending) and select the first one
            return applicableTemplates
                .OrderByDescending(t => t.UsageCount)
                .FirstOrDefault();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding template for pattern match: {PatternName} ({PatternId})", patternMatch.PatternName, patternMatch.PatternId);
            return null;
        }
    }

    /// <summary>
    /// Gets the file path for a metascript
    /// </summary>
    /// <param name="metascript">The metascript</param>
    /// <returns>The file path</returns>
    private string GetMetascriptFilePath(GeneratedMetascript metascript)
    {
        var language = metascript.Language.ToLowerInvariant();
        var fileName = $"{metascript.Id}.json";

        return Path.Combine(_metascriptDirectory, language, fileName);
    }

    /// <summary>
    /// Ensures the metascript generator service is initialized
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task EnsureInitializedAsync()
    {
        if (!_isInitialized)
        {
            await InitializeAsync();
        }
    }
}
