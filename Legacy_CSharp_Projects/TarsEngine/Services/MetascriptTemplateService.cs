using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for managing metascript templates
/// </summary>
public class MetascriptTemplateService
{
    private readonly ILogger _logger;
    private readonly string _templateDirectory;
    private readonly Dictionary<string, MetascriptTemplate> _templates = new();
    private readonly Dictionary<string, List<string>> _templatesByTag = new();
    private readonly Dictionary<string, List<string>> _templatesByLanguage = new();
    private bool _isInitialized = false;

    /// <summary>
    /// Initializes a new instance of the <see cref="MetascriptTemplateService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="templateDirectory">The directory containing template files</param>
    public MetascriptTemplateService(ILogger logger, string templateDirectory)
    {
        _logger = logger;
        _templateDirectory = templateDirectory;
    }

    /// <summary>
    /// Initializes the template service
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing template service from directory: {TemplateDirectory}", _templateDirectory);

            if (!Directory.Exists(_templateDirectory))
            {
                _logger.LogWarning("Template directory not found: {TemplateDirectory}", _templateDirectory);
                Directory.CreateDirectory(_templateDirectory);
            }

            // Clear existing templates
            _templates.Clear();
            _templatesByTag.Clear();
            _templatesByLanguage.Clear();

            // Load templates from files
            var templateFiles = Directory.GetFiles(_templateDirectory, "*.json", SearchOption.AllDirectories);
            _logger.LogInformation("Found {TemplateFileCount} template files", templateFiles.Length);

            foreach (var file in templateFiles)
            {
                try
                {
                    var json = await File.ReadAllTextAsync(file);
                    var template = JsonSerializer.Deserialize<MetascriptTemplate>(json, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });

                    if (template != null)
                    {
                        AddTemplateToIndex(template);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error loading template file: {FilePath}", file);
                }
            }

            _logger.LogInformation("Initialized template service with {TemplateCount} templates", _templates.Count);
            _isInitialized = true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing template service");
        }
    }

    /// <summary>
    /// Gets all templates
    /// </summary>
    /// <param name="language">Optional language filter</param>
    /// <returns>The list of templates</returns>
    public async Task<List<MetascriptTemplate>> GetTemplatesAsync(string? language = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting templates with language filter: {Language}", language ?? "all");

            if (string.IsNullOrEmpty(language))
            {
                return _templates.Values.ToList();
            }

            if (_templatesByLanguage.TryGetValue(language.ToLowerInvariant(), out var templateIds))
            {
                return templateIds
                    .Where(id => _templates.ContainsKey(id))
                    .Select(id => _templates[id])
                    .ToList();
            }

            return new List<MetascriptTemplate>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting templates");
            return new List<MetascriptTemplate>();
        }
    }

    /// <summary>
    /// Gets a template by ID
    /// </summary>
    /// <param name="templateId">The template ID</param>
    /// <returns>The template, or null if not found</returns>
    public async Task<MetascriptTemplate?> GetTemplateAsync(string templateId)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting template by ID: {TemplateId}", templateId);

            if (_templates.TryGetValue(templateId, out var template))
            {
                return template;
            }

            _logger.LogWarning("Template not found: {TemplateId}", templateId);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting template: {TemplateId}", templateId);
            return null;
        }
    }

    /// <summary>
    /// Gets templates by tag
    /// </summary>
    /// <param name="tag">The tag</param>
    /// <returns>The list of templates</returns>
    public async Task<List<MetascriptTemplate>> GetTemplatesByTagAsync(string tag)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting templates by tag: {Tag}", tag);

            if (_templatesByTag.TryGetValue(tag.ToLowerInvariant(), out var templateIds))
            {
                return templateIds
                    .Where(id => _templates.ContainsKey(id))
                    .Select(id => _templates[id])
                    .ToList();
            }

            return new List<MetascriptTemplate>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting templates by tag: {Tag}", tag);
            return new List<MetascriptTemplate>();
        }
    }

    /// <summary>
    /// Adds a new template
    /// </summary>
    /// <param name="template">The template to add</param>
    /// <returns>True if the template was added successfully, false otherwise</returns>
    public async Task<bool> AddTemplateAsync(MetascriptTemplate template)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Adding template: {TemplateName} ({TemplateId})", template.Name, template.Id);

            if (_templates.ContainsKey(template.Id))
            {
                _logger.LogWarning("Template already exists: {TemplateId}", template.Id);
                return false;
            }

            // Validate template
            if (!ValidateTemplate(template))
            {
                _logger.LogWarning("Template validation failed: {TemplateName} ({TemplateId})", template.Name, template.Id);
                return false;
            }

            // Add template to index
            AddTemplateToIndex(template);

            // Save template to file
            await SaveTemplateToFileAsync(template);

            _logger.LogInformation("Template added: {TemplateName} ({TemplateId})", template.Name, template.Id);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding template: {TemplateName} ({TemplateId})", template.Name, template.Id);
            return false;
        }
    }

    /// <summary>
    /// Updates an existing template
    /// </summary>
    /// <param name="template">The template to update</param>
    /// <returns>True if the template was updated successfully, false otherwise</returns>
    public async Task<bool> UpdateTemplateAsync(MetascriptTemplate template)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Updating template: {TemplateName} ({TemplateId})", template.Name, template.Id);

            if (!_templates.ContainsKey(template.Id))
            {
                _logger.LogWarning("Template not found: {TemplateId}", template.Id);
                return false;
            }

            // Validate template
            if (!ValidateTemplate(template))
            {
                _logger.LogWarning("Template validation failed: {TemplateName} ({TemplateId})", template.Name, template.Id);
                return false;
            }

            // Remove old template from index
            RemoveTemplateFromIndex(_templates[template.Id]);

            // Update template
            template.UpdatedAt = DateTime.UtcNow;
            AddTemplateToIndex(template);

            // Save template to file
            await SaveTemplateToFileAsync(template);

            _logger.LogInformation("Template updated: {TemplateName} ({TemplateId})", template.Name, template.Id);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating template: {TemplateName} ({TemplateId})", template.Name, template.Id);
            return false;
        }
    }

    /// <summary>
    /// Removes a template
    /// </summary>
    /// <param name="templateId">The ID of the template to remove</param>
    /// <returns>True if the template was removed successfully, false otherwise</returns>
    public async Task<bool> RemoveTemplateAsync(string templateId)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Removing template: {TemplateId}", templateId);

            if (!_templates.TryGetValue(templateId, out var template))
            {
                _logger.LogWarning("Template not found: {TemplateId}", templateId);
                return false;
            }

            // Remove template from index
            RemoveTemplateFromIndex(template);

            // Remove template file
            var filePath = GetTemplateFilePath(template);
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }

            _logger.LogInformation("Template removed: {TemplateId}", templateId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing template: {TemplateId}", templateId);
            return false;
        }
    }

    /// <summary>
    /// Gets all available tags
    /// </summary>
    /// <returns>The list of tags</returns>
    public async Task<List<string>> GetTagsAsync()
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting all tags");
            return _templatesByTag.Keys.ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting tags");
            return new List<string>();
        }
    }

    /// <summary>
    /// Gets all available languages
    /// </summary>
    /// <returns>The list of languages</returns>
    public async Task<List<string>> GetLanguagesAsync()
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting all languages");
            return _templatesByLanguage.Keys.ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting languages");
            return new List<string>();
        }
    }

    /// <summary>
    /// Validates a template
    /// </summary>
    /// <param name="template">The template to validate</param>
    /// <returns>True if the template is valid, false otherwise</returns>
    public bool ValidateTemplate(MetascriptTemplate template)
    {
        try
        {
            _logger.LogInformation("Validating template: {TemplateName} ({TemplateId})", template.Name, template.Id);

            // Check required fields
            if (string.IsNullOrWhiteSpace(template.Name))
            {
                _logger.LogWarning("Template name is required");
                return false;
            }

            if (string.IsNullOrWhiteSpace(template.Code))
            {
                _logger.LogWarning("Template code is required");
                return false;
            }

            // Validate parameters
            foreach (var parameter in template.Parameters)
            {
                if (string.IsNullOrWhiteSpace(parameter.Name))
                {
                    _logger.LogWarning("Parameter name is required");
                    return false;
                }

                // Validate parameter type-specific constraints
                switch (parameter.Type)
                {
                    case MetascriptParameterType.Integer:
                    case MetascriptParameterType.Float:
                        if (parameter.MinValue.HasValue && parameter.MaxValue.HasValue && parameter.MinValue > parameter.MaxValue)
                        {
                            _logger.LogWarning("Parameter {ParameterName} has invalid range: {MinValue} > {MaxValue}", parameter.Name, parameter.MinValue, parameter.MaxValue);
                            return false;
                        }
                        break;

                    case MetascriptParameterType.String:
                        if (parameter.MinLength.HasValue && parameter.MaxLength.HasValue && parameter.MinLength > parameter.MaxLength)
                        {
                            _logger.LogWarning("Parameter {ParameterName} has invalid length range: {MinLength} > {MaxLength}", parameter.Name, parameter.MinLength, parameter.MaxLength);
                            return false;
                        }
                        break;

                    case MetascriptParameterType.Enum:
                        if (parameter.AllowedValues == null || parameter.AllowedValues.Count == 0)
                        {
                            _logger.LogWarning("Parameter {ParameterName} is of type Enum but has no allowed values", parameter.Name);
                            return false;
                        }
                        break;

                    case MetascriptParameterType.Regex:
                        if (!string.IsNullOrEmpty(parameter.Pattern))
                        {
                            try
                            {
                                _ = new Regex(parameter.Pattern);
                            }
                            catch (Exception ex)
                            {
                                _logger.LogWarning(ex, "Parameter {ParameterName} has invalid regex pattern: {Pattern}", parameter.Name, parameter.Pattern);
                                return false;
                            }
                        }
                        break;
                }
            }

            // Validate template code
            if (!ValidateTemplateCode(template.Code, template.Parameters))
            {
                _logger.LogWarning("Template code validation failed");
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating template: {TemplateName} ({TemplateId})", template.Name, template.Id);
            return false;
        }
    }

    /// <summary>
    /// Validates template code
    /// </summary>
    /// <param name="code">The template code</param>
    /// <param name="parameters">The template parameters</param>
    /// <returns>True if the template code is valid, false otherwise</returns>
    private bool ValidateTemplateCode(string code, List<MetascriptParameter> parameters)
    {
        try
        {
            // Check for parameter placeholders
            var placeholderRegex = new Regex(@"\$\{([^}]+)\}|\$([a-zA-Z0-9_]+)");
            var matches = placeholderRegex.Matches(code);
            var placeholders = new HashSet<string>();

            foreach (Match match in matches)
            {
                var placeholderName = match.Groups[1].Success ? match.Groups[1].Value : match.Groups[2].Value;
                placeholders.Add(placeholderName);
            }

            // Check that all required parameters have placeholders
            foreach (var parameter in parameters.Where(p => p.IsRequired))
            {
                if (!placeholders.Contains(parameter.Name))
                {
                    _logger.LogWarning("Required parameter {ParameterName} is not used in the template code", parameter.Name);
                    return false;
                }
            }

            // Check that all placeholders have corresponding parameters
            foreach (var placeholder in placeholders)
            {
                if (!parameters.Any(p => p.Name == placeholder))
                {
                    _logger.LogWarning("Placeholder {Placeholder} has no corresponding parameter", placeholder);
                    return false;
                }
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error validating template code");
            return false;
        }
    }

    /// <summary>
    /// Ensures the template service is initialized
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task EnsureInitializedAsync()
    {
        if (!_isInitialized)
        {
            await InitializeAsync();
        }
    }

    /// <summary>
    /// Adds a template to the index
    /// </summary>
    /// <param name="template">The template to add</param>
    private void AddTemplateToIndex(MetascriptTemplate template)
    {
        _templates[template.Id] = template;

        // Index by language
        var language = template.Language.ToLowerInvariant();
        if (!_templatesByLanguage.ContainsKey(language))
        {
            _templatesByLanguage[language] = new List<string>();
        }
        _templatesByLanguage[language].Add(template.Id);

        // Index by tags
        foreach (var tag in template.Tags)
        {
            var tagKey = tag.ToLowerInvariant();
            if (!_templatesByTag.ContainsKey(tagKey))
            {
                _templatesByTag[tagKey] = new List<string>();
            }
            _templatesByTag[tagKey].Add(template.Id);
        }
    }

    /// <summary>
    /// Removes a template from the index
    /// </summary>
    /// <param name="template">The template to remove</param>
    private void RemoveTemplateFromIndex(MetascriptTemplate template)
    {
        _templates.Remove(template.Id);

        // Remove from language index
        var language = template.Language.ToLowerInvariant();
        if (_templatesByLanguage.TryGetValue(language, out var languageTemplates))
        {
            languageTemplates.Remove(template.Id);
            if (languageTemplates.Count == 0)
            {
                _templatesByLanguage.Remove(language);
            }
        }

        // Remove from tag index
        foreach (var tag in template.Tags)
        {
            var tagKey = tag.ToLowerInvariant();
            if (_templatesByTag.TryGetValue(tagKey, out var tagTemplates))
            {
                tagTemplates.Remove(template.Id);
                if (tagTemplates.Count == 0)
                {
                    _templatesByTag.Remove(tagKey);
                }
            }
        }
    }

    /// <summary>
    /// Saves a template to a file
    /// </summary>
    /// <param name="template">The template to save</param>
    /// <returns>A task representing the asynchronous operation</returns>
    private async Task SaveTemplateToFileAsync(MetascriptTemplate template)
    {
        var filePath = GetTemplateFilePath(template);
        var directory = Path.GetDirectoryName(filePath);
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory!);
        }

        var json = JsonSerializer.Serialize(template, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        await File.WriteAllTextAsync(filePath, json);
    }

    /// <summary>
    /// Gets the file path for a template
    /// </summary>
    /// <param name="template">The template</param>
    /// <returns>The file path</returns>
    private string GetTemplateFilePath(MetascriptTemplate template)
    {
        var language = template.Language.ToLowerInvariant();
        var fileName = $"{template.Id}.json";

        return Path.Combine(_templateDirectory, language, fileName);
    }
}
