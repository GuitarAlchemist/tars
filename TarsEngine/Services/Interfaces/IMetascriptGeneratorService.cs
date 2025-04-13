using TarsEngine.Models;

namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Interface for the metascript generator service
/// </summary>
public interface IMetascriptGeneratorService
{
    /// <summary>
    /// Generates a metascript from a pattern match
    /// </summary>
    /// <param name="patternMatch">The pattern match</param>
    /// <param name="options">Optional generation options</param>
    /// <returns>The generated metascript</returns>
    Task<GeneratedMetascript> GenerateMetascriptAsync(PatternMatch patternMatch, Dictionary<string, string>? options = null);

    /// <summary>
    /// Generates metascripts from a list of pattern matches
    /// </summary>
    /// <param name="patternMatches">The pattern matches</param>
    /// <param name="options">Optional generation options</param>
    /// <returns>The list of generated metascripts</returns>
    Task<List<GeneratedMetascript>> GenerateMetascriptsAsync(List<PatternMatch> patternMatches, Dictionary<string, string>? options = null);

    /// <summary>
    /// Validates a metascript
    /// </summary>
    /// <param name="metascript">The metascript to validate</param>
    /// <param name="options">Optional validation options</param>
    /// <returns>The validation result</returns>
    Task<MetascriptValidationResult> ValidateMetascriptAsync(GeneratedMetascript metascript, Dictionary<string, string>? options = null);

    /// <summary>
    /// Executes a metascript
    /// </summary>
    /// <param name="metascript">The metascript to execute</param>
    /// <param name="options">Optional execution options</param>
    /// <returns>The execution result</returns>
    Task<MetascriptExecutionResult> ExecuteMetascriptAsync(GeneratedMetascript metascript, Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets all available templates
    /// </summary>
    /// <param name="language">Optional language filter</param>
    /// <returns>The list of available templates</returns>
    Task<List<MetascriptTemplate>> GetTemplatesAsync(string? language = null);

    /// <summary>
    /// Gets a template by ID
    /// </summary>
    /// <param name="templateId">The template ID</param>
    /// <returns>The template, or null if not found</returns>
    Task<MetascriptTemplate?> GetTemplateAsync(string templateId);

    /// <summary>
    /// Adds a new template
    /// </summary>
    /// <param name="template">The template to add</param>
    /// <returns>True if the template was added successfully, false otherwise</returns>
    Task<bool> AddTemplateAsync(MetascriptTemplate template);

    /// <summary>
    /// Updates an existing template
    /// </summary>
    /// <param name="template">The template to update</param>
    /// <returns>True if the template was updated successfully, false otherwise</returns>
    Task<bool> UpdateTemplateAsync(MetascriptTemplate template);

    /// <summary>
    /// Removes a template
    /// </summary>
    /// <param name="templateId">The ID of the template to remove</param>
    /// <returns>True if the template was removed successfully, false otherwise</returns>
    Task<bool> RemoveTemplateAsync(string templateId);

    /// <summary>
    /// Gets all generated metascripts
    /// </summary>
    /// <param name="options">Optional filter options</param>
    /// <returns>The list of generated metascripts</returns>
    Task<List<GeneratedMetascript>> GetMetascriptsAsync(Dictionary<string, string>? options = null);

    /// <summary>
    /// Gets a metascript by ID
    /// </summary>
    /// <param name="metascriptId">The metascript ID</param>
    /// <returns>The metascript, or null if not found</returns>
    Task<GeneratedMetascript?> GetMetascriptAsync(string metascriptId);

    /// <summary>
    /// Saves a metascript
    /// </summary>
    /// <param name="metascript">The metascript to save</param>
    /// <returns>True if the metascript was saved successfully, false otherwise</returns>
    Task<bool> SaveMetascriptAsync(GeneratedMetascript metascript);

    /// <summary>
    /// Removes a metascript
    /// </summary>
    /// <param name="metascriptId">The ID of the metascript to remove</param>
    /// <returns>True if the metascript was removed successfully, false otherwise</returns>
    Task<bool> RemoveMetascriptAsync(string metascriptId);

    /// <summary>
    /// Gets the available generation options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    Task<Dictionary<string, string>> GetAvailableOptionsAsync();

    /// <summary>
    /// Gets the supported metascript languages
    /// </summary>
    /// <returns>The list of supported metascript languages</returns>
    Task<List<string>> GetSupportedLanguagesAsync();
}
