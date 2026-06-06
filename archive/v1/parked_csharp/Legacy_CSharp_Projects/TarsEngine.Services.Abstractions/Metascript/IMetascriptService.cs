using TarsEngine.Services.Abstractions.Common;
using TarsEngine.Services.Abstractions.Models.Metascript;

namespace TarsEngine.Services.Abstractions.Metascript
{
    /// <summary>
    /// Interface for services that work with Metascript.
    /// </summary>
    public interface IMetascriptService : IService
    {
        /// <summary>
        /// Executes a Metascript.
        /// </summary>
        /// <param name="script">The Metascript to execute.</param>
        /// <param name="parameters">The parameters to pass to the script.</param>
        /// <returns>The result of the execution.</returns>
        Task<MetascriptExecutionResult> ExecuteAsync(string script, Dictionary<string, object>? parameters = null);

        /// <summary>
        /// Validates a Metascript.
        /// </summary>
        /// <param name="script">The Metascript to validate.</param>
        /// <returns>The validation result.</returns>
        Task<MetascriptValidationResult> ValidateAsync(string script);

        /// <summary>
        /// Gets a Metascript template by name.
        /// </summary>
        /// <param name="templateName">The name of the template to retrieve.</param>
        /// <returns>The Metascript template, or null if not found.</returns>
        Task<MetascriptTemplate?> GetTemplateAsync(string templateName);

        /// <summary>
        /// Creates a new Metascript template.
        /// </summary>
        /// <param name="template">The template to create.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task CreateTemplateAsync(MetascriptTemplate template);

        /// <summary>
        /// Updates an existing Metascript template.
        /// </summary>
        /// <param name="template">The template to update.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task UpdateTemplateAsync(MetascriptTemplate template);

        /// <summary>
        /// Deletes a Metascript template by name.
        /// </summary>
        /// <param name="templateName">The name of the template to delete.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task DeleteTemplateAsync(string templateName);
    }
}
