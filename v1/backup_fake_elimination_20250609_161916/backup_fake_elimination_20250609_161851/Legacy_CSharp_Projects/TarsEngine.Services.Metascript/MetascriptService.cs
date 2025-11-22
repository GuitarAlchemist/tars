using Microsoft.Extensions.Logging;
using TarsEngine.Services.Abstractions.Metascript;
using TarsEngine.Services.Abstractions.Models.Metascript;
using TarsEngine.Services.Core.Base;

namespace TarsEngine.Services.Metascript
{
    /// <summary>
    /// Implementation of the IMetascriptService interface.
    /// </summary>
    public class MetascriptService : ServiceBase, IMetascriptService
    {
        private readonly Dictionary<string, MetascriptTemplate> _templates = new();

        /// <summary>
        /// Initializes a new instance of the <see cref="MetascriptService"/> class.
        /// </summary>
        /// <param name="logger">The logger instance.</param>
        public MetascriptService(ILogger<MetascriptService> logger)
            : base(logger)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Metascript Service";

        /// <inheritdoc/>
        public Task CreateTemplateAsync(MetascriptTemplate template)
        {
            Logger.LogInformation("Creating Metascript template: {TemplateName}", template.Name);
            
            if (_templates.ContainsKey(template.Name))
            {
                Logger.LogWarning("Template with name {TemplateName} already exists", template.Name);
                throw new InvalidOperationException($"Template with name {template.Name} already exists");
            }
            
            _templates[template.Name] = template;
            
            return Task.CompletedTask;
        }

        /// <inheritdoc/>
        public Task DeleteTemplateAsync(string templateName)
        {
            Logger.LogInformation("Deleting Metascript template: {TemplateName}", templateName);
            
            if (!_templates.ContainsKey(templateName))
            {
                Logger.LogWarning("Template with name {TemplateName} not found", templateName);
                throw new KeyNotFoundException($"Template with name {templateName} not found");
            }
            
            _templates.Remove(templateName);
            
            return Task.CompletedTask;
        }

        /// <inheritdoc/>
        public async Task<MetascriptExecutionResult> ExecuteAsync(string script, Dictionary<string, object>? parameters = null)
        {
            Logger.LogInformation("Executing Metascript, parameters: {ParameterCount}", parameters?.Count ?? 0);
            
            var startTime = DateTime.UtcNow;
            
            // Validate the script first
            var validationResult = await ValidateAsync(script);
            
            if (!validationResult.IsValid)
            {
                Logger.LogWarning("Script validation failed with {ErrorCount} errors", validationResult.Errors.Count);
                
                return new MetascriptExecutionResult
                {
                    Success = false,
                    Status = MetascriptExecutionStatus.Failed,
                    ErrorMessage = "Script validation failed: " + string.Join("; ", validationResult.Errors),
                    StartTimestamp = startTime,
                    EndTimestamp = DateTime.UtcNow
                };
            }
            
            // Simulate script execution
            await Task.Delay(100);
            
            // For demonstration purposes, we'll just return a success result
            var endTime = DateTime.UtcNow;
            var executionTime = (long)(endTime - startTime).TotalMilliseconds;
            
            return new MetascriptExecutionResult
            {
                Success = true,
                Status = MetascriptExecutionStatus.Completed,
                Output = $"Script executed successfully with {parameters?.Count ?? 0} parameters",
                StartTimestamp = startTime,
                EndTimestamp = endTime,
                ExecutionTimeMs = executionTime,
                Logs = new List<string> { "Script execution started", "Script execution completed" }
            };
        }

        /// <inheritdoc/>
        public Task<MetascriptTemplate?> GetTemplateAsync(string templateName)
        {
            Logger.LogInformation("Getting Metascript template: {TemplateName}", templateName);
            
            _templates.TryGetValue(templateName, out var template);
            
            return Task.FromResult(template);
        }

        /// <inheritdoc/>
        public Task UpdateTemplateAsync(MetascriptTemplate template)
        {
            Logger.LogInformation("Updating Metascript template: {TemplateName}", template.Name);
            
            if (!_templates.ContainsKey(template.Name))
            {
                Logger.LogWarning("Template with name {TemplateName} not found", template.Name);
                throw new KeyNotFoundException($"Template with name {template.Name} not found");
            }
            
            template.UpdatedAt = DateTime.UtcNow;
            _templates[template.Name] = template;
            
            return Task.CompletedTask;
        }

        /// <inheritdoc/>
        public Task<MetascriptValidationResult> ValidateAsync(string script)
        {
            Logger.LogInformation("Validating Metascript");
            
            // For demonstration purposes, we'll just do some basic validation
            var result = new MetascriptValidationResult
            {
                IsValid = true,
                Status = MetascriptValidationStatus.Completed,
                ValidationTimestamp = DateTime.UtcNow
            };
            
            // Check for some basic issues
            if (string.IsNullOrWhiteSpace(script))
            {
                result.IsValid = false;
                result.Status = MetascriptValidationStatus.Error;
                result.Errors.Add("Script cannot be empty");
            }
            
            // Check for balanced braces
            if (CountOccurrences(script, '{') != CountOccurrences(script, '}'))
            {
                result.IsValid = false;
                result.Status = MetascriptValidationStatus.Error;
                result.Errors.Add("Unbalanced braces in script");
            }
            
            // Check for balanced parentheses
            if (CountOccurrences(script, '(') != CountOccurrences(script, ')'))
            {
                result.IsValid = false;
                result.Status = MetascriptValidationStatus.Error;
                result.Errors.Add("Unbalanced parentheses in script");
            }
            
            // Check for balanced brackets
            if (CountOccurrences(script, '[') != CountOccurrences(script, ']'))
            {
                result.IsValid = false;
                result.Status = MetascriptValidationStatus.Error;
                result.Errors.Add("Unbalanced brackets in script");
            }
            
            return Task.FromResult(result);
        }
        
        private int CountOccurrences(string text, char character)
        {
            return text.Count(c => c == character);
        }
    }
}
