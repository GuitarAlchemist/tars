using System;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsEngine.Metascripts
{
    /// <summary>
    /// Implementation of the <see cref="IMetascriptExecutor"/> interface.
    /// </summary>
    public class MetascriptExecutor : IMetascriptExecutor
    {
        private readonly ILogger<MetascriptExecutor> _logger;
        private readonly IMetascriptService _metascriptService;

        /// <summary>
        /// Initializes a new instance of the <see cref="MetascriptExecutor"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="metascriptService">The metascript service.</param>
        public MetascriptExecutor(
            ILogger<MetascriptExecutor> logger,
            IMetascriptService metascriptService)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _metascriptService = metascriptService ?? throw new ArgumentNullException(nameof(metascriptService));
        }

        /// <inheritdoc/>
        public async Task<MetascriptExecutionResult> ExecuteMetascriptAsync(string metascriptPath, object? parameters = null)
        {
            try
            {
                _logger.LogInformation("Executing metascript: {MetascriptPath}", metascriptPath);

                // Check if the file exists
                if (!File.Exists(metascriptPath))
                {
                    _logger.LogError("Metascript file not found: {MetascriptPath}", metascriptPath);
                    return new MetascriptExecutionResult
                    {
                        Success = false,
                        ErrorMessage = $"Metascript file not found: {metascriptPath}"
                    };
                }

                // Read the metascript content
                var metascriptContent = await File.ReadAllTextAsync(metascriptPath);

                // Convert parameters to JSON if provided
                string parametersJson = string.Empty;
                if (parameters != null)
                {
                    parametersJson = JsonSerializer.Serialize(parameters);
                }

                // Execute the metascript
                var result = await _metascriptService.ExecuteMetascriptAsync(metascriptContent);

                // Convert the result to a MetascriptExecutionResult
                return new MetascriptExecutionResult
                {
                    Success = true,
                    Output = result?.ToString()
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error executing metascript: {MetascriptPath}", metascriptPath);
                return new MetascriptExecutionResult
                {
                    Success = false,
                    ErrorMessage = ex.Message
                };
            }
        }
    }
}
