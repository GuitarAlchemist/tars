using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsApp.Services.Interfaces;
using IEngineImprovementService = TarsEngine.Services.Interfaces.IImprovementService;

namespace TarsApp.Services
{
    public class ImprovementPrioritizerService : IImprovementPrioritizerService
    {
        private readonly ILogger<ImprovementPrioritizerService> _logger;
        private readonly IEngineImprovementService _improvementService;

        public ImprovementPrioritizerService(
            ILogger<ImprovementPrioritizerService> logger,
            IEngineImprovementService improvementService)
        {
            _logger = logger;
            _improvementService = improvementService;
        }

        public async Task<List<string>> GetImprovementCategoriesAsync()
        {
            try
            {
                return await _improvementService.GetImprovementCategoriesAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving improvement categories");
                return new List<string>();
            }
        }

        public async Task<List<string>> GetImprovementPrioritiesAsync()
        {
            try
            {
                return await _improvementService.GetImprovementPrioritiesAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving improvement priorities");
                return new List<string>();
            }
        }

        public async Task<List<string>> GetImprovementTagsAsync()
        {
            try
            {
                return await _improvementService.GetImprovementTagsAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving improvement tags");
                return new List<string>();
            }
        }

        public async Task<bool> PrioritizeImprovementAsync(string executionId, string improvementId, int priority)
        {
            try
            {
                return await _improvementService.PrioritizeImprovementAsync(executionId, improvementId, priority);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error prioritizing improvement {ImprovementId} for execution {ExecutionId}", improvementId, executionId);
                return false;
            }
        }

        public async Task<bool> TagImprovementAsync(string executionId, string improvementId, string tag)
        {
            try
            {
                return await _improvementService.TagImprovementAsync(executionId, improvementId, tag);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error tagging improvement {ImprovementId} for execution {ExecutionId}", improvementId, executionId);
                return false;
            }
        }

        public async Task<bool> RemoveTagFromImprovementAsync(string executionId, string improvementId, string tag)
        {
            try
            {
                return await _improvementService.RemoveTagFromImprovementAsync(executionId, improvementId, tag);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error removing tag from improvement {ImprovementId} for execution {ExecutionId}", improvementId, executionId);
                return false;
            }
        }
    }
}
