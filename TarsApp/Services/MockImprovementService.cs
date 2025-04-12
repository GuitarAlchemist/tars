using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Services.Interfaces;

namespace TarsApp.Services
{
    public class MockImprovementService : IImprovementService
    {
        private readonly ILogger<MockImprovementService> _logger;
        private readonly Dictionary<string, Dictionary<string, int>> _improvementPriorities = new();
        private readonly Dictionary<string, Dictionary<string, List<string>>> _improvementTags = new();

        public MockImprovementService(ILogger<MockImprovementService> logger)
        {
            _logger = logger;
        }

        public Task<List<string>> GetImprovementCategoriesAsync()
        {
            return Task.FromResult(new List<string>
            {
                "Performance",
                "Security",
                "Maintainability",
                "Reliability",
                "Usability"
            });
        }

        public Task<List<string>> GetImprovementPrioritiesAsync()
        {
            return Task.FromResult(new List<string>
            {
                "Critical",
                "High",
                "Medium",
                "Low"
            });
        }

        public Task<List<string>> GetImprovementTagsAsync()
        {
            return Task.FromResult(new List<string>
            {
                "Bug",
                "Feature",
                "Enhancement",
                "Refactoring",
                "Documentation",
                "Testing"
            });
        }

        public Task<bool> PrioritizeImprovementAsync(string executionId, string improvementId, int priority)
        {
            try
            {
                if (!_improvementPriorities.ContainsKey(executionId))
                {
                    _improvementPriorities[executionId] = new Dictionary<string, int>();
                }

                _improvementPriorities[executionId][improvementId] = priority;
                return Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error prioritizing improvement {ImprovementId} for execution {ExecutionId}", improvementId, executionId);
                return Task.FromResult(false);
            }
        }

        public Task<bool> TagImprovementAsync(string executionId, string improvementId, string tag)
        {
            try
            {
                if (!_improvementTags.ContainsKey(executionId))
                {
                    _improvementTags[executionId] = new Dictionary<string, List<string>>();
                }

                if (!_improvementTags[executionId].ContainsKey(improvementId))
                {
                    _improvementTags[executionId][improvementId] = new List<string>();
                }

                if (!_improvementTags[executionId][improvementId].Contains(tag))
                {
                    _improvementTags[executionId][improvementId].Add(tag);
                }

                return Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error tagging improvement {ImprovementId} for execution {ExecutionId}", improvementId, executionId);
                return Task.FromResult(false);
            }
        }

        public Task<bool> RemoveTagFromImprovementAsync(string executionId, string improvementId, string tag)
        {
            try
            {
                if (_improvementTags.ContainsKey(executionId) &&
                    _improvementTags[executionId].ContainsKey(improvementId) &&
                    _improvementTags[executionId][improvementId].Contains(tag))
                {
                    _improvementTags[executionId][improvementId].Remove(tag);
                    return Task.FromResult(true);
                }

                return Task.FromResult(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error removing tag from improvement {ImprovementId} for execution {ExecutionId}", improvementId, executionId);
                return Task.FromResult(false);
            }
        }
    }
}
