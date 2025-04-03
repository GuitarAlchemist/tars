using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for learning from transformations
    /// </summary>
    public class LearningService
    {
        private readonly ILogger<LearningService> _logger;

        public LearningService(ILogger<LearningService> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Gets learning statistics
        /// </summary>
        public Task<LearningStatistics> GetStatisticsAsync()
        {
            _logger.LogInformation("Getting learning statistics");

            // Create sample statistics
            var stats = new LearningStatistics
            {
                TotalFeedbackCount = 10,
                TotalPatternCount = 5,
                AverageFeedbackRating = 4.5,
                TopPatterns = new List<PatternStatistics>
                {
                    new PatternStatistics { Description = "Null check pattern", Score = 4.8, UsageCount = 15 },
                    new PatternStatistics { Description = "LINQ pattern", Score = 4.5, UsageCount = 10 }
                },
                FeedbackByType = new Dictionary<string, int>
                {
                    { "Positive", 8 },
                    { "Negative", 2 }
                }
            };

            return Task.FromResult(stats);
        }
    }

    /// <summary>
    /// Learning statistics
    /// </summary>
    public class LearningStatistics
    {
        public int TotalFeedbackCount { get; set; }
        public int TotalPatternCount { get; set; }
        public double AverageFeedbackRating { get; set; }
        public List<PatternStatistics> TopPatterns { get; set; } = new List<PatternStatistics>();
        public Dictionary<string, int> FeedbackByType { get; set; } = new Dictionary<string, int>();
    }

    /// <summary>
    /// Pattern statistics
    /// </summary>
    public class PatternStatistics
    {
        public string Description { get; set; }
        public double Score { get; set; }
        public int UsageCount { get; set; }
    }
}
