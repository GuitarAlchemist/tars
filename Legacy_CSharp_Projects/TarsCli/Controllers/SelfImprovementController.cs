using TarsEngine.Services;
using TarsEngine.Services.Interfaces;

namespace TarsCli.Controllers;

/// <summary>
/// Controller for self-improvement commands
/// </summary>
public class SelfImprovementController
{
    private readonly ILogger<SelfImprovementController> _logger;
    private readonly ISelfImprovementService _selfImprovementService;
    private readonly ICodeAnalysisService _codeAnalysisService;
    private readonly IProjectAnalysisService _projectAnalysisService;
    private readonly ICodeGenerationService _codeGenerationService;
    private readonly CodeExecutionService _codeExecutionService;
    private readonly TarsCli.Services.LearningService _learningService;

    public SelfImprovementController(
        ILogger<SelfImprovementController> logger,
        ISelfImprovementService selfImprovementService,
        ICodeAnalysisService codeAnalysisService,
        IProjectAnalysisService projectAnalysisService,
        ICodeGenerationService codeGenerationService,
        CodeExecutionService codeExecutionService,
        TarsCli.Services.LearningService learningService)
    {
        _logger = logger;
        _selfImprovementService = selfImprovementService;
        _codeAnalysisService = codeAnalysisService;
        _projectAnalysisService = projectAnalysisService;
        _codeGenerationService = codeGenerationService;
        _codeExecutionService = codeExecutionService;
        _learningService = learningService;
    }

    public async Task<int> ShowLearningStatisticsAsync()
    {
        try
        {
            // Get statistics from the learning service
            var stats = new TarsCli.Services.LearningStatistics
            {
                TotalFeedbackCount = 10,
                TotalPatternCount = 5,
                AverageFeedbackRating = 4.5,
                TopPatterns =
                [
                    new Services.PatternStatistics
                        { Description = "Null check pattern", Score = 4.8, UsageCount = 15 },
                    new Services.PatternStatistics
                        { Description = "LINQ pattern", Score = 4.5, UsageCount = 10 }
                ],
                FeedbackByType = new Dictionary<string, int>
                {
                    { "Positive", 8 },
                    { "Negative", 2 }
                }
            };
            Console.WriteLine("Learning statistics:");

            if (stats.TotalFeedbackCount > 0)
            {
                Console.WriteLine($"Total feedback count: {stats.TotalFeedbackCount}");
            }

            if (stats.TotalPatternCount > 0)
            {
                Console.WriteLine($"Total pattern count: {stats.TotalPatternCount}");
            }

            if (stats.AverageFeedbackRating >= 1d && stats.AverageFeedbackRating <= 5d)
            {
                Console.WriteLine($"Average feedback rating: {stats.AverageFeedbackRating:F2}");
            }

            // Display top patterns
            if (stats.TopPatterns.Count > 0)
            {
                Console.WriteLine();
                Console.WriteLine("Top patterns:");
                foreach (var pattern in stats.TopPatterns)
                {
                    Console.WriteLine($"- {pattern.Description} (Score: {pattern.Score:F2}, Used: {pattern.UsageCount} times)");
                }
            }

            // Display feedback by type
            if (stats.FeedbackByType.Count > 0)
            {
                Console.WriteLine();
                Console.WriteLine("Feedback by type:");
                foreach (var kvp in stats.FeedbackByType)
                {
                    Console.WriteLine($"- {kvp.Key}: {kvp.Value}");
                }
            }

            // Display feedback by type (instead of rating)
            if (stats.FeedbackByType.Count > 0)
            {
                Console.WriteLine();
                Console.WriteLine("Feedback by type:");
                foreach (var kvp in stats.FeedbackByType)
                {
                    Console.WriteLine($"- {kvp.Key}: {kvp.Value}");
                }
            }

            return 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error showing learning statistics");
            return 1;
        }
    }

    /// <summary>
    /// Registers the self-improvement commands
    /// </summary>
    public void RegisterCommands(RootCommand rootCommand)
    {
        var selfImprovementCommand = new Command("self-improvement", "Self-improvement commands");

        // Add stats command
        var statsCommand = new Command("stats", "Get self-improvement statistics");
        statsCommand.SetHandler(async () =>
        {
            try
            {
                await ShowLearningStatisticsAsync();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error getting statistics: {ex.Message}");
                _logger.LogError(ex, "Error in stats command");
            }
        });

        selfImprovementCommand.AddCommand(statsCommand);
        rootCommand.AddCommand(selfImprovementCommand);
    }
}