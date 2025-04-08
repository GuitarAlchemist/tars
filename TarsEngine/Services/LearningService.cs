using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Services;

/// <summary>
/// Service for learning from feedback and improving over time
/// </summary>
public class LearningService
{
    private readonly ILogger<LearningService> _logger;
    private readonly string _learningDataPath;
    private LearningData _learningData;

    public LearningService(ILogger<LearningService> logger)
    {
        _logger = logger;

        // Set the path for learning data
        string appDataPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "TARS");
        _learningDataPath = Path.Combine(appDataPath, "learning_data.json");

        // Ensure the directory exists
        Directory.CreateDirectory(appDataPath);

        // Load learning data
        LoadLearningData();
    }

    /// <summary>
    /// Records feedback on a code generation or improvement
    /// </summary>
    /// <param name="feedback">The feedback to record</param>
    /// <returns>True if the feedback was recorded successfully</returns>
    public async Task<bool> RecordFeedbackAsync(CodeFeedback feedback)
    {
        try
        {
            _logger.LogInformation($"Recording feedback for {feedback.Type} with rating {feedback.Rating}");

            // Add the feedback to the learning data
            _learningData.Feedback.Add(feedback);

            // Update pattern scores based on the feedback
            UpdatePatternScores(feedback);

            // Save the learning data
            await SaveLearningDataAsync();

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error recording feedback: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Gets the top patterns for a specific context
    /// </summary>
    /// <param name="context">The context to get patterns for</param>
    /// <param name="count">The number of patterns to return</param>
    /// <returns>A list of the top patterns</returns>
    public List<LearningCodePattern> GetTopPatternsForContext(string context, int count = 5)
    {
        try
        {
            _logger.LogInformation($"Getting top {count} patterns for context: {context}");

            // Filter patterns by context
            var matchingPatterns = _learningData.Patterns
                .Where(p => p.Context.Equals(context, StringComparison.OrdinalIgnoreCase))
                .OrderByDescending(p => p.Score)
                .Take(count)
                .ToList();

            return matchingPatterns;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting top patterns: {ex.Message}");
            return [];
        }
    }

    /// <summary>
    /// Adds a new pattern to the learning data
    /// </summary>
    /// <param name="pattern">The pattern to add</param>
    /// <returns>True if the pattern was added successfully</returns>
    public async Task<bool> AddPatternAsync(LearningCodePattern pattern)
    {
        try
        {
            _logger.LogInformation($"Adding new pattern for context: {pattern.Context}");

            // Check if the pattern already exists
            var existingPattern = _learningData.Patterns
                .FirstOrDefault(p => p.Context.Equals(pattern.Context, StringComparison.OrdinalIgnoreCase) &&
                                     p.Pattern.Equals(pattern.Pattern, StringComparison.OrdinalIgnoreCase));

            if (existingPattern != null)
            {
                // Update the existing pattern
                existingPattern.Score += 1;
                existingPattern.UsageCount += 1;
                existingPattern.LastUsed = DateTime.Now;
            }
            else
            {
                // Add the new pattern
                pattern.Id = Guid.NewGuid().ToString();
                pattern.CreatedAt = DateTime.Now;
                pattern.LastUsed = DateTime.Now;
                pattern.UsageCount = 1;

                _learningData.Patterns.Add(pattern);
            }

            // Save the learning data
            await SaveLearningDataAsync();

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error adding pattern: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Gets learning statistics
    /// </summary>
    /// <returns>Learning statistics</returns>
    public LearningStatistics GetStatistics()
    {
        try
        {
            _logger.LogInformation("Getting learning statistics");

            var stats = new LearningStatistics
            {
                TotalFeedbackCount = _learningData.Feedback.Count,
                TotalPatternCount = _learningData.Patterns.Count,
                AverageFeedbackRating = _learningData.Feedback.Any() ? _learningData.Feedback.Average(f => f.Rating) : 0,
                TopPatterns = _learningData.Patterns
                    .OrderByDescending(p => p.Score)
                    .Take(5)
                    .ToList(),
                FeedbackByType = _learningData.Feedback
                    .GroupBy(f => f.Type)
                    .ToDictionary(g => g.Key, g => g.Count()),
                FeedbackByRating = _learningData.Feedback
                    .GroupBy(f => f.Rating)
                    .ToDictionary(g => g.Key, g => g.Count())
            };

            return stats;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting statistics: {ex.Message}");
            return new LearningStatistics();
        }
    }

    /// <summary>
    /// Analyzes feedback to extract patterns
    /// </summary>
    /// <returns>True if patterns were extracted successfully</returns>
    public async Task<bool> AnalyzeFeedbackForPatternsAsync()
    {
        try
        {
            _logger.LogInformation("Analyzing feedback for patterns");

            // Get positive feedback (rating >= 4)
            var positiveFeedback = _learningData.Feedback
                .Where(f => f.Rating >= 4)
                .ToList();

            // Group feedback by context
            var feedbackByContext = positiveFeedback
                .GroupBy(f => f.Context)
                .ToDictionary(g => g.Key, g => g.ToList());

            // Extract patterns from each context group
            foreach (var contextGroup in feedbackByContext)
            {
                string context = contextGroup.Key;
                var feedbackItems = contextGroup.Value;

                // Extract common patterns from the code
                var patterns = ExtractPatternsFromCode(feedbackItems);

                // Add each pattern to the learning data
                foreach (var pattern in patterns)
                {
                    pattern.Context = context;
                    await AddPatternAsync(pattern);
                }
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error analyzing feedback for patterns: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Loads learning data from disk
    /// </summary>
    private void LoadLearningData()
    {
        try
        {
            if (File.Exists(_learningDataPath))
            {
                string json = File.ReadAllText(_learningDataPath);
                _learningData = JsonSerializer.Deserialize<LearningData>(json);
            }
            else
            {
                _learningData = new LearningData
                {
                    Feedback = [],
                    Patterns = []
                };
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error loading learning data: {ex.Message}");

            // Create a new learning data object if loading fails
            _learningData = new LearningData
            {
                Feedback = [],
                Patterns = []
            };
        }
    }

    /// <summary>
    /// Saves learning data to disk
    /// </summary>
    private async Task SaveLearningDataAsync()
    {
        try
        {
            string json = JsonSerializer.Serialize(_learningData, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            await File.WriteAllTextAsync(_learningDataPath, json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error saving learning data: {ex.Message}");
        }
    }

    /// <summary>
    /// Updates pattern scores based on feedback
    /// </summary>
    private void UpdatePatternScores(CodeFeedback feedback)
    {
        try
        {
            // Find patterns that match the context
            var matchingPatterns = _learningData.Patterns
                .Where(p => p.Context.Equals(feedback.Context, StringComparison.OrdinalIgnoreCase))
                .ToList();

            foreach (var pattern in matchingPatterns)
            {
                // Check if the pattern is present in the code
                if (feedback.Code.Contains(pattern.Pattern))
                {
                    // Update the pattern score based on the feedback rating
                    double scoreAdjustment = (feedback.Rating - 3) / 2.0; // Range: -1.0 to +1.0
                    pattern.Score += scoreAdjustment;
                    pattern.UsageCount += 1;
                    pattern.LastUsed = DateTime.Now;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error updating pattern scores: {ex.Message}");
        }
    }

    /// <summary>
    /// Extracts patterns from code feedback
    /// </summary>
    private List<LearningCodePattern> ExtractPatternsFromCode(List<CodeFeedback> feedbackItems)
    {
        var patterns = new List<LearningCodePattern>();

        try
        {
            // This is a simplified implementation
            // In a real implementation, we would use more sophisticated pattern extraction techniques

            // Extract common code blocks
            var codeBlocks = new List<string>();
            foreach (var feedback in feedbackItems)
            {
                // Split the code into lines
                var lines = feedback.Code.Split(["\r\n", "\r", "\n"], StringSplitOptions.None);

                // Extract blocks of 3-5 lines
                for (int i = 0; i < lines.Length - 2; i++)
                {
                    for (int blockSize = 3; blockSize <= Math.Min(5, lines.Length - i); blockSize++)
                    {
                        var block = string.Join(Environment.NewLine, lines.Skip(i).Take(blockSize));
                        codeBlocks.Add(block);
                    }
                }
            }

            // Count occurrences of each block
            var blockCounts = codeBlocks
                .GroupBy(b => b)
                .Select(g => new { Block = g.Key, Count = g.Count() })
                .Where(b => b.Count > 1) // Only consider blocks that appear multiple times
                .OrderByDescending(b => b.Count)
                .Take(10) // Take the top 10 blocks
                .ToList();

            // Create patterns from the blocks
            foreach (var block in blockCounts)
            {
                patterns.Add(new LearningCodePattern
                {
                    Pattern = block.Block,
                    Description = $"Common code pattern (found in {block.Count} feedback items)",
                    Score = block.Count / (double)feedbackItems.Count, // Score based on frequency
                    UsageCount = block.Count
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error extracting patterns from code: {ex.Message}");
        }

        return patterns;
    }
}

/// <summary>
/// Represents learning data
/// </summary>
public class LearningData
{
    public List<CodeFeedback> Feedback { get; set; } = [];
    public List<LearningCodePattern> Patterns { get; set; } = [];
}

/// <summary>
/// Represents feedback on code generation or improvement
/// </summary>
public class CodeFeedback
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public DateTime Timestamp { get; set; } = DateTime.Now;
    public string Type { get; set; } // "Generation", "Improvement", "Test", etc.
    public string Context { get; set; } // Language, project type, etc.
    public string Code { get; set; } // The code that was generated or improved
    public int Rating { get; set; } // 1-5 rating
    public string Comment { get; set; } // Optional comment
}

/// <summary>
/// Represents a code pattern
/// </summary>
public class LearningCodePattern
{
    public string Id { get; set; }
    public string Context { get; set; } // Language, project type, etc.
    public string Pattern { get; set; } // The code pattern
    public string Description { get; set; } // Description of the pattern
    public double Score { get; set; } // Score for the pattern (higher is better)
    public int UsageCount { get; set; } // Number of times the pattern has been used
    public DateTime CreatedAt { get; set; }
    public DateTime LastUsed { get; set; }
}

/// <summary>
/// Represents learning statistics
/// </summary>
public class LearningStatistics
{
    public int TotalFeedbackCount { get; set; }
    public int TotalPatternCount { get; set; }
    public double AverageFeedbackRating { get; set; }
    public List<LearningCodePattern> TopPatterns { get; set; } = [];
    public Dictionary<string, int> FeedbackByType { get; set; } = new Dictionary<string, int>();
    public Dictionary<int, int> FeedbackByRating { get; set; } = new Dictionary<int, int>();
}