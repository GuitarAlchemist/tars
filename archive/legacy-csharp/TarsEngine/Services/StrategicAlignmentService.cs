using System.Text.Json;
using Microsoft.Extensions.Logging;
using TarsEngine.Models;

namespace TarsEngine.Services;

/// <summary>
/// Service for managing strategic goals and alignment
/// </summary>
public class StrategicAlignmentService
{
    private readonly ILogger _logger;
    private readonly string _goalsDirectory;
    private readonly Dictionary<string, StrategicGoal> _goals = new();
    private bool _isInitialized = false;

    /// <summary>
    /// Initializes a new instance of the <see cref="StrategicAlignmentService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="goalsDirectory">The directory for storing strategic goals</param>
    public StrategicAlignmentService(ILogger logger, string goalsDirectory)
    {
        _logger = logger;
        _goalsDirectory = goalsDirectory;
    }

    /// <summary>
    /// Initializes the strategic alignment service
    /// </summary>
    /// <returns>A task representing the asynchronous operation</returns>
    public async Task InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing strategic alignment service from directory: {GoalsDirectory}", _goalsDirectory);

            if (!Directory.Exists(_goalsDirectory))
            {
                _logger.LogWarning("Goals directory not found: {GoalsDirectory}", _goalsDirectory);
                Directory.CreateDirectory(_goalsDirectory);
            }

            // Clear existing goals
            _goals.Clear();

            // Load goals from files
            var goalFiles = Directory.GetFiles(_goalsDirectory, "*.json", SearchOption.AllDirectories);
            _logger.LogInformation("Found {GoalFileCount} goal files", goalFiles.Length);

            foreach (var file in goalFiles)
            {
                try
                {
                    var json = await File.ReadAllTextAsync(file);
                    var goal = JsonSerializer.Deserialize<StrategicGoal>(json, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });

                    if (goal != null)
                    {
                        _goals[goal.Id] = goal;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error loading goal file: {FilePath}", file);
                }
            }

            // Create default goals if none exist
            if (_goals.Count == 0)
            {
                await CreateDefaultGoalsAsync();
            }

            _logger.LogInformation("Initialized strategic alignment service with {GoalCount} goals", _goals.Count);
            _isInitialized = true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing strategic alignment service");
        }
    }

    /// <summary>
    /// Gets all strategic goals
    /// </summary>
    /// <param name="options">Optional filter options</param>
    /// <returns>The list of strategic goals</returns>
    public async Task<List<StrategicGoal>> GetGoalsAsync(Dictionary<string, string>? options = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting strategic goals");

            // Apply filters
            var goals = _goals.Values.ToList();

            if (options != null)
            {
                // Filter by category
                if (options.TryGetValue("Category", out var category) && !string.IsNullOrEmpty(category))
                {
                    goals = goals.Where(g => g.Category.Equals(category, StringComparison.OrdinalIgnoreCase)).ToList();
                }

                // Filter by active status
                if (options.TryGetValue("IsActive", out var isActiveStr) && bool.TryParse(isActiveStr, out var isActive))
                {
                    goals = goals.Where(g => g.IsActive == isActive).ToList();
                }

                // Filter by tag
                if (options.TryGetValue("Tag", out var tag) && !string.IsNullOrEmpty(tag))
                {
                    goals = goals.Where(g => g.Tags.Contains(tag, StringComparer.OrdinalIgnoreCase)).ToList();
                }

                // Filter by keyword
                if (options.TryGetValue("Keyword", out var keyword) && !string.IsNullOrEmpty(keyword))
                {
                    goals = goals.Where(g => g.Keywords.Contains(keyword, StringComparer.OrdinalIgnoreCase)).ToList();
                }

                // Filter by minimum weight
                if (options.TryGetValue("MinWeight", out var minWeightStr) && double.TryParse(minWeightStr, out var minWeight))
                {
                    goals = goals.Where(g => g.Weight >= minWeight).ToList();
                }

                // Sort by weight
                if (options.TryGetValue("SortBy", out var sortBy) && !string.IsNullOrEmpty(sortBy))
                {
                    switch (sortBy.ToLowerInvariant())
                    {
                        case "weight":
                            goals = goals.OrderByDescending(g => g.Weight).ToList();
                            break;
                        case "name":
                            goals = goals.OrderBy(g => g.Name).ToList();
                            break;
                        case "category":
                            goals = goals.OrderBy(g => g.Category).ToList();
                            break;
                        case "date":
                            goals = goals.OrderByDescending(g => g.CreatedAt).ToList();
                            break;
                    }
                }

                // Limit results
                if (options.TryGetValue("Limit", out var limitStr) && int.TryParse(limitStr, out var limit) && limit > 0)
                {
                    goals = goals.Take(limit).ToList();
                }
            }

            _logger.LogInformation("Got {GoalCount} strategic goals", goals.Count);
            return goals;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting strategic goals");
            return [];
        }
    }

    /// <summary>
    /// Gets a strategic goal by ID
    /// </summary>
    /// <param name="goalId">The goal ID</param>
    /// <returns>The strategic goal, or null if not found</returns>
    public async Task<StrategicGoal?> GetGoalAsync(string goalId)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Getting strategic goal by ID: {GoalId}", goalId);

            if (_goals.TryGetValue(goalId, out var goal))
            {
                return goal;
            }

            _logger.LogWarning("Strategic goal not found: {GoalId}", goalId);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting strategic goal: {GoalId}", goalId);
            return null;
        }
    }

    /// <summary>
    /// Adds a strategic goal
    /// </summary>
    /// <param name="goal">The goal to add</param>
    /// <returns>True if the goal was added successfully, false otherwise</returns>
    public async Task<bool> AddGoalAsync(StrategicGoal goal)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Adding strategic goal: {GoalName} ({GoalId})", goal.Name, goal.Id);

            if (_goals.ContainsKey(goal.Id))
            {
                _logger.LogWarning("Strategic goal already exists: {GoalId}", goal.Id);
                return false;
            }

            // Add goal to memory
            _goals[goal.Id] = goal;

            // Save goal to file
            await SaveGoalToFileAsync(goal);

            _logger.LogInformation("Added strategic goal: {GoalName} ({GoalId})", goal.Name, goal.Id);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding strategic goal: {GoalName} ({GoalId})", goal.Name, goal.Id);
            return false;
        }
    }

    /// <summary>
    /// Updates a strategic goal
    /// </summary>
    /// <param name="goal">The goal to update</param>
    /// <returns>True if the goal was updated successfully, false otherwise</returns>
    public async Task<bool> UpdateGoalAsync(StrategicGoal goal)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Updating strategic goal: {GoalName} ({GoalId})", goal.Name, goal.Id);

            if (!_goals.ContainsKey(goal.Id))
            {
                _logger.LogWarning("Strategic goal not found: {GoalId}", goal.Id);
                return false;
            }

            // Update goal in memory
            goal.UpdatedAt = DateTime.UtcNow;
            _goals[goal.Id] = goal;

            // Save goal to file
            await SaveGoalToFileAsync(goal);

            _logger.LogInformation("Updated strategic goal: {GoalName} ({GoalId})", goal.Name, goal.Id);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating strategic goal: {GoalName} ({GoalId})", goal.Name, goal.Id);
            return false;
        }
    }

    /// <summary>
    /// Removes a strategic goal
    /// </summary>
    /// <param name="goalId">The ID of the goal to remove</param>
    /// <returns>True if the goal was removed successfully, false otherwise</returns>
    public async Task<bool> RemoveGoalAsync(string goalId)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Removing strategic goal: {GoalId}", goalId);

            if (!_goals.TryGetValue(goalId, out var goal))
            {
                _logger.LogWarning("Strategic goal not found: {GoalId}", goalId);
                return false;
            }

            // Remove goal from memory
            _goals.Remove(goalId);

            // Remove goal file
            var filePath = GetGoalFilePath(goal);
            if (File.Exists(filePath))
            {
                File.Delete(filePath);
            }

            _logger.LogInformation("Removed strategic goal: {GoalId}", goalId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error removing strategic goal: {GoalId}", goalId);
            return false;
        }
    }

    /// <summary>
    /// Gets the available goal options
    /// </summary>
    /// <returns>The dictionary of available options and their descriptions</returns>
    public Dictionary<string, string> GetAvailableOptions()
    {
        return new Dictionary<string, string>
        {
            { "Category", "Filter goals by category" },
            { "IsActive", "Filter goals by active status (true/false)" },
            { "Tag", "Filter goals by tag" },
            { "Keyword", "Filter goals by keyword" },
            { "MinWeight", "Minimum weight for goals" },
            { "SortBy", "Sort goals by weight, name, category, or date" },
            { "Limit", "Limit the number of goals returned" }
        };
    }

    /// <summary>
    /// Calculates the alignment score for an improvement
    /// </summary>
    /// <param name="improvement">The improvement</param>
    /// <param name="options">Optional alignment options</param>
    /// <returns>The alignment score (0.0 to 1.0)</returns>
    public async Task<double> CalculateAlignmentScoreAsync(PrioritizedImprovement improvement, Dictionary<string, string>? options = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Calculating alignment score for improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);

            if (improvement.StrategicGoals.Count == 0)
            {
                return 0.0;
            }

            // Calculate alignment score based on strategic goals
            var alignmentScore = 0.0;
            var totalWeight = 0.0;

            foreach (var goalId in improvement.StrategicGoals)
            {
                if (_goals.TryGetValue(goalId, out var goal) && goal.IsActive)
                {
                    alignmentScore += goal.Weight;
                    totalWeight += 1.0;
                }
            }

            // Normalize alignment score
            if (totalWeight > 0)
            {
                alignmentScore /= totalWeight;
            }

            // Apply minimum alignment score
            if (options != null && options.TryGetValue("MinAlignmentScore", out var minScoreStr) && double.TryParse(minScoreStr, out var minScore))
            {
                alignmentScore = Math.Max(alignmentScore, minScore);
            }

            // Apply maximum alignment score
            if (options != null && options.TryGetValue("MaxAlignmentScore", out var maxScoreStr) && double.TryParse(maxScoreStr, out var maxScore))
            {
                alignmentScore = Math.Min(alignmentScore, maxScore);
            }

            _logger.LogInformation("Calculated alignment score for improvement: {ImprovementName} ({ImprovementId}), Score: {AlignmentScore}", improvement.Name, improvement.Id, alignmentScore);
            return alignmentScore;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating alignment score for improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return 0.0;
        }
    }

    /// <summary>
    /// Finds strategic goals for an improvement
    /// </summary>
    /// <param name="improvement">The improvement</param>
    /// <param name="options">Optional search options</param>
    /// <returns>The list of strategic goals</returns>
    public async Task<List<StrategicGoal>> FindGoalsForImprovementAsync(PrioritizedImprovement improvement, Dictionary<string, string>? options = null)
    {
        await EnsureInitializedAsync();

        try
        {
            _logger.LogInformation("Finding strategic goals for improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);

            var goals = new List<StrategicGoal>();
            var activeGoals = _goals.Values.Where(g => g.IsActive).ToList();

            // Find goals by category
            var categoryGoals = activeGoals.Where(g => g.Category.Equals(improvement.Category.ToString(), StringComparison.OrdinalIgnoreCase)).ToList();
            goals.AddRange(categoryGoals);

            // Find goals by tags
            foreach (var tag in improvement.Tags)
            {
                var tagGoals = activeGoals.Where(g => g.Tags.Contains(tag, StringComparer.OrdinalIgnoreCase)).ToList();
                foreach (var goal in tagGoals)
                {
                    if (!goals.Contains(goal))
                    {
                        goals.Add(goal);
                    }
                }
            }

            // Find goals by keywords
            var keywords = new List<string>();
            keywords.Add(improvement.Name);
            keywords.Add(improvement.Description);
            keywords.AddRange(improvement.Tags);

            foreach (var keyword in keywords)
            {
                var keywordGoals = activeGoals.Where(g => g.Keywords.Any(k => keyword.Contains(k, StringComparison.OrdinalIgnoreCase))).ToList();
                foreach (var goal in keywordGoals)
                {
                    if (!goals.Contains(goal))
                    {
                        goals.Add(goal);
                    }
                }
            }

            // Sort goals by weight
            goals = goals.OrderByDescending(g => g.Weight).ToList();

            // Limit results
            if (options != null && options.TryGetValue("Limit", out var limitStr) && int.TryParse(limitStr, out var limit) && limit > 0)
            {
                goals = goals.Take(limit).ToList();
            }

            _logger.LogInformation("Found {GoalCount} strategic goals for improvement: {ImprovementName} ({ImprovementId})", goals.Count, improvement.Name, improvement.Id);
            return goals;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding strategic goals for improvement: {ImprovementName} ({ImprovementId})", improvement.Name, improvement.Id);
            return [];
        }
    }

    private async Task CreateDefaultGoalsAsync()
    {
        try
        {
            _logger.LogInformation("Creating default strategic goals");

            var defaultGoals = new List<StrategicGoal>
            {
                new()
                {
                    Name = "Improve Code Quality",
                    Description = "Improve the overall quality of the codebase",
                    Category = "Quality",
                    Weight = 0.8,
                    Tags = ["quality", "maintainability", "reliability"],
                    Keywords = ["quality", "clean", "maintainable", "reliable", "testable"]
                },
                new()
                {
                    Name = "Enhance Performance",
                    Description = "Improve the performance of the application",
                    Category = "Performance",
                    Weight = 0.9,
                    Tags = ["performance", "optimization", "speed"],
                    Keywords = ["performance", "fast", "efficient", "optimize", "speed"]
                },
                new()
                {
                    Name = "Strengthen Security",
                    Description = "Enhance the security of the application",
                    Category = "Security",
                    Weight = 1.0,
                    Tags = ["security", "vulnerability", "protection"],
                    Keywords = ["security", "secure", "vulnerability", "protect", "encryption"]
                },
                new()
                {
                    Name = "Improve Architecture",
                    Description = "Enhance the architecture of the application",
                    Category = "Architecture",
                    Weight = 0.8,
                    Tags = ["architecture", "design", "structure"],
                    Keywords = ["architecture", "design", "structure", "pattern", "solid"]
                },
                new()
                {
                    Name = "Enhance User Experience",
                    Description = "Improve the user experience of the application",
                    Category = "UX",
                    Weight = 0.7,
                    Tags = ["ux", "usability", "user-friendly"],
                    Keywords = ["ux", "user", "experience", "usability", "friendly", "interface"]
                },
                new()
                {
                    Name = "Increase Test Coverage",
                    Description = "Improve the test coverage of the codebase",
                    Category = "Testing",
                    Weight = 0.8,
                    Tags = ["testing", "coverage", "quality"],
                    Keywords = ["test", "coverage", "unit", "integration", "automated"]
                },
                new()
                {
                    Name = "Improve Documentation",
                    Description = "Enhance the documentation of the codebase",
                    Category = "Documentation",
                    Weight = 0.6,
                    Tags = ["documentation", "comments", "readme"],
                    Keywords = ["documentation", "comment", "readme", "wiki", "guide"]
                },
                new()
                {
                    Name = "Enhance Scalability",
                    Description = "Improve the scalability of the application",
                    Category = "Scalability",
                    Weight = 0.8,
                    Tags = ["scalability", "performance", "architecture"],
                    Keywords = ["scale", "scalable", "load", "throughput", "capacity"]
                },
                new()
                {
                    Name = "Reduce Technical Debt",
                    Description = "Reduce the technical debt in the codebase",
                    Category = "Maintenance",
                    Weight = 0.7,
                    Tags = ["technical-debt", "maintenance", "quality"],
                    Keywords = ["debt", "technical", "maintenance", "legacy", "refactor"]
                },
                new()
                {
                    Name = "Improve Accessibility",
                    Description = "Enhance the accessibility of the application",
                    Category = "Accessibility",
                    Weight = 0.6,
                    Tags = ["accessibility", "a11y", "usability"],
                    Keywords = ["accessibility", "a11y", "wcag", "aria", "inclusive"]
                }
            };

            foreach (var goal in defaultGoals)
            {
                await AddGoalAsync(goal);
            }

            _logger.LogInformation("Created {GoalCount} default strategic goals", defaultGoals.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating default strategic goals");
        }
    }

    private async Task SaveGoalToFileAsync(StrategicGoal goal)
    {
        var filePath = GetGoalFilePath(goal);
        var directory = Path.GetDirectoryName(filePath);
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory!);
        }

        var json = JsonSerializer.Serialize(goal, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        await File.WriteAllTextAsync(filePath, json);
    }

    private string GetGoalFilePath(StrategicGoal goal)
    {
        var category = string.IsNullOrEmpty(goal.Category) ? "Other" : goal.Category;
        var fileName = $"{goal.Id}.json";

        return Path.Combine(_goalsDirectory, category, fileName);
    }

    private async Task EnsureInitializedAsync()
    {
        if (!_isInitialized)
        {
            await InitializeAsync();
        }
    }
}
