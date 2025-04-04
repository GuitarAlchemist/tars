using Microsoft.Extensions.Configuration;
using System.Text.Json;

namespace TarsCli.Services;

/// <summary>
/// Service for generating and managing personalized learning plans
/// </summary>
public class LearningPlanService
{
    private readonly ILogger<LearningPlanService> _logger;
    private readonly IConfiguration _configuration;
    private readonly OllamaService _ollamaService;
    private readonly string _learningPlansDirectory;

    public LearningPlanService(
        ILogger<LearningPlanService> logger,
        IConfiguration configuration,
        OllamaService ollamaService)
    {
        _logger = logger;
        _configuration = configuration;
        _ollamaService = ollamaService;

        // Set up the learning plans directory
        var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        if (string.IsNullOrEmpty(appDataPath))
        {
            appDataPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars");
        }
        else
        {
            appDataPath = Path.Combine(appDataPath, "TARS");
        }

        _learningPlansDirectory = Path.Combine(appDataPath, "LearningPlans");

        if (!Directory.Exists(_learningPlansDirectory))
        {
            Directory.CreateDirectory(_learningPlansDirectory);
        }
    }

    /// <summary>
    /// Generates a personalized learning plan based on user preferences, skill level, and goals
    /// </summary>
    public async Task<LearningPlan> GenerateLearningPlan(
        string name,
        string topic,
        SkillLevel skillLevel,
        List<string> goals,
        List<string> preferences,
        int estimatedHours,
        string model = "")
    {
        _logger.LogInformation($"Generating learning plan for topic: {topic}, skill level: {skillLevel}");

        try
        {
            // Use default model if not specified
            if (string.IsNullOrEmpty(model))
            {
                model = _configuration["Ollama:DefaultModel"] ?? "llama3";
            }

            // Create a prompt for the AI model
            var prompt = CreateLearningPlanPrompt(topic, skillLevel, goals, preferences, estimatedHours);

            // Generate the learning plan content using Ollama
            var response = await _ollamaService.GenerateCompletion(prompt, model);

            // Parse the response into a structured learning plan
            var learningPlan = ParseLearningPlanResponse(response, name, topic, skillLevel);

            // Save the learning plan
            await SaveLearningPlan(learningPlan);

            return learningPlan;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating learning plan for topic: {topic}");
            throw;
        }
    }

    /// <summary>
    /// Creates a prompt for generating a learning plan
    /// </summary>
    private string CreateLearningPlanPrompt(
        string topic,
        SkillLevel skillLevel,
        List<string> goals,
        List<string> preferences,
        int estimatedHours)
    {
        return $@"
You are an expert educational content creator and curriculum designer. Create a personalized learning plan with the following details:

TOPIC: {topic}
SKILL LEVEL: {skillLevel}
GOALS: {string.Join(", ", goals)}
PREFERENCES: {string.Join(", ", preferences)}
ESTIMATED HOURS: {estimatedHours}

The learning plan should include:
1. A brief introduction to the topic
2. A list of prerequisites (if any)
3. A structured sequence of modules, each containing:
   - Module title
   - Learning objectives
   - Estimated time to complete
   - Resources (books, articles, videos, exercises)
   - Assessment criteria
4. A timeline for completion
5. Milestones to track progress
6. Recommended practice exercises or projects

Format your response as a structured JSON object with the following schema:
{{
  ""introduction"": ""string"",
  ""prerequisites"": [""string""],
  ""modules"": [
    {{
      ""title"": ""string"",
      ""objectives"": [""string""],
      ""estimatedHours"": number,
      ""resources"": [
        {{
          ""type"": ""string"",
          ""title"": ""string"",
          ""url"": ""string"",
          ""description"": ""string""
        }}
      ],
      ""assessment"": ""string""
    }}
  ],
  ""timeline"": [
    {{
      ""week"": number,
      ""activities"": [""string""]
    }}
  ],
  ""milestones"": [
    {{
      ""title"": ""string"",
      ""description"": ""string"",
      ""completionCriteria"": ""string""
    }}
  ],
  ""practiceProjects"": [
    {{
      ""title"": ""string"",
      ""description"": ""string"",
      ""difficulty"": ""string"",
      ""estimatedHours"": number
    }}
  ]
}}

Ensure the learning plan is tailored to the specified skill level and preferences, and can be realistically completed within the estimated hours.
";
    }

    /// <summary>
    /// Parses the AI response into a structured learning plan
    /// </summary>
    private LearningPlan ParseLearningPlanResponse(string response, string name, string topic, SkillLevel skillLevel)
    {
        try
        {
            // Extract JSON from the response (in case the model includes additional text)
            var jsonStart = response.IndexOf('{');
            var jsonEnd = response.LastIndexOf('}');

            if (jsonStart >= 0 && jsonEnd >= 0 && jsonEnd > jsonStart)
            {
                var jsonContent = response.Substring(jsonStart, jsonEnd - jsonStart + 1);
                var planContent = JsonSerializer.Deserialize<LearningPlanContent>(jsonContent, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                // Create a new learning plan with metadata and content
                return new LearningPlan
                {
                    Id = Guid.NewGuid().ToString(),
                    Name = name,
                    Topic = topic,
                    SkillLevel = skillLevel,
                    CreatedDate = DateTime.UtcNow,
                    LastModifiedDate = DateTime.UtcNow,
                    Content = planContent
                };
            }
            else
            {
                _logger.LogError("Failed to extract JSON from the response");
                throw new FormatException("The AI response did not contain valid JSON");
            }
        }
        catch (JsonException ex)
        {
            _logger.LogError(ex, "Error parsing learning plan JSON");
            throw new FormatException("The AI response contained invalid JSON", ex);
        }
    }

    /// <summary>
    /// Saves a learning plan to disk
    /// </summary>
    public async Task SaveLearningPlan(LearningPlan plan)
    {
        var filePath = Path.Combine(_learningPlansDirectory, $"{plan.Id}.json");

        var options = new JsonSerializerOptions
        {
            WriteIndented = true
        };

        var json = JsonSerializer.Serialize(plan, options);
        await File.WriteAllTextAsync(filePath, json);

        _logger.LogInformation($"Learning plan saved: {filePath}");
    }

    /// <summary>
    /// Gets a learning plan by ID
    /// </summary>
    public async Task<LearningPlan> GetLearningPlan(string id)
    {
        var filePath = Path.Combine(_learningPlansDirectory, $"{id}.json");

        if (!File.Exists(filePath))
        {
            _logger.LogError($"Learning plan not found: {id}");
            throw new FileNotFoundException($"Learning plan not found: {id}");
        }

        var json = await File.ReadAllTextAsync(filePath);
        return JsonSerializer.Deserialize<LearningPlan>(json);
    }

    /// <summary>
    /// Gets all learning plans
    /// </summary>
    public async Task<List<LearningPlan>> GetLearningPlans()
    {
        var plans = new List<LearningPlan>();

        foreach (var file in Directory.GetFiles(_learningPlansDirectory, "*.json"))
        {
            try
            {
                var json = await File.ReadAllTextAsync(file);
                var plan = JsonSerializer.Deserialize<LearningPlan>(json);
                plans.Add(plan);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error loading learning plan: {file}");
            }
        }

        return plans.OrderByDescending(p => p.LastModifiedDate).ToList();
    }

    /// <summary>
    /// Updates an existing learning plan
    /// </summary>
    public async Task<LearningPlan> UpdateLearningPlan(string id, LearningPlan updatedPlan)
    {
        var existingPlan = await GetLearningPlan(id);

        // Update the plan properties
        existingPlan.Name = updatedPlan.Name;
        existingPlan.Topic = updatedPlan.Topic;
        existingPlan.SkillLevel = updatedPlan.SkillLevel;
        existingPlan.Content = updatedPlan.Content;
        existingPlan.LastModifiedDate = DateTime.UtcNow;

        // Save the updated plan
        await SaveLearningPlan(existingPlan);

        return existingPlan;
    }

    /// <summary>
    /// Deletes a learning plan
    /// </summary>
    public Task DeleteLearningPlan(string id)
    {
        var filePath = Path.Combine(_learningPlansDirectory, $"{id}.json");

        if (!File.Exists(filePath))
        {
            _logger.LogError($"Learning plan not found: {id}");
            throw new FileNotFoundException($"Learning plan not found: {id}");
        }

        File.Delete(filePath);
        _logger.LogInformation($"Learning plan deleted: {id}");

        return Task.CompletedTask;
    }
}

/// <summary>
/// Represents a user's skill level
/// </summary>
public enum SkillLevel
{
    Beginner,
    Intermediate,
    Advanced,
    Expert
}

/// <summary>
/// Represents a learning plan
/// </summary>
public class LearningPlan
{
    public required string Id { get; set; }
    public required string Name { get; set; }
    public required string Topic { get; set; }
    public SkillLevel SkillLevel { get; set; }
    public DateTime CreatedDate { get; set; }
    public DateTime LastModifiedDate { get; set; }
    public required LearningPlanContent Content { get; set; }
}

/// <summary>
/// Represents the content of a learning plan
/// </summary>
public class LearningPlanContent
{
    public required string Introduction { get; set; }
    public List<string> Prerequisites { get; set; } = new List<string>();
    public List<Module> Modules { get; set; } = new List<Module>();
    public List<TimelineItem> Timeline { get; set; } = new List<TimelineItem>();
    public List<Milestone> Milestones { get; set; } = new List<Milestone>();
    public List<PracticeProject> PracticeProjects { get; set; } = new List<PracticeProject>();
}

/// <summary>
/// Represents a module in a learning plan
/// </summary>
public class Module
{
    public required string Title { get; set; }
    public List<string> Objectives { get; set; } = new List<string>();
    public int EstimatedHours { get; set; }
    public List<Resource> Resources { get; set; } = new List<Resource>();
    public required string Assessment { get; set; }
}

/// <summary>
/// Represents a resource in a module
/// </summary>
public class Resource
{
    public required string Type { get; set; }
    public required string Title { get; set; }
    public required string Url { get; set; }
    public required string Description { get; set; }
}

/// <summary>
/// Represents a timeline item in a learning plan
/// </summary>
public class TimelineItem
{
    public int Week { get; set; }
    public List<string> Activities { get; set; } = new List<string>();
}

/// <summary>
/// Represents a milestone in a learning plan
/// </summary>
public class Milestone
{
    public required string Title { get; set; }
    public required string Description { get; set; }
    public required string CompletionCriteria { get; set; }
}

/// <summary>
/// Represents a practice project in a learning plan
/// </summary>
public class PracticeProject
{
    public required string Title { get; set; }
    public required string Description { get; set; }
    public required string Difficulty { get; set; }
    public int EstimatedHours { get; set; }
}