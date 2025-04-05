using Microsoft.Extensions.Configuration;
using System.Text.Json;
using System.Text.Json.Serialization;

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

                // Clean up the JSON content to handle potential issues
                jsonContent = CleanupJsonContent(jsonContent);

                // Try to deserialize the content
                LearningPlanContent? planContent = null;
                try
                {
                    planContent = JsonSerializer.Deserialize<LearningPlanContent>(jsonContent, JsonSerializerConfig.AiResponseOptions);
                }
                catch (JsonException ex)
                {
                    _logger.LogWarning(ex, "Error deserializing learning plan content, attempting fallback parsing");
                    planContent = FallbackParseLearningPlanContent(jsonContent);
                }

                if (planContent == null)
                {
                    _logger.LogError("Failed to parse learning plan content");
                    throw new FormatException("The AI response could not be parsed into a learning plan");
                }

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
        catch (Exception ex) when (ex is not FormatException)
        {
            _logger.LogError(ex, "Unexpected error parsing learning plan");
            throw new FormatException("An unexpected error occurred while parsing the learning plan", ex);
        }
    }

    /// <summary>
    /// Cleans up JSON content to handle common issues
    /// </summary>
    private string CleanupJsonContent(string jsonContent)
    {
        // Replace any invalid escape sequences
        jsonContent = jsonContent.Replace("\\n", "\n");

        // Remove any trailing commas before closing brackets or braces
        jsonContent = System.Text.RegularExpressions.Regex.Replace(jsonContent, @",\s*([\]\}])", "$1");

        return jsonContent;
    }

    /// <summary>
    /// Fallback method to parse learning plan content when standard deserialization fails
    /// </summary>
    private LearningPlanContent? FallbackParseLearningPlanContent(string jsonContent)
    {
        try
        {
            // Try to parse as a dynamic object first
            using var doc = JsonDocument.Parse(jsonContent);
            var root = doc.RootElement;

            var content = new LearningPlanContent
            {
                Introduction = GetStringProperty(root, "introduction") ?? "No introduction provided."
            };

            // Try to parse prerequisites
            if (root.TryGetProperty("prerequisites", out var prereqsElement) && prereqsElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var prereq in prereqsElement.EnumerateArray())
                {
                    if (prereq.ValueKind == JsonValueKind.String)
                    {
                        content.Prerequisites.Add(prereq.GetString() ?? "");
                    }
                }
            }

            // Return the partially parsed content
            return content;
        }
        catch
        {
            // If all parsing attempts fail, return null
            return null;
        }
    }

    /// <summary>
    /// Helper method to safely get a string property from a JsonElement
    /// </summary>
    private string? GetStringProperty(JsonElement element, string propertyName)
    {
        if (element.TryGetProperty(propertyName, out var property) && property.ValueKind == JsonValueKind.String)
        {
            return property.GetString();
        }
        return null;
    }

    /// <summary>
    /// Saves a learning plan to disk
    /// </summary>
    public async Task SaveLearningPlan(LearningPlan plan)
    {
        var filePath = Path.Combine(_learningPlansDirectory, $"{plan.Id}.json");

        var json = JsonSerializer.Serialize(plan, JsonSerializerConfig.DefaultOptions);
        await File.WriteAllTextAsync(filePath, json);

        _logger.LogInformation($"Learning plan saved: {Path.GetFullPath(filePath)}");
    }

    /// <summary>
    /// Gets a learning plan by ID
    /// </summary>
    public async Task<LearningPlan> GetLearningPlan(string id)
    {
        _logger.LogInformation($"Loading learning plan: {id}");

        var filePath = Path.Combine(_learningPlansDirectory, $"{id}.json");
        _logger.LogInformation($"Learning plan file path: {Path.GetFullPath(filePath)}");

        if (!File.Exists(filePath))
        {
            _logger.LogError($"Learning plan not found: {id}");
            throw new FileNotFoundException($"Learning plan not found: {id}");
        }

        var json = await File.ReadAllTextAsync(filePath);
        return JsonSerializer.Deserialize<LearningPlan>(json, JsonSerializerConfig.DefaultOptions);
    }

    /// <summary>
    /// Gets all learning plans
    /// </summary>
    public async Task<List<LearningPlan>> GetLearningPlans()
    {
        _logger.LogInformation($"Loading all learning plans from directory: {Path.GetFullPath(_learningPlansDirectory)}");

        var plans = new List<LearningPlan>();

        foreach (var file in Directory.GetFiles(_learningPlansDirectory, "*.json"))
        {
            try
            {
                _logger.LogInformation($"Loading learning plan from file: {Path.GetFullPath(file)}");
                var json = await File.ReadAllTextAsync(file);
                var plan = JsonSerializer.Deserialize<LearningPlan>(json, JsonSerializerConfig.DefaultOptions);
                plans.Add(plan);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error loading learning plan: {Path.GetFullPath(file)}");
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
[JsonConverter(typeof(JsonStringEnumConverter))]
public class LearningPlan
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = string.Empty;
    public string Topic { get; set; } = string.Empty;
    public SkillLevel SkillLevel { get; set; } = SkillLevel.Beginner;
    public DateTime CreatedDate { get; set; } = DateTime.UtcNow;
    public DateTime LastModifiedDate { get; set; } = DateTime.UtcNow;
    public LearningPlanContent Content { get; set; } = new LearningPlanContent();
}

/// <summary>
/// Represents the content of a learning plan
/// </summary>
public class LearningPlanContent
{
    public string Introduction { get; set; } = string.Empty;
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
    public string Title { get; set; } = string.Empty;
    public List<string> Objectives { get; set; } = new List<string>();
    public int EstimatedHours { get; set; }
    public List<Resource> Resources { get; set; } = new List<Resource>();
    public string Assessment { get; set; } = string.Empty;
}

/// <summary>
/// Represents a resource in a module
/// </summary>
public class Resource
{
    public string Type { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public string Url { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
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
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string CompletionCriteria { get; set; } = string.Empty;
}

/// <summary>
/// Represents a practice project in a learning plan
/// </summary>
public class PracticeProject
{
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string Difficulty { get; set; } = string.Empty;
    public int EstimatedHours { get; set; }
}