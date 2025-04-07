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

            // Create a template-based learning plan instead of parsing JSON
            var learningPlan = CreateTemplateLearningPlan(name, topic, skillLevel, goals, preferences, estimatedHours, response);

            // Save the learning plan
            await SaveLearningPlan(learningPlan);

            return learningPlan;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating learning plan for topic: {topic}");

            // Fallback to a default template if there's an error
            var defaultPlan = CreateDefaultLearningPlan(name, topic, skillLevel, goals, preferences, estimatedHours);
            await SaveLearningPlan(defaultPlan);
            return defaultPlan;
        }
    }

    /// <summary>
    /// Creates a template-based learning plan using AI-generated content
    /// </summary>
    private LearningPlan CreateTemplateLearningPlan(
        string name,
        string topic,
        SkillLevel skillLevel,
        List<string> goals,
        List<string> preferences,
        int estimatedHours,
        string aiResponse)
    {
        _logger.LogInformation($"Creating template learning plan for topic: {topic}");

        // Extract key information from the AI response
        var introduction = ExtractSection(aiResponse, "introduction", "prerequisites");
        var prerequisites = ExtractListItems(aiResponse, "prerequisites");
        var modules = ExtractListItems(aiResponse, "modules");
        var timeline = ExtractListItems(aiResponse, "timeline");

        // Create a learning plan with the extracted content
        var plan = new LearningPlan
        {
            Id = Guid.NewGuid().ToString(),
            Name = name,
            Topic = topic,
            SkillLevel = skillLevel,
            CreatedDate = DateTime.UtcNow,
            LastModifiedDate = DateTime.UtcNow,
            Content = new LearningPlanContent
            {
                Introduction = introduction.Length > 0 ? introduction : $"A comprehensive learning plan for {topic} designed for {skillLevel} skill level.",
                Prerequisites = prerequisites.Count > 0 ? prerequisites : new List<string> { "Basic understanding of programming concepts" },
                Modules = CreateModulesFromText(modules, estimatedHours),
                Timeline = CreateTimelineFromText(timeline),
                Milestones = CreateDefaultMilestones(topic),
                PracticeProjects = CreateDefaultProjects(topic)
            }
        };

        return plan;
    }

    /// <summary>
    /// Creates a default learning plan when AI generation fails
    /// </summary>
    private LearningPlan CreateDefaultLearningPlan(
        string name,
        string topic,
        SkillLevel skillLevel,
        List<string> goals,
        List<string> preferences,
        int estimatedHours)
    {
        _logger.LogInformation($"Creating default learning plan for topic: {topic}");

        // Create a basic learning plan with default content
        var plan = new LearningPlan
        {
            Id = Guid.NewGuid().ToString(),
            Name = name,
            Topic = topic,
            SkillLevel = skillLevel,
            CreatedDate = DateTime.UtcNow,
            LastModifiedDate = DateTime.UtcNow,
            Content = new LearningPlanContent
            {
                Introduction = $"A comprehensive learning plan for {topic} designed for {skillLevel} skill level.",
                Prerequisites = new List<string> { "Basic understanding of programming concepts" },
                Modules = new List<Module>
                {
                    new Module
                    {
                        Title = "Getting Started with " + topic,
                        Objectives = new List<string> { "Understand the basics", "Set up your development environment" },
                        EstimatedHours = estimatedHours / 4,
                        Resources = new List<Resource>
                        {
                            new Resource
                            {
                                Type = "Article",
                                Title = "Introduction to " + topic,
                                Url = "https://example.com/intro",
                                Description = "A beginner-friendly introduction to the topic."
                            }
                        },
                        Assessment = "Complete a simple project demonstrating basic concepts."
                    },
                    new Module
                    {
                        Title = "Advanced Concepts in " + topic,
                        Objectives = new List<string> { "Master advanced techniques", "Build a complex project" },
                        EstimatedHours = estimatedHours / 2,
                        Resources = new List<Resource>
                        {
                            new Resource
                            {
                                Type = "Video",
                                Title = "Advanced " + topic + " Techniques",
                                Url = "https://example.com/advanced",
                                Description = "In-depth coverage of advanced concepts."
                            }
                        },
                        Assessment = "Build a comprehensive project showcasing advanced skills."
                    }
                },
                Timeline = new List<TimelineItem>
                {
                    new TimelineItem
                    {
                        Week = "1-2",
                        Activities = new List<string> { "Complete Module 1", "Set up development environment" }
                    },
                    new TimelineItem
                    {
                        Week = "3-4",
                        Activities = new List<string> { "Complete Module 2", "Work on final project" }
                    }
                },
                Milestones = new List<Milestone>
                {
                    new Milestone
                    {
                        Title = "Basic Proficiency",
                        Description = "Demonstrate understanding of fundamental concepts",
                        CompletionCriteria = "Complete Module 1 and pass assessment"
                    },
                    new Milestone
                    {
                        Title = "Advanced Mastery",
                        Description = "Demonstrate ability to apply advanced concepts",
                        CompletionCriteria = "Complete Module 2 and final project"
                    }
                },
                PracticeProjects = new List<PracticeProject>
                {
                    new PracticeProject
                    {
                        Title = "Simple " + topic + " Application",
                        Description = "Build a basic application to practice fundamental concepts",
                        Difficulty = "Beginner",
                        EstimatedHours = estimatedHours / 5
                    },
                    new PracticeProject
                    {
                        Title = "Advanced " + topic + " Project",
                        Description = "Build a comprehensive project showcasing advanced skills",
                        Difficulty = "Advanced",
                        EstimatedHours = estimatedHours / 3
                    }
                }
            }
        };

        return plan;
    }

    /// <summary>
    /// Helper method to extract a section from the AI response
    /// </summary>
    private string ExtractSection(string text, string sectionName, string nextSectionName)
    {
        try
        {
            // Look for patterns like "Introduction:" or "## Introduction" or "**Introduction**"
            var patterns = new[]
            {
                $@"(?i){sectionName}:\s*([^#]*?)(?:\s*{nextSectionName}:|\s*##|\s*\*\*|$)",
                $@"(?i)##\s*{sectionName}\s*([^#]*?)(?:\s*##|$)",
                $@"(?i)\*\*{sectionName}\*\*\s*([^*]*?)(?:\s*\*\*|$)"
            };

            foreach (var pattern in patterns)
            {
                var match = System.Text.RegularExpressions.Regex.Match(text, pattern, System.Text.RegularExpressions.RegexOptions.Singleline);
                if (match.Success && match.Groups.Count > 1)
                {
                    return match.Groups[1].Value.Trim();
                }
            }

            return "";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error extracting section {sectionName}");
            return "";
        }
    }

    /// <summary>
    /// Helper method to extract list items from the AI response
    /// </summary>
    private List<string> ExtractListItems(string text, string sectionName)
    {
        var items = new List<string>();
        try
        {
            // Extract the section first
            var section = ExtractSection(text, sectionName, "");
            if (string.IsNullOrEmpty(section))
            {
                return items;
            }

            // Look for list items (numbered or bulleted)
            var listItemPattern = @"(?m)^\s*(?:\d+\.|-|\*|•)\s*(.+)$";
            var matches = System.Text.RegularExpressions.Regex.Matches(section, listItemPattern);

            foreach (System.Text.RegularExpressions.Match match in matches)
            {
                if (match.Groups.Count > 1)
                {
                    items.Add(match.Groups[1].Value.Trim());
                }
            }

            return items;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error extracting list items for section {sectionName}");
            return items;
        }
    }

    /// <summary>
    /// Helper method to create modules from extracted text
    /// </summary>
    private List<Module> CreateModulesFromText(List<string> moduleTexts, int totalEstimatedHours)
    {
        var modules = new List<Module>();
        try
        {
            // If no modules were extracted, return an empty list
            if (moduleTexts.Count == 0)
            {
                return modules;
            }

            // Calculate hours per module
            var hoursPerModule = totalEstimatedHours / Math.Max(1, moduleTexts.Count);

            // Create a module for each extracted text
            for (int i = 0; i < moduleTexts.Count; i++)
            {
                var moduleText = moduleTexts[i];
                modules.Add(new Module
                {
                    Title = moduleText,
                    Objectives = new List<string> { "Master the concepts in this module" },
                    EstimatedHours = hoursPerModule,
                    Resources = new List<Resource>
                    {
                        new Resource
                        {
                            Type = "Article",
                            Title = moduleText,
                            Url = "https://example.com/resource",
                            Description = "A resource for this module."
                        }
                    },
                    Assessment = "Complete the exercises for this module."
                });
            }

            return modules;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating modules from text");
            return modules;
        }
    }

    /// <summary>
    /// Helper method to create timeline from extracted text
    /// </summary>
    private List<TimelineItem> CreateTimelineFromText(List<string> timelineTexts)
    {
        var timeline = new List<TimelineItem>();
        try
        {
            // If no timeline items were extracted, return an empty list
            if (timelineTexts.Count == 0)
            {
                return timeline;
            }

            // Create a timeline item for each extracted text
            for (int i = 0; i < timelineTexts.Count; i++)
            {
                var weekNumber = i + 1;
                timeline.Add(new TimelineItem
                {
                    Week = weekNumber.ToString(),
                    Activities = new List<string> { timelineTexts[i] }
                });
            }

            return timeline;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating timeline from text");
            return timeline;
        }
    }

    /// <summary>
    /// Helper method to create default milestones
    /// </summary>
    private List<Milestone> CreateDefaultMilestones(string topic)
    {
        return new List<Milestone>
        {
            new Milestone
            {
                Title = "Basic Proficiency in " + topic,
                Description = "Demonstrate understanding of fundamental concepts",
                CompletionCriteria = "Complete introductory modules and exercises"
            },
            new Milestone
            {
                Title = "Advanced Mastery of " + topic,
                Description = "Demonstrate ability to apply advanced concepts",
                CompletionCriteria = "Complete advanced modules and final project"
            }
        };
    }

    /// <summary>
    /// Helper method to create default practice projects
    /// </summary>
    private List<PracticeProject> CreateDefaultProjects(string topic)
    {
        return new List<PracticeProject>
        {
            new PracticeProject
            {
                Title = "Simple " + topic + " Application",
                Description = "Build a basic application to practice fundamental concepts",
                Difficulty = "Beginner",
                EstimatedHours = 5
            },
            new PracticeProject
            {
                Title = "Advanced " + topic + " Project",
                Description = "Build a comprehensive project showcasing advanced skills",
                Difficulty = "Advanced",
                EstimatedHours = 10
            }
        };
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
        try
        {
            // Replace any invalid escape sequences
            jsonContent = jsonContent.Replace("\\n", "\n");

            // Remove any trailing commas before closing brackets or braces
            jsonContent = System.Text.RegularExpressions.Regex.Replace(jsonContent, @",\s*([\]\}])", "$1");

            // Fix common JSON formatting issues
            // Replace single quotes with double quotes for JSON compatibility
            jsonContent = System.Text.RegularExpressions.Regex.Replace(jsonContent, @"'([^']*)'\s*:", "\"$1\":");

            // Fix unquoted property names
            jsonContent = System.Text.RegularExpressions.Regex.Replace(jsonContent, @"([a-zA-Z0-9_]+)\s*:", "\"$1\":");

            // Fix unquoted string values
            jsonContent = System.Text.RegularExpressions.Regex.Replace(jsonContent, @":\s*([a-zA-Z][a-zA-Z0-9_\s-]*)\s*([,\}\]])", ": \"$1\"$2");

            // Fix invalid numeric values with hyphens (like "1-2" weeks)
            jsonContent = System.Text.RegularExpressions.Regex.Replace(jsonContent, @":\s*(\d+)-(\d+)\s*([,\}\]])", ": \"$1-$2\"$3");

            // Fix missing quotes around numeric values with hyphens
            jsonContent = System.Text.RegularExpressions.Regex.Replace(jsonContent, @"""week"":\s*(\d+)-(\d+)", "\"week\": \"$1-$2\"");

            // Log the cleaned JSON for debugging
            _logger.LogDebug($"Cleaned JSON content: {jsonContent}");

            return jsonContent;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error cleaning JSON content");
            return jsonContent; // Return original content if cleanup fails
        }
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

            // Try to parse timeline
            if (root.TryGetProperty("timeline", out var timelineElement) && timelineElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in timelineElement.EnumerateArray())
                {
                    var timelineItem = new TimelineItem();

                    // Handle week property - could be a number or a string like "1-2"
                    if (item.TryGetProperty("week", out var weekElement))
                    {
                        if (weekElement.ValueKind == JsonValueKind.Number)
                        {
                            timelineItem.Week = weekElement.GetInt32().ToString();
                        }
                        else if (weekElement.ValueKind == JsonValueKind.String)
                        {
                            timelineItem.Week = weekElement.GetString() ?? "";
                        }
                    }

                    // Handle activities
                    if (item.TryGetProperty("activities", out var activitiesElement) && activitiesElement.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var activity in activitiesElement.EnumerateArray())
                        {
                            if (activity.ValueKind == JsonValueKind.String)
                            {
                                timelineItem.Activities.Add(activity.GetString() ?? "");
                            }
                        }
                    }

                    content.Timeline.Add(timelineItem);
                }
            }

            // Try to parse modules
            if (root.TryGetProperty("modules", out var modulesElement) && modulesElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var moduleElement in modulesElement.EnumerateArray())
                {
                    var module = new Module();

                    // Get module title
                    if (moduleElement.TryGetProperty("title", out var titleElement) && titleElement.ValueKind == JsonValueKind.String)
                    {
                        module.Title = titleElement.GetString() ?? "";
                    }

                    // Get module objectives
                    if (moduleElement.TryGetProperty("objectives", out var objectivesElement) && objectivesElement.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var objective in objectivesElement.EnumerateArray())
                        {
                            if (objective.ValueKind == JsonValueKind.String)
                            {
                                module.Objectives.Add(objective.GetString() ?? "");
                            }
                        }
                    }

                    // Get estimated hours
                    if (moduleElement.TryGetProperty("estimatedHours", out var hoursElement) && hoursElement.ValueKind == JsonValueKind.Number)
                    {
                        module.EstimatedHours = hoursElement.GetInt32();
                    }

                    // Get assessment
                    if (moduleElement.TryGetProperty("assessment", out var assessmentElement) && assessmentElement.ValueKind == JsonValueKind.String)
                    {
                        module.Assessment = assessmentElement.GetString() ?? "";
                    }

                    content.Modules.Add(module);
                }
            }

            // Return the partially parsed content
            return content;
        }
        catch (Exception ex)
        {
            // Log the exception for debugging
            _logger.LogError(ex, "Error in fallback parsing of learning plan content");

            // Create a minimal valid learning plan content as a last resort
            return new LearningPlanContent
            {
                Introduction = "A learning plan for " + (jsonContent.Contains("CLI") ? "building CLI applications" : "software development"),
                Prerequisites = new List<string> { "Basic programming knowledge", "Familiarity with C#" },
                Modules = new List<Module>
                {
                    new Module
                    {
                        Title = "Getting Started",
                        Objectives = new List<string> { "Set up development environment", "Create a basic project" },
                        EstimatedHours = 4
                    },
                    new Module
                    {
                        Title = "Core Concepts",
                        Objectives = new List<string> { "Learn fundamental principles", "Practice with exercises" },
                        EstimatedHours = 8
                    }
                },
                Timeline = new List<TimelineItem>
                {
                    new TimelineItem
                    {
                        Week = "1",
                        Activities = new List<string> { "Complete Module 1", "Start Module 2" }
                    },
                    new TimelineItem
                    {
                        Week = "2",
                        Activities = new List<string> { "Complete Module 2", "Work on project" }
                    }
                }
            };
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
[JsonConverter(typeof(JsonStringEnumConverter))]
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
    // Changed from int to string to handle ranges like "1-2"
    public string Week { get; set; } = string.Empty;
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