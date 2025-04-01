using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace TarsCli.Services;

/// <summary>
/// Service for generating structured course content
/// </summary>
public class CourseGeneratorService
{
    private readonly ILogger<CourseGeneratorService> _logger;
    private readonly IConfiguration _configuration;
    private readonly OllamaService _ollamaService;
    private readonly string _coursesDirectory;

    public CourseGeneratorService(
        ILogger<CourseGeneratorService> logger,
        IConfiguration configuration,
        OllamaService ollamaService)
    {
        _logger = logger;
        _configuration = configuration;
        _ollamaService = ollamaService;

        // Set up the courses directory
        var appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        if (string.IsNullOrEmpty(appDataPath))
        {
            appDataPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars");
        }
        else
        {
            appDataPath = Path.Combine(appDataPath, "TARS");
        }

        _coursesDirectory = Path.Combine(appDataPath, "Courses");

        if (!Directory.Exists(_coursesDirectory))
        {
            Directory.CreateDirectory(_coursesDirectory);
        }
    }

    /// <summary>
    /// Generates a structured course based on topic and difficulty level
    /// </summary>
    public async Task<Course> GenerateCourse(
        string title,
        string description,
        string topic,
        DifficultyLevel difficultyLevel,
        int estimatedHours,
        List<string> targetAudience,
        string model = "")
    {
        _logger.LogInformation($"Generating course for topic: {topic}, difficulty level: {difficultyLevel}");

        try
        {
            // Use default model if not specified
            if (string.IsNullOrEmpty(model))
            {
                model = _configuration["Ollama:DefaultModel"] ?? "llama3";
            }

            // Create a prompt for the AI model
            var prompt = CreateCoursePrompt(title, description, topic, difficultyLevel, estimatedHours, targetAudience);

            // Generate the course content using Ollama
            var response = await _ollamaService.GenerateCompletion(prompt, model);

            // Parse the response into a structured course
            var course = ParseCourseResponse(response, title, description, topic, difficultyLevel);

            // Save the course
            await SaveCourse(course);

            return course;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating course for topic: {topic}");
            throw;
        }
    }

    /// <summary>
    /// Creates a prompt for generating a course
    /// </summary>
    private string CreateCoursePrompt(
        string title,
        string description,
        string topic,
        DifficultyLevel difficultyLevel,
        int estimatedHours,
        List<string> targetAudience)
    {
        return $@"
You are an expert educational content creator and curriculum designer. Create a structured course with the following details:

TITLE: {title}
DESCRIPTION: {description}
TOPIC: {topic}
DIFFICULTY LEVEL: {difficultyLevel}
ESTIMATED HOURS: {estimatedHours}
TARGET AUDIENCE: {string.Join(", ", targetAudience)}

The course should include:
1. A comprehensive overview of the topic
2. A list of learning objectives
3. A structured sequence of lessons, each containing:
   - Lesson title
   - Learning objectives
   - Estimated time to complete
   - Content (theory, concepts, examples)
   - Exercises or activities
   - Quiz questions with answers
4. A final assessment or project
5. A list of additional resources for further learning

Format your response as a structured JSON object with the following schema:
{{
  ""overview"": ""string"",
  ""learningObjectives"": [""string""],
  ""lessons"": [
    {{
      ""title"": ""string"",
      ""objectives"": [""string""],
      ""estimatedMinutes"": number,
      ""content"": ""string"",
      ""exercises"": [
        {{
          ""title"": ""string"",
          ""description"": ""string"",
          ""difficulty"": ""string""
        }}
      ],
      ""quizQuestions"": [
        {{
          ""question"": ""string"",
          ""options"": [""string""],
          ""correctAnswerIndex"": number,
          ""explanation"": ""string""
        }}
      ]
    }}
  ],
  ""finalAssessment"": {{
    ""title"": ""string"",
    ""description"": ""string"",
    ""criteria"": [""string""],
    ""estimatedHours"": number
  }},
  ""additionalResources"": [
    {{
      ""title"": ""string"",
      ""type"": ""string"",
      ""url"": ""string"",
      ""description"": ""string""
    }}
  ]
}}

Ensure the course is tailored to the specified difficulty level and target audience, and can be realistically completed within the estimated hours.
";
    }

    /// <summary>
    /// Parses the AI response into a structured course
    /// </summary>
    private Course ParseCourseResponse(string response, string title, string description, string topic, DifficultyLevel difficultyLevel)
    {
        try
        {
            // Extract JSON from the response (in case the model includes additional text)
            var jsonStart = response.IndexOf('{');
            var jsonEnd = response.LastIndexOf('}');

            if (jsonStart >= 0 && jsonEnd >= 0 && jsonEnd > jsonStart)
            {
                var jsonContent = response.Substring(jsonStart, jsonEnd - jsonStart + 1);
                var courseContent = JsonSerializer.Deserialize<CourseContent>(jsonContent, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                // Create a new course with metadata and content
                return new Course
                {
                    Id = Guid.NewGuid().ToString(),
                    Title = title,
                    Description = description,
                    Topic = topic,
                    DifficultyLevel = difficultyLevel,
                    CreatedDate = DateTime.UtcNow,
                    LastModifiedDate = DateTime.UtcNow,
                    Content = courseContent
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
            _logger.LogError(ex, "Error parsing course JSON");
            throw new FormatException("The AI response contained invalid JSON", ex);
        }
    }

    /// <summary>
    /// Saves a course to disk
    /// </summary>
    public async Task SaveCourse(Course course)
    {
        var filePath = Path.Combine(_coursesDirectory, $"{course.Id}.json");

        var options = new JsonSerializerOptions
        {
            WriteIndented = true
        };

        var json = JsonSerializer.Serialize(course, options);
        await File.WriteAllTextAsync(filePath, json);

        _logger.LogInformation($"Course saved: {filePath}");
    }

    /// <summary>
    /// Gets a course by ID
    /// </summary>
    public async Task<Course> GetCourse(string id)
    {
        var filePath = Path.Combine(_coursesDirectory, $"{id}.json");

        if (!File.Exists(filePath))
        {
            _logger.LogError($"Course not found: {id}");
            throw new FileNotFoundException($"Course not found: {id}");
        }

        var json = await File.ReadAllTextAsync(filePath);
        return JsonSerializer.Deserialize<Course>(json);
    }

    /// <summary>
    /// Gets all courses
    /// </summary>
    public async Task<List<Course>> GetCourses()
    {
        var courses = new List<Course>();

        foreach (var file in Directory.GetFiles(_coursesDirectory, "*.json"))
        {
            try
            {
                var json = await File.ReadAllTextAsync(file);
                var course = JsonSerializer.Deserialize<Course>(json);
                courses.Add(course);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, $"Error loading course: {file}");
            }
        }

        return courses.OrderByDescending(c => c.LastModifiedDate).ToList();
    }

    /// <summary>
    /// Updates an existing course
    /// </summary>
    public async Task<Course> UpdateCourse(string id, Course updatedCourse)
    {
        var existingCourse = await GetCourse(id);

        // Update the course properties
        existingCourse.Title = updatedCourse.Title;
        existingCourse.Description = updatedCourse.Description;
        existingCourse.Topic = updatedCourse.Topic;
        existingCourse.DifficultyLevel = updatedCourse.DifficultyLevel;
        existingCourse.Content = updatedCourse.Content;
        existingCourse.LastModifiedDate = DateTime.UtcNow;

        // Save the updated course
        await SaveCourse(existingCourse);

        return existingCourse;
    }

    /// <summary>
    /// Deletes a course
    /// </summary>
    public Task DeleteCourse(string id)
    {
        var filePath = Path.Combine(_coursesDirectory, $"{id}.json");

        if (!File.Exists(filePath))
        {
            _logger.LogError($"Course not found: {id}");
            throw new FileNotFoundException($"Course not found: {id}");
        }

        File.Delete(filePath);
        _logger.LogInformation($"Course deleted: {id}");

        return Task.CompletedTask;
    }

    /// <summary>
    /// Exports a course to markdown format
    /// </summary>
    public async Task<string> ExportCourseToMarkdown(string id)
    {
        var course = await GetCourse(id);

        var markdown = new System.Text.StringBuilder();

        // Add course header
        markdown.AppendLine($"# {course.Title}");
        markdown.AppendLine();
        markdown.AppendLine($"*{course.Description}*");
        markdown.AppendLine();
        markdown.AppendLine($"**Topic:** {course.Topic}");
        markdown.AppendLine($"**Difficulty:** {course.DifficultyLevel}");
        markdown.AppendLine($"**Created:** {course.CreatedDate:yyyy-MM-dd}");
        markdown.AppendLine();

        // Add overview
        markdown.AppendLine("## Overview");
        markdown.AppendLine();
        markdown.AppendLine(course.Content.Overview);
        markdown.AppendLine();

        // Add learning objectives
        markdown.AppendLine("## Learning Objectives");
        markdown.AppendLine();
        foreach (var objective in course.Content.LearningObjectives)
        {
            markdown.AppendLine($"- {objective}");
        }
        markdown.AppendLine();

        // Add lessons
        markdown.AppendLine("## Lessons");
        markdown.AppendLine();

        for (int i = 0; i < course.Content.Lessons.Count; i++)
        {
            var lesson = course.Content.Lessons[i];

            markdown.AppendLine($"### Lesson {i + 1}: {lesson.Title}");
            markdown.AppendLine();
            markdown.AppendLine($"**Estimated time:** {lesson.EstimatedMinutes} minutes");
            markdown.AppendLine();

            markdown.AppendLine("#### Objectives");
            markdown.AppendLine();
            foreach (var objective in lesson.Objectives)
            {
                markdown.AppendLine($"- {objective}");
            }
            markdown.AppendLine();

            markdown.AppendLine("#### Content");
            markdown.AppendLine();
            markdown.AppendLine(lesson.Content);
            markdown.AppendLine();

            markdown.AppendLine("#### Exercises");
            markdown.AppendLine();
            foreach (var exercise in lesson.Exercises)
            {
                markdown.AppendLine($"##### {exercise.Title}");
                markdown.AppendLine();
                markdown.AppendLine($"*Difficulty: {exercise.Difficulty}*");
                markdown.AppendLine();
                markdown.AppendLine(exercise.Description);
                markdown.AppendLine();
            }

            markdown.AppendLine("#### Quiz");
            markdown.AppendLine();
            for (int j = 0; j < lesson.QuizQuestions.Count; j++)
            {
                var quiz = lesson.QuizQuestions[j];

                markdown.AppendLine($"**Question {j + 1}:** {quiz.Question}");
                markdown.AppendLine();

                for (int k = 0; k < quiz.Options.Count; k++)
                {
                    var option = quiz.Options[k];
                    markdown.AppendLine($"{k + 1}. {option}");
                }

                markdown.AppendLine();
                markdown.AppendLine($"*Answer: {quiz.Options[quiz.CorrectAnswerIndex]}*");
                markdown.AppendLine();
                markdown.AppendLine($"*Explanation: {quiz.Explanation}*");
                markdown.AppendLine();
            }
        }

        // Add final assessment
        markdown.AppendLine("## Final Assessment");
        markdown.AppendLine();
        markdown.AppendLine($"### {course.Content.FinalAssessment.Title}");
        markdown.AppendLine();
        markdown.AppendLine($"**Estimated time:** {course.Content.FinalAssessment.EstimatedHours} hours");
        markdown.AppendLine();
        markdown.AppendLine(course.Content.FinalAssessment.Description);
        markdown.AppendLine();

        markdown.AppendLine("#### Assessment Criteria");
        markdown.AppendLine();
        foreach (var criterion in course.Content.FinalAssessment.Criteria)
        {
            markdown.AppendLine($"- {criterion}");
        }
        markdown.AppendLine();

        // Add additional resources
        markdown.AppendLine("## Additional Resources");
        markdown.AppendLine();
        foreach (var resource in course.Content.AdditionalResources)
        {
            markdown.AppendLine($"- [{resource.Title}]({resource.Url}) - {resource.Type}: {resource.Description}");
        }

        return markdown.ToString();
    }
}

/// <summary>
/// Represents a difficulty level
/// </summary>
public enum DifficultyLevel
{
    Beginner,
    Intermediate,
    Advanced,
    Expert
}

/// <summary>
/// Represents a course
/// </summary>
public class Course
{
    public required string Id { get; set; }
    public required string Title { get; set; }
    public required string Description { get; set; }
    public required string Topic { get; set; }
    public DifficultyLevel DifficultyLevel { get; set; }
    public DateTime CreatedDate { get; set; }
    public DateTime LastModifiedDate { get; set; }
    public required CourseContent Content { get; set; }
}

/// <summary>
/// Represents the content of a course
/// </summary>
public class CourseContent
{
    public required string Overview { get; set; }
    public List<string> LearningObjectives { get; set; } = new List<string>();
    public List<Lesson> Lessons { get; set; } = new List<Lesson>();
    public required FinalAssessment FinalAssessment { get; set; }
    public List<AdditionalResource> AdditionalResources { get; set; } = new List<AdditionalResource>();
}

/// <summary>
/// Represents a lesson in a course
/// </summary>
public class Lesson
{
    public required string Title { get; set; }
    public List<string> Objectives { get; set; } = new List<string>();
    public int EstimatedMinutes { get; set; }
    public required string Content { get; set; }
    public List<Exercise> Exercises { get; set; } = new List<Exercise>();
    public List<QuizQuestion> QuizQuestions { get; set; } = new List<QuizQuestion>();
}

/// <summary>
/// Represents an exercise in a lesson
/// </summary>
public class Exercise
{
    public required string Title { get; set; }
    public required string Description { get; set; }
    public required string Difficulty { get; set; }
}

/// <summary>
/// Represents a quiz question in a lesson
/// </summary>
public class QuizQuestion
{
    public required string Question { get; set; }
    public List<string> Options { get; set; } = new List<string>();
    public int CorrectAnswerIndex { get; set; }
    public required string Explanation { get; set; }
}

/// <summary>
/// Represents a final assessment in a course
/// </summary>
public class FinalAssessment
{
    public required string Title { get; set; }
    public required string Description { get; set; }
    public List<string> Criteria { get; set; } = new List<string>();
    public int EstimatedHours { get; set; }
}

/// <summary>
/// Represents an additional resource in a course
/// </summary>
public class AdditionalResource
{
    public required string Title { get; set; }
    public required string Type { get; set; }
    public required string Url { get; set; }
    public required string Description { get; set; }
}