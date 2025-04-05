using System.Text.Json;
using Microsoft.Extensions.Configuration;

namespace TarsCli.Services;

/// <summary>
/// Service for demonstrating TARS capabilities
/// </summary>
public class DemoService
{
    private readonly ILogger<DemoService> _logger;
    private readonly IConfiguration _configuration;
    private readonly SelfImprovementService _selfImprovementService;
    private readonly OllamaService _ollamaService;
    private readonly HuggingFaceService _huggingFaceService;
    private readonly LanguageSpecificationService _languageSpecificationService;
    private readonly DocumentationService _documentationService;
    private readonly ChatBotService _chatBotService;
    private readonly DeepThinkingService _deepThinkingService;
    private readonly LearningPlanService _learningPlanService;
    private readonly CourseGeneratorService _courseGeneratorService;
    private readonly TutorialOrganizerService _tutorialOrganizerService;
    private readonly TarsSpeechService _tarsSpeechService;
    private readonly McpService _mcpService;
    private readonly string _demoDir;

    public DemoService(
        ILogger<DemoService> logger,
        IConfiguration configuration,
        SelfImprovementService selfImprovementService,
        OllamaService ollamaService,
        HuggingFaceService huggingFaceService,
        LanguageSpecificationService languageSpecificationService,
        DocumentationService documentationService,
        ChatBotService chatBotService,
        DeepThinkingService deepThinkingService,
        LearningPlanService learningPlanService,
        CourseGeneratorService courseGeneratorService,
        TutorialOrganizerService tutorialOrganizerService,
        TarsSpeechService tarsSpeechService,
        McpService mcpService)
    {
        _logger = logger;
        _configuration = configuration;
        _selfImprovementService = selfImprovementService;
        _ollamaService = ollamaService;
        _huggingFaceService = huggingFaceService;
        _languageSpecificationService = languageSpecificationService;
        _documentationService = documentationService;
        _chatBotService = chatBotService;
        _deepThinkingService = deepThinkingService;
        _learningPlanService = learningPlanService;
        _courseGeneratorService = courseGeneratorService;
        _tutorialOrganizerService = tutorialOrganizerService;
        _tarsSpeechService = tarsSpeechService;
        _mcpService = mcpService;

        // Create demo directory
        var projectRoot = _configuration["Tars:ProjectRoot"] ?? Directory.GetCurrentDirectory();
        _demoDir = Path.Combine(projectRoot, "demo");
        Directory.CreateDirectory(_demoDir);
    }

    /// <summary>
    /// Run a demonstration of TARS capabilities
    /// </summary>
    public async Task<bool> RunDemoAsync(string demoType, string model)
    {
        try
        {
            _logger.LogInformation($"Running {demoType} demo with model {model}");

            CliSupport.WriteHeader($"TARS Demonstration - {demoType}");
            Console.WriteLine($"Model: {model}");
            Console.WriteLine();

            switch (demoType.ToLowerInvariant())
            {
                case "self-improvement":
                    return await RunSelfImprovementDemoAsync(model);
                case "code-generation":
                    return await RunCodeGenerationDemoAsync(model);
                case "language-specs":
                    return await RunLanguageSpecsDemoAsync();
                case "chatbot":
                    return await RunChatBotDemoAsync(model);
                case "deep-thinking":
                    return await RunDeepThinkingDemoAsync(model);
                case "learning-plan":
                    return await RunLearningPlanDemoAsync(model);
                case "course-generator":
                    return await RunCourseGeneratorDemoAsync(model);
                case "tutorial-organizer":
                    return await RunTutorialOrganizerDemoAsync(model);
                case "speech":
                    return await RunSpeechDemoAsync(model);
                case "mcp":
                    return await RunMcpDemoAsync(model);
                case "metascript":
                    return await RunMetascriptDemoAsync(model);
                case "all":
                    var success = await RunSelfImprovementDemoAsync(model);
                    if (!success) return false;

                    success = await RunCodeGenerationDemoAsync(model);
                    if (!success) return false;

                    success = await RunLanguageSpecsDemoAsync();
                    if (!success) return false;

                    success = await RunChatBotDemoAsync(model);
                    if (!success) return false;

                    success = await RunDeepThinkingDemoAsync(model);
                    if (!success) return false;

                    success = await RunLearningPlanDemoAsync(model);
                    if (!success) return false;

                    success = await RunCourseGeneratorDemoAsync(model);
                    if (!success) return false;

                    success = await RunTutorialOrganizerDemoAsync(model);
                    if (!success) return false;

                    success = await RunSpeechDemoAsync(model);
                    if (!success) return false;

                    success = await RunMcpDemoAsync(model);
                    if (!success) return false;

                    success = await RunMetascriptDemoAsync(model);
                    if (!success) return false;

                    return true;
                default:
                    CliSupport.WriteColorLine($"Unknown demo type: {demoType}", ConsoleColor.Red);
                    Console.WriteLine("Available demo types:");
                    Console.WriteLine("  - self-improvement: Demonstrate code analysis and improvement");
                    Console.WriteLine("  - code-generation: Demonstrate code generation capabilities");
                    Console.WriteLine("  - language-specs: Demonstrate language specification generation");
                    Console.WriteLine("  - chatbot: Demonstrate chat bot capabilities");
                    Console.WriteLine("  - deep-thinking: Demonstrate deep thinking exploration");
                    Console.WriteLine("  - learning-plan: Demonstrate learning plan generation");
                    Console.WriteLine("  - course-generator: Demonstrate course generation");
                    Console.WriteLine("  - tutorial-organizer: Demonstrate tutorial organization");
                    Console.WriteLine("  - speech: Demonstrate text-to-speech capabilities");
                    Console.WriteLine("  - mcp: Demonstrate Model Context Protocol integration");
                    Console.WriteLine("  - metascript: Demonstrate TARS Metascript capabilities");
                    Console.WriteLine("  - all: Run all demos");
                    return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error running {demoType} demo");
            CliSupport.WriteColorLine($"Error running demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Run a demonstration of self-improvement capabilities
    /// </summary>
    private async Task<bool> RunSelfImprovementDemoAsync(string model)
    {
        try
        {
            CliSupport.WriteColorLine("Self-Improvement Demonstration", ConsoleColor.Cyan);
            Console.WriteLine("This demo will show how TARS can analyze and improve code.");
            Console.WriteLine();

            // Create a demo file with code issues
            var demoFilePath = Path.Combine(_demoDir, "demo_code_with_issues.cs");
            await File.WriteAllTextAsync(demoFilePath, GetDemoCodeWithIssues());

            CliSupport.WriteColorLine("Created demo file with code issues:", ConsoleColor.Yellow);
            Console.WriteLine(demoFilePath);
            Console.WriteLine();

            // Display the original code
            CliSupport.WriteColorLine("Original Code:", ConsoleColor.Yellow);
            var originalCode = await File.ReadAllTextAsync(demoFilePath);
            Console.WriteLine(originalCode);
            Console.WriteLine();

            // Step 1: Analyze the code
            CliSupport.WriteColorLine("Step 1: Analyzing the code...", ConsoleColor.Green);
            Console.WriteLine();

            var analyzeSuccess = await _selfImprovementService.AnalyzeFile(demoFilePath, model);
            if (!analyzeSuccess)
            {
                CliSupport.WriteColorLine("Failed to analyze the code.", ConsoleColor.Red);
                return false;
            }

            Console.WriteLine();

            // Step 2: Propose improvements
            CliSupport.WriteColorLine("Step 2: Proposing improvements...", ConsoleColor.Green);
            Console.WriteLine();

            var proposeSuccess = await _selfImprovementService.ProposeImprovement(demoFilePath, model, true);
            if (!proposeSuccess)
            {
                CliSupport.WriteColorLine("Failed to propose improvements.", ConsoleColor.Red);
                return false;
            }

            Console.WriteLine();

            // Step 3: Show the improved code
            CliSupport.WriteColorLine("Step 3: Improved Code:", ConsoleColor.Green);
            var improvedCode = await File.ReadAllTextAsync(demoFilePath);
            Console.WriteLine(improvedCode);

            CliSupport.WriteColorLine("Self-Improvement Demo Completed Successfully!", ConsoleColor.Cyan);
            Console.WriteLine();

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running self-improvement demo");
            CliSupport.WriteColorLine($"Error running self-improvement demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Run a demonstration of code generation capabilities
    /// </summary>
    private async Task<bool> RunCodeGenerationDemoAsync(string model)
    {
        try
        {
            CliSupport.WriteColorLine("Code Generation Demonstration", ConsoleColor.Cyan);
            Console.WriteLine("This demo will show how TARS can generate code based on natural language descriptions.");
            Console.WriteLine();

            // Step 1: Generate a simple class
            CliSupport.WriteColorLine("Step 1: Generating a simple class...", ConsoleColor.Green);
            Console.WriteLine();

            var prompt = "Generate a C# class called 'WeatherForecast' with properties for Date (DateTime), TemperatureC (int), TemperatureF (calculated from TemperatureC), and Summary (string). Include appropriate XML documentation comments.";
            Console.WriteLine($"Prompt: {prompt}");
            Console.WriteLine();

            var response = await _ollamaService.GenerateCompletion(prompt, model);

            // Extract the code from the response
            var code = ExtractCodeFromResponse(response);

            // Save the generated code to a file
            var demoFilePath = Path.Combine(_demoDir, "WeatherForecast.cs");
            await File.WriteAllTextAsync(demoFilePath, code);

            CliSupport.WriteColorLine("Generated Code:", ConsoleColor.Yellow);
            Console.WriteLine(code);
            Console.WriteLine();

            // Step 2: Generate a more complex example
            CliSupport.WriteColorLine("Step 2: Generating a more complex example...", ConsoleColor.Green);
            Console.WriteLine();

            prompt = "Generate a C# implementation of a simple in-memory cache with generic type support. Include methods for Add, Get, Remove, and Clear. Implement a time-based expiration mechanism for cache entries.";
            Console.WriteLine($"Prompt: {prompt}");
            Console.WriteLine();

            response = await _ollamaService.GenerateCompletion(prompt, model);

            // Extract the code from the response
            code = ExtractCodeFromResponse(response);

            // Save the generated code to a file
            demoFilePath = Path.Combine(_demoDir, "MemoryCache.cs");
            await File.WriteAllTextAsync(demoFilePath, code);

            CliSupport.WriteColorLine("Generated Code:", ConsoleColor.Yellow);
            Console.WriteLine(code);
            Console.WriteLine();

            CliSupport.WriteColorLine("Code Generation Demo Completed Successfully!", ConsoleColor.Cyan);
            Console.WriteLine();

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running code generation demo");
            CliSupport.WriteColorLine($"Error running code generation demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Run a demonstration of language specification generation
    /// </summary>
    private async Task<bool> RunLanguageSpecsDemoAsync()
    {
        try
        {
            CliSupport.WriteColorLine("Language Specification Demonstration", ConsoleColor.Cyan);
            Console.WriteLine("This demo will show how TARS can generate language specifications for its DSL.");
            Console.WriteLine();

            // Step 1: Generate EBNF specification
            CliSupport.WriteColorLine("Step 1: Generating EBNF specification...", ConsoleColor.Green);
            Console.WriteLine();

            var ebnf = await _languageSpecificationService.GenerateEbnfAsync();
            var ebnfPath = Path.Combine(_demoDir, "tars_grammar.ebnf");
            await File.WriteAllTextAsync(ebnfPath, ebnf);

            CliSupport.WriteColorLine($"EBNF specification saved to: {ebnfPath}", ConsoleColor.Yellow);
            Console.WriteLine("Preview:");
            Console.WriteLine(ebnf.Substring(0, Math.Min(500, ebnf.Length)) + "...");
            Console.WriteLine();

            // Step 2: Generate BNF specification
            CliSupport.WriteColorLine("Step 2: Generating BNF specification...", ConsoleColor.Green);
            Console.WriteLine();

            var bnf = await _languageSpecificationService.GenerateBnfAsync();
            var bnfPath = Path.Combine(_demoDir, "tars_grammar.bnf");
            await File.WriteAllTextAsync(bnfPath, bnf);

            CliSupport.WriteColorLine($"BNF specification saved to: {bnfPath}", ConsoleColor.Yellow);
            Console.WriteLine("Preview:");
            Console.WriteLine(bnf.Substring(0, Math.Min(500, bnf.Length)) + "...");
            Console.WriteLine();

            // Step 3: Generate JSON schema
            CliSupport.WriteColorLine("Step 3: Generating JSON schema...", ConsoleColor.Green);
            Console.WriteLine();

            var schema = await _languageSpecificationService.GenerateJsonSchemaAsync();
            var schemaPath = Path.Combine(_demoDir, "tars_schema.json");
            await File.WriteAllTextAsync(schemaPath, schema);

            CliSupport.WriteColorLine($"JSON schema saved to: {schemaPath}", ConsoleColor.Yellow);
            Console.WriteLine("Preview:");
            Console.WriteLine(schema.Substring(0, Math.Min(500, schema.Length)) + "...");
            Console.WriteLine();

            // Step 4: Generate markdown documentation
            CliSupport.WriteColorLine("Step 4: Generating markdown documentation...", ConsoleColor.Green);
            Console.WriteLine();

            var docs = await _languageSpecificationService.GenerateMarkdownDocumentationAsync();
            var docsPath = Path.Combine(_demoDir, "tars_dsl_docs.md");
            await File.WriteAllTextAsync(docsPath, docs);

            CliSupport.WriteColorLine($"Markdown documentation saved to: {docsPath}", ConsoleColor.Yellow);
            Console.WriteLine("Preview:");
            Console.WriteLine(docs.Substring(0, Math.Min(500, docs.Length)) + "...");
            Console.WriteLine();

            CliSupport.WriteColorLine("Language Specification Demo Completed Successfully!", ConsoleColor.Cyan);
            Console.WriteLine();

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running language specification demo");
            CliSupport.WriteColorLine($"Error running language specification demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Extract code from a response
    /// </summary>
    private string ExtractCodeFromResponse(string response)
    {
        // Look for code blocks
        var codeBlockStart = response.IndexOf("```csharp");
        if (codeBlockStart == -1)
        {
            codeBlockStart = response.IndexOf("```c#");
        }
        if (codeBlockStart == -1)
        {
            codeBlockStart = response.IndexOf("```");
        }

        if (codeBlockStart != -1)
        {
            var codeStart = response.IndexOf('\n', codeBlockStart) + 1;
            var codeBlockEnd = response.IndexOf("```", codeStart);

            if (codeBlockEnd != -1)
            {
                return response.Substring(codeStart, codeBlockEnd - codeStart).Trim();
            }
        }

        // If no code block is found, return the entire response
        return response;
    }

    /// <summary>
    /// Get demo code with issues
    /// </summary>
    private string GetDemoCodeWithIssues()
    {
        return @"using System;
using System.Collections.Generic;

namespace DemoCode
{
    public class Program
    {
        public static void Main(string[] args)
        {
            // This is a demo program with some issues to be improved
            Console.WriteLine(""Hello, World!"");

            // Issue 1: Magic numbers
            int timeout = 300;

            // Issue 2: Inefficient string concatenation in loop
            string result = """";
            for (int i = 0; i < 100; i++)
            {
                result += i.ToString();
            }

            // Issue 3: Empty catch block
            try
            {
                int x = int.Parse(""abc"");
            }
            catch (Exception)
            {
                // Empty catch block
            }

            // Issue 4: Unused variable
            var unusedList = new List<string>();

            Console.WriteLine(result);
            Console.WriteLine($""Timeout is set to {timeout} seconds"");
        }
    }
}";
    }

    /// <summary>
    /// Run a demonstration of chat bot capabilities
    /// </summary>
    private async Task<bool> RunChatBotDemoAsync(string model)
    {
        try
        {
            CliSupport.WriteColorLine("Step 1: Demonstrating chat bot capabilities...", ConsoleColor.Green);
            Console.WriteLine();

            // Prepare a few example messages
            var messages = new List<string>
            {
                "Hello, how can you help me with TARS?",
                "What features does TARS have?",
                "Can you explain how the deep thinking feature works?"
            };

            // Send each message and get a response
            foreach (var message in messages)
            {
                Console.WriteLine($"User: {message}");
                var response = await _chatBotService.SendMessageAsync(message, false);
                CliSupport.WriteColorLine($"TARS: {response}", ConsoleColor.Cyan);
                Console.WriteLine();

                // Save the conversation to a file
                var demoFilePath = Path.Combine(_demoDir, "ChatBotDemo.txt");
                await File.AppendAllTextAsync(demoFilePath, $"User: {message}\n");
                await File.AppendAllTextAsync(demoFilePath, $"TARS: {response}\n\n");

                // Pause briefly between messages
                await Task.Delay(1000);
            }

            CliSupport.WriteColorLine("Chat bot demo completed successfully!", ConsoleColor.Green);
            Console.WriteLine($"Conversation saved to {Path.Combine(_demoDir, "ChatBotDemo.txt")}");
            Console.WriteLine();

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running chat bot demo");
            CliSupport.WriteColorLine($"Error running chat bot demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Run a demonstration of deep thinking capabilities
    /// </summary>
    private async Task<bool> RunDeepThinkingDemoAsync(string model)
    {
        try
        {
            CliSupport.WriteColorLine("Step 1: Generating a deep thinking exploration...", ConsoleColor.Green);
            Console.WriteLine();

            var topic = "The future of AI and human collaboration";
            Console.WriteLine($"Topic: {topic}");
            Console.WriteLine();

            var result = await _deepThinkingService.GenerateDeepThinkingExplorationAsync(topic, null, model);
            var filePath = await _deepThinkingService.SaveDeepThinkingExplorationAsync(result);

            CliSupport.WriteColorLine("Deep thinking exploration generated successfully!", ConsoleColor.Green);
            Console.WriteLine($"Saved to: {filePath}");
            Console.WriteLine();

            // Display a preview of the content
            CliSupport.WriteColorLine("Preview:", ConsoleColor.Yellow);
            var previewLength = Math.Min(result.Content.Length, 500);
            Console.WriteLine(result.Content.Substring(0, previewLength) + "...");
            Console.WriteLine();

            // Copy the file to the demo directory
            var demoFilePath = Path.Combine(_demoDir, "DeepThinkingDemo.md");
            await File.WriteAllTextAsync(demoFilePath, result.Content);

            CliSupport.WriteColorLine("Step 2: Generating related topics...", ConsoleColor.Green);
            Console.WriteLine();

            var relatedTopics = await _deepThinkingService.GenerateRelatedTopicsAsync(topic, 3, model);

            CliSupport.WriteColorLine("Related topics:", ConsoleColor.Yellow);
            foreach (var relatedTopic in relatedTopics)
            {
                Console.WriteLine($"- {relatedTopic}");
            }
            Console.WriteLine();

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running deep thinking demo");
            CliSupport.WriteColorLine($"Error running deep thinking demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Run a demonstration of learning plan generation
    /// </summary>
    private async Task<bool> RunLearningPlanDemoAsync(string model)
    {
        try
        {
            CliSupport.WriteColorLine("Step 1: Generating a learning plan...", ConsoleColor.Green);
            Console.WriteLine();

            var name = "TARS Development Learning Plan";
            var topic = "Building AI-powered CLI applications";
            var skillLevel = SkillLevel.Intermediate;
            var goals = new List<string> { "Learn C# CLI development", "Understand AI integration patterns", "Build a functional prototype" };
            var preferences = new List<string> { "Hands-on exercises", "Project-based learning" };
            var estimatedHours = 40;

            Console.WriteLine($"Name: {name}");
            Console.WriteLine($"Topic: {topic}");
            Console.WriteLine($"Skill Level: {skillLevel}");
            Console.WriteLine($"Goals: {string.Join(", ", goals)}");
            Console.WriteLine($"Preferences: {string.Join(", ", preferences)}");
            Console.WriteLine($"Estimated Hours: {estimatedHours}");
            Console.WriteLine();

            var learningPlan = await _learningPlanService.GenerateLearningPlan(name, topic, skillLevel, goals, preferences, estimatedHours, model);

            CliSupport.WriteColorLine("Learning plan generated successfully!", ConsoleColor.Green);
            Console.WriteLine($"ID: {learningPlan.Id}");
            Console.WriteLine();

            // Display a preview of the learning plan
            CliSupport.WriteColorLine("Introduction:", ConsoleColor.Yellow);
            Console.WriteLine(learningPlan.Content.Introduction);
            Console.WriteLine();

            CliSupport.WriteColorLine("Modules:", ConsoleColor.Yellow);
            foreach (var module in learningPlan.Content.Modules)
            {
                Console.WriteLine($"- {module.Title} ({module.EstimatedHours} hours)");
            }
            Console.WriteLine();

            // Save a copy to the demo directory
            var demoFilePath = Path.Combine(_demoDir, "LearningPlanDemo.json");
            var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(learningPlan, options);
            await File.WriteAllTextAsync(demoFilePath, json);

            // Log the full path of the demo file
            _logger.LogInformation($"Learning plan demo saved to: {Path.GetFullPath(demoFilePath)}");
            Console.WriteLine($"Learning plan demo saved to: {Path.GetFullPath(demoFilePath)}");

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running learning plan demo");
            CliSupport.WriteColorLine($"Error running learning plan demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Run a demonstration of course generation
    /// </summary>
    private async Task<bool> RunCourseGeneratorDemoAsync(string model)
    {
        try
        {
            CliSupport.WriteColorLine("Step 1: Generating a course...", ConsoleColor.Green);
            Console.WriteLine();

            var title = "Introduction to TARS Development";
            var description = "A comprehensive course on developing with the TARS framework";
            var topic = "TARS Development";
            var difficultyLevel = DifficultyLevel.Intermediate;
            var estimatedHours = 30;
            var targetAudience = new List<string> { "Software Developers", "AI Enthusiasts" };

            Console.WriteLine($"Title: {title}");
            Console.WriteLine($"Description: {description}");
            Console.WriteLine($"Topic: {topic}");
            Console.WriteLine($"Difficulty: {difficultyLevel}");
            Console.WriteLine($"Estimated Hours: {estimatedHours}");
            Console.WriteLine($"Target Audience: {string.Join(", ", targetAudience)}");
            Console.WriteLine();

            var course = await _courseGeneratorService.GenerateCourse(title, description, topic, difficultyLevel, estimatedHours, targetAudience, model);

            CliSupport.WriteColorLine("Course generated successfully!", ConsoleColor.Green);
            Console.WriteLine($"ID: {course.Id}");
            Console.WriteLine();

            // Display a preview of the course
            CliSupport.WriteColorLine("Overview:", ConsoleColor.Yellow);
            Console.WriteLine(course.Content.Overview);
            Console.WriteLine();

            CliSupport.WriteColorLine("Lessons:", ConsoleColor.Yellow);
            foreach (var lesson in course.Content.Lessons)
            {
                Console.WriteLine($"- {lesson.Title} ({lesson.EstimatedMinutes} minutes)");
            }
            Console.WriteLine();

            // Save a copy to the demo directory
            var demoFilePath = Path.Combine(_demoDir, "CourseDemo.json");
            var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(course, options);
            await File.WriteAllTextAsync(demoFilePath, json);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running course generator demo");
            CliSupport.WriteColorLine($"Error running course generator demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Run a demonstration of tutorial organization
    /// </summary>
    private async Task<bool> RunTutorialOrganizerDemoAsync(string model)
    {
        try
        {
            CliSupport.WriteColorLine("Step 1: Adding a tutorial...", ConsoleColor.Green);
            Console.WriteLine();

            var title = "Getting Started with TARS CLI";
            var description = "A beginner's guide to using the TARS command-line interface";
            var content = "# Getting Started with TARS CLI\n\nThis tutorial will guide you through the basics of using the TARS command-line interface.\n\n## Installation\n\nFirst, make sure you have .NET 8.0 or later installed on your system.\n\n```bash\ndotnet tool install --global TarsCli\n```\n\n## Basic Commands\n\nHere are some basic commands to get you started:\n\n```bash\ntarscli --help\ntarscli demo --type all\n```\n\n## Next Steps\n\nExplore more advanced features like self-improvement and deep thinking.";
            var category = "Getting Started";
            var difficultyLevel = DifficultyLevel.Beginner;
            var tags = new List<string> { "CLI", "Beginner", "Tutorial" };
            var prerequisites = new List<string> { "Basic command-line knowledge" };

            Console.WriteLine($"Title: {title}");
            Console.WriteLine($"Description: {description}");
            Console.WriteLine($"Category: {category}");
            Console.WriteLine($"Difficulty: {difficultyLevel}");
            Console.WriteLine($"Tags: {string.Join(", ", tags)}");
            Console.WriteLine($"Prerequisites: {string.Join(", ", prerequisites)}");
            Console.WriteLine();

            var tutorial = await _tutorialOrganizerService.AddTutorial(title, description, content, category, difficultyLevel, tags, prerequisites);

            CliSupport.WriteColorLine("Tutorial added successfully!", ConsoleColor.Green);
            Console.WriteLine($"ID: {tutorial.Id}");
            Console.WriteLine();

            // Add a second tutorial
            CliSupport.WriteColorLine("Step 2: Adding another tutorial...", ConsoleColor.Green);
            Console.WriteLine();

            title = "Advanced TARS Features";
            description = "Exploring advanced features of the TARS framework";
            content = "# Advanced TARS Features\n\nThis tutorial explores advanced features of the TARS framework.\n\n## Deep Thinking\n\nThe deep thinking feature allows TARS to generate in-depth explorations on complex topics.\n\n```bash\ntarscli deep-thinking --topic \"AI Ethics\" --model llama3\n```\n\n## Self-Improvement\n\nTARS can analyze and improve code automatically.\n\n```bash\ntarscli self-analyze --file path/to/code.cs --model codellama\n```\n\n## Conclusion\n\nThese advanced features make TARS a powerful tool for AI-assisted development.";
            category = "Advanced Features";
            difficultyLevel = DifficultyLevel.Advanced;
            tags = ["Advanced", "Deep Thinking", "Self-Improvement"];
            prerequisites = ["Completion of Getting Started tutorial"];

            Console.WriteLine($"Title: {title}");
            Console.WriteLine($"Description: {description}");
            Console.WriteLine($"Category: {category}");
            Console.WriteLine($"Difficulty: {difficultyLevel}");
            Console.WriteLine($"Tags: {string.Join(", ", tags)}");
            Console.WriteLine($"Prerequisites: {string.Join(", ", prerequisites)}");
            Console.WriteLine();

            var tutorial2 = await _tutorialOrganizerService.AddTutorial(title, description, content, category, difficultyLevel, tags, prerequisites);

            CliSupport.WriteColorLine("Second tutorial added successfully!", ConsoleColor.Green);
            Console.WriteLine($"ID: {tutorial2.Id}");
            Console.WriteLine();

            // List tutorials
            CliSupport.WriteColorLine("Step 3: Listing tutorials...", ConsoleColor.Green);
            Console.WriteLine();

            var tutorials = await _tutorialOrganizerService.GetTutorials();

            CliSupport.WriteColorLine("All Tutorials:", ConsoleColor.Yellow);
            foreach (var t in tutorials)
            {
                Console.WriteLine($"- {t.Title} ({t.Category}, {t.DifficultyLevel})");
            }
            Console.WriteLine();

            // Save copies to the demo directory
            var demoFilePath1 = Path.Combine(_demoDir, "TutorialDemo1.md");
            var demoFilePath2 = Path.Combine(_demoDir, "TutorialDemo2.md");
            await File.WriteAllTextAsync(demoFilePath1, content);
            await File.WriteAllTextAsync(demoFilePath2, content);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running tutorial organizer demo");
            CliSupport.WriteColorLine($"Error running tutorial organizer demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Run a demonstration of text-to-speech capabilities
    /// </summary>
    private async Task<bool> RunSpeechDemoAsync(string model)
    {
        try
        {
            CliSupport.WriteColorLine("Step 1: Demonstrating text-to-speech...", ConsoleColor.Green);
            Console.WriteLine();

            var text = "Hello! I am TARS, an AI assistant designed to help with development tasks. I can generate code, improve existing code, create learning plans, and much more.";
            Console.WriteLine($"Text: {text}");
            Console.WriteLine();

            CliSupport.WriteColorLine("Converting text to speech...", ConsoleColor.Yellow);
            _tarsSpeechService.Speak(text);

            // List available voices
            CliSupport.WriteColorLine("\nStep 2: Listing available voices...", ConsoleColor.Green);
            Console.WriteLine();

            var voices = _tarsSpeechService.GetAvailableVoices();

            CliSupport.WriteColorLine("Available Voices:", ConsoleColor.Yellow);
            foreach (var voice in voices)
            {
                Console.WriteLine($"- {voice}");
            }
            Console.WriteLine();

            // Save demo info to a file
            var demoFilePath = Path.Combine(_demoDir, "SpeechDemo.txt");
            await File.WriteAllTextAsync(demoFilePath, $"Text: {text}\n\nAvailable Voices:\n{string.Join("\n", voices.Select(v => $"- {v}"))}\n");

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running speech demo");
            CliSupport.WriteColorLine($"Error running speech demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Run a demonstration of Model Context Protocol integration
    /// </summary>
    private async Task<bool> RunMcpDemoAsync(string model)
    {
        try
        {
            CliSupport.WriteColorLine("Step 1: Demonstrating MCP integration...", ConsoleColor.Green);
            Console.WriteLine();

            var command = "echo Hello from MCP!";
            Console.WriteLine($"Command: {command}");
            Console.WriteLine();

            var result = await _mcpService.ExecuteCommand(command);

            CliSupport.WriteColorLine("Command Result:", ConsoleColor.Yellow);
            Console.WriteLine(result);
            Console.WriteLine();

            // Generate code using MCP
            CliSupport.WriteColorLine("Step 2: Generating code with MCP...", ConsoleColor.Green);
            Console.WriteLine();

            var filePath = Path.Combine(_demoDir, "McpGeneratedCode.cs");
            var codeContent = "// This code was generated by MCP\nusing System;\n\nnamespace McpDemo\n{\n    public class HelloWorld\n    {\n        public static void Main()\n        {\n            Console.WriteLine(\"Hello from MCP!\");\n        }\n    }\n}";

            Console.WriteLine($"Generating code to: {filePath}");
            Console.WriteLine();

            var codeResult = await _mcpService.GenerateCode(filePath, codeContent);

            CliSupport.WriteColorLine("Code Generation Result:", ConsoleColor.Yellow);
            Console.WriteLine(codeResult);
            Console.WriteLine();

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running MCP demo");
            CliSupport.WriteColorLine($"Error running MCP demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Run a demonstration of TARS Metascript capabilities
    /// </summary>
    private async Task<bool> RunMetascriptDemoAsync(string model)
    {
        try
        {
            CliSupport.WriteColorLine("Step 1: Demonstrating TARS Metascript capabilities...", ConsoleColor.Green);
            Console.WriteLine();

            Console.WriteLine("TARS Metascripts allow you to create complex workflows that combine");
            Console.WriteLine("multiple AI capabilities, including collaboration with other AI systems.");
            Console.WriteLine();

            CliSupport.WriteColorLine("Step 2: Loading a sample metascript for TARS-Augment collaboration...", ConsoleColor.Green);
            Console.WriteLine();

            var metascriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Metascripts", "tars_augment_collaboration.tars");
            if (!File.Exists(metascriptPath))
            {
                metascriptPath = Path.Combine("TarsCli", "Metascripts", "tars_augment_collaboration.tars");
                if (!File.Exists(metascriptPath))
                {
                    metascriptPath = Path.Combine("Examples", "metascripts", "tars_augment_collaboration.tars");
                    if (!File.Exists(metascriptPath))
                    {
                        Console.WriteLine("Error: Could not find the sample metascript file.");
                        return false;
                    }
                }
            }

            var metascriptContent = await File.ReadAllTextAsync(metascriptPath);
            CliSupport.WriteColorLine("Metascript loaded successfully:", ConsoleColor.Yellow);
            Console.WriteLine("----------------------------------------");
            // Show just the first few lines
            var metascriptPreview = string.Join("\n", metascriptContent.Split('\n').Take(15));
            Console.WriteLine(metascriptPreview + "\n...");
            Console.WriteLine("----------------------------------------");
            Console.WriteLine();

            CliSupport.WriteColorLine("Step 3: Executing the metascript (simulation)...", ConsoleColor.Green);
            Console.WriteLine();

            Console.WriteLine("Starting collaboration with Augment on task: Implement a WebGPU renderer for a 3D scene with dynamic lighting");
            await Task.Delay(800);

            Console.WriteLine("[MCP] Sending request to Augment for code generation...");
            await Task.Delay(1500);

            Console.WriteLine("Received code from Augment, now enhancing with TARS capabilities");
            await Task.Delay(800);

            Console.WriteLine("TARS analysis complete. Sending optimization suggestions back to Augment.");
            await Task.Delay(1000);

            Console.WriteLine("[MCP] Sending optimization suggestions to Augment...");
            await Task.Delay(1500);

            Console.WriteLine("Collaboration complete. Final optimized code ready.");
            await Task.Delay(500);

            Console.WriteLine("Saving the final code to a file...");
            await Task.Delay(800);

            CliSupport.WriteColorLine("Metascript execution completed successfully.", ConsoleColor.Yellow);
            Console.WriteLine();
            Console.WriteLine("This demo shows how TARS can use metascripts to orchestrate complex workflows");
            Console.WriteLine("and collaborate with other AI systems like Augment Code via MCP.");
            Console.WriteLine();

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running Metascript demo");
            CliSupport.WriteColorLine($"Error running Metascript demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }
}