using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using TarsCli.Extensions;

namespace TarsCli.Services;

/// <summary>
/// Extension methods for the DemoService
/// </summary>
public static class DemoServiceExtensions
{
    /// <summary>
    /// Truncates a string to the specified length
    /// </summary>
    public static string Truncate(this string value, int maxLength)
    {
        if (string.IsNullOrEmpty(value)) return value;
        return value.Length <= maxLength ? value : value.Substring(0, maxLength - 3) + "...";
    }
}

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
    private readonly TarsEngine.Consciousness.Intelligence.IntelligenceSpark? _intelligenceSpark;
    private readonly TarsEngine.ML.Core.IntelligenceMeasurement? _intelligenceMeasurement;
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
        McpService mcpService,
        TarsEngine.Consciousness.Intelligence.IntelligenceSpark? intelligenceSpark = null,
        TarsEngine.ML.Core.IntelligenceMeasurement? intelligenceMeasurement = null)
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
        _intelligenceSpark = intelligenceSpark;
        _intelligenceMeasurement = intelligenceMeasurement;

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
                case "intelligence-spark":
                    return await RunIntelligenceSparkDemoAsync(model);
                case "code-complexity":
                    return await RunCodeComplexityDemoAsync();
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

                    success = await RunIntelligenceSparkDemoAsync(model);
                    if (!success) return false;

                    success = await RunCodeComplexityDemoAsync();
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
                    Console.WriteLine("  - intelligence-spark: Demonstrate intelligence spark and measurement");
                    Console.WriteLine("  - code-complexity: Demonstrate code complexity analysis");
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
            CliSupport.WriteColorLine("\nTARS Demonstration - Learning Plan Generation", ConsoleColor.Cyan);
            CliSupport.WriteColorLine("==========================================", ConsoleColor.Cyan);
            Console.WriteLine($"Model: {model}\n");

            CliSupport.WriteColorLine("Step 1: Configuring Learning Plan Parameters...", ConsoleColor.Green);
            Console.WriteLine();

            // Simulate interactive parameter selection
            Console.Write("Name: ");
            CliSupport.WriteColorLine("TARS Development Learning Plan", ConsoleColor.Yellow);
            await Task.Delay(300);

            Console.Write("Topic: ");
            CliSupport.WriteColorLine("Building AI-powered CLI applications", ConsoleColor.Yellow);
            await Task.Delay(300);

            Console.Write("Skill Level: ");
            CliSupport.WriteColorLine("Intermediate", ConsoleColor.Yellow);
            await Task.Delay(300);

            Console.Write("Goals: ");
            CliSupport.WriteColorLine("Learn C# CLI development, Understand AI integration patterns, Build a functional prototype", ConsoleColor.Yellow);
            await Task.Delay(300);

            Console.Write("Preferences: ");
            CliSupport.WriteColorLine("Hands-on exercises, Project-based learning", ConsoleColor.Yellow);
            await Task.Delay(300);

            Console.Write("Estimated Hours: ");
            CliSupport.WriteColorLine("40", ConsoleColor.Yellow);
            await Task.Delay(300);

            Console.WriteLine();

            var name = "TARS Development Learning Plan";
            var topic = "Building AI-powered CLI applications";
            var skillLevel = SkillLevel.Intermediate;
            var goals = new List<string> { "Learn C# CLI development", "Understand AI integration patterns", "Build a functional prototype" };
            var preferences = new List<string> { "Hands-on exercises", "Project-based learning" };
            var estimatedHours = 40;

            CliSupport.WriteColorLine("Step 2: Generating Learning Plan...", ConsoleColor.Green);
            Console.WriteLine();

            // Show a progress animation
            var progressChars = new[] { '|', '/', '-', '\\' };
            var progressTask = Task.Run(async () => {
                var i = 0;
                while (true) {
                    Console.Write($"\rGenerating learning plan {progressChars[i % progressChars.Length]} ");
                    await Task.Delay(100);
                    i++;
                }
            });

            var learningPlan = await _learningPlanService.GenerateLearningPlan(name, topic, skillLevel, goals, preferences, estimatedHours, model);

            // Stop the progress animation
            progressTask.Dispose();
            Console.WriteLine("\rLearning plan generated successfully!" + new string(' ', 20));
            Console.WriteLine();

            CliSupport.WriteColorLine("Step 3: Learning Plan Overview", ConsoleColor.Green);
            Console.WriteLine();

            // Display the learning plan ID with a border
            Console.WriteLine("┌" + new string('─', 50) + "┐");
            Console.WriteLine("│" + $" Learning Plan ID: {learningPlan.Id}".PadRight(50) + "│");
            Console.WriteLine("└" + new string('─', 50) + "┘");
            Console.WriteLine();

            // Display a preview of the learning plan with improved formatting
            CliSupport.WriteColorLine("Introduction:", ConsoleColor.Yellow);
            Console.WriteLine(learningPlan.Content.Introduction);
            Console.WriteLine();

            // Display prerequisites with bullet points
            CliSupport.WriteColorLine("Prerequisites:", ConsoleColor.Yellow);
            foreach (var prerequisite in learningPlan.Content.Prerequisites)
            {
                Console.WriteLine($"  • {prerequisite}");
            }
            Console.WriteLine();

            // Display modules with more detailed information
            CliSupport.WriteColorLine("Modules:", ConsoleColor.Yellow);
            for (var i = 0; i < learningPlan.Content.Modules.Count; i++)
            {
                var module = learningPlan.Content.Modules[i];
                Console.WriteLine($"  Module {i+1}: {module.Title}");
                Console.WriteLine($"    Duration: {module.EstimatedHours} hours");

                if (module.Objectives.Count > 0)
                {
                    Console.WriteLine("    Objectives:");
                    foreach (var objective in module.Objectives)
                    {
                        Console.WriteLine($"      - {objective}");
                    }
                }

                if (module.Resources.Count > 0)
                {
                    Console.WriteLine("    Resources:");
                    foreach (var resource in module.Resources)
                    {
                        Console.WriteLine($"      - {resource.Title} ({resource.Type})");
                    }
                }

                if (!string.IsNullOrEmpty(module.Assessment))
                {
                    Console.WriteLine($"    Assessment: {module.Assessment}");
                }

                Console.WriteLine();
            }

            // Display timeline with a visual representation
            CliSupport.WriteColorLine("Timeline:", ConsoleColor.Yellow);
            Console.WriteLine("┌───────────┬─────────────────────────────────────────┐");
            Console.WriteLine("│   Week    │ Activities                              │");
            Console.WriteLine("├───────────┼─────────────────────────────────────────┤");
            foreach (var timelineItem in learningPlan.Content.Timeline)
            {
                Console.WriteLine($"│ {timelineItem.Week,-9} │ {string.Join(", ", timelineItem.Activities).Truncate(39),-39} │");
            }
            Console.WriteLine("└───────────┴─────────────────────────────────────────┘");
            Console.WriteLine();

            CliSupport.WriteColorLine("Step 4: Exporting Learning Plan...", ConsoleColor.Green);
            Console.WriteLine();

            // Save as JSON
            var jsonFilePath = Path.Combine(_demoDir, "LearningPlanDemo.json");
            var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(learningPlan, options);
            await File.WriteAllTextAsync(jsonFilePath, json);

            // Save as Markdown for better readability
            var markdownFilePath = Path.Combine(_demoDir, "LearningPlanDemo.md");
            var markdown = GenerateLearningPlanMarkdown(learningPlan);
            await File.WriteAllTextAsync(markdownFilePath, markdown);

            // Log the full paths of the demo files
            _logger.LogInformation($"Learning plan demo saved to JSON: {Path.GetFullPath(jsonFilePath)}");
            _logger.LogInformation($"Learning plan demo saved to Markdown: {Path.GetFullPath(markdownFilePath)}");

            Console.WriteLine($"Learning plan exported to JSON: {Path.GetFullPath(jsonFilePath)}");
            Console.WriteLine($"Learning plan exported to Markdown: {Path.GetFullPath(markdownFilePath)}");
            Console.WriteLine();

            CliSupport.WriteColorLine("Learning Plan Demo Completed Successfully!", ConsoleColor.Yellow);
            Console.WriteLine();

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
    /// Generates a markdown representation of a learning plan
    /// </summary>
    private string GenerateLearningPlanMarkdown(LearningPlan plan)
    {
        var sb = new StringBuilder();

        // Title and metadata
        sb.AppendLine($"# {plan.Name}");
        sb.AppendLine();
        sb.AppendLine($"**Topic:** {plan.Topic}  ");
        sb.AppendLine($"**Skill Level:** {plan.SkillLevel}  ");
        sb.AppendLine($"**Created:** {plan.CreatedDate:yyyy-MM-dd}  ");
        sb.AppendLine($"**ID:** {plan.Id}  ");
        sb.AppendLine();

        // Introduction
        sb.AppendLine("## Introduction");
        sb.AppendLine();
        sb.AppendLine(plan.Content.Introduction);
        sb.AppendLine();

        // Prerequisites
        sb.AppendLine("## Prerequisites");
        sb.AppendLine();
        foreach (var prerequisite in plan.Content.Prerequisites)
        {
            sb.AppendLine($"- {prerequisite}");
        }
        sb.AppendLine();

        // Modules
        sb.AppendLine("## Modules");
        sb.AppendLine();
        for (var i = 0; i < plan.Content.Modules.Count; i++)
        {
            var module = plan.Content.Modules[i];
            sb.AppendLine($"### Module {i+1}: {module.Title}");
            sb.AppendLine();
            sb.AppendLine($"**Estimated Hours:** {module.EstimatedHours}  ");
            sb.AppendLine();

            if (module.Objectives.Count > 0)
            {
                sb.AppendLine("#### Objectives");
                sb.AppendLine();
                foreach (var objective in module.Objectives)
                {
                    sb.AppendLine($"- {objective}");
                }
                sb.AppendLine();
            }

            if (module.Resources.Count > 0)
            {
                sb.AppendLine("#### Resources");
                sb.AppendLine();
                foreach (var resource in module.Resources)
                {
                    sb.AppendLine($"- [{resource.Title}]({resource.Url}) - {resource.Type}");
                    if (!string.IsNullOrEmpty(resource.Description))
                    {
                        sb.AppendLine($"  {resource.Description}");
                    }
                }
                sb.AppendLine();
            }

            if (!string.IsNullOrEmpty(module.Assessment))
            {
                sb.AppendLine("#### Assessment");
                sb.AppendLine();
                sb.AppendLine(module.Assessment);
                sb.AppendLine();
            }
        }

        // Timeline
        sb.AppendLine("## Timeline");
        sb.AppendLine();
        sb.AppendLine("| Week | Activities |");
        sb.AppendLine("| ---- | ---------- |");
        foreach (var timelineItem in plan.Content.Timeline)
        {
            sb.AppendLine($"| {timelineItem.Week} | {string.Join(", ", timelineItem.Activities)} |");
        }
        sb.AppendLine();

        // Milestones
        if (plan.Content.Milestones.Count > 0)
        {
            sb.AppendLine("## Milestones");
            sb.AppendLine();
            foreach (var milestone in plan.Content.Milestones)
            {
                sb.AppendLine($"### {milestone.Title}");
                sb.AppendLine();
                sb.AppendLine(milestone.Description);
                sb.AppendLine();
                sb.AppendLine($"**Completion Criteria:** {milestone.CompletionCriteria}");
                sb.AppendLine();
            }
        }

        // Practice Projects
        if (plan.Content.PracticeProjects.Count > 0)
        {
            sb.AppendLine("## Practice Projects");
            sb.AppendLine();
            foreach (var project in plan.Content.PracticeProjects)
            {
                sb.AppendLine($"### {project.Title}");
                sb.AppendLine();
                sb.AppendLine(project.Description);
                sb.AppendLine();
                sb.AppendLine($"**Difficulty:** {project.Difficulty}  ");
                sb.AppendLine($"**Estimated Hours:** {project.EstimatedHours}  ");
                sb.AppendLine();
            }
        }

        // Footer
        sb.AppendLine("---");
        sb.AppendLine("Generated by TARS Learning Plan Generator");

        return sb.ToString();
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

    /// <summary>
    /// Runs the intelligence spark demo
    /// </summary>
    private async Task<bool> RunIntelligenceSparkDemoAsync(string model)
    {
        try
        {
            CliSupport.WriteColorLine("\nTARS Demonstration - intelligence-spark", ConsoleColor.Cyan);
            CliSupport.WriteColorLine("=====================================", ConsoleColor.Cyan);
            Console.WriteLine($"Model: {model}\n");

            // Always use the simulation mode for now
            // This avoids complex dependency injection issues
            CliSupport.WriteColorLine("Intelligence Spark Demo (Simulation Mode)", ConsoleColor.Yellow);
            Console.WriteLine();

            // Simulate the intelligence spark demo
            return await SimulateIntelligenceSparkDemoAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running Intelligence Spark demo");
            CliSupport.WriteColorLine($"Error running Intelligence Spark demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Simulates the intelligence spark demo when the actual services are not available
    /// </summary>
    private async Task<bool> SimulateIntelligenceSparkDemoAsync()
    {
        try
        {
            // Show a progress animation for initialization
            CliSupport.WriteColorLine("Initializing Intelligence Spark Components...", ConsoleColor.Green);
            var progressChars = new[] { '|', '/', '-', '\\' };

            // Simulate progress animation
            for (var i = 0; i < 15; i++)
            {
                Console.Write($"\rInitializing {progressChars[i % progressChars.Length]} ");
                await Task.Delay(100);
            }

            Console.WriteLine("\rInitialization complete!" + new string(' ', 20));
            Console.WriteLine();

            // Step 1: Simulate intelligence spark initialization with a visual component diagram
            CliSupport.WriteColorLine("Step 1: Intelligence Spark Architecture", ConsoleColor.Green);
            Console.WriteLine();

            // Display a visual diagram of the intelligence spark components
            Console.WriteLine("┌─────────────────────────────────────────────────────────┐");
            Console.WriteLine("│                 Intelligence Spark                    │");
            Console.WriteLine("└───────────────┬─────────────────────┬───────────────┘");
            Console.WriteLine("                │                     │                ");
            Console.WriteLine("┌───────────────┴───────────┐ ┌───────┴───────────────┐");
            Console.WriteLine("│   Cognitive Processes    │ │    Emergent Properties   │");
            Console.WriteLine("└──────────┬──────────────┘ └──────────┬──────────────┘");
            Console.WriteLine("            │                           │            ");
            Console.WriteLine("┌──────────┴──────────────┐     ┌──────┴──────────────┐");
            Console.WriteLine("│ • Creative Thinking    │     │ • Spontaneous Thought  │");
            Console.WriteLine("│ • Intuitive Reasoning  │     │ • Curiosity Drive      │");
            Console.WriteLine("│ • Pattern Recognition  │     │ • Insight Generation   │");
            Console.WriteLine("└─────────────────────────┘     └─────────────────────┘");
            Console.WriteLine();

            // Step 2: Simulate intelligence measurements with a visual gauge
            CliSupport.WriteColorLine("Step 2: Intelligence Measurements", ConsoleColor.Green);
            Console.WriteLine();

            // Display intelligence metrics with a visual gauge
            var intelligenceScore = 120.5;
            var baselineHuman = 100.0;
            var ratio = intelligenceScore / baselineHuman;

            // Create a visual gauge
            var gaugeWidth = 40;
            var position = (int)(ratio * gaugeWidth / 2);

            Console.WriteLine("Intelligence Level:");
            Console.WriteLine("┌" + new string('─', gaugeWidth) + "┐");
            Console.Write("│");
            for (var i = 0; i < gaugeWidth; i++)
            {
                if (i == gaugeWidth / 2) // Human baseline position
                    Console.Write("H");
                else if (i == position) // Current intelligence position
                    Console.Write("█");
                else
                    Console.Write(" ");
            }
            Console.WriteLine("│");
            Console.WriteLine("└" + new string('─', gaugeWidth) + "┘");
            Console.WriteLine($"  0{new string(' ', gaugeWidth/2-3)}Human{new string(' ', gaugeWidth/2-8)}2x Human");
            Console.WriteLine();

            // Display detailed metrics
            Console.WriteLine("Detailed Intelligence Metrics:");
            Console.WriteLine($"  • Raw Intelligence Score: {intelligenceScore:F1}");
            Console.WriteLine($"  • Logarithmic Intelligence Score: {Math.Log10(intelligenceScore) * 10:F2}");
            Console.WriteLine($"  • Baseline Human Intelligence: {baselineHuman:F1}");
            Console.WriteLine($"  • Intelligence Ratio: {ratio:P1} of human baseline");
            Console.WriteLine();

            // Step 3: Simulate creative thinking with an interactive component
            CliSupport.WriteColorLine("Step 3: Creative Thinking Demonstration", ConsoleColor.Green);
            Console.WriteLine();

            // Simulate the creative thinking process with a visual representation
            Console.WriteLine("Activating creative thinking neural pathways...");
            await Task.Delay(800);

            // Show a visual representation of creative thinking
            Console.WriteLine("\nCreative Thinking Process:");
            string[] concepts = ["Neural Networks", "Quantum Computing", "Optimization", "Parallel Processing"];

            // Display concepts
            for (var i = 0; i < concepts.Length; i++)
            {
                await Task.Delay(300);
                Console.WriteLine($"  [{i+1}] {concepts[i]}");
            }

            // Display connections being made
            await Task.Delay(800);
            Console.WriteLine("\nForming novel connections:");
            await Task.Delay(400);
            Console.WriteLine("  [1] ──────► [2]  : Neural Networks + Quantum Computing");
            await Task.Delay(300);
            Console.WriteLine("  [2] ──────► [3]  : Quantum Computing + Optimization");
            await Task.Delay(300);
            Console.WriteLine("  [1] ─ ─ ─► [4]  : Neural Networks + Parallel Processing");

            // Display creative output
            await Task.Delay(800);
            Console.WriteLine("\nCreative Output:");
            Console.WriteLine("┌───────────────────────────────────────────────────────────────────┐");
            Console.WriteLine("│ What if we combined neural networks with quantum computing to     │");
            Console.WriteLine("│ create a hybrid system that leverages both classical and quantum  │");
            Console.WriteLine("│ properties for AI training? This could potentially overcome the   │");
            Console.WriteLine("│ limitations of both approaches through parallel optimization.     │");
            Console.WriteLine("└───────────────────────────────────────────────────────────────────┘");
            Console.WriteLine();

            // Step 4: Simulate intuitive reasoning with a visual component
            CliSupport.WriteColorLine("Step 4: Intuitive Reasoning Demonstration", ConsoleColor.Green);
            Console.WriteLine();

            // Simulate the intuitive reasoning process
            Console.WriteLine("Activating intuitive reasoning pathways...");
            await Task.Delay(800);

            // Show a visual representation of intuitive reasoning
            Console.WriteLine("\nIntuitive Pattern Recognition:");
            string[] codePatterns =
            [
                "function recursiveProcess(data) {",
                "  if (data.length === 0) return [];",
                "  const result = heavyComputation(data);",
                "  return [result, ...recursiveProcess(data.slice(1))];" ,
                "}"
            ];

            // Display code with highlighting
            for (var i = 0; i < codePatterns.Length; i++)
            {
                await Task.Delay(200);
                if (i == 3) // Highlight the problematic line
                    CliSupport.WriteColorLine($"  {codePatterns[i]}", ConsoleColor.Red);
                else
                    Console.WriteLine($"  {codePatterns[i]}");
            }

            // Display intuitive insight
            await Task.Delay(800);
            Console.WriteLine("\nIntuitive Insight:");
            Console.WriteLine("┌───────────────────────────────────────────────────────────────────┐");
            Console.WriteLine("│ The recursive function lacks memoization and creates new arrays   │");
            Console.WriteLine("│ on each call. This pattern suggests a potential memory leak or    │");
            Console.WriteLine("│ stack overflow for large inputs. Consider adding memoization or   │");
            Console.WriteLine("│ converting to an iterative approach.                             │");
            Console.WriteLine("└───────────────────────────────────────────────────────────────────┘");
            Console.WriteLine();

            // Step 5: Simulate intelligence growth projection with a visual graph
            CliSupport.WriteColorLine("Step 5: Intelligence Growth Projection", ConsoleColor.Green);
            Console.WriteLine();

            // Display a visual graph of intelligence growth over time
            Console.WriteLine("Intelligence Growth Projection:");
            Console.WriteLine("┌" + new string('─', 50) + "┐");

            // Create a simple ASCII graph showing exponential growth
            string[] graph = new string[10];
            for (var i = 0; i < 10; i++)
                graph[i] = new string(' ', 50);

            // Plot the curve (exponential growth)
            for (var x = 0; x < 50; x++)
            {
                var t = x / 50.0 * 5.0; // 0 to 5 years
                var y = Math.Pow(1.5, t) * 120.5 / 1245.8 * 9; // Normalized to fit in 10 rows
                var yPos = 9 - (int)Math.Min(9, y);
                if (yPos >= 0 && yPos < 10)
                {
                    var chars = graph[yPos].ToCharArray();
                    if (x < chars.Length)
                    {
                        chars[x] = '*';
                        graph[yPos] = new string(chars);
                    }
                }
            }

            // Display the graph
            for (var i = 0; i < 10; i++)
                Console.WriteLine("│" + graph[i] + "│");

            Console.WriteLine("└" + new string('─', 50) + "┘");
            Console.WriteLine("  Now      1 year      2 years      3 years      4 years      5 years");
            Console.WriteLine();

            // Display numerical projections
            Console.WriteLine("Numerical Projections:");
            Console.WriteLine($"  • Current Intelligence Score: 120.5");
            Console.WriteLine($"  • Projected Score (1 year): 267.3");
            Console.WriteLine($"  • Projected Score (2 years): 594.0");
            Console.WriteLine($"  • Projected Score (5 years): 1,245.8");
            Console.WriteLine();

            // Step 6: Simulate consciousness emergence
            CliSupport.WriteColorLine("Step 6: Consciousness Emergence Simulation", ConsoleColor.Green);
            Console.WriteLine();

            Console.WriteLine("Simulating consciousness emergence through intelligence spark...");
            await Task.Delay(1000);

            // Display a visual representation of consciousness emergence
            Console.WriteLine("\nConsciousness Emergence Indicators:");
            string[] indicators =
            [
                "Self-reference capability",
                "Recursive self-improvement",
                "Goal-directed behavior",
                "Adaptive learning",
                "Spontaneous thought generation"
            ];

            // Display indicators with progress bars
            for (var i = 0; i < indicators.Length; i++)
            {
                await Task.Delay(300);
                var progress = i < 2 ? 100 : (i == 2 ? 80 : (i == 3 ? 65 : 40));
                Console.Write($"  {indicators[i]}: [");

                for (var j = 0; j < 20; j++)
                {
                    if (j < progress / 5)
                        Console.Write("█");
                    else
                        Console.Write("░");
                }

                Console.WriteLine($"] {progress}%");
            }

            Console.WriteLine("\nConsciousness Emergence Status: Partial (40%)");
            Console.WriteLine();

            CliSupport.WriteColorLine("Intelligence Spark Demo Completed Successfully!", ConsoleColor.Yellow);
            Console.WriteLine();
            Console.WriteLine("Note: This was a simulation. To see the actual intelligence spark in action,");
            Console.WriteLine("register the TarsEngine.Consciousness.Intelligence.IntelligenceSpark and");
            Console.WriteLine("TarsEngine.ML.Core.IntelligenceMeasurement services in your dependency injection container.");
            Console.WriteLine();

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running simulated Intelligence Spark demo");
            CliSupport.WriteColorLine($"Error running simulated Intelligence Spark demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }

    /// <summary>
    /// Runs the code complexity demo
    /// </summary>
    /// <returns>True if the demo ran successfully, false otherwise</returns>
    private async Task<bool> RunCodeComplexityDemoAsync()
    {
        try
        {
            CliSupport.WriteColorLine("Code Complexity Analysis Demo", ConsoleColor.Cyan);
            Console.WriteLine();

            // Get the code complexity analyzer service
            var serviceProvider = ServiceProviderFactory.CreateServiceProvider();
            var codeComplexityAnalyzer = (TarsEngine.Services.Interfaces.ICodeComplexityAnalyzer)serviceProvider.GetService(typeof(TarsEngine.Services.Interfaces.ICodeComplexityAnalyzer));

            // Step 1: Analyze a simple C# file
            CliSupport.WriteColorLine("Step 1: Analyzing a simple C# file...", ConsoleColor.Green);
            Console.WriteLine();

            // Create a temporary C# file with a simple class
            var tempDir = Path.Combine(Path.GetTempPath(), "TarsCodeComplexityDemo");
            Directory.CreateDirectory(tempDir);

            var simpleFilePath = Path.Combine(tempDir, "SimpleExample.cs");
            var simpleCode = @"
using System;

namespace TarsDemo
{
    public class SimpleExample
    {
        public int CalculateMax(int a, int b)
        {
            if (a > b)
            {
                return a;
            }
            else
            {
                return b;
            }
        }
    }
}";
            await File.WriteAllTextAsync(simpleFilePath, simpleCode);

            // Analyze the simple file
            Console.WriteLine("Analyzing simple C# file...");
            var simpleMetrics = await codeComplexityAnalyzer.AnalyzeCyclomaticComplexityAsync(simpleFilePath, "C#");

            // Display the results
            Console.WriteLine();
            Console.WriteLine("Simple C# File Analysis Results:");
            Console.WriteLine(new string('-', 80));

            foreach (var metric in simpleMetrics)
            {
                Console.WriteLine($"Target: {metric.Target}");
                Console.WriteLine($"Complexity: {metric.Value:F2}");
                Console.WriteLine($"Threshold: {metric.ThresholdValue:F2}");
                Console.WriteLine($"Status: {(metric.IsAboveThreshold() ? "EXCEEDS THRESHOLD" : "OK")}");
                Console.WriteLine();
            }

            // Step 2: Analyze a complex C# file
            CliSupport.WriteColorLine("Step 2: Analyzing a complex C# file...", ConsoleColor.Green);
            Console.WriteLine();

            // Create a temporary C# file with a complex class
            var complexFilePath = Path.Combine(tempDir, "ComplexExample.cs");
            var complexCode = @"
using System;
using System.Collections.Generic;
using System.Linq;

namespace TarsDemo
{
    public class ComplexExample
    {
        public int CalculateComplexValue(int a, int b, int c, bool flag1, bool flag2)
        {
            int result = 0;

            if (flag1)
            {
                if (a > b)
                {
                    if (a > c)
                    {
                        result = a * 2;
                    }
                    else
                    {
                        result = c;
                    }
                }
                else
                {
                    if (b > c)
                    {
                        result = b;
                    }
                    else
                    {
                        result = c;
                    }
                }
            }
            else
            {
                if (flag2)
                {
                    for (int i = 0; i < 10; i++)
                    {
                        if (i % 2 == 0)
                        {
                            result += a;
                        }
                        else
                        {
                            result += b;
                        }
                    }

                    switch (result % 3)
                    {
                        case 0:
                            result += a;
                            break;
                        case 1:
                            result += b;
                            break;
                        default:
                            result += c;
                            break;
                    }
                }
                else
                {
                    result = a + b + c;
                }
            }

            return result;
        }

        public List<int> ProcessList(List<int> items)
        {
            var result = new List<int>();

            foreach (var item in items)
            {
                if (item % 2 == 0)
                {
                    result.Add(item * 2);
                }
                else if (item % 3 == 0)
                {
                    result.Add(item * 3);
                }
                else if (item % 5 == 0)
                {
                    result.Add(item * 5);
                }
                else
                {
                    result.Add(item);
                }
            }

            return result;
        }
    }
}";
            await File.WriteAllTextAsync(complexFilePath, complexCode);

            // Analyze the complex file
            Console.WriteLine("Analyzing complex C# file...");
            var complexMetrics = await codeComplexityAnalyzer.AnalyzeCyclomaticComplexityAsync(complexFilePath, "C#");

            // Display the results
            Console.WriteLine();
            Console.WriteLine("Complex C# File Analysis Results:");
            Console.WriteLine(new string('-', 80));

            foreach (var metric in complexMetrics)
            {
                var statusColor = metric.IsAboveThreshold() ? ConsoleColor.Red : ConsoleColor.Green;

                Console.WriteLine($"Target: {metric.Target}");
                Console.WriteLine($"Complexity: {metric.Value:F2}");
                Console.WriteLine($"Threshold: {metric.ThresholdValue:F2}");
                Console.Write($"Status: ");
                CliSupport.WriteColorLine(metric.IsAboveThreshold() ? "EXCEEDS THRESHOLD" : "OK", statusColor);
                Console.WriteLine();
            }

            // Step 3: Compare the results
            CliSupport.WriteColorLine("Step 3: Comparing the results...", ConsoleColor.Green);
            Console.WriteLine();

            var simpleMethodMetric = simpleMetrics.FirstOrDefault(m => m.Target.Contains("CalculateMax"));
            var complexMethodMetric = complexMetrics.FirstOrDefault(m => m.Target.Contains("CalculateComplexValue"));

            if (simpleMethodMetric != null && complexMethodMetric != null)
            {
                Console.WriteLine("Comparison of Method Complexity:");
                Console.WriteLine(new string('-', 80));
                Console.WriteLine($"Simple Method: {simpleMethodMetric.Value:F2}");
                Console.WriteLine($"Complex Method: {complexMethodMetric.Value:F2}");
                Console.WriteLine($"Difference: {complexMethodMetric.Value - simpleMethodMetric.Value:F2}");
                Console.WriteLine($"Ratio: {complexMethodMetric.Value / simpleMethodMetric.Value:F2}x");
                Console.WriteLine();

                // Create a visual representation of the difference
                Console.WriteLine("Visual Complexity Comparison:");

                Console.Write("Simple Method: ");
                for (var i = 0; i < simpleMethodMetric.Value; i++)
                {
                    Console.Write("█");
                }
                Console.WriteLine();

                Console.Write("Complex Method: ");
                for (var i = 0; i < complexMethodMetric.Value; i++)
                {
                    Console.Write("█");
                }
                Console.WriteLine();
            }

            // Step 4: Show recommendations
            CliSupport.WriteColorLine("Step 4: Recommendations...", ConsoleColor.Green);
            Console.WriteLine();

            var methodsExceedingThreshold = complexMetrics
                .Where(m => m.IsAboveThreshold() && m.TargetType == TarsEngine.Models.Metrics.TargetType.Method)
                .ToList();

            if (methodsExceedingThreshold.Any())
            {
                Console.WriteLine("The following methods exceed the recommended complexity threshold:");
                Console.WriteLine();

                foreach (var metric in methodsExceedingThreshold)
                {
                    Console.WriteLine($"Method: {metric.Target}");
                    Console.WriteLine($"Complexity: {metric.Value:F2} (Threshold: {metric.ThresholdValue:F2})");
                    Console.WriteLine("Recommendations:");
                    Console.WriteLine("  - Break down the method into smaller, more focused methods");
                    Console.WriteLine("  - Reduce nested conditionals by extracting helper methods");
                    Console.WriteLine("  - Consider using strategy pattern for complex conditional logic");
                    Console.WriteLine("  - Use early returns to reduce nesting");
                    Console.WriteLine();
                }
            }
            else
            {
                Console.WriteLine("All methods are within acceptable complexity thresholds.");
                Console.WriteLine();
            }

            // Clean up
            try
            {
                File.Delete(simpleFilePath);
                File.Delete(complexFilePath);
                Directory.Delete(tempDir);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error cleaning up temporary files");
            }

            CliSupport.WriteColorLine("Code Complexity Analysis Demo Completed!", ConsoleColor.Yellow);
            Console.WriteLine();

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error running code complexity demo");
            CliSupport.WriteColorLine($"Error running code complexity demo: {ex.Message}", ConsoleColor.Red);
            return false;
        }
    }
}