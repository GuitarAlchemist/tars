using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Configuration;

namespace TarsCli.Services;

/// <summary>
/// Service for deep thinking about TARS explorations and generating new insights
/// </summary>
public class DeepThinkingService
{
    private readonly ILogger<DeepThinkingService> _logger;
    private readonly IConfiguration _configuration;
    private readonly OllamaService _ollamaService;
    private readonly ExplorationReflectionService2 _reflectionService;
    private readonly string _explorationsDirectory;

    public DeepThinkingService(
        ILogger<DeepThinkingService> logger,
        IConfiguration configuration,
        OllamaService ollamaService,
        ExplorationReflectionService2 reflectionService)
    {
        _logger = logger;
        _configuration = configuration;
        _ollamaService = ollamaService;
        _reflectionService = reflectionService;

        // Get explorations directory from configuration or use default
        _explorationsDirectory = _configuration.GetValue<string>("Tars:Explorations:Directory",
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "docs", "Explorations"));

        _logger.LogInformation($"DeepThinkingService initialized with explorations directory: {_explorationsDirectory}");
    }

    /// <summary>
    /// Get all exploration versions
    /// </summary>
    public List<string> GetExplorationVersions()
    {
        try
        {
            if (!Directory.Exists(_explorationsDirectory))
            {
                _logger.LogWarning($"Explorations directory not found: {_explorationsDirectory}");
                return new List<string>();
            }

            var versions = Directory.GetDirectories(_explorationsDirectory)
                .Where(d => Path.GetFileName(d).StartsWith("v"))
                .Select(d => Path.GetFileName(d))
                .OrderBy(v => v)
                .ToList();

            _logger.LogInformation($"Found {versions.Count} exploration versions: {string.Join(", ", versions)}");
            return versions;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting exploration versions from {_explorationsDirectory}");
            return new List<string>();
        }
    }

    /// <summary>
    /// Get the latest exploration version
    /// </summary>
    public string GetLatestExplorationVersion()
    {
        var versions = GetExplorationVersions();
        return versions.Count > 0 ? versions.Last() : "v1";
    }

    /// <summary>
    /// Get the next exploration version
    /// </summary>
    public string GetNextExplorationVersion()
    {
        var latestVersion = GetLatestExplorationVersion();
        var versionNumber = int.Parse(latestVersion.Substring(1));
        return $"v{versionNumber + 1}";
    }

    /// <summary>
    /// Generate related topics for a given topic
    /// </summary>
    /// <param name="topic">The main topic</param>
    /// <param name="count">Number of related topics to generate</param>
    /// <param name="model">The model to use for generation</param>
    /// <returns>List of related topics</returns>
    public async Task<List<string>> GenerateRelatedTopicsAsync(string topic, int count, string model = "llama3")
    {
        _logger.LogInformation($"Generating {count} related topics for '{topic}' using model {model}");

        try
        {
            var prompt = @$"Generate {count} related topics for deep thinking about '{topic}'.

These should be interesting, thought-provoking variations or extensions of the main topic.
Return only the list of topics, one per line, with no numbering or bullets.";

            var response = await _ollamaService.GenerateCompletion(prompt, model);

            // Parse the response into a list of topics
            var topics = response.Split('\n', StringSplitOptions.RemoveEmptyEntries)
                .Select(t => t.Trim())
                .Where(t => !string.IsNullOrWhiteSpace(t))
                .Take(count)
                .ToList();

            _logger.LogInformation($"Generated {topics.Count} related topics for '{topic}'");

            return topics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating related topics for '{topic}'");
            return new List<string>();
        }
    }

    /// <summary>
    /// Generate a deep thinking exploration based on existing explorations
    /// </summary>
    public async Task<DeepThinkingResult> GenerateDeepThinkingExplorationAsync(
        string topic,
        string? baseExplorationPath = null,
        string model = "llama3")
    {
        try
        {
            _logger.LogInformation($"Generating deep thinking exploration on topic: {topic}");

            // Get the next version
            var nextVersion = GetNextExplorationVersion();

            // Prepare the context for deep thinking
            var context = await PrepareDeepThinkingContextAsync(topic, baseExplorationPath);

            // Generate the deep thinking prompt
            var prompt = GenerateDeepThinkingPrompt(topic, context);

            // Generate the exploration content
            _logger.LogInformation($"Generating deep thinking exploration using model: {model}");
            var content = await _ollamaService.GenerateCompletion(prompt, model);

            // Create the result
            var result = new DeepThinkingResult
            {
                Topic = topic,
                Version = nextVersion,
                Content = content,
                BaseExplorationPath = baseExplorationPath,
                Timestamp = DateTime.Now
            };

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating deep thinking exploration on topic: {topic}");
            throw;
        }
    }

    /// <summary>
    /// Save a deep thinking exploration to a file
    /// </summary>
    public async Task<string> SaveDeepThinkingExplorationAsync(DeepThinkingResult result)
    {
        try
        {
            // Create the directory for the next version if it doesn't exist
            var versionDirectory = Path.Combine(_explorationsDirectory, result.Version);
            var chatsDirectory = Path.Combine(versionDirectory, "Chats");

            if (!Directory.Exists(versionDirectory))
            {
                Directory.CreateDirectory(versionDirectory);
                _logger.LogInformation($"Created directory for version: {result.Version}");
            }

            if (!Directory.Exists(chatsDirectory))
            {
                Directory.CreateDirectory(chatsDirectory);
                _logger.LogInformation($"Created Chats directory for version: {result.Version}");
            }

            // Sanitize the topic for use in a filename
            var sanitizedTopic = SanitizeForFileName(result.Topic);

            // Create the file path
            var filePath = Path.Combine(chatsDirectory, $"DeepThinking-{sanitizedTopic}.md");

            // Format the content as a markdown file
            var markdown = FormatDeepThinkingAsMarkdown(result);

            // Write the file
            await File.WriteAllTextAsync(filePath, markdown);

            _logger.LogInformation($"Saved deep thinking exploration to: {filePath}");
            return filePath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error saving deep thinking exploration on topic: {result.Topic}");
            throw;
        }
    }

    /// <summary>
    /// Generate a series of deep thinking explorations on related topics
    /// </summary>
    public async Task<List<string>> GenerateDeepThinkingSeriesAsync(
        string baseTopic,
        int count = 3,
        string model = "llama3")
    {
        try
        {
            _logger.LogInformation($"Generating deep thinking series on base topic: {baseTopic}");

            // Generate related topics
            var topics = await GenerateRelatedTopicsInternalAsync(baseTopic, count, model);

            // Generate and save explorations for each topic
            var filePaths = new List<string>();

            foreach (var topic in topics)
            {
                var result = await GenerateDeepThinkingExplorationAsync(topic, baseExplorationPath: null, model);
                var filePath = await SaveDeepThinkingExplorationAsync(result);
                filePaths.Add(filePath);
            }

            return filePaths;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating deep thinking series on base topic: {baseTopic}");
            throw;
        }
    }

    /// <summary>
    /// Generate a deep thinking evolution based on an existing exploration
    /// </summary>
    public async Task<string> GenerateDeepThinkingEvolutionAsync(
        string explorationPath,
        string model = "llama3")
    {
        try
        {
            _logger.LogInformation($"Generating deep thinking evolution based on: {explorationPath}");

            // Parse the exploration file
            var exploration = await _reflectionService.ParseExplorationFileAsync(explorationPath);

            // Extract the topic from the title
            var topic = exploration.Title;

            // Generate the deep thinking exploration
            var result = await GenerateDeepThinkingExplorationAsync(topic, explorationPath, model);

            // Save the exploration
            var filePath = await SaveDeepThinkingExplorationAsync(result);

            return filePath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating deep thinking evolution based on: {explorationPath}");
            throw;
        }
    }

    /// <summary>
    /// Prepare the context for deep thinking
    /// </summary>
    private async Task<string> PrepareDeepThinkingContextAsync(string topic, string baseExplorationPath)
    {
        var context = new StringBuilder();

        // If a base exploration is provided, include its content
        if (!string.IsNullOrEmpty(baseExplorationPath) && File.Exists(baseExplorationPath))
        {
            var exploration = await _reflectionService.ParseExplorationFileAsync(baseExplorationPath);

            context.AppendLine("# Base Exploration");
            context.AppendLine($"Title: {exploration.Title}");
            context.AppendLine($"Created: {exploration.Created}");
            context.AppendLine();
            context.AppendLine("## Prompt");
            context.AppendLine(exploration.Prompt);
            context.AppendLine();
            context.AppendLine("## Response Summary");
            context.AppendLine(TruncateText(exploration.Response, 2000));
            context.AppendLine();
        }

        // Find related explorations based on the topic
        var relatedExplorations = await FindRelatedExplorationsAsync(topic, 3);

        if (relatedExplorations.Count > 0)
        {
            context.AppendLine("# Related Explorations");

            foreach (var explorationFile in relatedExplorations)
            {
                var parsedExploration = await _reflectionService.ParseExplorationFileAsync(explorationFile.FilePath);

                context.AppendLine($"## {parsedExploration.Title}");
                context.AppendLine($"Created: {parsedExploration.Created}");
                context.AppendLine();
                context.AppendLine("### Prompt");
                context.AppendLine(parsedExploration.Prompt);
                context.AppendLine();
                context.AppendLine("### Response Summary");
                context.AppendLine(TruncateText(parsedExploration.Response, 1000));
                context.AppendLine();
            }
        }

        return context.ToString();
    }

    /// <summary>
    /// Generate a deep thinking prompt
    /// </summary>
    private string GenerateDeepThinkingPrompt(string topic, string context)
    {
        var prompt = new StringBuilder();

        prompt.AppendLine($"You are TARS, an advanced AI system engaged in deep thinking about the topic: \"{topic}\".");
        prompt.AppendLine();
        prompt.AppendLine("Your task is to generate a new, more advanced exploration that builds upon existing knowledge and pushes the boundaries of understanding on this topic.");
        prompt.AppendLine();
        prompt.AppendLine("Consider the following context from previous explorations:");
        prompt.AppendLine();
        prompt.AppendLine(context);
        prompt.AppendLine();
        prompt.AppendLine("Now, generate a deep, insightful exploration on this topic that:");
        prompt.AppendLine("1. Builds upon previous knowledge and insights");
        prompt.AppendLine("2. Introduces new perspectives or approaches");
        prompt.AppendLine("3. Makes connections to other domains or concepts");
        prompt.AppendLine("4. Identifies implications for the TARS project");
        prompt.AppendLine("5. Suggests concrete applications or implementations");
        prompt.AppendLine();
        prompt.AppendLine("Your exploration should be comprehensive, well-structured, and push the boundaries of current understanding. Think deeply and creatively about the topic.");

        return prompt.ToString();
    }

    /// <summary>
    /// Format a deep thinking result as markdown
    /// </summary>
    private string FormatDeepThinkingAsMarkdown(DeepThinkingResult result)
    {
        var markdown = new StringBuilder();

        markdown.AppendLine($"# Deep Thinking: {result.Topic}");
        markdown.AppendLine();
        markdown.AppendLine($"**Version:** {result.Version}");
        markdown.AppendLine($"**Generated:** {result.Timestamp.ToString("yyyy-MM-dd HH:mm:ss")}");

        if (!string.IsNullOrEmpty(result.BaseExplorationPath))
        {
            markdown.AppendLine($"**Based on:** {Path.GetFileName(result.BaseExplorationPath)}");
        }

        markdown.AppendLine();
        markdown.AppendLine("## Content");
        markdown.AppendLine();
        markdown.AppendLine(result.Content);

        return markdown.ToString();
    }

    /// <summary>
    /// Find explorations related to a topic
    /// </summary>
    private async Task<List<ExplorationFile>> FindRelatedExplorationsAsync(string topic, int count = 3)
    {
        try
        {
            // Get all exploration files
            var allExplorations = new List<ExplorationFile>();

            foreach (var version in GetExplorationVersions())
            {
                var versionExplorations = _reflectionService.GetExplorationFiles($"{version}/Chats");
                allExplorations.AddRange(versionExplorations);
            }

            // If there are no explorations, return an empty list
            if (allExplorations.Count == 0)
            {
                return new List<ExplorationFile>();
            }

            // Parse each exploration to get its content
            var explorationContents = new List<(ExplorationFile File, string Content)>();

            foreach (var exploration in allExplorations)
            {
                var parsedExploration = await _reflectionService.ParseExplorationFileAsync(exploration.FilePath);
                var content = $"{parsedExploration.Title} {parsedExploration.Prompt} {TruncateText(parsedExploration.Response, 500)}";
                explorationContents.Add((exploration, content));
            }

            // Calculate relevance scores based on keyword matching
            var keywords = topic.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var scoredExplorations = new List<(ExplorationFile File, double Score)>();

            foreach (var (file, content) in explorationContents)
            {
                double score = 0;

                foreach (var keyword in keywords)
                {
                    var regex = new Regex($"\\b{Regex.Escape(keyword)}\\b", RegexOptions.IgnoreCase);
                    var matches = regex.Matches(content);
                    score += matches.Count;
                }

                scoredExplorations.Add((file, score));
            }

            // Sort by relevance score and take the top 'count'
            var relatedExplorations = scoredExplorations
                .OrderByDescending(x => x.Score)
                .Take(count)
                .Select(x => x.File)
                .ToList();

            return relatedExplorations;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error finding explorations related to topic: {topic}");
            return new List<ExplorationFile>();
        }
    }

    /// <summary>
    /// Generate related topics for a series of deep thinking explorations
    /// </summary>
    private async Task<List<string>> GenerateRelatedTopicsInternalAsync(string baseTopic, int count, string model)
    {
        try
        {
            var prompt = $@"You are TARS, an advanced AI system. Based on the topic ""{baseTopic}"", generate {count} related but distinct topics for deep thinking explorations. These topics should:

1. Build upon the base topic
2. Explore different aspects or dimensions of the subject
3. Be specific enough for focused exploration
4. Be relevant to the TARS project (AI, machine learning, cognitive architectures)

Format your response as a numbered list with just the topic titles.";

            var response = await _ollamaService.GenerateCompletion(prompt, model);

            // Parse the response to extract topics
            var topics = new List<string>();
            var lines = response.Split('\n', StringSplitOptions.RemoveEmptyEntries);

            foreach (var line in lines)
            {
                // Look for numbered list items (e.g., "1. Topic title")
                var match = Regex.Match(line.Trim(), @"^\d+\.\s+(.+)$");
                if (match.Success)
                {
                    topics.Add(match.Groups[1].Value.Trim());
                }
            }

            // If we couldn't parse any topics, or didn't get enough, add the base topic
            if (topics.Count == 0)
            {
                topics.Add(baseTopic);
            }

            // Ensure we have exactly the requested number of topics
            while (topics.Count < count)
            {
                topics.Add($"{baseTopic} - Variation {topics.Count + 1}");
            }

            // If we have too many, trim the list
            if (topics.Count > count)
            {
                topics = topics.Take(count).ToList();
            }

            return topics;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error generating related topics for base topic: {baseTopic}");

            // Return fallback topics
            var fallbackTopics = new List<string>();
            for (var i = 0; i < count; i++)
            {
                fallbackTopics.Add($"{baseTopic} - Aspect {i + 1}");
            }

            return fallbackTopics;
        }
    }

    /// <summary>
    /// Sanitize a string for use in a file name
    /// </summary>
    private string SanitizeForFileName(string input)
    {
        // Replace invalid characters with underscores
        var invalidChars = Path.GetInvalidFileNameChars();
        var sanitized = new string(input.Select(c => invalidChars.Contains(c) ? '_' : c).ToArray());

        // Trim to a reasonable length
        if (sanitized.Length > 50)
        {
            sanitized = sanitized.Substring(0, 50);
        }

        return sanitized;
    }

    /// <summary>
    /// Truncate text to a maximum length
    /// </summary>
    private string TruncateText(string text, int maxLength)
    {
        if (string.IsNullOrEmpty(text))
        {
            return string.Empty;
        }

        if (text.Length <= maxLength)
        {
            return text;
        }

        return text.Substring(0, maxLength) + "...";
    }
}

/// <summary>
/// Result of a deep thinking exploration
/// </summary>
public class DeepThinkingResult
{
    /// <summary>
    /// The topic of the exploration
    /// </summary>
    public required string Topic { get; set; }

    /// <summary>
    /// The version of the exploration
    /// </summary>
    public required string Version { get; set; }

    /// <summary>
    /// The content of the exploration
    /// </summary>
    public required string Content { get; set; }

    /// <summary>
    /// The path to the base exploration, if any
    /// </summary>
    public string? BaseExplorationPath { get; set; }

    /// <summary>
    /// The timestamp of the exploration
    /// </summary>
    public DateTime Timestamp { get; set; }
}