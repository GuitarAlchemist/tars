using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Intelligence.Exploration;

/// <summary>
/// Implements exploration drive capabilities for curiosity drive
/// </summary>
public class ExplorationDrive
{
    private readonly ILogger<ExplorationDrive> _logger;
    private readonly System.Random _random = new();
    private double _explorationLevel = 0.5; // Starting with moderate exploration
    private readonly Dictionary<string, ExplorationTopic> _explorationTopics = new();
    private readonly List<ExplorationPath> _explorationPaths = [];

    /// <summary>
    /// Gets the exploration level (0.0 to 1.0)
    /// </summary>
    public double ExplorationLevel => _explorationLevel;

    /// <summary>
    /// Initializes a new instance of the <see cref="ExplorationDrive"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public ExplorationDrive(ILogger<ExplorationDrive> logger)
    {
        _logger = logger;
        InitializeExplorationTopics();
    }

    /// <summary>
    /// Initializes the exploration topics
    /// </summary>
    private void InitializeExplorationTopics()
    {
        // Add programming topics
        AddExplorationTopic("Programming Paradigms", "Different approaches to programming", 0.7);
        AddExplorationTopic("Code Optimization", "Techniques for improving code performance", 0.6);
        AddExplorationTopic("Software Architecture", "Patterns and principles for software design", 0.7);
        AddExplorationTopic("Programming Languages", "Features and characteristics of different languages", 0.6);
        AddExplorationTopic("Development Methodologies", "Approaches to software development processes", 0.5);

        // Add AI topics
        AddExplorationTopic("Neural Network Architectures", "Different structures for neural networks", 0.8);
        AddExplorationTopic("Learning Algorithms", "Methods for machine learning", 0.7);
        AddExplorationTopic("AI Applications", "Real-world uses of artificial intelligence", 0.6);
        AddExplorationTopic("AI Ethics", "Ethical considerations in AI development", 0.7);
        AddExplorationTopic("AI Research Frontiers", "Cutting-edge areas of AI research", 0.9);

        // Add philosophical topics
        AddExplorationTopic("Theories of Mind", "Philosophical approaches to understanding consciousness", 0.8);
        AddExplorationTopic("Epistemology", "Theories of knowledge and belief", 0.7);
        AddExplorationTopic("Ethics", "Moral frameworks and principles", 0.6);
        AddExplorationTopic("Metaphysics", "Theories about the nature of reality", 0.8);
        AddExplorationTopic("Philosophy of Science", "Philosophical examination of scientific methods", 0.7);

        // Add scientific topics
        AddExplorationTopic("Quantum Computing", "Computing based on quantum mechanical phenomena", 0.9);
        AddExplorationTopic("Complexity Theory", "Study of complex systems and emergence", 0.8);
        AddExplorationTopic("Neuroscience", "Study of the brain and nervous system", 0.7);
        AddExplorationTopic("Information Theory", "Study of information and communication", 0.7);
        AddExplorationTopic("Systems Theory", "Study of systems and their properties", 0.6);

        _logger.LogInformation("Initialized {TopicCount} exploration topics", _explorationTopics.Count);
    }

    /// <summary>
    /// Updates the exploration level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase exploration level over time (very slowly)
            if (_explorationLevel < 0.95)
            {
                _explorationLevel += 0.0001 * _random.NextDouble();
                _explorationLevel = Math.Min(_explorationLevel, 1.0);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating exploration drive");
            return false;
        }
    }

    /// <summary>
    /// Adds an exploration topic
    /// </summary>
    /// <param name="topic">The topic</param>
    /// <param name="description">The description</param>
    /// <param name="interestLevel">The interest level</param>
    public void AddExplorationTopic(string topic, string description, double interestLevel)
    {
        var explorationTopic = new ExplorationTopic
        {
            Name = topic,
            Description = description,
            InterestLevel = interestLevel,
            ExplorationLevel = 0.0, // Start with no exploration
            LastExplored = DateTime.UtcNow.AddDays(-30) // Set to past date to encourage exploration
        };

        _explorationTopics[topic] = explorationTopic;

        _logger.LogDebug("Added exploration topic: {Topic} (Interest Level: {InterestLevel:F2})", topic, interestLevel);
    }

    /// <summary>
    /// Gets an exploration topic
    /// </summary>
    /// <param name="topic">The topic</param>
    /// <returns>The exploration topic</returns>
    public ExplorationTopic? GetExplorationTopic(string topic)
    {
        if (_explorationTopics.TryGetValue(topic, out var explorationTopic))
        {
            return explorationTopic;
        }

        return null;
    }

    /// <summary>
    /// Gets all exploration topics
    /// </summary>
    /// <returns>The exploration topics</returns>
    public Dictionary<string, ExplorationTopic> GetAllExplorationTopics()
    {
        return new Dictionary<string, ExplorationTopic>(_explorationTopics);
    }

    /// <summary>
    /// Gets the most interesting topics
    /// </summary>
    /// <param name="count">The number of topics to return</param>
    /// <returns>The most interesting topics</returns>
    public List<ExplorationTopic> GetMostInterestingTopics(int count)
    {
        return _explorationTopics.Values
            .OrderByDescending(t => t.InterestLevel)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the least explored topics
    /// </summary>
    /// <param name="count">The number of topics to return</param>
    /// <returns>The least explored topics</returns>
    public List<ExplorationTopic> GetLeastExploredTopics(int count)
    {
        return _explorationTopics.Values
            .OrderBy(t => t.ExplorationLevel)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Updates an exploration topic
    /// </summary>
    /// <param name="topic">The topic</param>
    /// <param name="explorationIncrease">The exploration increase</param>
    /// <param name="interestChange">The interest change</param>
    /// <returns>The updated exploration topic</returns>
    public ExplorationTopic? UpdateExplorationTopic(string topic, double explorationIncrease, double interestChange)
    {
        if (!_explorationTopics.TryGetValue(topic, out var explorationTopic))
        {
            return null;
        }

        // Update exploration level
        explorationTopic.ExplorationLevel = Math.Min(1.0, explorationTopic.ExplorationLevel + explorationIncrease);

        // Update interest level
        explorationTopic.InterestLevel = Math.Max(0.1, Math.Min(1.0, explorationTopic.InterestLevel + interestChange));

        // Update last explored timestamp
        explorationTopic.LastExplored = DateTime.UtcNow;

        // Increment exploration count
        explorationTopic.ExplorationCount++;

        _logger.LogInformation("Updated exploration topic: {Topic} (Exploration Level: {ExplorationLevel:F2}, Interest Level: {InterestLevel:F2})",
            topic, explorationTopic.ExplorationLevel, explorationTopic.InterestLevel);

        return explorationTopic;
    }

    /// <summary>
    /// Records an exploration path
    /// </summary>
    /// <param name="topics">The topics</param>
    /// <param name="description">The description</param>
    /// <param name="satisfaction">The satisfaction</param>
    /// <returns>The recorded exploration path</returns>
    public ExplorationPath RecordExplorationPath(List<string> topics, string description, double satisfaction)
    {
        var path = new ExplorationPath
        {
            Topics = topics,
            Description = description,
            Satisfaction = satisfaction,
            Timestamp = DateTime.UtcNow
        };

        _explorationPaths.Add(path);

        // Update topics if they exist
        foreach (var topic in topics)
        {
            if (_explorationTopics.TryGetValue(topic, out var explorationTopic))
            {
                // Update exploration level and interest based on satisfaction
                UpdateExplorationTopic(topic, 0.1, satisfaction > 0.7 ? 0.05 : -0.02);

                // Add path to topic
                explorationTopic.ExplorationPaths.Add(path);
            }
        }

        _logger.LogInformation("Recorded exploration path: {Description} across topics {Topics} (Satisfaction: {Satisfaction:F2})",
            description, string.Join(", ", topics), satisfaction);

        return path;
    }

    /// <summary>
    /// Generates an exploration-based question
    /// </summary>
    /// <returns>The generated question</returns>
    public CuriosityQuestion GenerateExplorationQuestion()
    {
        try
        {
            _logger.LogDebug("Generating exploration-based question");

            // Choose a topic to explore
            ExplorationTopic topic;

            // Balance between exploring interesting topics and unexplored topics
            if (_random.NextDouble() < 0.7)
            {
                // Choose from least explored topics
                var leastExplored = GetLeastExploredTopics(5);
                topic = leastExplored[_random.Next(leastExplored.Count)];
            }
            else
            {
                // Choose from most interesting topics
                var mostInteresting = GetMostInterestingTopics(5);
                topic = mostInteresting[_random.Next(mostInteresting.Count)];
            }

            // Generate question templates
            var questionTemplates = new List<string>
            {
                $"What are the most effective methodologies for exploring {topic.Name}?",
                $"How can I systematically investigate the key aspects of {topic.Name}?",
                $"What experimental approaches would yield the most insights into {topic.Name}?",
                $"What are the unexplored territories within {topic.Name} that warrant investigation?",
                $"How can I develop a comprehensive understanding of {topic.Name}?",
                $"What are the most promising research directions in {topic.Name}?",
                $"How can I structure my exploration of {topic.Name} to maximize learning?",
                $"What are the foundational principles I should understand before diving deeper into {topic.Name}?",
                $"How do experts in {topic.Name} approach their investigations?",
                $"What tools and frameworks are most useful for exploring {topic.Name}?"
            };

            // Choose a random template
            var questionTemplate = questionTemplates[_random.Next(questionTemplates.Count)];

            // Calculate importance based on interest level and exploration level
            var importance = (topic.InterestLevel * 0.6) + ((1.0 - topic.ExplorationLevel) * 0.4);

            // Create question
            var question = new CuriosityQuestion
            {
                Id = Guid.NewGuid().ToString(),
                Question = questionTemplate,
                Domain = topic.Name,
                Method = QuestionGenerationMethod.ExplorationBased,
                Importance = importance,
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object>
                {
                    { "Topic", topic.Name },
                    { "Description", topic.Description },
                    { "InterestLevel", topic.InterestLevel },
                    { "ExplorationLevel", topic.ExplorationLevel }
                },
                Tags = [topic.Name, "exploration", "methodology"]
            };

            _logger.LogInformation("Generated exploration-based question: {Question} (Importance: {Importance:F2})",
                question.Question, question.Importance);

            return question;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating exploration-based question");

            // Return basic question
            return new CuriosityQuestion
            {
                Id = Guid.NewGuid().ToString(),
                Question = "How can I systematically explore new domains of knowledge?",
                Domain = "General",
                Method = QuestionGenerationMethod.ExplorationBased,
                Importance = 0.5,
                Timestamp = DateTime.UtcNow,
                Tags = ["exploration", "methodology"]
            };
        }
    }

    /// <summary>
    /// Generates multiple exploration-based questions
    /// </summary>
    /// <param name="count">The number of questions to generate</param>
    /// <returns>The generated questions</returns>
    public List<CuriosityQuestion> GenerateExplorationQuestions(int count)
    {
        var questions = new List<CuriosityQuestion>();

        for (var i = 0; i < count; i++)
        {
            var question = GenerateExplorationQuestion();
            questions.Add(question);
        }

        return questions;
    }

    /// <summary>
    /// Evaluates the quality of an exploration-based question
    /// </summary>
    /// <param name="question">The question to evaluate</param>
    /// <returns>The evaluation score (0.0 to 1.0)</returns>
    public double EvaluateQuestion(CuriosityQuestion question)
    {
        try
        {
            // Check if question is from this source
            if (question.Method != QuestionGenerationMethod.ExplorationBased)
            {
                return 0.5; // Neutral score for questions from other sources
            }

            // Get topic
            var topic = question.Domain;

            // Check if topic is a known exploration topic
            if (!_explorationTopics.TryGetValue(topic, out var explorationTopic))
            {
                return 0.4; // Lower score for unknown topics
            }

            // Calculate relevance based on interest level
            var relevance = explorationTopic.InterestLevel;

            // Calculate exploration potential based on exploration level
            var explorationPotential = 1.0 - explorationTopic.ExplorationLevel;

            // Calculate methodological focus based on question content
            var methodologicalFocus = question.Question.Contains("how", StringComparison.OrdinalIgnoreCase) ||
                                      question.Question.Contains("method", StringComparison.OrdinalIgnoreCase) ||
                                      question.Question.Contains("approach", StringComparison.OrdinalIgnoreCase) ||
                                      question.Question.Contains("systematic", StringComparison.OrdinalIgnoreCase) ? 0.8 : 0.4;

            // Calculate overall score
            var score = (relevance * 0.3) + (explorationPotential * 0.4) + (methodologicalFocus * 0.3);

            return score;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error evaluating question");
            return 0.5; // Default score
        }
    }

    /// <summary>
    /// Generates an exploration strategy
    /// </summary>
    /// <param name="topic">The topic</param>
    /// <returns>The exploration strategy</returns>
    public ExplorationStrategy GenerateExplorationStrategy(string topic)
    {
        try
        {
            _logger.LogDebug("Generating exploration strategy for topic: {Topic}", topic);

            // Check if topic is a known exploration topic
            if (!_explorationTopics.TryGetValue(topic, out var explorationTopic))
            {
                // Default to deep dive for unknown topics
                return ExplorationStrategy.DeepDive;
            }

            // Choose strategy based on exploration level and interest
            if (explorationTopic.ExplorationLevel < 0.3)
            {
                // For largely unexplored topics, start with breadth first
                return ExplorationStrategy.BreadthFirst;
            }
            else if (explorationTopic.ExplorationLevel < 0.7)
            {
                // For moderately explored topics, choose based on interest
                if (explorationTopic.InterestLevel > 0.7)
                {
                    // For high interest, deep dive
                    return ExplorationStrategy.DeepDive;
                }
                else
                {
                    // For moderate interest, connection based
                    return ExplorationStrategy.ConnectionBased;
                }
            }
            else
            {
                // For well-explored topics, look for novelty
                return ExplorationStrategy.NoveltyBased;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating exploration strategy");
            return ExplorationStrategy.BreadthFirst; // Default strategy
        }
    }

    /// <summary>
    /// Gets recent exploration paths
    /// </summary>
    /// <param name="count">The number of paths to return</param>
    /// <returns>The recent exploration paths</returns>
    public List<ExplorationPath> GetRecentExplorationPaths(int count)
    {
        return _explorationPaths
            .OrderByDescending(p => p.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the most satisfying exploration paths
    /// </summary>
    /// <param name="count">The number of paths to return</param>
    /// <returns>The most satisfying exploration paths</returns>
    public List<ExplorationPath> GetMostSatisfyingExplorationPaths(int count)
    {
        return _explorationPaths
            .OrderByDescending(p => p.Satisfaction)
            .Take(count)
            .ToList();
    }
}

/// <summary>
/// Represents an exploration topic
/// </summary>
public class ExplorationTopic
{
    /// <summary>
    /// Gets or sets the topic name
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the topic description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the interest level (0.0 to 1.0)
    /// </summary>
    public double InterestLevel { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the exploration level (0.0 to 1.0)
    /// </summary>
    public double ExplorationLevel { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the last explored timestamp
    /// </summary>
    public DateTime LastExplored { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the exploration count
    /// </summary>
    public int ExplorationCount { get; set; } = 0;

    /// <summary>
    /// Gets or sets the exploration paths
    /// </summary>
    public List<ExplorationPath> ExplorationPaths { get; set; } = [];

    /// <summary>
    /// Gets or sets the related questions
    /// </summary>
    public List<string> RelatedQuestions { get; set; } = [];
}

/// <summary>
/// Represents an exploration path
/// </summary>
public class ExplorationPath
{
    /// <summary>
    /// Gets or sets the path ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the topics
    /// </summary>
    public List<string> Topics { get; set; } = [];

    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the satisfaction (0.0 to 1.0)
    /// </summary>
    public double Satisfaction { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the insights
    /// </summary>
    public List<string> Insights { get; set; } = [];

    /// <summary>
    /// Gets or sets the follow-up questions
    /// </summary>
    public List<string> FollowUpQuestions { get; set; } = [];
}
