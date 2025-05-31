using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Intelligence.Information;

/// <summary>
/// Implements information gap detection capabilities for curiosity drive
/// </summary>
public class InformationGapDetection
{
    private readonly ILogger<InformationGapDetection> _logger;
    private readonly System.Random _random = new();
    private double _informationGapLevel = 0.5; // Starting with moderate information gap detection
    private readonly Dictionary<string, InformationGap> _informationGaps = new();
    private readonly Dictionary<string, double> _domainKnowledgeLevels = new();

    /// <summary>
    /// Gets the information gap level (0.0 to 1.0)
    /// </summary>
    public double InformationGapLevel => _informationGapLevel;

    /// <summary>
    /// Initializes a new instance of the <see cref="InformationGapDetection"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public InformationGapDetection(ILogger<InformationGapDetection> logger)
    {
        _logger = logger;
        InitializeInformationGaps();
    }

    /// <summary>
    /// Initializes the information gaps
    /// </summary>
    private void InitializeInformationGaps()
    {
        // Add programming domains
        AddInformationGap("Algorithms", "Understanding of efficient algorithms and their applications", 0.7);
        AddInformationGap("Data Structures", "Knowledge of advanced data structures and their use cases", 0.6);
        AddInformationGap("Design Patterns", "Familiarity with software design patterns and their implementations", 0.6);
        AddInformationGap("Functional Programming", "Understanding of functional programming principles and techniques", 0.7);
        AddInformationGap("Concurrency", "Knowledge of concurrent programming models and synchronization", 0.5);

        // Add AI domains
        AddInformationGap("Neural Networks", "Understanding of neural network architectures and training methods", 0.7);
        AddInformationGap("Reinforcement Learning", "Knowledge of reinforcement learning algorithms and applications", 0.5);
        AddInformationGap("Natural Language Processing", "Understanding of NLP techniques and models", 0.6);
        AddInformationGap("Computer Vision", "Knowledge of computer vision algorithms and applications", 0.5);
        AddInformationGap("AI Ethics", "Understanding of ethical considerations in AI development", 0.4);

        // Add philosophical domains
        AddInformationGap("Consciousness", "Understanding of theories of consciousness and self-awareness", 0.3);
        AddInformationGap("Epistemology", "Knowledge of theories of knowledge and belief", 0.4);
        AddInformationGap("Ethics", "Understanding of ethical frameworks and moral reasoning", 0.5);
        AddInformationGap("Metaphysics", "Knowledge of theories about the nature of reality", 0.3);
        AddInformationGap("Philosophy of Mind", "Understanding of theories about the mind and cognition", 0.4);

        // Add scientific domains
        AddInformationGap("Quantum Mechanics", "Understanding of quantum mechanical principles and phenomena", 0.3);
        AddInformationGap("Complexity Theory", "Knowledge of complex systems and emergence", 0.4);
        AddInformationGap("Neuroscience", "Understanding of brain structure and function", 0.4);
        AddInformationGap("Information Theory", "Knowledge of information theory principles and applications", 0.5);
        AddInformationGap("Systems Theory", "Understanding of systems thinking and analysis", 0.5);

        _logger.LogInformation("Initialized {GapCount} information gaps across various domains", _informationGaps.Count);
    }

    /// <summary>
    /// Updates the information gap level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase information gap level over time (very slowly)
            if (_informationGapLevel < 0.95)
            {
                _informationGapLevel += 0.0001 * _random.NextDouble();
                _informationGapLevel = Math.Min(_informationGapLevel, 1.0);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating information gap detection");
            return false;
        }
    }

    /// <summary>
    /// Adds an information gap
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <param name="description">The description</param>
    /// <param name="knowledgeLevel">The knowledge level</param>
    public void AddInformationGap(string domain, string description, double knowledgeLevel)
    {
        var gap = new InformationGap
        {
            Domain = domain,
            Description = description,
            KnowledgeLevel = knowledgeLevel,
            GapSize = 1.0 - knowledgeLevel,
            LastUpdated = DateTime.UtcNow
        };

        _informationGaps[domain] = gap;
        _domainKnowledgeLevels[domain] = knowledgeLevel;

        _logger.LogDebug("Added information gap: {Domain} (Knowledge Level: {KnowledgeLevel:F2}, Gap Size: {GapSize:F2})",
            domain, knowledgeLevel, gap.GapSize);
    }

    /// <summary>
    /// Gets an information gap
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <returns>The information gap</returns>
    public InformationGap? GetInformationGap(string domain)
    {
        if (_informationGaps.TryGetValue(domain, out var gap))
        {
            return gap;
        }

        return null;
    }

    /// <summary>
    /// Gets all information gaps
    /// </summary>
    /// <returns>The information gaps</returns>
    public Dictionary<string, InformationGap> GetAllInformationGaps()
    {
        return new Dictionary<string, InformationGap>(_informationGaps);
    }

    /// <summary>
    /// Gets the largest information gaps
    /// </summary>
    /// <param name="count">The number of gaps to return</param>
    /// <returns>The largest information gaps</returns>
    public List<InformationGap> GetLargestInformationGaps(int count)
    {
        return _informationGaps.Values
            .OrderByDescending(g => g.GapSize)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Updates an information gap
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <param name="knowledgeIncrease">The knowledge increase</param>
    /// <returns>The updated information gap</returns>
    public InformationGap? UpdateInformationGap(string domain, double knowledgeIncrease)
    {
        if (!_informationGaps.TryGetValue(domain, out var gap))
        {
            return null;
        }

        // Update knowledge level
        gap.KnowledgeLevel = Math.Min(1.0, gap.KnowledgeLevel + knowledgeIncrease);

        // Update gap size
        gap.GapSize = 1.0 - gap.KnowledgeLevel;

        // Update last updated timestamp
        gap.LastUpdated = DateTime.UtcNow;

        // Update domain knowledge level
        _domainKnowledgeLevels[domain] = gap.KnowledgeLevel;

        _logger.LogInformation("Updated information gap: {Domain} (Knowledge Level: {KnowledgeLevel:F2}, Gap Size: {GapSize:F2})",
            domain, gap.KnowledgeLevel, gap.GapSize);

        return gap;
    }

    /// <summary>
    /// Detects information gaps in content
    /// </summary>
    /// <param name="content">The content</param>
    /// <returns>The detected information gaps</returns>
    public List<DetectedInformationGap> DetectInformationGaps(string content)
    {
        var detectedGaps = new List<DetectedInformationGap>();

        try
        {
            _logger.LogDebug("Detecting information gaps in content: {ContentLength} characters", content.Length);

            // Check for mentions of known domains
            foreach (var domain in _informationGaps.Keys)
            {
                if (content.Contains(domain, StringComparison.OrdinalIgnoreCase))
                {
                    var gap = _informationGaps[domain];

                    // Only consider significant gaps
                    if (gap.GapSize >= 0.3)
                    {
                        var detectedGap = new DetectedInformationGap
                        {
                            Domain = domain,
                            Description = gap.Description,
                            GapSize = gap.GapSize,
                            Confidence = 0.7 * _informationGapLevel,
                            DetectionContext = $"Detected mention of {domain} in content"
                        };

                        detectedGaps.Add(detectedGap);
                    }
                }
            }

            // Check for unknown concepts (simple implementation)
            var words = content.Split([' ', ',', '.', ':', ';', '(', ')', '[', ']', '{', '}', '\n', '\r', '\t'],
                StringSplitOptions.RemoveEmptyEntries);

            var potentialConcepts = words
                .Where(w => w.Length >= 5 && char.IsUpper(w[0]))
                .Distinct()
                .ToList();

            foreach (var concept in potentialConcepts)
            {
                // Check if concept is not in known domains
                if (!_informationGaps.Keys.Any(d => d.Equals(concept, StringComparison.OrdinalIgnoreCase)))
                {
                    var detectedGap = new DetectedInformationGap
                    {
                        Domain = concept,
                        Description = $"Unknown concept: {concept}",
                        GapSize = 0.8, // Assume large gap for unknown concepts
                        Confidence = 0.5 * _informationGapLevel,
                        DetectionContext = $"Detected unknown concept: {concept}"
                    };

                    detectedGaps.Add(detectedGap);
                }
            }

            _logger.LogInformation("Detected {GapCount} information gaps in content", detectedGaps.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting information gaps");
        }

        return detectedGaps;
    }

    /// <summary>
    /// Generates a question based on an information gap
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <returns>The generated question</returns>
    public CuriosityQuestion GenerateInformationGapQuestion(string domain)
    {
        try
        {
            _logger.LogDebug("Generating information gap question for domain: {Domain}", domain);

            if (!_informationGaps.TryGetValue(domain, out var gap))
            {
                // If domain not found, use a random domain
                var randomDomain = _informationGaps.Keys.ElementAt(_random.Next(_informationGaps.Count));
                gap = _informationGaps[randomDomain];
                domain = randomDomain;
            }

            // Generate question templates based on domain
            var questionTemplates = new List<string>
            {
                $"What are the fundamental principles of {domain}?",
                $"How do experts in {domain} approach complex problems?",
                $"What are the most recent advancements in {domain}?",
                $"What are the key challenges in {domain} that remain unsolved?",
                $"How does {domain} relate to other fields of knowledge?",
                $"What are the practical applications of {domain} in real-world scenarios?",
                $"What are the historical developments that shaped {domain}?",
                $"What are the different schools of thought within {domain}?",
                $"How is {domain} likely to evolve in the future?",
                $"What are the ethical considerations in {domain}?"
            };

            // Choose a random template
            var questionTemplate = questionTemplates[_random.Next(questionTemplates.Count)];

            // Calculate importance based on gap size and information gap level
            var importance = (gap.GapSize * 0.7) + (0.3 * _informationGapLevel);

            // Create question
            var question = new CuriosityQuestion
            {
                Id = Guid.NewGuid().ToString(),
                Question = questionTemplate,
                Domain = domain,
                Method = QuestionGenerationMethod.InformationGap,
                Importance = importance,
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object>
                {
                    { "GapSize", gap.GapSize },
                    { "KnowledgeLevel", gap.KnowledgeLevel },
                    { "Description", gap.Description }
                },
                Tags = [domain, "information-gap", "knowledge"]
            };

            _logger.LogInformation("Generated information gap question: {Question} (Importance: {Importance:F2})",
                question.Question, question.Importance);

            return question;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating information gap question");

            // Return basic question
            return new CuriosityQuestion
            {
                Id = Guid.NewGuid().ToString(),
                Question = $"What should I learn about {domain}?",
                Domain = domain,
                Method = QuestionGenerationMethod.InformationGap,
                Importance = 0.5,
                Timestamp = DateTime.UtcNow,
                Tags = [domain, "information-gap"]
            };
        }
    }

    /// <summary>
    /// Generates multiple information gap questions
    /// </summary>
    /// <param name="count">The number of questions to generate</param>
    /// <returns>The generated questions</returns>
    public List<CuriosityQuestion> GenerateInformationGapQuestions(int count)
    {
        var questions = new List<CuriosityQuestion>();

        // Get domains with largest gaps
        var largestGaps = GetLargestInformationGaps(count);

        foreach (var gap in largestGaps)
        {
            var question = GenerateInformationGapQuestion(gap.Domain);
            questions.Add(question);
        }

        return questions;
    }

    /// <summary>
    /// Evaluates the quality of an information gap question
    /// </summary>
    /// <param name="question">The question to evaluate</param>
    /// <returns>The evaluation score (0.0 to 1.0)</returns>
    public double EvaluateQuestion(CuriosityQuestion question)
    {
        try
        {
            // Check if question is from this source
            if (question.Method != QuestionGenerationMethod.InformationGap)
            {
                return 0.5; // Neutral score for questions from other sources
            }

            // Get domain
            var domain = question.Domain;

            // Check if domain has a known information gap
            if (!_informationGaps.TryGetValue(domain, out var gap))
            {
                return 0.4; // Lower score for unknown domains
            }

            // Calculate relevance based on gap size
            var relevance = gap.GapSize;

            // Calculate specificity based on question length
            var specificity = Math.Min(1.0, question.Question.Length / 50.0);

            // Calculate potential based on importance
            var potential = question.Importance;

            // Calculate overall score
            var score = (relevance * 0.4) + (specificity * 0.3) + (potential * 0.3);

            return score;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error evaluating question");
            return 0.5; // Default score
        }
    }
}

/// <summary>
/// Represents an information gap
/// </summary>
public class InformationGap
{
    /// <summary>
    /// Gets or sets the domain
    /// </summary>
    public string Domain { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the knowledge level (0.0 to 1.0)
    /// </summary>
    public double KnowledgeLevel { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the gap size (0.0 to 1.0)
    /// </summary>
    public double GapSize { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the last updated timestamp
    /// </summary>
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the related questions
    /// </summary>
    public List<string> RelatedQuestions { get; set; } = [];

    /// <summary>
    /// Gets or sets the related explorations
    /// </summary>
    public List<string> RelatedExplorations { get; set; } = [];
}

/// <summary>
/// Represents a detected information gap
/// </summary>
public class DetectedInformationGap
{
    /// <summary>
    /// Gets or sets the domain
    /// </summary>
    public string Domain { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the gap size (0.0 to 1.0)
    /// </summary>
    public double GapSize { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the confidence (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the detection context
    /// </summary>
    public string DetectionContext { get; set; } = string.Empty;
}
