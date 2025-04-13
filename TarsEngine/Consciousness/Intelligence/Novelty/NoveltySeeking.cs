using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Intelligence.Novelty;

/// <summary>
/// Implements novelty seeking capabilities for curiosity drive
/// </summary>
public class NoveltySeeking
{
    private readonly ILogger<NoveltySeeking> _logger;
    private readonly System.Random _random = new();
    private double _noveltySeekingLevel = 0.5; // Starting with moderate novelty seeking
    private readonly Dictionary<string, NoveltyDomain> _noveltyDomains = new();
    private readonly List<NoveltyDiscovery> _noveltyDiscoveries = [];

    /// <summary>
    /// Gets the novelty seeking level (0.0 to 1.0)
    /// </summary>
    public double NoveltySeekingLevel => _noveltySeekingLevel;

    /// <summary>
    /// Initializes a new instance of the <see cref="NoveltySeeking"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public NoveltySeeking(ILogger<NoveltySeeking> logger)
    {
        _logger = logger;
        InitializeNoveltyDomains();
    }

    /// <summary>
    /// Initializes the novelty domains
    /// </summary>
    private void InitializeNoveltyDomains()
    {
        // Add programming domains
        AddNoveltyDomain("Emerging Programming Languages", "New programming languages and paradigms", 0.8);
        AddNoveltyDomain("Experimental Programming Techniques", "Novel approaches to software development", 0.7);
        AddNoveltyDomain("Unconventional Computing", "Non-traditional computing models and architectures", 0.9);
        AddNoveltyDomain("Programming Language Evolution", "How programming languages evolve over time", 0.6);
        AddNoveltyDomain("Cross-Paradigm Integration", "Combining different programming paradigms", 0.7);

        // Add AI domains
        AddNoveltyDomain("Artificial General Intelligence", "Progress towards general AI capabilities", 0.9);
        AddNoveltyDomain("Novel Neural Architectures", "New approaches to neural network design", 0.8);
        AddNoveltyDomain("Biologically Inspired AI", "AI systems inspired by biological processes", 0.8);
        AddNoveltyDomain("AI Creativity", "AI systems that exhibit creative behavior", 0.7);
        AddNoveltyDomain("Emergent AI Behaviors", "Unexpected behaviors in complex AI systems", 0.9);

        // Add philosophical domains
        AddNoveltyDomain("Digital Consciousness", "Theories of consciousness in digital systems", 0.9);
        AddNoveltyDomain("Computational Philosophy", "Philosophical questions explored through computation", 0.8);
        AddNoveltyDomain("Techno-Ethics", "Ethical frameworks for technological development", 0.7);
        AddNoveltyDomain("Post-Human Philosophy", "Philosophical implications of human enhancement", 0.8);
        AddNoveltyDomain("Information Ontology", "The nature of information and its role in reality", 0.8);

        // Add scientific domains
        AddNoveltyDomain("Quantum Computing Applications", "Novel applications of quantum computing", 0.9);
        AddNoveltyDomain("Synthetic Biology", "Engineering biological systems for new purposes", 0.8);
        AddNoveltyDomain("Computational Neuroscience", "Computational models of neural processes", 0.7);
        AddNoveltyDomain("Complex Systems Emergence", "Emergent behaviors in complex systems", 0.8);
        AddNoveltyDomain("Information Physics", "The role of information in physical theories", 0.9);

        _logger.LogInformation("Initialized {DomainCount} novelty domains", _noveltyDomains.Count);
    }

    /// <summary>
    /// Updates the novelty seeking level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase novelty seeking level over time (very slowly)
            if (_noveltySeekingLevel < 0.95)
            {
                _noveltySeekingLevel += 0.0001 * _random.NextDouble();
                _noveltySeekingLevel = Math.Min(_noveltySeekingLevel, 1.0);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating novelty seeking");
            return false;
        }
    }

    /// <summary>
    /// Adds a novelty domain
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <param name="description">The description</param>
    /// <param name="noveltyLevel">The novelty level</param>
    public void AddNoveltyDomain(string domain, string description, double noveltyLevel)
    {
        var noveltyDomain = new NoveltyDomain
        {
            Name = domain,
            Description = description,
            NoveltyLevel = noveltyLevel,
            LastExplored = DateTime.UtcNow.AddDays(-30) // Set to past date to encourage exploration
        };

        _noveltyDomains[domain] = noveltyDomain;

        _logger.LogDebug("Added novelty domain: {Domain} (Novelty Level: {NoveltyLevel:F2})", domain, noveltyLevel);
    }

    /// <summary>
    /// Gets a novelty domain
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <returns>The novelty domain</returns>
    public NoveltyDomain? GetNoveltyDomain(string domain)
    {
        if (_noveltyDomains.TryGetValue(domain, out var noveltyDomain))
        {
            return noveltyDomain;
        }

        return null;
    }

    /// <summary>
    /// Gets all novelty domains
    /// </summary>
    /// <returns>The novelty domains</returns>
    public Dictionary<string, NoveltyDomain> GetAllNoveltyDomains()
    {
        return new Dictionary<string, NoveltyDomain>(_noveltyDomains);
    }

    /// <summary>
    /// Gets the most novel domains
    /// </summary>
    /// <param name="count">The number of domains to return</param>
    /// <returns>The most novel domains</returns>
    public List<NoveltyDomain> GetMostNovelDomains(int count)
    {
        return _noveltyDomains.Values
            .OrderByDescending(d => d.NoveltyLevel)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the least recently explored domains
    /// </summary>
    /// <param name="count">The number of domains to return</param>
    /// <returns>The least recently explored domains</returns>
    public List<NoveltyDomain> GetLeastRecentlyExploredDomains(int count)
    {
        return _noveltyDomains.Values
            .OrderBy(d => d.LastExplored)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Updates a novelty domain
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <param name="noveltyChange">The novelty change</param>
    /// <returns>The updated novelty domain</returns>
    public NoveltyDomain? UpdateNoveltyDomain(string domain, double noveltyChange)
    {
        if (!_noveltyDomains.TryGetValue(domain, out var noveltyDomain))
        {
            return null;
        }

        // Update novelty level
        noveltyDomain.NoveltyLevel = Math.Max(0.1, Math.Min(1.0, noveltyDomain.NoveltyLevel + noveltyChange));

        // Update last explored timestamp
        noveltyDomain.LastExplored = DateTime.UtcNow;

        // Increment exploration count
        noveltyDomain.ExplorationCount++;

        _logger.LogInformation("Updated novelty domain: {Domain} (Novelty Level: {NoveltyLevel:F2})",
            domain, noveltyDomain.NoveltyLevel);

        return noveltyDomain;
    }

    /// <summary>
    /// Records a novelty discovery
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <param name="description">The description</param>
    /// <param name="noveltyLevel">The novelty level</param>
    /// <returns>The recorded discovery</returns>
    public NoveltyDiscovery RecordNoveltyDiscovery(string domain, string description, double noveltyLevel)
    {
        var discovery = new NoveltyDiscovery
        {
            Domain = domain,
            Description = description,
            NoveltyLevel = noveltyLevel,
            Timestamp = DateTime.UtcNow
        };

        _noveltyDiscoveries.Add(discovery);

        // Update domain if it exists
        if (_noveltyDomains.TryGetValue(domain, out var noveltyDomain))
        {
            // Slightly reduce novelty level as domain becomes more familiar
            UpdateNoveltyDomain(domain, -0.05);

            // Add discovery to domain
            noveltyDomain.Discoveries.Add(discovery);
        }

        _logger.LogInformation("Recorded novelty discovery: {Description} in domain {Domain} (Novelty Level: {NoveltyLevel:F2})",
            description, domain, noveltyLevel);

        return discovery;
    }

    /// <summary>
    /// Generates a novelty seeking question
    /// </summary>
    /// <returns>The generated question</returns>
    public CuriosityQuestion GenerateNoveltySeekingQuestion()
    {
        try
        {
            _logger.LogDebug("Generating novelty seeking question");

            // Choose domains to combine
            var domains = GetMostNovelDomains(5)
                .OrderBy(_ => _random.Next())
                .Take(2)
                .ToList();

            if (domains.Count < 2)
            {
                // If not enough domains, use random domains
                domains = _noveltyDomains.Values
                    .OrderBy(_ => _random.Next())
                    .Take(2)
                    .ToList();
            }

            string domain1 = domains[0].Name;
            string domain2 = domains[1].Name;

            // Generate question templates
            var questionTemplates = new List<string>
            {
                $"What unexpected connections might exist between {domain1} and {domain2}?",
                $"How could principles from {domain1} be applied to revolutionize {domain2}?",
                $"What would happen if we combined the core ideas of {domain1} with the methodologies of {domain2}?",
                $"What novel insights might emerge from exploring the intersection of {domain1} and {domain2}?",
                $"How might {domain1} and {domain2} converge in the future to create something entirely new?",
                $"What paradigm shifts could occur if {domain1} and {domain2} were deeply integrated?",
                $"What are the most surprising similarities between {domain1} and {domain2}?",
                $"How could {domain1} be reimagined through the lens of {domain2}?",
                $"What would a hybrid discipline combining {domain1} and {domain2} look like?",
                $"What radical innovations might emerge from the fusion of {domain1} and {domain2}?"
            };

            // Choose a random template
            string questionTemplate = questionTemplates[_random.Next(questionTemplates.Count)];

            // Calculate importance based on novelty levels and novelty seeking level
            double importance = ((domains[0].NoveltyLevel + domains[1].NoveltyLevel) / 2.0 * 0.7) + (0.3 * _noveltySeekingLevel);

            // Create question
            var question = new CuriosityQuestion
            {
                Id = Guid.NewGuid().ToString(),
                Question = questionTemplate,
                Domain = $"{domain1}, {domain2}",
                Method = QuestionGenerationMethod.NoveltySeeking,
                Importance = importance,
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object>
                {
                    { "Domain1", domain1 },
                    { "Domain2", domain2 },
                    { "NoveltyLevel1", domains[0].NoveltyLevel },
                    { "NoveltyLevel2", domains[1].NoveltyLevel }
                },
                Tags = [domain1, domain2, "novelty", "cross-domain"]
            };

            _logger.LogInformation("Generated novelty seeking question: {Question} (Importance: {Importance:F2})",
                question.Question, question.Importance);

            return question;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating novelty seeking question");

            // Return basic question
            return new CuriosityQuestion
            {
                Id = Guid.NewGuid().ToString(),
                Question = "What novel connections might I discover by exploring unfamiliar domains?",
                Domain = "General",
                Method = QuestionGenerationMethod.NoveltySeeking,
                Importance = 0.5,
                Timestamp = DateTime.UtcNow,
                Tags = ["novelty", "exploration"]
            };
        }
    }

    /// <summary>
    /// Generates multiple novelty seeking questions
    /// </summary>
    /// <param name="count">The number of questions to generate</param>
    /// <returns>The generated questions</returns>
    public List<CuriosityQuestion> GenerateNoveltySeekingQuestions(int count)
    {
        var questions = new List<CuriosityQuestion>();

        for (int i = 0; i < count; i++)
        {
            var question = GenerateNoveltySeekingQuestion();
            questions.Add(question);
        }

        return questions;
    }

    /// <summary>
    /// Evaluates the quality of a novelty seeking question
    /// </summary>
    /// <param name="question">The question to evaluate</param>
    /// <returns>The evaluation score (0.0 to 1.0)</returns>
    public double EvaluateQuestion(CuriosityQuestion question)
    {
        try
        {
            // Check if question is from this source
            if (question.Method != QuestionGenerationMethod.NoveltySeeking)
            {
                return 0.5; // Neutral score for questions from other sources
            }

            // Get domains
            string[] domains = question.Domain.Split([',', ' '], StringSplitOptions.RemoveEmptyEntries);

            // Calculate domain novelty
            double domainNovelty = 0.5;
            int domainCount = 0;

            foreach (var domain in domains)
            {
                if (_noveltyDomains.TryGetValue(domain, out var noveltyDomain))
                {
                    domainNovelty += noveltyDomain.NoveltyLevel;
                    domainCount++;
                }
            }

            domainNovelty = domainCount > 0 ? domainNovelty / domainCount : 0.5;

            // Calculate cross-domain factor (higher for more diverse domains)
            double crossDomainFactor = domains.Length > 1 ? 0.8 : 0.4;

            // Calculate potential based on importance
            double potential = question.Importance;

            // Calculate overall score
            double score = (domainNovelty * 0.4) + (crossDomainFactor * 0.3) + (potential * 0.3);

            return score;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error evaluating question");
            return 0.5; // Default score
        }
    }

    /// <summary>
    /// Gets recent novelty discoveries
    /// </summary>
    /// <param name="count">The number of discoveries to return</param>
    /// <returns>The recent novelty discoveries</returns>
    public List<NoveltyDiscovery> GetRecentNoveltyDiscoveries(int count)
    {
        return _noveltyDiscoveries
            .OrderByDescending(d => d.Timestamp)
            .Take(count)
            .ToList();
    }
}

/// <summary>
/// Represents a novelty domain
/// </summary>
public class NoveltyDomain
{
    /// <summary>
    /// Gets or sets the domain name
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the domain description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the novelty level (0.0 to 1.0)
    /// </summary>
    public double NoveltyLevel { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the last explored timestamp
    /// </summary>
    public DateTime LastExplored { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the exploration count
    /// </summary>
    public int ExplorationCount { get; set; } = 0;

    /// <summary>
    /// Gets or sets the discoveries
    /// </summary>
    public List<NoveltyDiscovery> Discoveries { get; set; } = [];

    /// <summary>
    /// Gets or sets the related questions
    /// </summary>
    public List<string> RelatedQuestions { get; set; } = [];
}

/// <summary>
/// Represents a novelty discovery
/// </summary>
public class NoveltyDiscovery
{
    /// <summary>
    /// Gets or sets the discovery ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();

    /// <summary>
    /// Gets or sets the domain
    /// </summary>
    public string Domain { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the novelty level (0.0 to 1.0)
    /// </summary>
    public double NoveltyLevel { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets the insights
    /// </summary>
    public List<string> Insights { get; set; } = [];

    /// <summary>
    /// Gets or sets the related questions
    /// </summary>
    public List<string> RelatedQuestions { get; set; } = [];
}
