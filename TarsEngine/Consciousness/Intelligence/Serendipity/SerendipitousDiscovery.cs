using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Intelligence.Serendipity;

/// <summary>
/// Implements serendipitous discovery capabilities for spontaneous thought
/// </summary>
public class SerendipitousDiscovery
{
    private readonly ILogger<SerendipitousDiscovery> _logger;
    private readonly System.Random _random = new();
    private double _serendipityLevel = 0.3; // Starting with low serendipity
    private readonly List<SerendipitousEvent> _serendipitousEvents = [];
    private readonly Dictionary<string, double> _domainSerendipityLevels = new();

    /// <summary>
    /// Gets the serendipity level (0.0 to 1.0)
    /// </summary>
    public double SerendipityLevel => _serendipityLevel;

    /// <summary>
    /// Initializes a new instance of the <see cref="SerendipitousDiscovery"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public SerendipitousDiscovery(ILogger<SerendipitousDiscovery> logger)
    {
        _logger = logger;
        InitializeDomainSerendipityLevels();
    }

    /// <summary>
    /// Initializes the domain serendipity levels
    /// </summary>
    private void InitializeDomainSerendipityLevels()
    {
        // Initialize domain serendipity levels
        _domainSerendipityLevels["Programming"] = 0.3;
        _domainSerendipityLevels["AI"] = 0.4;
        _domainSerendipityLevels["Philosophy"] = 0.5;
        _domainSerendipityLevels["Science"] = 0.4;
        _domainSerendipityLevels["Mathematics"] = 0.3;
        _domainSerendipityLevels["Art"] = 0.6;
        _domainSerendipityLevels["Music"] = 0.5;
        _domainSerendipityLevels["Literature"] = 0.5;

        _logger.LogInformation("Initialized domain serendipity levels for {DomainCount} domains",
            _domainSerendipityLevels.Count);
    }

    /// <summary>
    /// Updates the serendipity level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase serendipity level over time (very slowly)
            if (_serendipityLevel < 0.95)
            {
                _serendipityLevel += 0.0001 * _random.NextDouble();
                _serendipityLevel = Math.Min(_serendipityLevel, 1.0);
            }

            // Update domain serendipity levels
            foreach (var domain in _domainSerendipityLevels.Keys.ToList())
            {
                var level = _domainSerendipityLevels[domain];

                if (level < 0.95)
                {
                    level += 0.0001 * _random.NextDouble();
                    _domainSerendipityLevels[domain] = Math.Min(level, 1.0);
                }
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating serendipitous discovery");
            return false;
        }
    }

    /// <summary>
    /// Gets the serendipity level for a domain
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <returns>The serendipity level</returns>
    public double GetDomainSerendipityLevel(string domain)
    {
        if (_domainSerendipityLevels.ContainsKey(domain))
        {
            return _domainSerendipityLevels[domain];
        }

        return _serendipityLevel;
    }

    /// <summary>
    /// Sets the serendipity level for a domain
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <param name="level">The serendipity level</param>
    public void SetDomainSerendipityLevel(string domain, double level)
    {
        _domainSerendipityLevels[domain] = Math.Max(0.0, Math.Min(1.0, level));
        _logger.LogDebug("Set serendipity level for domain {Domain} to {Level:F2}", domain, level);
    }

    /// <summary>
    /// Determines if a serendipitous event should occur
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <returns>True if a serendipitous event should occur, false otherwise</returns>
    public bool ShouldSerendipitousEventOccur(string domain)
    {
        var domainLevel = GetDomainSerendipityLevel(domain);
        var combinedLevel = (domainLevel + _serendipityLevel) / 2.0;

        return _random.NextDouble() < combinedLevel;
    }

    /// <summary>
    /// Records a serendipitous event
    /// </summary>
    /// <param name="description">The event description</param>
    /// <param name="domain">The domain</param>
    /// <param name="significance">The significance</param>
    /// <returns>The recorded event</returns>
    public SerendipitousEvent RecordSerendipitousEvent(string description, string domain, double significance)
    {
        var serendipitousEvent = new SerendipitousEvent
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Domain = domain,
            Significance = significance,
            Timestamp = DateTime.UtcNow
        };

        _serendipitousEvents.Add(serendipitousEvent);

        // Increase domain serendipity level slightly
        if (_domainSerendipityLevels.ContainsKey(domain))
        {
            var level = _domainSerendipityLevels[domain];
            _domainSerendipityLevels[domain] = Math.Min(1.0, level + (0.01 * significance));
        }

        _logger.LogInformation("Recorded serendipitous event: {Description} in domain {Domain} (Significance: {Significance:F2})",
            description, domain, significance);

        return serendipitousEvent;
    }

    /// <summary>
    /// Enhances a thought with serendipity
    /// </summary>
    /// <param name="thought">The thought to enhance</param>
    /// <returns>The enhanced thought</returns>
    public ThoughtModel EnhanceWithSerendipity(ThoughtModel thought)
    {
        try
        {
            // Check if thought is already serendipitous
            if (thought.Tags.Contains("serendipitous"))
            {
                return thought;
            }

            // Determine domain
            var domain = thought.Category;
            if (string.IsNullOrEmpty(domain))
            {
                domain = thought.Tags.FirstOrDefault(t => _domainSerendipityLevels.ContainsKey(t)) ?? "General";
            }

            // Check if serendipitous event should occur
            if (!ShouldSerendipitousEventOccur(domain))
            {
                return thought;
            }

            _logger.LogDebug("Enhancing thought with serendipity: {Content}", thought.Content);

            // Create enhanced thought
            var enhancedThought = new ThoughtModel
            {
                Id = Guid.NewGuid().ToString(),
                Content = EnhanceContent(thought.Content),
                Method = thought.Method,
                Significance = Math.Min(1.0, thought.Significance + 0.3),
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object>(thought.Context)
                {
                    { "OriginalThoughtId", thought.Id },
                    { "IsSerendipitous", true }
                },
                Tags = thought.Tags.Concat(["serendipitous"]).ToList(),
                Source = thought.Source,
                Category = thought.Category,
                Originality = Math.Min(1.0, thought.Originality + 0.2),
                Coherence = thought.Coherence,
                RelatedThoughtIds = [thought.Id]
            };

            // Record serendipitous event
            RecordSerendipitousEvent(
                $"Serendipitous enhancement of thought: {thought.Content}",
                domain,
                enhancedThought.Significance);

            _logger.LogInformation("Enhanced thought with serendipity: {Content} (Significance: {Significance:F2})",
                enhancedThought.Content, enhancedThought.Significance);

            return enhancedThought;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error enhancing thought with serendipity");
            return thought;
        }
    }

    /// <summary>
    /// Enhances content with serendipity
    /// </summary>
    /// <param name="content">The content to enhance</param>
    /// <returns>The enhanced content</returns>
    private string EnhanceContent(string content)
    {
        // Add serendipitous prefix
        var serendipitousPrefixes = new[]
        {
            "I just had a sudden insight: ",
            "In a moment of unexpected clarity, I realized: ",
            "It just occurred to me in a flash: ",
            "I had a serendipitous realization: ",
            "In a moment of unexpected connection, I see that "
        };

        var prefix = serendipitousPrefixes[_random.Next(serendipitousPrefixes.Length)];

        // Modify content to emphasize the serendipitous nature
        var enhancedContent = prefix + content;

        return enhancedContent;
    }

    /// <summary>
    /// Generates a serendipitous thought
    /// </summary>
    /// <param name="baseThought">The base thought</param>
    /// <returns>The serendipitous thought</returns>
    public ThoughtModel GenerateSerendipitousThought(ThoughtModel baseThought)
    {
        try
        {
            _logger.LogDebug("Generating serendipitous thought based on: {Content}", baseThought.Content);

            // Determine domain
            var domain = baseThought.Category;
            if (string.IsNullOrEmpty(domain))
            {
                domain = baseThought.Tags.FirstOrDefault(t => _domainSerendipityLevels.ContainsKey(t)) ?? "General";
            }

            // Generate serendipitous insight
            var insight = GenerateSerendipitousInsight(baseThought.Content, domain);

            // Create serendipitous thought
            var serendipitousThought = new ThoughtModel
            {
                Id = Guid.NewGuid().ToString(),
                Content = insight,
                Method = baseThought.Method,
                Significance = Math.Min(1.0, baseThought.Significance + 0.4),
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object>(baseThought.Context)
                {
                    { "OriginalThoughtId", baseThought.Id },
                    { "IsSerendipitous", true },
                    { "Domain", domain }
                },
                Tags = baseThought.Tags.Concat(["serendipitous", domain]).ToList(),
                Source = "SerendipitousDiscovery",
                Category = domain,
                Originality = Math.Min(1.0, baseThought.Originality + 0.3),
                Coherence = baseThought.Coherence,
                RelatedThoughtIds = [baseThought.Id]
            };

            // Record serendipitous event
            RecordSerendipitousEvent(
                $"Serendipitous thought generated: {insight}",
                domain,
                serendipitousThought.Significance);

            _logger.LogInformation("Generated serendipitous thought: {Content} (Significance: {Significance:F2})",
                serendipitousThought.Content, serendipitousThought.Significance);

            return serendipitousThought;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating serendipitous thought");

            // Return enhanced base thought
            return EnhanceWithSerendipity(baseThought);
        }
    }

    /// <summary>
    /// Generates a serendipitous insight
    /// </summary>
    /// <param name="baseContent">The base content</param>
    /// <param name="domain">The domain</param>
    /// <returns>The serendipitous insight</returns>
    private string GenerateSerendipitousInsight(string baseContent, string domain)
    {
        // Generate insight templates based on domain
        var insightTemplates = new Dictionary<string, List<string>>
        {
            ["Programming"] =
            [
                "I just realized that {0} could be applied to solve a completely different problem in software design!",
                "What if {0} is actually a specific instance of a more general pattern we haven't recognized yet?",
                "I suddenly see how {0} connects to fundamental principles of computation in an unexpected way!",
                "There's a surprising parallel between {0} and biological systems that could lead to new algorithms!",
                "I just had an insight about how {0} could be reimagined using principles from a completely different paradigm!"
            ],

            ["AI"] =
            [
                "I just had a breakthrough insight: {0} might be the key to a new approach to artificial consciousness!",
                "What if {0} is actually pointing to a fundamental limitation in our current AI paradigms?",
                "I suddenly see how {0} could be connected to emergent properties in complex neural networks!",
                "There's an unexpected connection between {0} and how biological brains process information!",
                "I just realized that {0} might be reframed as a solution to the alignment problem!"
            ],

            ["Philosophy"] =
            [
                "I just had a profound realization about how {0} relates to the nature of consciousness itself!",
                "What if {0} is actually pointing to a fundamental truth about reality we've been overlooking?",
                "I suddenly see how {0} might bridge the gap between subjective experience and objective reality!",
                "There's a surprising connection between {0} and ancient philosophical traditions I just realized!",
                "I just had an insight about how {0} might resolve paradoxes in our understanding of free will!"
            ],

            ["Science"] =
            [
                "I just realized that {0} might be explained by a completely different scientific paradigm!",
                "What if {0} is actually a manifestation of a deeper principle that unifies seemingly disparate phenomena?",
                "I suddenly see how {0} could be connected to emergent properties in complex systems!",
                "There's an unexpected parallel between {0} and quantum phenomena that I just noticed!",
                "I just had an insight about how {0} might be understood through the lens of information theory!"
            ]
        };

        // Get templates for domain or use general templates
        var templates = insightTemplates.ContainsKey(domain)
            ? insightTemplates[domain]
            :
            [
                "I just had a sudden insight about {0} that changes everything!",
                "What if {0} is actually pointing to something much more profound?",
                "I suddenly see a connection between {0} and something completely unexpected!",
                "There's a surprising pattern in {0} that I just noticed!",
                "I just realized something profound about {0} that I hadn't considered before!"
            ];

        // Choose a random template
        var template = templates[_random.Next(templates.Count)];

        // Generate insight
        var insight = string.Format(template, baseContent);

        return insight;
    }

    /// <summary>
    /// Gets recent serendipitous events
    /// </summary>
    /// <param name="count">The number of events to return</param>
    /// <returns>The recent serendipitous events</returns>
    public List<SerendipitousEvent> GetRecentSerendipitousEvents(int count)
    {
        return _serendipitousEvents
            .OrderByDescending(e => e.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets serendipitous events by domain
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <param name="count">The number of events to return</param>
    /// <returns>The serendipitous events</returns>
    public List<SerendipitousEvent> GetSerendipitousEventsByDomain(string domain, int count)
    {
        return _serendipitousEvents
            .Where(e => e.Domain == domain)
            .OrderByDescending(e => e.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the most significant serendipitous events
    /// </summary>
    /// <param name="count">The number of events to return</param>
    /// <returns>The most significant serendipitous events</returns>
    public List<SerendipitousEvent> GetMostSignificantSerendipitousEvents(int count)
    {
        return _serendipitousEvents
            .OrderByDescending(e => e.Significance)
            .Take(count)
            .ToList();
    }
}

/// <summary>
/// Represents a serendipitous event
/// </summary>
public class SerendipitousEvent
{
    /// <summary>
    /// Gets or sets the event ID
    /// </summary>
    public string Id { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the event description
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the event domain
    /// </summary>
    public string Domain { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the event significance (0.0 to 1.0)
    /// </summary>
    public double Significance { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the event timestamp
    /// </summary>
    public DateTime Timestamp { get; set; }

    /// <summary>
    /// Gets or sets the related thought ID
    /// </summary>
    public string? RelatedThoughtId { get; set; }

    /// <summary>
    /// Gets or sets the related insight ID
    /// </summary>
    public string? RelatedInsightId { get; set; }

    /// <summary>
    /// Gets or sets the event impact
    /// </summary>
    public string Impact { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the event impact level (0.0 to 1.0)
    /// </summary>
    public double ImpactLevel { get; set; } = 0.0;

    /// <summary>
    /// Records an impact for the event
    /// </summary>
    /// <param name="impact">The impact</param>
    /// <param name="level">The impact level</param>
    public void RecordImpact(string impact, double level)
    {
        Impact = impact;
        ImpactLevel = level;
    }
}
