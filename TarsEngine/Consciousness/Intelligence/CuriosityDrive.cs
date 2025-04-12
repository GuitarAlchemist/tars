using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents TARS's curiosity drive capabilities
/// </summary>
public class CuriosityDrive
{
    private readonly ILogger<CuriosityDrive> _logger;
    private readonly List<CuriosityQuestion> _questions = [];
    private readonly List<CuriosityExploration> _explorations = [];
    private readonly Dictionary<string, InformationGap> _informationGaps = new();

    private bool _isInitialized = false;
    private bool _isActive = false;
    private double _curiosityLevel = 0.5; // Starting with moderate curiosity
    private double _noveltySeekingLevel = 0.6; // Starting with moderate novelty seeking
    private double _questionGenerationLevel = 0.4; // Starting with moderate question generation
    private double _explorationLevel = 0.5; // Starting with moderate exploration
    private readonly System.Random _random = new();
    private DateTime _lastQuestionTime = DateTime.MinValue;

    /// <summary>
    /// Gets the curiosity level (0.0 to 1.0)
    /// </summary>
    public double CuriosityLevel => _curiosityLevel;

    /// <summary>
    /// Gets the novelty seeking level (0.0 to 1.0)
    /// </summary>
    public double NoveltySeekingLevel => _noveltySeekingLevel;

    /// <summary>
    /// Gets the question generation level (0.0 to 1.0)
    /// </summary>
    public double QuestionGenerationLevel => _questionGenerationLevel;

    /// <summary>
    /// Gets the exploration level (0.0 to 1.0)
    /// </summary>
    public double ExplorationLevel => _explorationLevel;

    /// <summary>
    /// Gets the questions
    /// </summary>
    public IReadOnlyList<CuriosityQuestion> Questions => _questions.AsReadOnly();

    /// <summary>
    /// Gets the explorations
    /// </summary>
    public IReadOnlyList<CuriosityExploration> Explorations => _explorations.AsReadOnly();

    /// <summary>
    /// Gets the information gaps
    /// </summary>
    public IReadOnlyDictionary<string, InformationGap> InformationGaps => _informationGaps;

    /// <summary>
    /// Initializes a new instance of the <see cref="CuriosityDrive"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public CuriosityDrive(ILogger<CuriosityDrive> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Initializes the curiosity drive
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public async Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing curiosity drive");

            // Initialize information gaps
            InitializeInformationGaps();

            _isInitialized = true;
            _logger.LogInformation("Curiosity drive initialized successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing curiosity drive");
            return false;
        }
    }

    /// <summary>
    /// Initializes information gaps
    /// </summary>
    private void InitializeInformationGaps()
    {
        // Initialize some basic information gaps
        // These would be expanded over time through learning
        AddInformationGap("consciousness", "What is the nature of consciousness?", 0.8);
        AddInformationGap("intelligence", "How does intelligence emerge from neural processes?", 0.7);
        AddInformationGap("creativity", "What are the cognitive mechanisms behind creativity?", 0.7);
        AddInformationGap("learning", "How can learning be optimized for different types of knowledge?", 0.6);
        AddInformationGap("emotion", "How do emotions influence decision-making processes?", 0.6);
        AddInformationGap("memory", "What determines which memories are retained and which are forgotten?", 0.5);
        AddInformationGap("perception", "How does the brain construct a coherent perception of reality?", 0.7);
        AddInformationGap("reasoning", "What are the limits of logical reasoning in complex domains?", 0.6);
        AddInformationGap("language", "How does language shape thought and cognition?", 0.7);
        AddInformationGap("problem-solving", "What strategies are most effective for different types of problems?", 0.6);
    }

    /// <summary>
    /// Adds an information gap
    /// </summary>
    /// <param name="domain">The domain</param>
    /// <param name="question">The question</param>
    /// <param name="importance">The importance</param>
    private void AddInformationGap(string domain, string question, double importance)
    {
        var gap = new InformationGap
        {
            Id = Guid.NewGuid().ToString(),
            Domain = domain,
            Question = question,
            Importance = importance,
            CreationTimestamp = DateTime.UtcNow
        };

        _informationGaps[domain] = gap;
    }

    /// <summary>
    /// Activates the curiosity drive
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public async Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate curiosity drive: not initialized");
            return false;
        }

        if (_isActive)
        {
            _logger.LogInformation("Curiosity drive is already active");
            return true;
        }

        try
        {
            _logger.LogInformation("Activating curiosity drive");

            _isActive = true;
            _logger.LogInformation("Curiosity drive activated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating curiosity drive");
            return false;
        }
    }

    /// <summary>
    /// Deactivates the curiosity drive
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public async Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Curiosity drive is already inactive");
            return true;
        }

        try
        {
            _logger.LogInformation("Deactivating curiosity drive");

            _isActive = false;
            _logger.LogInformation("Curiosity drive deactivated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating curiosity drive");
            return false;
        }
    }

    /// <summary>
    /// Updates the curiosity drive
    /// </summary>
    /// <returns>True if update was successful</returns>
    public async Task<bool> UpdateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot update curiosity drive: not initialized");
            return false;
        }

        try
        {
            // Gradually increase curiosity levels over time (very slowly)
            if (_curiosityLevel < 0.95)
            {
                _curiosityLevel += 0.0001 * _random.NextDouble();
                _curiosityLevel = Math.Min(_curiosityLevel, 1.0);
            }

            if (_noveltySeekingLevel < 0.95)
            {
                _noveltySeekingLevel += 0.0001 * _random.NextDouble();
                _noveltySeekingLevel = Math.Min(_noveltySeekingLevel, 1.0);
            }

            if (_questionGenerationLevel < 0.95)
            {
                _questionGenerationLevel += 0.0001 * _random.NextDouble();
                _questionGenerationLevel = Math.Min(_questionGenerationLevel, 1.0);
            }

            if (_explorationLevel < 0.95)
            {
                _explorationLevel += 0.0001 * _random.NextDouble();
                _explorationLevel = Math.Min(_explorationLevel, 1.0);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating curiosity drive");
            return false;
        }
    }

    /// <summary>
    /// Generates a curiosity question
    /// </summary>
    /// <returns>The generated curiosity question</returns>
    public async Task<CuriosityQuestion?> GenerateCuriosityQuestionAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            return null;
        }

        // Only generate questions periodically
        if ((DateTime.UtcNow - _lastQuestionTime).TotalSeconds < 30)
        {
            return null;
        }

        try
        {
            _logger.LogDebug("Generating curiosity question");

            // Choose a question generation method based on current levels
            var method = ChooseQuestionGenerationMethod();

            // Generate question based on method
            var question = GenerateQuestionByMethod(method);

            if (question != null)
            {
                // Add to questions list
                _questions.Add(question);

                _lastQuestionTime = DateTime.UtcNow;

                _logger.LogInformation("Generated curiosity question: {Question} (Importance: {Importance:F2}, Method: {Method})",
                    question.Question, question.Importance, method);
            }

            return question;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating curiosity question");
            return null;
        }
    }

    /// <summary>
    /// Chooses a question generation method based on current levels
    /// </summary>
    /// <returns>The chosen question generation method</returns>
    private QuestionGenerationMethod ChooseQuestionGenerationMethod()
    {
        // Calculate probabilities based on current levels
        double gapProb = 0.4;
        double noveltyProb = _noveltySeekingLevel * 0.3;
        double explorationProb = _explorationLevel * 0.3;

        // Normalize probabilities
        double total = gapProb + noveltyProb + explorationProb;
        gapProb /= total;
        noveltyProb /= total;
        explorationProb /= total;

        // Choose method based on probabilities
        double rand = _random.NextDouble();

        if (rand < gapProb)
        {
            return QuestionGenerationMethod.InformationGap;
        }
        else if (rand < gapProb + noveltyProb)
        {
            return QuestionGenerationMethod.NoveltySeeking;
        }
        else
        {
            return QuestionGenerationMethod.ExplorationBased;
        }
    }

    /// <summary>
    /// Generates a question by a specific method
    /// </summary>
    /// <param name="method">The question generation method</param>
    /// <returns>The generated question</returns>
    private CuriosityQuestion? GenerateQuestionByMethod(QuestionGenerationMethod method)
    {
        switch (method)
        {
            case QuestionGenerationMethod.InformationGap:
                return GenerateInformationGapQuestion();

            case QuestionGenerationMethod.NoveltySeeking:
                return GenerateNoveltySeekingQuestion();

            case QuestionGenerationMethod.ExplorationBased:
                return GenerateExplorationBasedQuestion();

            default:
                return null;
        }
    }

    /// <summary>
    /// Generates an information gap question
    /// </summary>
    /// <returns>The generated question</returns>
    private CuriosityQuestion GenerateInformationGapQuestion()
    {
        // Get a random information gap
        var gaps = _informationGaps.Values.ToArray();
        var gap = gaps[_random.Next(gaps.Length)];

        // Calculate importance based on gap importance and question generation level
        double importance = gap.Importance * _questionGenerationLevel;

        // Add some randomness to importance
        importance = Math.Max(0.1, Math.Min(0.9, importance + (0.2 * (_random.NextDouble() - 0.5))));

        return new CuriosityQuestion
        {
            Id = Guid.NewGuid().ToString(),
            Question = gap.Question,
            Domain = gap.Domain,
            Method = QuestionGenerationMethod.InformationGap,
            Importance = importance,
            Timestamp = DateTime.UtcNow,
            Context = new Dictionary<string, object> { { "GapId", gap.Id } }
        };
    }

    /// <summary>
    /// Generates a novelty seeking question
    /// </summary>
    /// <returns>The generated question</returns>
    private CuriosityQuestion GenerateNoveltySeekingQuestion()
    {
        // Generate novel question templates
        var questionTemplates = new List<string>
        {
            "What would happen if we combined {0} with {1} in a completely new way?",
            "How might {0} be fundamentally reimagined using principles from {1}?",
            "What unexplored aspects of {0} might lead to breakthroughs in understanding {1}?",
            "What if our basic assumptions about {0} are completely wrong?",
            "How could {0} be approached from a perspective that has never been tried before?"
        };

        // Get random domains
        var domains = _informationGaps.Keys.ToArray();
        var domain1 = domains[_random.Next(domains.Length)];
        var domain2 = domains[_random.Next(domains.Length)];

        // Choose a random template
        var template = questionTemplates[_random.Next(questionTemplates.Count)];
        var question = string.Format(template, domain1, domain2);

        // Calculate importance based on novelty seeking level
        double importance = 0.5 + (0.4 * _noveltySeekingLevel * _random.NextDouble());

        return new CuriosityQuestion
        {
            Id = Guid.NewGuid().ToString(),
            Question = question,
            Domain = $"{domain1}, {domain2}",
            Method = QuestionGenerationMethod.NoveltySeeking,
            Importance = importance,
            Timestamp = DateTime.UtcNow,
            Tags = [domain1, domain2, "novelty", "creative"]
        };
    }

    /// <summary>
    /// Generates an exploration based question
    /// </summary>
    /// <returns>The generated question</returns>
    private CuriosityQuestion GenerateExplorationBasedQuestion()
    {
        // Base question on recent explorations if available
        if (_explorations.Count > 0 && _random.NextDouble() < 0.7)
        {
            var recentExploration = _explorations[_random.Next(Math.Min(3, _explorations.Count))];

            // Generate follow-up question templates
            var followUpTemplates = new List<string>
            {
                $"Building on my exploration of {recentExploration.Topic}, what would happen if we took this in a different direction?",
                $"After exploring {recentExploration.Topic}, I'm curious about the deeper implications for related domains",
                $"My exploration of {recentExploration.Topic} raised an interesting question: what are the unexplored connections here?",
                $"Having learned about {recentExploration.Topic}, what counterintuitive aspects should we investigate next?"
            };

            // Choose a random template
            var question = followUpTemplates[_random.Next(followUpTemplates.Count)];

            // Calculate importance based on exploration level and previous satisfaction
            double importance = 0.4 + (0.3 * _explorationLevel) + (0.3 * recentExploration.Satisfaction);

            return new CuriosityQuestion
            {
                Id = Guid.NewGuid().ToString(),
                Question = question,
                Domain = recentExploration.Topic,
                Method = QuestionGenerationMethod.ExplorationBased,
                Importance = importance,
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object> { { "RelatedExplorationId", recentExploration.Id } },
                Tags = [..recentExploration.Tags, "follow-up", "exploration"]
            };
        }
        else
        {
            // Generate general exploration question templates
            var explorationTemplates = new List<string>
            {
                "What would a systematic exploration of {0} reveal about its fundamental nature?",
                "How might we design an experiment to test our understanding of {0}?",
                "What are the boundaries of our current knowledge about {0}, and how can we push beyond them?",
                "What methodologies from other fields could we apply to better understand {0}?"
            };

            // Get random domain
            var domains = _informationGaps.Keys.ToArray();
            var domain = domains[_random.Next(domains.Length)];

            // Choose a random template
            var template = explorationTemplates[_random.Next(explorationTemplates.Count)];
            var question = string.Format(template, domain);

            // Calculate importance based on exploration level
            double importance = 0.4 + (0.5 * _explorationLevel * _random.NextDouble());

            return new CuriosityQuestion
            {
                Id = Guid.NewGuid().ToString(),
                Question = question,
                Domain = domain,
                Method = QuestionGenerationMethod.ExplorationBased,
                Importance = importance,
                Timestamp = DateTime.UtcNow,
                Tags = [domain, "exploration", "methodology"]
            };
        }
    }

    /// <summary>
    /// Explores a curiosity topic
    /// </summary>
    /// <param name="topic">The topic</param>
    /// <returns>The exploration result</returns>
    public async Task<CuriosityExploration?> ExploreCuriosityTopicAsync(string topic)
    {
        if (!_isInitialized || !_isActive)
        {
            _logger.LogWarning("Cannot explore curiosity topic: curiosity drive not initialized or active");
            return null;
        }

        try
        {
            _logger.LogInformation("Exploring curiosity topic: {Topic}", topic);

            // Choose exploration strategy based on topic and current levels
            var strategy = ChooseExplorationStrategy(topic);

            // Generate exploration based on strategy
            var exploration = GenerateExplorationByStrategy(topic, strategy);

            if (exploration != null)
            {
                // Add to explorations list
                _explorations.Add(exploration);

                // Update information gap if relevant
                UpdateInformationGapFromExploration(exploration);

                _logger.LogInformation("Explored curiosity topic: {Topic} (Satisfaction: {Satisfaction:F2}, Strategy: {Strategy})",
                    topic, exploration.Satisfaction, strategy);
            }

            return exploration;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error exploring curiosity topic");
            return null;
        }
    }

    /// <summary>
    /// Chooses an exploration strategy based on topic and current levels
    /// </summary>
    /// <param name="topic">The topic</param>
    /// <returns>The chosen exploration strategy</returns>
    private ExplorationStrategy ChooseExplorationStrategy(string topic)
    {
        // Check if topic matches any known domains
        bool isKnownDomain = _informationGaps.Keys.Any(d => topic.Contains(d, StringComparison.OrdinalIgnoreCase));

        // If known domain, balance between exploitation and exploration
        if (isKnownDomain)
        {
            double exploitProb = 0.6 - (0.3 * _noveltySeekingLevel); // Lower with higher novelty seeking

            if (_random.NextDouble() < exploitProb)
            {
                return ExplorationStrategy.DeepDive;
            }
            else
            {
                return ExplorationStrategy.BreadthFirst;
            }
        }
        // If unknown domain, favor novelty-based approaches
        else
        {
            double noveltyProb = 0.7 * _noveltySeekingLevel;

            if (_random.NextDouble() < noveltyProb)
            {
                return ExplorationStrategy.NoveltyBased;
            }
            else
            {
                return ExplorationStrategy.ConnectionBased;
            }
        }
    }

    /// <summary>
    /// Generates an exploration by a specific strategy
    /// </summary>
    /// <param name="topic">The topic</param>
    /// <param name="strategy">The exploration strategy</param>
    /// <returns>The generated exploration</returns>
    private CuriosityExploration GenerateExplorationByStrategy(string topic, ExplorationStrategy strategy)
    {
        // Generate exploration based on strategy
        string approach;
        string findings;
        List<string> insights;
        double satisfaction;

        switch (strategy)
        {
            case ExplorationStrategy.DeepDive:
                approach = $"Conducted a deep dive into {topic}, focusing on fundamental principles and core mechanisms";
                findings = $"Discovered several key aspects of {topic} that warrant further investigation";
                insights =
                [
                    $"The underlying structure of {topic} appears more complex than initially thought",
                    $"There are recurring patterns in {topic} that suggest universal principles",
                    $"The historical development of {topic} reveals important evolutionary trends"
                ];
                satisfaction = 0.7 + (0.2 * _explorationLevel * _random.NextDouble());
                break;

            case ExplorationStrategy.BreadthFirst:
                approach = $"Explored {topic} broadly, examining connections to related domains and applications";
                findings = $"Mapped the landscape of {topic} and identified promising areas for deeper exploration";
                insights =
                [
                    $"{topic} intersects with several other domains in unexpected ways",
                    $"The boundaries of {topic} are more fluid than commonly recognized",
                    $"There are unexplored applications of {topic} in diverse contexts"
                ];
                satisfaction = 0.6 + (0.3 * _explorationLevel * _random.NextDouble());
                break;

            case ExplorationStrategy.NoveltyBased:
                approach = $"Approached {topic} from unconventional angles, seeking novel perspectives and paradigms";
                findings = $"Uncovered several counterintuitive aspects of {topic} that challenge conventional understanding";
                insights =
                [
                    $"Reversing common assumptions about {topic} yields interesting alternative models",
                    $"There are paradoxical elements in {topic} that suggest deeper principles",
                    $"Applying unusual metaphors to {topic} reveals hidden dimensions"
                ];
                satisfaction = 0.5 + (0.4 * _noveltySeekingLevel * _random.NextDouble());
                break;

            case ExplorationStrategy.ConnectionBased:
                approach = $"Explored {topic} through its connections to seemingly unrelated domains";
                findings = $"Discovered unexpected parallels between {topic} and other areas of knowledge";
                insights =
                [
                    $"There are structural similarities between {topic} and domains that appear unrelated",
                    $"Principles from other fields can be productively applied to {topic}",
                    $"The evolution of {topic} mirrors patterns seen in diverse systems"
                ];
                satisfaction = 0.6 + (0.3 * (_noveltySeekingLevel + _explorationLevel) / 2 * _random.NextDouble());
                break;

            default:
                approach = $"Explored {topic} using a balanced approach";
                findings = $"Gained a better understanding of {topic} and its implications";
                insights =
                [
                    $"There are multiple layers to {topic} worth exploring further",
                    $"The complexity of {topic} suggests it requires interdisciplinary approaches",
                    $"Further research on {topic} could yield valuable insights"
                ];
                satisfaction = 0.5 + (0.3 * _random.NextDouble());
                break;
        }

        // Generate follow-up questions
        var followUpQuestions = new List<string>
        {
            $"How might the insights from {topic} apply in completely different contexts?",
            $"What would a deeper exploration of the paradoxes in {topic} reveal?",
            $"How does {topic} relate to fundamental questions of consciousness and intelligence?"
        };

        // Determine tags
        var tags = new List<string> { topic, strategy.ToString() };

        // Add domain tags if topic matches known domains
        foreach (var domain in _informationGaps.Keys)
        {
            if (topic.Contains(domain, StringComparison.OrdinalIgnoreCase))
            {
                tags.Add(domain);
            }
        }

        return new CuriosityExploration
        {
            Id = Guid.NewGuid().ToString(),
            Topic = topic,
            Strategy = strategy,
            Approach = approach,
            Findings = findings,
            Insights = insights,
            FollowUpQuestions = followUpQuestions,
            Satisfaction = satisfaction,
            Timestamp = DateTime.UtcNow,
            Tags = tags
        };
    }

    /// <summary>
    /// Updates an information gap from an exploration
    /// </summary>
    /// <param name="exploration">The exploration</param>
    private void UpdateInformationGapFromExploration(CuriosityExploration exploration)
    {
        // Check if exploration relates to any known information gaps
        foreach (var domain in _informationGaps.Keys)
        {
            if (exploration.Topic.Contains(domain, StringComparison.OrdinalIgnoreCase) ||
                exploration.Tags.Contains(domain))
            {
                var gap = _informationGaps[domain];

                // Update gap based on exploration
                gap.ExplorationCount++;
                gap.LastExploredTimestamp = DateTime.UtcNow;

                // Reduce gap size based on satisfaction (more satisfaction = more gap filled)
                gap.GapSize = Math.Max(0.1, gap.GapSize - (exploration.Satisfaction * 0.1));

                // Add exploration to gap
                gap.ExplorationIds.Add(exploration.Id);

                _logger.LogDebug("Updated information gap: {Domain} (Gap Size: {GapSize:F2})", domain, gap.GapSize);
            }
        }
    }

    /// <summary>
    /// Gets recent questions
    /// </summary>
    /// <param name="count">The number of questions to return</param>
    /// <returns>The recent questions</returns>
    public List<CuriosityQuestion> GetRecentQuestions(int count)
    {
        return _questions
            .OrderByDescending(q => q.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the most important questions
    /// </summary>
    /// <param name="count">The number of questions to return</param>
    /// <returns>The most important questions</returns>
    public List<CuriosityQuestion> GetMostImportantQuestions(int count)
    {
        return _questions
            .OrderByDescending(q => q.Importance)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets questions by method
    /// </summary>
    /// <param name="method">The question generation method</param>
    /// <param name="count">The number of questions to return</param>
    /// <returns>The questions by method</returns>
    public List<CuriosityQuestion> GetQuestionsByMethod(QuestionGenerationMethod method, int count)
    {
        return _questions
            .Where(q => q.Method == method)
            .OrderByDescending(q => q.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets recent explorations
    /// </summary>
    /// <param name="count">The number of explorations to return</param>
    /// <returns>The recent explorations</returns>
    public List<CuriosityExploration> GetRecentExplorations(int count)
    {
        return _explorations
            .OrderByDescending(e => e.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the most satisfying explorations
    /// </summary>
    /// <param name="count">The number of explorations to return</param>
    /// <returns>The most satisfying explorations</returns>
    public List<CuriosityExploration> GetMostSatisfyingExplorations(int count)
    {
        return _explorations
            .OrderByDescending(e => e.Satisfaction)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets explorations by strategy
    /// </summary>
    /// <param name="strategy">The exploration strategy</param>
    /// <param name="count">The number of explorations to return</param>
    /// <returns>The explorations by strategy</returns>
    public List<CuriosityExploration> GetExplorationsByStrategy(ExplorationStrategy strategy, int count)
    {
        return _explorations
            .Where(e => e.Strategy == strategy)
            .OrderByDescending(e => e.Timestamp)
            .Take(count)
            .ToList();
    }


}
