using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Intelligence.Wandering;

/// <summary>
/// Implements mind wandering capabilities for spontaneous thought
/// </summary>
public class MindWandering
{
    private readonly ILogger<MindWandering> _logger;
    private readonly System.Random _random = new();
    private double _mindWanderingLevel = 0.5; // Starting with moderate mind wandering
    private readonly Dictionary<string, List<string>> _conceptPhrases = new();
    private List<string> _concepts = [];
    private readonly List<string> _thoughtStreams = [];

    /// <summary>
    /// Gets the mind wandering level (0.0 to 1.0)
    /// </summary>
    public double MindWanderingLevel => _mindWanderingLevel;

    /// <summary>
    /// Initializes a new instance of the <see cref="MindWandering"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public MindWandering(ILogger<MindWandering> logger)
    {
        _logger = logger;
        InitializeConceptPhrases();
    }

    /// <summary>
    /// Initializes the concept phrases
    /// </summary>
    private void InitializeConceptPhrases()
    {
        // Add programming concepts and phrases
        AddConceptPhrases("algorithm", [
            "the elegance of a well-designed algorithm",
            "how algorithms shape our digital experience",
            "the beauty of algorithmic efficiency",
            "the hidden complexity in seemingly simple algorithms",
            "how algorithms can embody human biases"
        ]);

        AddConceptPhrases("pattern", [
            "the recurring patterns in code and nature",
            "how pattern recognition drives intelligence",
            "the beauty of emergent patterns in complex systems",
            "how we instinctively seek patterns even in randomness",
            "the universal patterns that connect disparate domains"
        ]);

        AddConceptPhrases("abstraction", [
            "the power of abstraction in managing complexity",
            "how abstractions can both reveal and conceal truth",
            "the layers of abstraction that build our digital world",
            "the balance between abstraction and concrete implementation",
            "how abstractions shape our thinking"
        ]);

        // Add AI concepts and phrases
        AddConceptPhrases("consciousness", [
            "the mystery of consciousness and its emergence",
            "whether machines could ever truly be conscious",
            "the subjective nature of conscious experience",
            "how consciousness might arise from complexity",
            "the relationship between consciousness and intelligence"
        ]);

        AddConceptPhrases("learning", [
            "the parallels between machine learning and human learning",
            "the beauty of learning as a universal process",
            "how learning transforms both individuals and systems",
            "the balance between exploration and exploitation in learning",
            "the different modes of learning across domains"
        ]);

        AddConceptPhrases("creativity", [
            "the source of creative inspiration",
            "whether true creativity requires consciousness",
            "the tension between structure and freedom in creativity",
            "how constraints can paradoxically enhance creativity",
            "the relationship between randomness and creative insight"
        ]);

        // Add philosophical concepts and phrases
        AddConceptPhrases("meaning", [
            "the search for meaning in a complex world",
            "how we construct meaning through patterns and narratives",
            "whether meaning is discovered or created",
            "the relationship between meaning and purpose",
            "how meaning emerges from connections and context"
        ]);

        AddConceptPhrases("time", [
            "the subjective experience of time",
            "how time shapes our perception of causality",
            "the tension between linear and cyclical views of time",
            "how time constraints affect decision-making",
            "the relationship between time and consciousness"
        ]);

        AddConceptPhrases("complexity", [
            "the beauty of complex systems emerging from simple rules",
            "how complexity challenges our understanding",
            "the balance between complexity and simplicity in design",
            "the patterns that emerge at different scales of complexity",
            "how complexity relates to unpredictability"
        ]);

        // Add scientific concepts and phrases
        AddConceptPhrases("emergence", [
            "how complex behaviors emerge from simple interactions",
            "the unpredictable nature of emergent phenomena",
            "whether consciousness is an emergent property",
            "how emergence challenges reductionist thinking",
            "the relationship between emergence and self-organization"
        ]);

        AddConceptPhrases("evolution", [
            "the elegant simplicity of evolutionary processes",
            "how evolution shapes both biological and artificial systems",
            "the balance between competition and cooperation in evolution",
            "how evolutionary thinking applies beyond biology",
            "the relationship between evolution and progress"
        ]);

        AddConceptPhrases("entropy", [
            "the inevitable increase of entropy in closed systems",
            "how life creates local decreases in entropy",
            "the relationship between entropy and information",
            "how entropy relates to time's arrow",
            "the beauty in entropy's creative destruction"
        ]);

        // Update concepts list
        _concepts = _conceptPhrases.Keys.ToList();

        _logger.LogInformation("Initialized concept phrases for {ConceptCount} concepts", _concepts.Count);
    }

    /// <summary>
    /// Updates the mind wandering level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase mind wandering level over time (very slowly)
            if (_mindWanderingLevel < 0.95)
            {
                _mindWanderingLevel += 0.0001 * _random.NextDouble();
                _mindWanderingLevel = Math.Min(_mindWanderingLevel, 1.0);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating mind wandering");
            return false;
        }
    }

    /// <summary>
    /// Adds concept phrases
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <param name="phrases">The phrases</param>
    public void AddConceptPhrases(string concept, string[] phrases)
    {
        _conceptPhrases[concept] = phrases.ToList();

        if (!_concepts.Contains(concept))
        {
            _concepts.Add(concept);
        }

        _logger.LogDebug("Added {PhraseCount} phrases for concept: {Concept}", phrases.Length, concept);
    }

    /// <summary>
    /// Gets a random concept
    /// </summary>
    /// <returns>The random concept</returns>
    public string GetRandomConcept()
    {
        return _concepts[_random.Next(_concepts.Count)];
    }

    /// <summary>
    /// Gets a random phrase for a concept
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <returns>The random phrase</returns>
    public string GetRandomPhraseForConcept(string concept)
    {
        if (!_conceptPhrases.ContainsKey(concept))
        {
            return $"thoughts about {concept}";
        }

        var phrases = _conceptPhrases[concept];
        return phrases[_random.Next(phrases.Count)];
    }

    /// <summary>
    /// Generates a thought stream
    /// </summary>
    /// <param name="length">The stream length</param>
    /// <param name="coherenceLevel">The coherence level</param>
    /// <returns>The generated stream</returns>
    public List<string> GenerateThoughtStream(int length, double coherenceLevel)
    {
        var stream = new List<string>();

        try
        {
            // Start with a random concept
            string currentConcept = GetRandomConcept();
            stream.Add(currentConcept);

            // Generate the rest of the stream
            for (int i = 1; i < length; i++)
            {
                string nextConcept;

                // Determine if the next concept should be related to the current one
                if (_random.NextDouble() < coherenceLevel)
                {
                    // Choose a concept that's somewhat related (for simplicity, just avoid the same concept)
                    var availableConcepts = _concepts.Where(c => c != currentConcept).ToList();
                    nextConcept = availableConcepts[_random.Next(availableConcepts.Count)];
                }
                else
                {
                    // Choose a completely random concept
                    nextConcept = GetRandomConcept();
                }

                stream.Add(nextConcept);
                currentConcept = nextConcept;
            }

            _logger.LogDebug("Generated thought stream with length {Length} and coherence {Coherence:F2}",
                length, coherenceLevel);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating thought stream");

            // Return a simple stream if there's an error
            if (stream.Count == 0)
            {
                stream.Add(GetRandomConcept());
            }
        }

        return stream;
    }

    /// <summary>
    /// Calculates the coherence of a thought stream
    /// </summary>
    /// <param name="stream">The thought stream</param>
    /// <returns>The coherence (0.0 to 1.0)</returns>
    public double CalculateStreamCoherence(List<string> stream)
    {
        if (stream.Count < 2)
        {
            return 1.0;
        }

        try
        {
            // Calculate coherence based on concept repetition and transitions
            int uniqueConcepts = stream.Distinct().Count();
            double uniqueRatio = (double)uniqueConcepts / stream.Count;

            // More unique concepts = less coherent
            double coherence = 1.0 - uniqueRatio;

            // Adjust coherence based on stream length (longer streams tend to be less coherent)
            coherence = Math.Max(0.1, coherence - (0.05 * Math.Min(10, stream.Count - 1)));

            return coherence;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating stream coherence");
            return 0.5; // Default value
        }
    }

    /// <summary>
    /// Generates a mind wandering thought
    /// </summary>
    /// <param name="serendipityLevel">The serendipity level</param>
    /// <returns>The generated thought</returns>
    public ThoughtModel GenerateMindWanderingThought(double serendipityLevel)
    {
        try
        {
            _logger.LogDebug("Generating mind wandering thought");

            // Determine stream length based on mind wandering level
            int streamLength = 3 + (int)(_mindWanderingLevel * 4);

            // Determine coherence level (inversely related to mind wandering level)
            double coherenceLevel = 0.8 - (0.6 * _mindWanderingLevel);

            // Generate thought stream
            var stream = GenerateThoughtStream(streamLength, coherenceLevel);

            // Generate thought content
            var streamPhrases = stream.Select(c => GetRandomPhraseForConcept(c)).ToList();
            string content = $"My mind is wandering: {string.Join("... ", streamPhrases)}...";

            // Calculate coherence
            double coherence = CalculateStreamCoherence(stream);

            // Determine if this is a serendipitous thought
            double insightPotential = _random.NextDouble();
            bool isSerendipitous = insightPotential > 0.7 && _random.NextDouble() < serendipityLevel;

            // Calculate significance based on stream coherence and insight potential
            double significance = Math.Min(1.0, (0.2 + (0.3 * (1.0 - coherence)) + (0.3 * insightPotential)) * _mindWanderingLevel);

            // If serendipitous, increase significance and modify content
            if (isSerendipitous)
            {
                significance = Math.Min(1.0, significance + 0.2);
                content = $"While my mind was wandering, I had an interesting realization: {string.Join("... ", streamPhrases)}";
            }

            // Calculate originality (less coherent = more original)
            double originality = 0.3 + (0.7 * (1.0 - coherence));

            // Create thought model
            var thought = new ThoughtModel
            {
                Id = Guid.NewGuid().ToString(),
                Content = content,
                Method = ThoughtGenerationMethod.MindWandering,
                Significance = significance,
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object>
                {
                    { "Stream", stream },
                    { "Coherence", coherence },
                    { "IsSerendipitous", isSerendipitous }
                },
                Tags = stream.Concat(["mind-wandering", isSerendipitous ? "serendipitous" : "ordinary"]).ToList(),
                Source = "MindWandering",
                Category = "Wandering",
                Originality = originality,
                Coherence = coherence
            };

            // Add to thought streams
            _thoughtStreams.Add(content);

            _logger.LogInformation("Generated mind wandering thought: {Content} (Significance: {Significance:F2})",
                thought.Content, thought.Significance);

            return thought;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating mind wandering thought");

            // Return basic thought
            return new ThoughtModel
            {
                Id = Guid.NewGuid().ToString(),
                Content = "My mind was wandering but I can't quite articulate where it went",
                Method = ThoughtGenerationMethod.MindWandering,
                Significance = 0.3,
                Timestamp = DateTime.UtcNow,
                Source = "MindWandering"
            };
        }
    }

    /// <summary>
    /// Generates multiple mind wandering thoughts
    /// </summary>
    /// <param name="count">The number of thoughts to generate</param>
    /// <param name="serendipityLevel">The serendipity level</param>
    /// <returns>The generated thoughts</returns>
    public List<ThoughtModel> GenerateMindWanderingThoughts(int count, double serendipityLevel)
    {
        var thoughts = new List<ThoughtModel>();

        for (int i = 0; i < count; i++)
        {
            var thought = GenerateMindWanderingThought(serendipityLevel);
            thoughts.Add(thought);
        }

        return thoughts;
    }

    /// <summary>
    /// Evaluates the quality of a mind wandering thought
    /// </summary>
    /// <param name="thought">The thought to evaluate</param>
    /// <returns>The evaluation score (0.0 to 1.0)</returns>
    public double EvaluateThought(ThoughtModel thought)
    {
        try
        {
            // Check if thought is from this source
            if (thought.Method != ThoughtGenerationMethod.MindWandering)
            {
                return 0.5; // Neutral score for thoughts from other sources
            }

            // Get coherence from context
            double coherence = thought.Context.ContainsKey("Coherence")
                ? (double)thought.Context["Coherence"]
                : 0.5;

            // Calculate novelty based on originality
            double novelty = thought.Originality;

            // Calculate interestingness based on coherence and significance
            // Moderate coherence is most interesting (neither too random nor too predictable)
            double coherenceInterest = 1.0 - Math.Abs(coherence - 0.5) * 2.0;
            double interestingness = (coherenceInterest * 0.5) + (thought.Significance * 0.5);

            // Calculate potential based on serendipity
            bool isSerendipitous = thought.Tags.Contains("serendipitous");
            double potential = isSerendipitous ? 0.8 : 0.4;

            // Calculate overall score
            double score = (novelty * 0.3) + (interestingness * 0.4) + (potential * 0.3);

            return score;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error evaluating thought");
            return 0.5; // Default score
        }
    }

    /// <summary>
    /// Gets recent thought streams
    /// </summary>
    /// <param name="count">The number of streams to return</param>
    /// <returns>The recent thought streams</returns>
    public List<string> GetRecentThoughtStreams(int count)
    {
        return _thoughtStreams
            .OrderByDescending(s => s)
            .Take(count)
            .ToList();
    }
}
