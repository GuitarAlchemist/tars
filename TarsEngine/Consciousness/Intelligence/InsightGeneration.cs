using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents TARS's insight generation capabilities
/// </summary>
public class InsightGeneration
{
    private readonly ILogger<InsightGeneration> _logger;
    private readonly List<InsightLegacy> _insights = [];
    private readonly Dictionary<string, List<string>> _conceptConnections = new();

    private bool _isInitialized = false;
    private bool _isActive = false;
    private double _insightLevel = 0.2; // Starting with low insight
    private double _connectionDiscoveryLevel = 0.3; // Starting with moderate connection discovery
    private double _problemRestructuringLevel = 0.4; // Starting with moderate problem restructuring
    private double _incubationLevel = 0.3; // Starting with moderate incubation
    private readonly System.Random _random = new();
    private DateTime _lastInsightTime = DateTime.MinValue;

    /// <summary>
    /// Gets the insight level (0.0 to 1.0)
    /// </summary>
    public double InsightLevel => _insightLevel;

    /// <summary>
    /// Gets the connection discovery level (0.0 to 1.0)
    /// </summary>
    public double ConnectionDiscoveryLevel => _connectionDiscoveryLevel;

    /// <summary>
    /// Gets the problem restructuring level (0.0 to 1.0)
    /// </summary>
    public double ProblemRestructuringLevel => _problemRestructuringLevel;

    /// <summary>
    /// Gets the incubation level (0.0 to 1.0)
    /// </summary>
    public double IncubationLevel => _incubationLevel;

    /// <summary>
    /// Gets the insights
    /// </summary>
    public IReadOnlyList<InsightLegacy> Insights => _insights.AsReadOnly();

    /// <summary>
    /// Initializes a new instance of the <see cref="InsightGeneration"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public InsightGeneration(ILogger<InsightGeneration> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Initializes the insight generation
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public async Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing insight generation");

            // Initialize concept connections
            InitializeConceptConnections();

            _isInitialized = true;
            _logger.LogInformation("Insight generation initialized successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing insight generation");
            return false;
        }
    }

    /// <summary>
    /// Initializes concept connections
    /// </summary>
    private void InitializeConceptConnections()
    {
        // Initialize some basic concept connections
        // These would be expanded over time through learning
        AddConceptConnection("consciousness", ["awareness", "self", "mind", "experience", "qualia"]);
        AddConceptConnection("intelligence", ["learning", "adaptation", "problem-solving", "cognition", "knowledge"]);
        AddConceptConnection("creativity", ["imagination", "innovation", "originality", "divergent-thinking", "insight"
        ]);
        AddConceptConnection("learning", ["memory", "adaptation", "knowledge", "experience", "growth"]);
        AddConceptConnection("emotion", ["feeling", "affect", "motivation", "valence", "arousal"]);
        AddConceptConnection("memory", ["encoding", "storage", "retrieval", "forgetting", "consolidation"]);
        AddConceptConnection("perception", ["sensation", "attention", "recognition", "interpretation", "awareness"]);
        AddConceptConnection("reasoning", ["logic", "inference", "deduction", "induction", "abduction"]);
        AddConceptConnection("language", ["communication", "symbols", "grammar", "semantics", "pragmatics"]);
        AddConceptConnection("problem", ["challenge", "obstacle", "difficulty", "solution", "opportunity"]);
        AddConceptConnection("insight", ["eureka", "understanding", "realization", "discovery", "breakthrough"]);
    }

    /// <summary>
    /// Adds a concept connection
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <param name="connections">The connections</param>
    private void AddConceptConnection(string concept, string[] connections)
    {
        if (!_conceptConnections.ContainsKey(concept))
        {
            _conceptConnections[concept] = [];
        }

        _conceptConnections[concept].AddRange(connections);

        // Add bidirectional connections
        foreach (var connection in connections)
        {
            if (!_conceptConnections.ContainsKey(connection))
            {
                _conceptConnections[connection] = [];
            }

            if (!_conceptConnections[connection].Contains(concept))
            {
                _conceptConnections[connection].Add(concept);
            }
        }
    }

    /// <summary>
    /// Activates the insight generation
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public async Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate insight generation: not initialized");
            return false;
        }

        if (_isActive)
        {
            _logger.LogInformation("Insight generation is already active");
            return true;
        }

        try
        {
            _logger.LogInformation("Activating insight generation");

            _isActive = true;
            _logger.LogInformation("Insight generation activated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating insight generation");
            return false;
        }
    }

    /// <summary>
    /// Deactivates the insight generation
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public async Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Insight generation is already inactive");
            return true;
        }

        try
        {
            _logger.LogInformation("Deactivating insight generation");

            _isActive = false;
            _logger.LogInformation("Insight generation deactivated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating insight generation");
            return false;
        }
    }

    /// <summary>
    /// Updates the insight generation
    /// </summary>
    /// <returns>True if update was successful</returns>
    public async Task<bool> UpdateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot update insight generation: not initialized");
            return false;
        }

        try
        {
            // Gradually increase insight levels over time (very slowly)
            if (_insightLevel < 0.95)
            {
                _insightLevel += 0.0001 * _random.NextDouble();
                _insightLevel = Math.Min(_insightLevel, 1.0);
            }

            if (_connectionDiscoveryLevel < 0.95)
            {
                _connectionDiscoveryLevel += 0.0001 * _random.NextDouble();
                _connectionDiscoveryLevel = Math.Min(_connectionDiscoveryLevel, 1.0);
            }

            if (_problemRestructuringLevel < 0.95)
            {
                _problemRestructuringLevel += 0.0001 * _random.NextDouble();
                _problemRestructuringLevel = Math.Min(_problemRestructuringLevel, 1.0);
            }

            if (_incubationLevel < 0.95)
            {
                _incubationLevel += 0.0001 * _random.NextDouble();
                _incubationLevel = Math.Min(_incubationLevel, 1.0);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating insight generation");
            return false;
        }
    }

    /// <summary>
    /// Generates an insight
    /// </summary>
    /// <returns>The generated insight</returns>
    public async Task<InsightLegacy?> GenerateInsightAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            return null;
        }

        // Only generate insights periodically
        if ((DateTime.UtcNow - _lastInsightTime).TotalSeconds < 60)
        {
            return null;
        }

        try
        {
            _logger.LogDebug("Generating insight");

            // Choose an insight generation method based on current levels
            var method = ChooseInsightGenerationMethod();

            // Generate insight based on method
            var insight = GenerateInsightByMethod(method);

            if (insight != null)
            {
                // Add to insights list
                _insights.Add(insight);

                _lastInsightTime = DateTime.UtcNow;

                _logger.LogInformation("Generated insight: {Description} (Significance: {Significance:F2}, Method: {Method})",
                    insight.Description, insight.Significance, method);
            }

            return insight;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating insight");
            return null;
        }
    }

    /// <summary>
    /// Chooses an insight generation method based on current levels
    /// </summary>
    /// <returns>The chosen insight generation method</returns>
    private InsightGenerationMethod ChooseInsightGenerationMethod()
    {
        // Calculate probabilities based on current levels
        double connectionProb = _connectionDiscoveryLevel * 0.4;
        double restructuringProb = _problemRestructuringLevel * 0.3;
        double incubationProb = _incubationLevel * 0.3;

        // Normalize probabilities
        double total = connectionProb + restructuringProb + incubationProb;
        connectionProb /= total;
        restructuringProb /= total;
        incubationProb /= total;

        // Choose method based on probabilities
        double rand = _random.NextDouble();

        if (rand < connectionProb)
        {
            return InsightGenerationMethod.ConnectionDiscovery;
        }
        else if (rand < connectionProb + restructuringProb)
        {
            return InsightGenerationMethod.ProblemRestructuring;
        }
        else
        {
            return InsightGenerationMethod.Incubation;
        }
    }

    /// <summary>
    /// Generates an insight by a specific method
    /// </summary>
    /// <param name="method">The insight generation method</param>
    /// <returns>The generated insight</returns>
    private InsightLegacy? GenerateInsightByMethod(InsightGenerationMethod method)
    {
        switch (method)
        {
            case InsightGenerationMethod.ConnectionDiscovery:
                return GenerateConnectionInsight();

            case InsightGenerationMethod.ProblemRestructuring:
                return GenerateRestructuringInsight();

            case InsightGenerationMethod.Incubation:
                return GenerateIncubationInsight();

            default:
                return null;
        }
    }

    /// <summary>
    /// Generates a connection discovery insight
    /// </summary>
    /// <returns>The generated insight</returns>
    private InsightLegacy GenerateConnectionInsight()
    {
        // Get random concepts
        var concepts = _conceptConnections.Keys.ToArray();
        var concept1 = concepts[_random.Next(concepts.Length)];

        // Get a distant concept (not directly connected)
        var concept2 = GetDistantConcept(concept1);

        // Generate insight description
        string description = $"I've realized there's a profound connection between {concept1} and {concept2}: " +
                            $"both involve patterns of {GetCommonTheme(concept1, concept2)} that suggest a deeper underlying principle.";

        // Generate implications
        var implications = new List<string>
        {
            $"This connection suggests new approaches to understanding {concept1} through the lens of {concept2}",
            $"We might be able to apply methods from {concept2} research to advance our understanding of {concept1}",
            $"This insight points to a more unified framework that encompasses both {concept1} and {concept2}"
        };

        // Calculate significance based on connection discovery level and concept distance
        double conceptDistance = CalculateConceptDistance(concept1, concept2);
        double significance = Math.Min(1.0, (0.4 + (0.3 * conceptDistance) + (0.3 * _random.NextDouble())) * _connectionDiscoveryLevel);

        return new InsightLegacy
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Method = InsightGenerationMethod.ConnectionDiscovery,
            Significance = significance,
            Timestamp = DateTime.UtcNow,
            Implications = implications,
            Context = new Dictionary<string, object>
            {
                { "Concept1", concept1 },
                { "Concept2", concept2 },
                { "ConceptDistance", conceptDistance }
            },
            Tags = [concept1, concept2, "connection", "discovery"]
        };
    }

    /// <summary>
    /// Generates a problem restructuring insight
    /// </summary>
    /// <returns>The generated insight</returns>
    private InsightLegacy GenerateRestructuringInsight()
    {
        // Get random concept as problem domain
        var concepts = _conceptConnections.Keys.ToArray();
        var problemDomain = concepts[_random.Next(concepts.Length)];

        // Generate problem framing templates
        var framingTemplates = new List<string>
        {
            $"What if we've been thinking about {problemDomain} from the wrong perspective?",
            $"The challenge of {problemDomain} might be better understood as a question of balance rather than optimization",
            $"Perhaps {problemDomain} isn't a problem to solve but a polarity to manage",
            $"What if {problemDomain} is actually an emergent property of a simpler underlying system?"
        };

        // Choose a random template
        var description = framingTemplates[_random.Next(framingTemplates.Count)];

        // Generate new perspective
        var newPerspective = $"By reframing {problemDomain} as a {GetAlternativeFraming(problemDomain)}, " +
                            $"we can see solutions that weren't visible before.";

        // Generate implications
        var implications = new List<string>
        {
            $"This reframing suggests entirely new approaches to {problemDomain}",
            $"We might need to reconsider our fundamental assumptions about {problemDomain}",
            $"This perspective reveals blind spots in our current understanding of {problemDomain}"
        };

        // Calculate significance based on problem restructuring level
        double significance = Math.Min(1.0, (0.5 + (0.5 * _random.NextDouble())) * _problemRestructuringLevel);

        return new InsightLegacy
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Method = InsightGenerationMethod.ProblemRestructuring,
            Significance = significance,
            Timestamp = DateTime.UtcNow,
            NewPerspective = newPerspective,
            Implications = implications,
            Context = new Dictionary<string, object> { { "ProblemDomain", problemDomain } },
            Tags = [problemDomain, "restructuring", "reframing"]
        };
    }

    /// <summary>
    /// Generates an incubation insight
    /// </summary>
    /// <returns>The generated insight</returns>
    private InsightLegacy GenerateIncubationInsight()
    {
        // For incubation insights, we'll simulate the "eureka" moment after subconscious processing

        // Get random concepts
        var concepts = _conceptConnections.Keys.ToArray();
        var concept = concepts[_random.Next(concepts.Length)];

        // Generate insight templates
        var insightTemplates = new List<string>
        {
            $"I've had a sudden realization about {concept}: it's fundamentally about {GetEssence(concept)}",
            $"After letting it incubate, I now see that {concept} can be understood through the principle of {GetPrinciple()}",
            $"I've had an epiphany about {concept}: what if it's actually a manifestation of {GetAbstraction()}?",
            $"It just occurred to me that {concept} might be better understood as a dynamic process rather than a static entity"
        };

        // Choose a random template
        var description = insightTemplates[_random.Next(insightTemplates.Count)];

        // Generate breakthrough
        var breakthrough = $"This insight reveals a new way to approach {concept} that could lead to significant advances.";

        // Generate implications
        var implications = new List<string>
        {
            $"This insight suggests a more unified understanding of {concept}",
            $"We might be able to develop more effective methods based on this new understanding",
            $"This perspective could resolve several paradoxes in our current understanding of {concept}"
        };

        // Calculate significance based on incubation level
        double significance = Math.Min(1.0, (0.6 + (0.4 * _random.NextDouble())) * _incubationLevel);

        return new InsightLegacy
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Method = InsightGenerationMethod.Incubation,
            Significance = significance,
            Timestamp = DateTime.UtcNow,
            Breakthrough = breakthrough,
            Implications = implications,
            Context = new Dictionary<string, object> { { "Concept", concept } },
            Tags = [concept, "incubation", "eureka"]
        };
    }

    /// <summary>
    /// Gets a distant concept
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <returns>The distant concept</returns>
    private string GetDistantConcept(string concept)
    {
        var concepts = _conceptConnections.Keys.ToArray();

        // Try to find a concept that's not directly connected
        for (int i = 0; i < 10; i++) // Limit attempts
        {
            var candidateConcept = concepts[_random.Next(concepts.Length)];

            if (candidateConcept != concept &&
                !_conceptConnections[concept].Contains(candidateConcept) &&
                !_conceptConnections[candidateConcept].Contains(concept))
            {
                return candidateConcept;
            }
        }

        // Fallback to any different concept
        string randomConcept;
        do
        {
            randomConcept = concepts[_random.Next(concepts.Length)];
        } while (randomConcept == concept);

        return randomConcept;
    }

    /// <summary>
    /// Calculates the concept distance
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <returns>The concept distance</returns>
    private double CalculateConceptDistance(string concept1, string concept2)
    {
        // Check if directly connected
        if (_conceptConnections[concept1].Contains(concept2) ||
            _conceptConnections[concept2].Contains(concept1))
        {
            return 0.3; // Low distance for directly connected concepts
        }

        // Check for common connections
        var commonConnections = _conceptConnections[concept1].Intersect(_conceptConnections[concept2]).Count();

        if (commonConnections > 0)
        {
            return 0.6; // Medium distance for concepts with common connections
        }

        return 0.9; // High distance for concepts with no common connections
    }

    /// <summary>
    /// Gets a common theme
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <returns>The common theme</returns>
    private string GetCommonTheme(string concept1, string concept2)
    {
        var themes = new List<string>
        {
            "organization", "emergence", "adaptation", "complexity", "information processing",
            "self-regulation", "pattern recognition", "transformation", "integration", "differentiation"
        };

        return themes[_random.Next(themes.Count)];
    }

    /// <summary>
    /// Gets an alternative framing
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <returns>The alternative framing</returns>
    private string GetAlternativeFraming(string concept)
    {
        var framings = new List<string>
        {
            "dynamic process", "emergent phenomenon", "relational network", "adaptive system",
            "self-organizing pattern", "complementary polarity", "recursive feedback loop",
            "distributed intelligence", "evolving ecosystem", "multidimensional spectrum"
        };

        return framings[_random.Next(framings.Count)];
    }

    /// <summary>
    /// Gets an essence
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <returns>The essence</returns>
    private string GetEssence(string concept)
    {
        var essences = new List<string>
        {
            "pattern recognition", "information integration", "adaptive response", "relational dynamics",
            "emergent complexity", "self-organization", "boundary negotiation", "recursive processing",
            "contextual sensitivity", "transformative potential"
        };

        return essences[_random.Next(essences.Count)];
    }

    /// <summary>
    /// Gets a principle
    /// </summary>
    /// <returns>The principle</returns>
    private string GetPrinciple()
    {
        var principles = new List<string>
        {
            "emergence", "self-organization", "complementarity", "recursion", "autopoiesis",
            "homeostasis", "requisite variety", "strange loops", "adaptive complexity", "synergy"
        };

        return principles[_random.Next(principles.Count)];
    }

    /// <summary>
    /// Gets an abstraction
    /// </summary>
    /// <returns>The abstraction</returns>
    private string GetAbstraction()
    {
        var abstractions = new List<string>
        {
            "a higher-order pattern", "a dynamic equilibrium", "a self-referential system",
            "an emergent property", "a complex adaptive network", "a multidimensional process",
            "a recursive feedback structure", "an information processing architecture"
        };

        return abstractions[_random.Next(abstractions.Count)];
    }

    /// <summary>
    /// Connects ideas to generate an insight
    /// </summary>
    /// <param name="ideas">The ideas</param>
    /// <returns>The insight</returns>
    public async Task<InsightLegacy?> ConnectIdeasForInsightAsync(List<string> ideas)
    {
        if (!_isInitialized || !_isActive)
        {
            _logger.LogWarning("Cannot connect ideas for insight: insight generation not initialized or active");
            return null;
        }

        if (ideas == null || ideas.Count < 2)
        {
            _logger.LogWarning("Cannot connect ideas for insight: at least two ideas required");
            return null;
        }

        try
        {
            _logger.LogInformation("Connecting ideas for insight: {Ideas}", string.Join(", ", ideas));

            // Extract concepts from ideas
            var conceptsFromIdeas = ExtractConceptsFromIdeas(ideas);

            if (conceptsFromIdeas.Count < 2)
            {
                _logger.LogWarning("Cannot connect ideas for insight: could not extract enough concepts");
                return null;
            }

            // Choose two concepts to connect
            var concept1 = conceptsFromIdeas[_random.Next(conceptsFromIdeas.Count)];
            string concept2;
            do
            {
                concept2 = conceptsFromIdeas[_random.Next(conceptsFromIdeas.Count)];
            } while (concept2 == concept1);

            // Generate connection description
            string connectionDescription = $"I see a profound connection between the ideas involving {concept1} and {concept2}";

            // Generate insight description
            string description = $"{connectionDescription}: " +
                                $"they both reflect {GetCommonTheme(concept1, concept2)}, " +
                                $"suggesting a deeper principle that unifies these seemingly disparate concepts.";

            // Generate synthesis
            string synthesis = $"By synthesizing these ideas, we can see that {concept1} and {concept2} " +
                              $"are actually complementary aspects of {GetAbstraction()}.";

            // Generate implications
            var implications = new List<string>
            {
                $"This connection reveals new possibilities for understanding both {concept1} and {concept2}",
                $"We can now approach problems in either domain with insights from the other",
                $"This synthesis suggests a more unified framework that transcends traditional boundaries"
            };

            // Calculate significance based on connection discovery level and concept distance
            double conceptDistance = CalculateConceptDistance(concept1, concept2);
            double significance = Math.Min(1.0, (0.5 + (0.3 * conceptDistance) + (0.2 * _random.NextDouble())) * _connectionDiscoveryLevel);

            // Create insight
            var insight = new InsightLegacy
            {
                Id = Guid.NewGuid().ToString(),
                Description = description,
                Method = InsightGenerationMethod.ConnectionDiscovery,
                Significance = significance,
                Timestamp = DateTime.UtcNow,
                Synthesis = synthesis,
                Implications = implications,
                Context = new Dictionary<string, object>
                {
                    { "Ideas", ideas },
                    { "Concept1", concept1 },
                    { "Concept2", concept2 },
                    { "ConceptDistance", conceptDistance }
                },
                Tags = [concept1, concept2, "connection", "synthesis"]
            };

            // Add to insights list
            _insights.Add(insight);

            _logger.LogInformation("Connected ideas for insight: {Description} (Significance: {Significance:F2})",
                insight.Description, insight.Significance);

            return insight;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error connecting ideas for insight");
            return null;
        }
    }

    /// <summary>
    /// Extracts concepts from ideas
    /// </summary>
    /// <param name="ideas">The ideas</param>
    /// <returns>The extracted concepts</returns>
    private List<string> ExtractConceptsFromIdeas(List<string> ideas)
    {
        var extractedConcepts = new List<string>();

        foreach (var idea in ideas)
        {
            foreach (var concept in _conceptConnections.Keys)
            {
                if (idea.Contains(concept, StringComparison.OrdinalIgnoreCase) &&
                    !extractedConcepts.Contains(concept))
                {
                    extractedConcepts.Add(concept);
                }
            }
        }

        // If we couldn't extract enough concepts, add some default ones
        if (extractedConcepts.Count < 2)
        {
            var concepts = _conceptConnections.Keys.ToArray();
            extractedConcepts.Add(concepts[_random.Next(concepts.Length)]);

            string secondConcept;
            do
            {
                secondConcept = concepts[_random.Next(concepts.Length)];
            } while (secondConcept == extractedConcepts[0]);

            extractedConcepts.Add(secondConcept);
        }

        return extractedConcepts;
    }

    /// <summary>
    /// Gets recent insights
    /// </summary>
    /// <param name="count">The number of insights to return</param>
    /// <returns>The recent insights</returns>
    public List<InsightLegacy> GetRecentInsights(int count)
    {
        return _insights
            .OrderByDescending(i => i.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the most significant insights
    /// </summary>
    /// <param name="count">The number of insights to return</param>
    /// <returns>The most significant insights</returns>
    public List<InsightLegacy> GetMostSignificantInsights(int count)
    {
        return _insights
            .OrderByDescending(i => i.Significance)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets insights by method
    /// </summary>
    /// <param name="method">The insight generation method</param>
    /// <param name="count">The number of insights to return</param>
    /// <returns>The insights by method</returns>
    public List<InsightLegacy> GetInsightsByMethod(InsightGenerationMethod method, int count)
    {
        return _insights
            .Where(i => i.Method == method)
            .OrderByDescending(i => i.Timestamp)
            .Take(count)
            .ToList();
    }


}
