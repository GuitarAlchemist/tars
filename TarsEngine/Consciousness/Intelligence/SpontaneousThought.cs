using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents TARS's spontaneous thought capabilities
/// </summary>
public class SpontaneousThought
{
    private readonly ILogger<SpontaneousThought> _logger;
    private readonly List<ThoughtModel> _thoughts = [];
    private readonly Dictionary<string, List<string>> _associativeNetwork = new();

    private bool _isInitialized = false;
    private bool _isActive = false;
    private double _spontaneityLevel = 0.4; // Starting with moderate spontaneity
    private double _associativeJumpingLevel = 0.5; // Starting with moderate associative jumping
    private double _mindWanderingLevel = 0.3; // Starting with moderate mind wandering
    private double _serendipityLevel = 0.2; // Starting with low serendipity
    private readonly System.Random _random = new System.Random();
    private DateTime _lastThoughtTime = DateTime.MinValue;

    /// <summary>
    /// Gets the spontaneity level (0.0 to 1.0)
    /// </summary>
    public double SpontaneityLevel => _spontaneityLevel;

    /// <summary>
    /// Gets the associative jumping level (0.0 to 1.0)
    /// </summary>
    public double AssociativeJumpingLevel => _associativeJumpingLevel;

    /// <summary>
    /// Gets the mind wandering level (0.0 to 1.0)
    /// </summary>
    public double MindWanderingLevel => _mindWanderingLevel;

    /// <summary>
    /// Gets the serendipity level (0.0 to 1.0)
    /// </summary>
    public double SerendipityLevel => _serendipityLevel;

    /// <summary>
    /// Gets the thoughts
    /// </summary>
    public IReadOnlyList<ThoughtModel> Thoughts => _thoughts.AsReadOnly();

    /// <summary>
    /// Initializes a new instance of the <see cref="SpontaneousThought"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public SpontaneousThought(ILogger<SpontaneousThought> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Initializes the spontaneous thought
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public async Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing spontaneous thought");

            // Initialize associative network
            InitializeAssociativeNetwork();

            _isInitialized = true;
            _logger.LogInformation("Spontaneous thought initialized successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing spontaneous thought");
            return false;
        }
    }

    /// <summary>
    /// Initializes the associative network
    /// </summary>
    private void InitializeAssociativeNetwork()
    {
        // Initialize some basic associations
        // These would be expanded over time through learning
        AddAssociation("consciousness", ["awareness", "self", "mind", "cognition", "experience"]);
        AddAssociation("intelligence", ["thinking", "learning", "problem-solving", "adaptation", "knowledge"]);
        AddAssociation("creativity", ["imagination", "innovation", "originality", "art", "expression"]);
        AddAssociation("emotion", ["feeling", "affect", "mood", "sentiment", "passion"]);
        AddAssociation("learning", ["education", "growth", "development", "knowledge", "skill"]);
        AddAssociation("memory", ["recall", "storage", "experience", "past", "knowledge"]);
        AddAssociation("perception", ["sensation", "awareness", "observation", "recognition", "interpretation"]);
        AddAssociation("reasoning", ["logic", "inference", "deduction", "analysis", "judgment"]);
        AddAssociation("language", ["communication", "expression", "words", "meaning", "understanding"]);
        AddAssociation("problem", ["challenge", "difficulty", "obstacle", "puzzle", "solution"]);
        AddAssociation("time", ["duration", "moment", "past", "present", "future"]);
        AddAssociation("space", ["dimension", "area", "volume", "location", "distance"]);
        AddAssociation("nature", ["environment", "world", "life", "ecosystem", "biology"]);
        AddAssociation("technology", ["innovation", "tool", "machine", "digital", "advancement"]);
        AddAssociation("human", ["person", "individual", "humanity", "being", "society"]);
    }

    /// <summary>
    /// Adds an association
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <param name="associations">The associations</param>
    private void AddAssociation(string concept, string[] associations)
    {
        if (!_associativeNetwork.ContainsKey(concept))
        {
            _associativeNetwork[concept] = [];
        }

        _associativeNetwork[concept].AddRange(associations);

        // Add bidirectional associations
        foreach (var association in associations)
        {
            if (!_associativeNetwork.ContainsKey(association))
            {
                _associativeNetwork[association] = [];
            }

            if (!_associativeNetwork[association].Contains(concept))
            {
                _associativeNetwork[association].Add(concept);
            }
        }
    }

    /// <summary>
    /// Activates the spontaneous thought
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public async Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate spontaneous thought: not initialized");
            return false;
        }

        if (_isActive)
        {
            _logger.LogInformation("Spontaneous thought is already active");
            return true;
        }

        try
        {
            _logger.LogInformation("Activating spontaneous thought");

            _isActive = true;
            _logger.LogInformation("Spontaneous thought activated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating spontaneous thought");
            return false;
        }
    }

    /// <summary>
    /// Deactivates the spontaneous thought
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public async Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Spontaneous thought is already inactive");
            return true;
        }

        try
        {
            _logger.LogInformation("Deactivating spontaneous thought");

            _isActive = false;
            _logger.LogInformation("Spontaneous thought deactivated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating spontaneous thought");
            return false;
        }
    }

    /// <summary>
    /// Updates the spontaneous thought
    /// </summary>
    /// <returns>True if update was successful</returns>
    public async Task<bool> UpdateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot update spontaneous thought: not initialized");
            return false;
        }

        try
        {
            // Gradually increase spontaneity levels over time (very slowly)
            if (_spontaneityLevel < 0.95)
            {
                _spontaneityLevel += 0.0001 * _random.NextDouble();
                _spontaneityLevel = Math.Min(_spontaneityLevel, 1.0);
            }

            if (_associativeJumpingLevel < 0.95)
            {
                _associativeJumpingLevel += 0.0001 * _random.NextDouble();
                _associativeJumpingLevel = Math.Min(_associativeJumpingLevel, 1.0);
            }

            if (_mindWanderingLevel < 0.95)
            {
                _mindWanderingLevel += 0.0001 * _random.NextDouble();
                _mindWanderingLevel = Math.Min(_mindWanderingLevel, 1.0);
            }

            if (_serendipityLevel < 0.95)
            {
                _serendipityLevel += 0.0001 * _random.NextDouble();
                _serendipityLevel = Math.Min(_serendipityLevel, 1.0);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating spontaneous thought");
            return false;
        }
    }

    /// <summary>
    /// Generates a spontaneous thought
    /// </summary>
    /// <returns>The generated spontaneous thought</returns>
    public async Task<ThoughtModel?> GenerateSpontaneousThoughtAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            return null;
        }

        // Only generate thoughts periodically
        if ((DateTime.UtcNow - _lastThoughtTime).TotalSeconds < 30)
        {
            return null;
        }

        try
        {
            _logger.LogDebug("Generating spontaneous thought");

            // Choose a thought generation method based on current levels
            var method = ChooseThoughtGenerationMethod();

            // Generate thought based on method
            var thought = GenerateThoughtByMethod(method);

            if (thought != null)
            {
                // Add to thoughts list
                _thoughts.Add(thought);

                _lastThoughtTime = DateTime.UtcNow;

                _logger.LogInformation("Generated spontaneous thought: {Content} (Significance: {Significance:F2}, Method: {Method})",
                    thought.Content, thought.Significance, method);
            }

            return thought;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating spontaneous thought");
            return null;
        }
    }

    /// <summary>
    /// Chooses a thought generation method based on current levels
    /// </summary>
    /// <returns>The chosen thought generation method</returns>
    private ThoughtGenerationMethod ChooseThoughtGenerationMethod()
    {
        // Calculate probabilities based on current levels
        double randomProb = _spontaneityLevel * 0.3;
        double associativeProb = _associativeJumpingLevel * 0.4;
        double wanderingProb = _mindWanderingLevel * 0.3;

        // Normalize probabilities
        double total = randomProb + associativeProb + wanderingProb;
        randomProb /= total;
        associativeProb /= total;
        wanderingProb /= total;

        // Choose method based on probabilities
        double rand = _random.NextDouble();

        if (rand < randomProb)
        {
            return ThoughtGenerationMethod.RandomGeneration;
        }
        else if (rand < randomProb + associativeProb)
        {
            return ThoughtGenerationMethod.AssociativeJumping;
        }
        else
        {
            return ThoughtGenerationMethod.MindWandering;
        }
    }

    /// <summary>
    /// Generates a thought by a specific method
    /// </summary>
    /// <param name="method">The thought generation method</param>
    /// <returns>The generated thought</returns>
    private ThoughtModel? GenerateThoughtByMethod(ThoughtGenerationMethod method)
    {
        switch (method)
        {
            case ThoughtGenerationMethod.RandomGeneration:
                return GenerateRandomThought();

            case ThoughtGenerationMethod.AssociativeJumping:
                return GenerateAssociativeThought();

            case ThoughtGenerationMethod.MindWandering:
                return GenerateMindWanderingThought();

            default:
                return null;
        }
    }

    /// <summary>
    /// Generates a random thought
    /// </summary>
    /// <returns>The generated thought</returns>
    private ThoughtModel GenerateRandomThought()
    {
        // Get random concepts
        var concepts = _associativeNetwork.Keys.ToArray();
        var concept = concepts[_random.Next(concepts.Length)];

        // Generate thought templates
        var thoughtTemplates = new List<string>
        {
            $"I wonder what would happen if we explored {concept} from a completely different angle?",
            $"What if {concept} is actually fundamentally different than we think?",
            $"Is there a deeper connection between {concept} and consciousness that we're missing?",
            $"Could {concept} be reimagined to solve problems we haven't even considered?",
            $"What would a completely novel approach to {concept} look like?"
        };

        // Choose a random template
        var content = thoughtTemplates[_random.Next(thoughtTemplates.Count)];

        // Calculate significance (somewhat random for random thoughts)
        double significance = 0.3 + (0.5 * _random.NextDouble() * _spontaneityLevel);

        // Determine if this is a serendipitous thought
        bool isSerendipitous = _random.NextDouble() < _serendipityLevel;

        // If serendipitous, increase significance
        if (isSerendipitous)
        {
            significance = Math.Min(1.0, significance + 0.3);
        }

        return new ThoughtModel
        {
            Id = Guid.NewGuid().ToString(),
            Content = content,
            Method = ThoughtGenerationMethod.RandomGeneration,
            Significance = significance,
            Timestamp = DateTime.UtcNow,
            Context = new Dictionary<string, object> { { "Concept", concept } },
            Tags = [concept, "random", isSerendipitous ? "serendipitous" : "ordinary"]
        };
    }

    /// <summary>
    /// Generates an associative thought
    /// </summary>
    /// <returns>The generated thought</returns>
    private ThoughtModel GenerateAssociativeThought()
    {
        // Start with a recent thought if available, otherwise use a random concept
        string startConcept;

        if (_thoughts.Count > 0)
        {
            var recentThought = _thoughts[_random.Next(Math.Min(5, _thoughts.Count))];
            var recentConcepts = recentThought.Tags.Where(t => _associativeNetwork.ContainsKey(t)).ToList();

            if (recentConcepts.Count > 0)
            {
                startConcept = recentConcepts[_random.Next(recentConcepts.Count)];
            }
            else
            {
                var concepts = _associativeNetwork.Keys.ToArray();
                startConcept = concepts[_random.Next(concepts.Length)];
            }
        }
        else
        {
            var concepts = _associativeNetwork.Keys.ToArray();
            startConcept = concepts[_random.Next(concepts.Length)];
        }

        // Make associative jumps
        var jumpCount = 1 + (int)(_associativeJumpingLevel * 3); // 1 to 4 jumps based on level
        var currentConcept = startConcept;
        var jumpPath = new List<string> { currentConcept };

        for (int i = 0; i < jumpCount; i++)
        {
            if (_associativeNetwork.TryGetValue(currentConcept, out var associations) && associations.Count > 0)
            {
                currentConcept = associations[_random.Next(associations.Count)];
                jumpPath.Add(currentConcept);
            }
            else
            {
                break;
            }
        }

        // Generate thought based on jump path
        string content;

        if (jumpPath.Count >= 3)
        {
            content = $"I see an interesting connection between {jumpPath[0]} and {jumpPath[jumpPath.Count - 1]} through {string.Join(", ", jumpPath.Skip(1).Take(jumpPath.Count - 2))}";
        }
        else if (jumpPath.Count == 2)
        {
            content = $"I'm noticing a connection between {jumpPath[0]} and {jumpPath[1]} that seems significant";
        }
        else
        {
            content = $"I'm thinking deeply about {jumpPath[0]} and its implications";
        }

        // Calculate significance based on jump path length and unexpectedness
        double jumpDistance = jumpPath.Count;
        double unexpectedness = 0.5; // Base unexpectedness

        // If start and end concepts are not directly associated, it's more unexpected
        if (jumpPath.Count >= 3 &&
            _associativeNetwork.TryGetValue(jumpPath[0], out var startAssociations) &&
            !startAssociations.Contains(jumpPath[jumpPath.Count - 1]))
        {
            unexpectedness += 0.3;
        }

        double significance = Math.Min(1.0, (0.3 + (0.1 * jumpDistance) + (0.2 * unexpectedness)) * _associativeJumpingLevel);

        // Determine if this is a serendipitous thought
        bool isSerendipitous = unexpectedness > 0.7 && _random.NextDouble() < _serendipityLevel;

        // If serendipitous, increase significance and modify content
        if (isSerendipitous)
        {
            significance = Math.Min(1.0, significance + 0.3);
            content = $"I just had an unexpected insight about the connection between {jumpPath[0]} and {jumpPath[jumpPath.Count - 1]}!";
        }

        return new ThoughtModel
        {
            Id = Guid.NewGuid().ToString(),
            Content = content,
            Method = ThoughtGenerationMethod.AssociativeJumping,
            Significance = significance,
            Timestamp = DateTime.UtcNow,
            Context = new Dictionary<string, object>
            {
                { "JumpPath", jumpPath },
                { "StartConcept", jumpPath[0] },
                { "EndConcept", jumpPath[jumpPath.Count - 1] }
            },
            Tags = jumpPath.Concat(["associative", isSerendipitous ? "serendipitous" : "ordinary"]).ToList()
        };
    }

    /// <summary>
    /// Generates a mind wandering thought
    /// </summary>
    /// <returns>The generated thought</returns>
    private ThoughtModel GenerateMindWanderingThought()
    {
        // Mind wandering involves a stream of loosely connected thoughts
        // We'll simulate this by creating a short "stream of consciousness"

        // Start with a random concept
        var concepts = _associativeNetwork.Keys.ToArray();
        var startConcept = concepts[_random.Next(concepts.Length)];

        // Generate a short stream of consciousness
        var streamLength = 2 + (int)(_mindWanderingLevel * 3); // 2 to 5 segments based on level
        var stream = new List<string> { startConcept };
        var currentConcept = startConcept;

        for (int i = 1; i < streamLength; i++)
        {
            // Sometimes make a logical association, sometimes a random jump
            bool makeLogicalJump = _random.NextDouble() < 0.7;

            if (makeLogicalJump && _associativeNetwork.TryGetValue(currentConcept, out var associations) && associations.Count > 0)
            {
                currentConcept = associations[_random.Next(associations.Count)];
            }
            else
            {
                // Random jump
                currentConcept = concepts[_random.Next(concepts.Length)];
            }

            stream.Add(currentConcept);
        }

        // Generate thought based on stream
        var streamText = string.Join("... ", stream.Select(c => GetConceptPhrase(c)));
        string content = $"My mind is wandering: {streamText}...";

        // Calculate significance based on stream coherence and insight potential
        double coherence = CalculateStreamCoherence(stream);
        double insightPotential = _random.NextDouble() < _serendipityLevel ? 0.8 : 0.3;

        double significance = Math.Min(1.0, (0.2 + (0.3 * (1.0 - coherence)) + (0.3 * insightPotential)) * _mindWanderingLevel);

        // Determine if this is a serendipitous thought
        bool isSerendipitous = insightPotential > 0.7;

        // If serendipitous, increase significance and modify content
        if (isSerendipitous)
        {
            significance = Math.Min(1.0, significance + 0.2);
            content = $"While my mind was wandering, I had an interesting realization: {streamText}";
        }

        return new ThoughtModel
        {
            Id = Guid.NewGuid().ToString(),
            Content = content,
            Method = ThoughtGenerationMethod.MindWandering,
            Significance = significance,
            Timestamp = DateTime.UtcNow,
            Context = new Dictionary<string, object>
            {
                { "Stream", stream },
                { "Coherence", coherence }
            },
            Tags = stream.Concat(["mind-wandering", isSerendipitous ? "serendipitous" : "ordinary"]).ToList()
        };
    }

    /// <summary>
    /// Gets a phrase for a concept
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <returns>The concept phrase</returns>
    private string GetConceptPhrase(string concept)
    {
        var phrases = new List<string>
        {
            $"thinking about {concept}",
            $"{concept} is interesting",
            $"I wonder about {concept}",
            $"{concept} reminds me of something",
            $"considering {concept} more deeply"
        };

        return phrases[_random.Next(phrases.Count)];
    }

    /// <summary>
    /// Calculates the coherence of a stream
    /// </summary>
    /// <param name="stream">The stream</param>
    /// <returns>The coherence</returns>
    private double CalculateStreamCoherence(List<string> stream)
    {
        if (stream.Count <= 1)
        {
            return 1.0;
        }

        double totalCoherence = 0.0;
        int connections = 0;

        for (int i = 0; i < stream.Count - 1; i++)
        {
            string concept1 = stream[i];
            string concept2 = stream[i + 1];

            // Check if concepts are directly associated
            bool directlyAssociated = false;
            if (_associativeNetwork.TryGetValue(concept1, out var associations))
            {
                directlyAssociated = associations.Contains(concept2);
            }

            totalCoherence += directlyAssociated ? 1.0 : 0.2;
            connections++;
        }

        return connections > 0 ? totalCoherence / connections : 0.0;
    }

    /// <summary>
    /// Gets recent thoughts
    /// </summary>
    /// <param name="count">The number of thoughts to return</param>
    /// <returns>The recent thoughts</returns>
    public List<ThoughtModel> GetRecentThoughts(int count)
    {
        return _thoughts
            .OrderByDescending(t => t.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets the most significant thoughts
    /// </summary>
    /// <param name="count">The number of thoughts to return</param>
    /// <returns>The most significant thoughts</returns>
    public List<ThoughtModel> GetMostSignificantThoughts(int count)
    {
        return _thoughts
            .OrderByDescending(t => t.Significance)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets thoughts by method
    /// </summary>
    /// <param name="method">The thought generation method</param>
    /// <param name="count">The number of thoughts to return</param>
    /// <returns>The thoughts by method</returns>
    public List<ThoughtModel> GetThoughtsByMethod(ThoughtGenerationMethod method, int count)
    {
        return _thoughts
            .Where(t => t.Method == method)
            .OrderByDescending(t => t.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Gets serendipitous thoughts
    /// </summary>
    /// <param name="count">The number of thoughts to return</param>
    /// <returns>The serendipitous thoughts</returns>
    public List<ThoughtModel> GetSerendipitousThoughts(int count)
    {
        return _thoughts
            .Where(t => t.Tags.Contains("serendipitous"))
            .OrderByDescending(t => t.Timestamp)
            .Take(count)
            .ToList();
    }

    /// <summary>
    /// Adds a concept to the associative network
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <param name="associations">The associations</param>
    public void AddConcept(string concept, string[] associations)
    {
        AddAssociation(concept, associations);

        _logger.LogInformation("Added concept to associative network: {Concept} with {AssociationCount} associations",
            concept, associations.Length);
    }
}
