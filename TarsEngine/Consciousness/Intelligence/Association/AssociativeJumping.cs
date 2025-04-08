using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Consciousness.Intelligence;

namespace TarsEngine.Consciousness.Intelligence.Association;

/// <summary>
/// Implements associative jumping capabilities for spontaneous thought
/// </summary>
public class AssociativeJumping
{
    private readonly ILogger<AssociativeJumping> _logger;
    private readonly System.Random _random = new();
    private double _associativeJumpingLevel = 0.5; // Starting with moderate associative jumping
    private readonly Dictionary<string, Dictionary<string, double>> _associativeNetwork = new();
    private readonly Dictionary<string, string> _conceptCategories = new();

    /// <summary>
    /// Gets the associative jumping level (0.0 to 1.0)
    /// </summary>
    public double AssociativeJumpingLevel => _associativeJumpingLevel;

    /// <summary>
    /// Initializes a new instance of the <see cref="AssociativeJumping"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public AssociativeJumping(ILogger<AssociativeJumping> logger)
    {
        _logger = logger;
        InitializeAssociativeNetwork();
    }

    /// <summary>
    /// Initializes the associative network
    /// </summary>
    private void InitializeAssociativeNetwork()
    {
        // Add programming concepts and associations
        AddConcept("algorithm", "Programming");
        AddConcept("pattern", "Programming");
        AddConcept("abstraction", "Programming");
        AddConcept("modularity", "Programming");
        AddConcept("encapsulation", "Programming");
        AddConcept("inheritance", "Programming");
        AddConcept("polymorphism", "Programming");
        AddConcept("recursion", "Programming");
        AddConcept("iteration", "Programming");
        AddConcept("parallelism", "Programming");

        // Add associations between programming concepts
        AddAssociation("algorithm", "pattern", 0.7);
        AddAssociation("algorithm", "recursion", 0.6);
        AddAssociation("algorithm", "iteration", 0.8);
        AddAssociation("pattern", "abstraction", 0.5);
        AddAssociation("abstraction", "modularity", 0.7);
        AddAssociation("modularity", "encapsulation", 0.8);
        AddAssociation("inheritance", "polymorphism", 0.9);
        AddAssociation("recursion", "iteration", 0.7);
        AddAssociation("parallelism", "iteration", 0.6);

        // Add AI concepts and associations
        AddConcept("neural network", "AI");
        AddConcept("machine learning", "AI");
        AddConcept("deep learning", "AI");
        AddConcept("reinforcement learning", "AI");
        AddConcept("supervised learning", "AI");
        AddConcept("unsupervised learning", "AI");
        AddConcept("natural language processing", "AI");
        AddConcept("computer vision", "AI");
        AddConcept("generative AI", "AI");
        AddConcept("transformer", "AI");

        // Add associations between AI concepts
        AddAssociation("neural network", "deep learning", 0.9);
        AddAssociation("machine learning", "supervised learning", 0.8);
        AddAssociation("machine learning", "unsupervised learning", 0.8);
        AddAssociation("deep learning", "transformer", 0.7);
        AddAssociation("reinforcement learning", "supervised learning", 0.5);
        AddAssociation("natural language processing", "transformer", 0.8);
        AddAssociation("computer vision", "deep learning", 0.8);
        AddAssociation("generative AI", "transformer", 0.7);

        // Add philosophical concepts and associations
        AddConcept("consciousness", "Philosophy");
        AddConcept("free will", "Philosophy");
        AddConcept("determinism", "Philosophy");
        AddConcept("epistemology", "Philosophy");
        AddConcept("ontology", "Philosophy");
        AddConcept("ethics", "Philosophy");
        AddConcept("aesthetics", "Philosophy");
        AddConcept("metaphysics", "Philosophy");
        AddConcept("logic", "Philosophy");
        AddConcept("rationality", "Philosophy");

        // Add associations between philosophical concepts
        AddAssociation("consciousness", "free will", 0.8);
        AddAssociation("free will", "determinism", 0.9);
        AddAssociation("epistemology", "ontology", 0.7);
        AddAssociation("metaphysics", "ontology", 0.8);
        AddAssociation("logic", "rationality", 0.9);
        AddAssociation("ethics", "free will", 0.7);
        AddAssociation("consciousness", "metaphysics", 0.6);

        // Add cross-domain associations
        AddAssociation("neural network", "consciousness", 0.4);
        AddAssociation("machine learning", "determinism", 0.3);
        AddAssociation("algorithm", "logic", 0.6);
        AddAssociation("pattern", "ontology", 0.3);
        AddAssociation("abstraction", "metaphysics", 0.4);
        AddAssociation("natural language processing", "consciousness", 0.3);

        _logger.LogInformation("Initialized associative network with {ConceptCount} concepts and {AssociationCount} associations",
            _associativeNetwork.Count, _associativeNetwork.Sum(c => c.Value.Count));
    }

    /// <summary>
    /// Updates the associative jumping level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase associative jumping level over time (very slowly)
            if (_associativeJumpingLevel < 0.95)
            {
                _associativeJumpingLevel += 0.0001 * _random.NextDouble();
                _associativeJumpingLevel = Math.Min(_associativeJumpingLevel, 1.0);
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating associative jumping");
            return false;
        }
    }

    /// <summary>
    /// Adds a concept to the associative network
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <param name="category">The category</param>
    public void AddConcept(string concept, string category)
    {
        if (!_associativeNetwork.ContainsKey(concept))
        {
            _associativeNetwork[concept] = new Dictionary<string, double>();
            _conceptCategories[concept] = category;

            _logger.LogDebug("Added concept to associative network: {Concept} in category {Category}", concept, category);
        }
    }

    /// <summary>
    /// Adds an association between two concepts
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <param name="strength">The association strength</param>
    public void AddAssociation(string concept1, string concept2, double strength)
    {
        // Ensure both concepts exist
        if (!_associativeNetwork.ContainsKey(concept1))
        {
            AddConcept(concept1, "Unknown");
        }

        if (!_associativeNetwork.ContainsKey(concept2))
        {
            AddConcept(concept2, "Unknown");
        }

        // Add bidirectional association
        _associativeNetwork[concept1][concept2] = strength;
        _associativeNetwork[concept2][concept1] = strength;

        _logger.LogDebug("Added association: {Concept1} <-> {Concept2} with strength {Strength:F2}",
            concept1, concept2, strength);
    }

    /// <summary>
    /// Gets a random concept from the associative network
    /// </summary>
    /// <param name="category">The category (optional)</param>
    /// <returns>The random concept</returns>
    public string GetRandomConcept(string? category = null)
    {
        if (!string.IsNullOrEmpty(category))
        {
            var conceptsInCategory = _conceptCategories
                .Where(c => c.Value == category)
                .Select(c => c.Key)
                .ToList();

            if (conceptsInCategory.Count > 0)
            {
                return conceptsInCategory[_random.Next(conceptsInCategory.Count)];
            }
        }

        var concepts = _associativeNetwork.Keys.ToList();
        return concepts[_random.Next(concepts.Count)];
    }

    /// <summary>
    /// Gets the associated concepts for a concept
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <param name="minStrength">The minimum association strength</param>
    /// <returns>The associated concepts</returns>
    public Dictionary<string, double> GetAssociatedConcepts(string concept, double minStrength = 0.0)
    {
        if (!_associativeNetwork.ContainsKey(concept))
        {
            return new Dictionary<string, double>();
        }

        return _associativeNetwork[concept]
            .Where(a => a.Value >= minStrength)
            .ToDictionary(a => a.Key, a => a.Value);
    }

    /// <summary>
    /// Performs an associative jump from a concept
    /// </summary>
    /// <param name="startConcept">The start concept</param>
    /// <param name="jumpDistance">The jump distance</param>
    /// <returns>The jump path</returns>
    public List<string> PerformAssociativeJump(string startConcept, int jumpDistance)
    {
        var jumpPath = new List<string> { startConcept };

        try
        {
            string currentConcept = startConcept;

            for (int i = 0; i < jumpDistance; i++)
            {
                // Get associated concepts
                var associations = GetAssociatedConcepts(currentConcept);

                if (associations.Count == 0)
                {
                    break;
                }

                // Choose next concept based on association strength
                string nextConcept;

                // Occasionally make an unexpected jump
                if (_random.NextDouble() < 0.2 * _associativeJumpingLevel)
                {
                    // Choose a random concept that's not already in the path
                    var availableConcepts = _associativeNetwork.Keys
                        .Except(jumpPath)
                        .ToList();

                    if (availableConcepts.Count > 0)
                    {
                        nextConcept = availableConcepts[_random.Next(availableConcepts.Count)];
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    // Choose based on association strength, but avoid cycles
                    var availableAssociations = associations
                        .Where(a => !jumpPath.Contains(a.Key))
                        .ToList();

                    if (availableAssociations.Count == 0)
                    {
                        break;
                    }

                    // Choose with probability proportional to association strength
                    double totalStrength = availableAssociations.Sum(a => a.Value);
                    double randomValue = _random.NextDouble() * totalStrength;
                    double cumulativeStrength = 0.0;

                    nextConcept = availableAssociations.Last().Key; // Default

                    foreach (var association in availableAssociations)
                    {
                        cumulativeStrength += association.Value;

                        if (cumulativeStrength >= randomValue)
                        {
                            nextConcept = association.Key;
                            break;
                        }
                    }
                }

                // Add to jump path
                jumpPath.Add(nextConcept);
                currentConcept = nextConcept;
            }

            _logger.LogDebug("Performed associative jump from {StartConcept} with distance {JumpDistance}, path: {JumpPath}",
                startConcept, jumpDistance, string.Join(" -> ", jumpPath));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error performing associative jump");
        }

        return jumpPath;
    }

    /// <summary>
    /// Calculates the unexpectedness of a jump path
    /// </summary>
    /// <param name="jumpPath">The jump path</param>
    /// <returns>The unexpectedness (0.0 to 1.0)</returns>
    public double CalculateUnexpectedness(List<string> jumpPath)
    {
        if (jumpPath.Count < 2)
        {
            return 0.0;
        }

        try
        {
            // Calculate average association strength along the path
            double totalStrength = 0.0;
            int connections = 0;

            for (int i = 0; i < jumpPath.Count - 1; i++)
            {
                string concept1 = jumpPath[i];
                string concept2 = jumpPath[i + 1];

                if (_associativeNetwork.ContainsKey(concept1) && _associativeNetwork[concept1].ContainsKey(concept2))
                {
                    totalStrength += _associativeNetwork[concept1][concept2];
                    connections++;
                }
            }

            double avgStrength = connections > 0 ? totalStrength / connections : 0.0;

            // Calculate category diversity
            var categories = jumpPath
                .Where(c => _conceptCategories.ContainsKey(c))
                .Select(c => _conceptCategories[c])
                .Distinct()
                .Count();

            double categoryDiversity = Math.Min(1.0, categories / 3.0);

            // Calculate unexpectedness (lower average strength and higher diversity = more unexpected)
            double unexpectedness = ((1.0 - avgStrength) * 0.7) + (categoryDiversity * 0.3);

            return unexpectedness;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error calculating unexpectedness");
            return 0.5; // Default value
        }
    }

    /// <summary>
    /// Generates an associative thought
    /// </summary>
    /// <param name="serendipityLevel">The serendipity level</param>
    /// <returns>The generated thought</returns>
    public ThoughtModel GenerateAssociativeThought(double serendipityLevel)
    {
        try
        {
            _logger.LogDebug("Generating associative thought");

            // Choose a random starting concept
            string startConcept = GetRandomConcept();

            // Determine jump distance based on associative jumping level
            int jumpDistance = 2 + (int)(_associativeJumpingLevel * 3);

            // Perform associative jump
            var jumpPath = PerformAssociativeJump(startConcept, jumpDistance);

            // Calculate unexpectedness
            double unexpectedness = CalculateUnexpectedness(jumpPath);

            // Generate thought content
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

            // Calculate significance based on jump distance and unexpectedness
            double significance = Math.Min(1.0, (0.3 + (0.1 * jumpDistance) + (0.2 * unexpectedness)) * _associativeJumpingLevel);

            // Determine if this is a serendipitous thought
            bool isSerendipitous = unexpectedness > 0.7 && _random.NextDouble() < serendipityLevel;

            // If serendipitous, increase significance and modify content
            if (isSerendipitous)
            {
                significance = Math.Min(1.0, significance + 0.3);
                content = $"I just had an unexpected insight about the connection between {jumpPath[0]} and {jumpPath[jumpPath.Count - 1]}!";
            }

            // Calculate originality based on unexpectedness
            double originality = 0.3 + (0.7 * unexpectedness);

            // Calculate coherence based on average association strength
            double coherence = 0.4 + (0.6 * (1.0 - unexpectedness));

            // Get categories for tags
            var categories = jumpPath
                .Where(c => _conceptCategories.ContainsKey(c))
                .Select(c => _conceptCategories[c])
                .Distinct()
                .ToList();

            // Create thought model
            var thought = new ThoughtModel
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
                    { "EndConcept", jumpPath[jumpPath.Count - 1] },
                    { "Unexpectedness", unexpectedness },
                    { "IsSerendipitous", isSerendipitous }
                },
                Tags = jumpPath.Concat(categories).Concat(["associative", isSerendipitous ? "serendipitous" : "ordinary"
                ]).ToList(),
                Source = "AssociativeJumping",
                Category = categories.FirstOrDefault() ?? "General",
                Originality = originality,
                Coherence = coherence
            };

            _logger.LogInformation("Generated associative thought: {Content} (Significance: {Significance:F2})",
                thought.Content, thought.Significance);

            return thought;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating associative thought");

            // Return basic thought
            return new ThoughtModel
            {
                Id = Guid.NewGuid().ToString(),
                Content = "I had an associative thought but can't quite articulate it",
                Method = ThoughtGenerationMethod.AssociativeJumping,
                Significance = 0.3,
                Timestamp = DateTime.UtcNow,
                Source = "AssociativeJumping"
            };
        }
    }

    /// <summary>
    /// Generates multiple associative thoughts
    /// </summary>
    /// <param name="count">The number of thoughts to generate</param>
    /// <param name="serendipityLevel">The serendipity level</param>
    /// <returns>The generated thoughts</returns>
    public List<ThoughtModel> GenerateAssociativeThoughts(int count, double serendipityLevel)
    {
        var thoughts = new List<ThoughtModel>();

        for (int i = 0; i < count; i++)
        {
            var thought = GenerateAssociativeThought(serendipityLevel);
            thoughts.Add(thought);
        }

        return thoughts;
    }

    /// <summary>
    /// Evaluates the quality of an associative thought
    /// </summary>
    /// <param name="thought">The thought to evaluate</param>
    /// <returns>The evaluation score (0.0 to 1.0)</returns>
    public double EvaluateThought(ThoughtModel thought)
    {
        try
        {
            // Check if thought is from this source
            if (thought.Method != ThoughtGenerationMethod.AssociativeJumping)
            {
                return 0.5; // Neutral score for thoughts from other sources
            }

            // Get unexpectedness from context
            double unexpectedness = thought.Context.ContainsKey("Unexpectedness")
                ? (double)thought.Context["Unexpectedness"]
                : 0.5;

            // Get jump path from context
            var jumpPath = thought.Context.ContainsKey("JumpPath")
                ? (List<string>)thought.Context["JumpPath"]
                : [];

            // Calculate novelty based on unexpectedness
            double novelty = unexpectedness;

            // Calculate interestingness based on jump path length and significance
            double interestingness = jumpPath.Count > 0
                ? Math.Min(1.0, (jumpPath.Count / 5.0) * thought.Significance)
                : thought.Significance;

            // Calculate potential based on serendipity
            bool isSerendipitous = thought.Tags.Contains("serendipitous");
            double potential = isSerendipitous ? 0.8 : 0.5;

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
}
