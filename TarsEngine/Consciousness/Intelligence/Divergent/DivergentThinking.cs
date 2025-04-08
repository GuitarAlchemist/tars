using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Consciousness.Intelligence;

namespace TarsEngine.Consciousness.Intelligence.Divergent;

/// <summary>
/// Implements divergent thinking capabilities for creative idea generation
/// </summary>
public class DivergentThinking
{
    private readonly ILogger<DivergentThinking> _logger;
    private readonly System.Random _random = new();
    private double _divergentThinkingLevel = 0.5; // Starting with moderate divergent thinking
    private readonly List<string> _conceptLibrary = [];
    private readonly Dictionary<string, Dictionary<string, double>> _conceptAssociations = new();
    
    /// <summary>
    /// Gets the divergent thinking level (0.0 to 1.0)
    /// </summary>
    public double DivergentThinkingLevel => _divergentThinkingLevel;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="DivergentThinking"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public DivergentThinking(ILogger<DivergentThinking> logger)
    {
        _logger = logger;
        InitializeConceptLibrary();
    }
    
    /// <summary>
    /// Initializes the concept library with seed concepts
    /// </summary>
    private void InitializeConceptLibrary()
    {
        // Add seed concepts
        var seedConcepts = new List<string>
        {
            "algorithm", "pattern", "abstraction", "modularity", "encapsulation",
            "inheritance", "polymorphism", "recursion", "iteration", "parallelism",
            "concurrency", "asynchrony", "event", "message", "stream",
            "pipeline", "filter", "transformation", "validation", "verification",
            "testing", "debugging", "profiling", "optimization", "refactoring",
            "architecture", "design", "implementation", "deployment", "maintenance",
            "security", "performance", "reliability", "scalability", "usability",
            "accessibility", "internationalization", "localization", "documentation", "collaboration"
        };
        
        _conceptLibrary.AddRange(seedConcepts);
        
        // Initialize some random associations between concepts
        foreach (var concept in _conceptLibrary)
        {
            _conceptAssociations[concept] = new Dictionary<string, double>();
            
            foreach (var otherConcept in _conceptLibrary.Where(c => c != concept))
            {
                // Random association strength between 0.1 and 0.9
                double associationStrength = 0.1 + (0.8 * _random.NextDouble());
                _conceptAssociations[concept][otherConcept] = associationStrength;
            }
        }
        
        _logger.LogInformation("Initialized divergent thinking with {ConceptCount} concepts", _conceptLibrary.Count);
    }
    
    /// <summary>
    /// Updates the divergent thinking level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase divergent thinking level over time (very slowly)
            if (_divergentThinkingLevel < 0.95)
            {
                _divergentThinkingLevel += 0.0001 * _random.NextDouble();
                _divergentThinkingLevel = Math.Min(_divergentThinkingLevel, 1.0);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating divergent thinking");
            return false;
        }
    }
    
    /// <summary>
    /// Adds a concept to the library
    /// </summary>
    /// <param name="concept">The concept to add</param>
    public void AddConcept(string concept)
    {
        if (!_conceptLibrary.Contains(concept))
        {
            _conceptLibrary.Add(concept);
            _conceptAssociations[concept] = new Dictionary<string, double>();
            
            foreach (var otherConcept in _conceptLibrary.Where(c => c != concept))
            {
                // Random association strength between 0.1 and 0.9
                double associationStrength = 0.1 + (0.8 * _random.NextDouble());
                _conceptAssociations[concept][otherConcept] = associationStrength;
                
                // Add reverse association if it doesn't exist
                if (!_conceptAssociations[otherConcept].ContainsKey(concept))
                {
                    _conceptAssociations[otherConcept][concept] = associationStrength;
                }
            }
            
            _logger.LogInformation("Added concept to library: {Concept}", concept);
        }
    }
    
    /// <summary>
    /// Gets random concepts from the library
    /// </summary>
    /// <param name="count">The number of concepts to get</param>
    /// <returns>The random concepts</returns>
    public List<string> GetRandomConcepts(int count)
    {
        var concepts = new List<string>();
        
        // Ensure we don't try to get more concepts than exist
        count = Math.Min(count, _conceptLibrary.Count);
        
        // Get random concepts
        var shuffled = _conceptLibrary.OrderBy(c => _random.Next()).ToList();
        concepts.AddRange(shuffled.Take(count));
        
        return concepts;
    }
    
    /// <summary>
    /// Gets the association strength between two concepts
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <returns>The association strength (0.0 to 1.0)</returns>
    public double GetAssociationStrength(string concept1, string concept2)
    {
        // If either concept doesn't exist, return low association
        if (!_conceptAssociations.ContainsKey(concept1) || !_conceptAssociations[concept1].ContainsKey(concept2))
        {
            return 0.1;
        }
        
        return _conceptAssociations[concept1][concept2];
    }
    
    /// <summary>
    /// Generates a divergent thinking idea
    /// </summary>
    /// <returns>The generated creative idea</returns>
    public CreativeIdea GenerateDivergentIdea()
    {
        // Get random seed concepts
        var seedConcepts = GetRandomConcepts(2);
        
        // Generate multiple perspectives on the concepts
        var perspectives = new List<string>
        {
            $"Combining {seedConcepts[0]} with {seedConcepts[1]}",
            $"Using {seedConcepts[0]} to enhance {seedConcepts[1]}",
            $"Applying {seedConcepts[1]} principles to {seedConcepts[0]}",
            $"Reimagining {seedConcepts[0]} through the lens of {seedConcepts[1]}",
            $"Creating a new approach that integrates {seedConcepts[0]} and {seedConcepts[1]}",
            $"Developing a framework where {seedConcepts[0]} complements {seedConcepts[1]}",
            $"Building a system that leverages both {seedConcepts[0]} and {seedConcepts[1]}"
        };
        
        // Choose a random perspective
        var perspective = perspectives[_random.Next(perspectives.Count)];
        
        // Generate idea description
        string description = $"What if we {perspective}?";
        
        // Calculate originality based on association strength (lower association = higher originality)
        double associationStrength = GetAssociationStrength(seedConcepts[0], seedConcepts[1]);
        double originality = 0.5 + (0.5 * (1.0 - associationStrength)) * _divergentThinkingLevel;
        
        // Calculate value (somewhat random but influenced by divergent thinking level)
        double value = 0.3 + (0.7 * _random.NextDouble() * _divergentThinkingLevel);
        
        return new CreativeIdea
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Originality = originality,
            Value = value,
            Timestamp = DateTime.UtcNow,
            ProcessType = CreativeProcessType.DivergentThinking,
            Concepts = seedConcepts.ToList()
        };
    }
    
    /// <summary>
    /// Generates multiple alternative ideas for a problem
    /// </summary>
    /// <param name="problem">The problem description</param>
    /// <param name="count">The number of alternatives to generate</param>
    /// <returns>The generated alternative ideas</returns>
    public List<CreativeIdea> GenerateAlternatives(string problem, int count)
    {
        var alternatives = new List<CreativeIdea>();
        
        try
        {
            _logger.LogInformation("Generating {Count} alternatives for problem: {Problem}", count, problem);
            
            // Extract concepts from problem
            var problemConcepts = ExtractConcepts(problem);
            
            // Add problem concepts to library
            foreach (var concept in problemConcepts)
            {
                AddConcept(concept);
            }
            
            // Generate alternatives
            for (int i = 0; i < count; i++)
            {
                // Get additional concepts beyond problem concepts
                var additionalConcepts = GetRandomConcepts(2).Where(c => !problemConcepts.Contains(c)).ToList();
                
                // Generate alternative approaches
                var approaches = new List<string>
                {
                    $"Approach {i+1}: Consider the problem from the perspective of {additionalConcepts[0]}",
                    $"Approach {i+1}: Apply {additionalConcepts[0]} principles to solve the {problemConcepts.FirstOrDefault() ?? "problem"}",
                    $"Approach {i+1}: Reframe the problem as a {additionalConcepts[0]} challenge",
                    $"Approach {i+1}: Use {additionalConcepts[0]} and {additionalConcepts[1]} together to address the problem"
                };
                
                // Choose a random approach
                var description = approaches[_random.Next(approaches.Count)];
                
                // Calculate originality and value
                double originality = 0.4 + (0.6 * _random.NextDouble() * _divergentThinkingLevel);
                double value = 0.3 + (0.7 * _random.NextDouble() * _divergentThinkingLevel);
                
                // Create idea
                var idea = new CreativeIdea
                {
                    Id = Guid.NewGuid().ToString(),
                    Description = description,
                    Originality = originality,
                    Value = value,
                    Timestamp = DateTime.UtcNow,
                    ProcessType = CreativeProcessType.DivergentThinking,
                    Concepts = [..problemConcepts.Concat(additionalConcepts)],
                    Problem = problem
                };
                
                alternatives.Add(idea);
            }
            
            _logger.LogInformation("Generated {Count} alternatives for problem: {Problem}", alternatives.Count, problem);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating alternatives");
        }
        
        return alternatives;
    }
    
    /// <summary>
    /// Extracts concepts from text
    /// </summary>
    /// <param name="text">The text</param>
    /// <returns>The extracted concepts</returns>
    private List<string> ExtractConcepts(string text)
    {
        var concepts = new List<string>();
        
        // Simple concept extraction by splitting and filtering
        var words = text.Split([' ', ',', '.', ':', ';', '(', ')', '[', ']', '{', '}', '\n', '\r', '\t'], 
            StringSplitOptions.RemoveEmptyEntries);
        
        foreach (var word in words)
        {
            // Only consider words of reasonable length
            if (word.Length >= 4 && word.Length <= 20)
            {
                // Convert to lowercase
                var concept = word.ToLowerInvariant();
                
                // Add if not already in list
                if (!concepts.Contains(concept))
                {
                    concepts.Add(concept);
                }
            }
        }
        
        // If no concepts found, add some default ones
        if (concepts.Count == 0)
        {
            concepts.Add("problem");
            concepts.Add("solution");
        }
        
        return concepts;
    }
    
    /// <summary>
    /// Generates perspective shifts for a problem
    /// </summary>
    /// <param name="problem">The problem description</param>
    /// <returns>The generated perspective shifts</returns>
    public List<string> GeneratePerspectiveShifts(string problem)
    {
        var perspectives = new List<string>();
        
        try
        {
            // Extract concepts from problem
            var problemConcepts = ExtractConcepts(problem);
            
            // Generate perspective shifts
            perspectives.Add($"What if the {problemConcepts.FirstOrDefault() ?? "problem"} is actually an opportunity?");
            perspectives.Add($"How would a beginner approach this problem without preconceptions?");
            perspectives.Add($"What would happen if we reversed our assumptions about {problemConcepts.FirstOrDefault() ?? "the problem"}?");
            perspectives.Add($"How would we solve this if we had unlimited resources?");
            perspectives.Add($"What if the constraints were actually advantages?");
            perspectives.Add($"How would nature solve this problem?");
            perspectives.Add($"What if we approached this problem from the opposite direction?");
            
            _logger.LogInformation("Generated {Count} perspective shifts for problem: {Problem}", perspectives.Count, problem);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating perspective shifts");
        }
        
        return perspectives;
    }
    
    /// <summary>
    /// Evaluates the quality of a divergent idea
    /// </summary>
    /// <param name="idea">The idea to evaluate</param>
    /// <returns>The evaluation score (0.0 to 1.0)</returns>
    public double EvaluateIdea(CreativeIdea idea)
    {
        try
        {
            // Calculate novelty based on concept associations
            double totalAssociation = 0.0;
            int pairs = 0;
            
            for (int i = 0; i < idea.Concepts.Count; i++)
            {
                for (int j = i + 1; j < idea.Concepts.Count; j++)
                {
                    totalAssociation += GetAssociationStrength(idea.Concepts[i], idea.Concepts[j]);
                    pairs++;
                }
            }
            
            double avgAssociation = pairs > 0 ? totalAssociation / pairs : 0.5;
            double novelty = 1.0 - avgAssociation;
            
            // Calculate usefulness based on value
            double usefulness = idea.Value;
            
            // Calculate elaboration based on description length
            double elaboration = Math.Min(1.0, idea.Description.Length / 100.0);
            
            // Calculate flexibility based on number of concepts
            double flexibility = Math.Min(1.0, idea.Concepts.Count / 5.0);
            
            // Calculate overall score
            double score = (novelty * 0.3) + (usefulness * 0.3) + (elaboration * 0.2) + (flexibility * 0.2);
            
            return score;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error evaluating idea");
            return 0.5; // Default score
        }
    }
}

