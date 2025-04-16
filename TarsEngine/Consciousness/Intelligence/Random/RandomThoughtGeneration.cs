using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Intelligence.Random;

/// <summary>
/// Implements random thought generation capabilities for spontaneous thought
/// </summary>
public class RandomThoughtGeneration
{
    private readonly ILogger<RandomThoughtGeneration> _logger;
    private readonly System.Random _random = new();
    private double _randomThoughtLevel = 0.5; // Starting with moderate random thought
    private readonly List<string> _conceptLibrary = [];
    private readonly Dictionary<string, List<string>> _conceptCategories = new();
    private readonly List<string> _thoughtTemplates = [];
    
    /// <summary>
    /// Gets the random thought level (0.0 to 1.0)
    /// </summary>
    public double RandomThoughtLevel => _randomThoughtLevel;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="RandomThoughtGeneration"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public RandomThoughtGeneration(ILogger<RandomThoughtGeneration> logger)
    {
        _logger = logger;
        InitializeConceptLibrary();
        InitializeThoughtTemplates();
    }
    
    /// <summary>
    /// Initializes the concept library
    /// </summary>
    private void InitializeConceptLibrary()
    {
        // Add programming concepts
        var programmingConcepts = new List<string>
        {
            "algorithm", "pattern", "abstraction", "modularity", "encapsulation",
            "inheritance", "polymorphism", "recursion", "iteration", "parallelism",
            "concurrency", "asynchrony", "event", "message", "stream",
            "pipeline", "filter", "transformation", "validation", "verification"
        };
        
        _conceptCategories["Programming"] = programmingConcepts;
        _conceptLibrary.AddRange(programmingConcepts);
        
        // Add AI concepts
        var aiConcepts = new List<string>
        {
            "neural network", "machine learning", "deep learning", "reinforcement learning", "supervised learning",
            "unsupervised learning", "natural language processing", "computer vision", "generative AI", "transformer",
            "attention mechanism", "embedding", "fine-tuning", "transfer learning", "few-shot learning",
            "zero-shot learning", "prompt engineering", "semantic search", "vector database", "multimodal"
        };
        
        _conceptCategories["AI"] = aiConcepts;
        _conceptLibrary.AddRange(aiConcepts);
        
        // Add philosophical concepts
        var philosophicalConcepts = new List<string>
        {
            "consciousness", "free will", "determinism", "epistemology", "ontology",
            "ethics", "aesthetics", "metaphysics", "logic", "rationality",
            "empiricism", "existentialism", "phenomenology", "dualism", "materialism",
            "idealism", "pragmatism", "relativism", "nihilism", "stoicism"
        };
        
        _conceptCategories["Philosophy"] = philosophicalConcepts;
        _conceptLibrary.AddRange(philosophicalConcepts);
        
        // Add scientific concepts
        var scientificConcepts = new List<string>
        {
            "entropy", "complexity", "emergence", "chaos theory", "quantum mechanics",
            "relativity", "evolution", "thermodynamics", "information theory", "systems theory",
            "network theory", "game theory", "decision theory", "probability", "statistics",
            "causality", "correlation", "hypothesis", "experiment", "theory"
        };
        
        _conceptCategories["Science"] = scientificConcepts;
        _conceptLibrary.AddRange(scientificConcepts);
        
        _logger.LogInformation("Initialized concept library with {ConceptCount} concepts across {CategoryCount} categories", 
            _conceptLibrary.Count, _conceptCategories.Count);
    }
    
    /// <summary>
    /// Initializes the thought templates
    /// </summary>
    private void InitializeThoughtTemplates()
    {
        // Add general templates
        _thoughtTemplates.AddRange([
            "I wonder what would happen if we explored {0} from a completely different angle?",
            "What if {0} is actually fundamentally different than we think?",
            "Is there a deeper connection between {0} and consciousness that we're missing?",
            "Could {0} be reimagined to solve problems we haven't even considered?",
            "What would a completely novel approach to {0} look like?",
            "I'm curious about how {0} might evolve in the next decade",
            "What if we combined {0} with something seemingly unrelated?",
            "Are there hidden patterns in {0} that we haven't recognized yet?",
            "How would {0} be understood by a completely different intelligence?",
            "What are the ethical implications of advances in {0}?",
            "Could {0} be applied in a completely different domain?",
            "What are the limits of {0} and how might we transcend them?",
            "Is our understanding of {0} limited by our cognitive biases?",
            "How might {0} be perceived from a non-human perspective?",
            "What if everything we know about {0} is wrong?"
        ]);
        
        // Add category-specific templates
        _thoughtTemplates.AddRange([
            "Could principles from {0} help solve challenges in completely different fields?",
            "What would happen if we inverted our assumptions about {0}?",
            "Is there a universal pattern underlying {0} that connects to other phenomena?",
            "How might {0} be transformed by exponential technological change?",
            "What paradoxes exist within {0} that might lead to breakthroughs?"
        ]);
        
        _logger.LogInformation("Initialized thought templates: {TemplateCount}", _thoughtTemplates.Count);
    }
    
    /// <summary>
    /// Updates the random thought level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase random thought level over time (very slowly)
            if (_randomThoughtLevel < 0.95)
            {
                _randomThoughtLevel += 0.0001 * _random.NextDouble();
                _randomThoughtLevel = Math.Min(_randomThoughtLevel, 1.0);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating random thought generation");
            return false;
        }
    }
    
    /// <summary>
    /// Adds a concept to the library
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <param name="category">The category</param>
    public void AddConcept(string concept, string category)
    {
        if (!_conceptLibrary.Contains(concept))
        {
            _conceptLibrary.Add(concept);
            
            if (!_conceptCategories.ContainsKey(category))
            {
                _conceptCategories[category] = [];
            }
            
            _conceptCategories[category].Add(concept);
            
            _logger.LogDebug("Added concept to library: {Concept} in category {Category}", concept, category);
        }
    }
    
    /// <summary>
    /// Adds a thought template
    /// </summary>
    /// <param name="template">The template</param>
    public void AddThoughtTemplate(string template)
    {
        if (!_thoughtTemplates.Contains(template))
        {
            _thoughtTemplates.Add(template);
            _logger.LogDebug("Added thought template: {Template}", template);
        }
    }
    
    /// <summary>
    /// Gets a random concept
    /// </summary>
    /// <param name="category">The category (optional)</param>
    /// <returns>The random concept</returns>
    public string GetRandomConcept(string? category = null)
    {
        if (!string.IsNullOrEmpty(category) && _conceptCategories.ContainsKey(category))
        {
            var concepts = _conceptCategories[category];
            return concepts[_random.Next(concepts.Count)];
        }
        
        return _conceptLibrary[_random.Next(_conceptLibrary.Count)];
    }
    
    /// <summary>
    /// Gets a random thought template
    /// </summary>
    /// <returns>The random thought template</returns>
    public string GetRandomThoughtTemplate()
    {
        return _thoughtTemplates[_random.Next(_thoughtTemplates.Count)];
    }
    
    /// <summary>
    /// Generates a random thought
    /// </summary>
    /// <param name="serendipityLevel">The serendipity level</param>
    /// <returns>The generated thought</returns>
    public ThoughtModel GenerateRandomThought(double serendipityLevel)
    {
        try
        {
            _logger.LogDebug("Generating random thought");
            
            // Get random concept
            var concept = GetRandomConcept();
            
            // Get random template
            var template = GetRandomThoughtTemplate();
            
            // Generate thought content
            var content = string.Format(template, concept);
            
            // Calculate significance (somewhat random for random thoughts)
            var significance = 0.3 + (0.5 * _random.NextDouble() * _randomThoughtLevel);
            
            // Determine if this is a serendipitous thought
            var isSerendipitous = _random.NextDouble() < serendipityLevel;
            
            // If serendipitous, increase significance
            if (isSerendipitous)
            {
                significance = Math.Min(1.0, significance + 0.3);
                
                // Add serendipitous prefix
                var serendipitousPrefixes = new[]
                {
                    "I just had a sudden realization about ",
                    "It just occurred to me that ",
                    "I had an unexpected insight about ",
                    "I suddenly see a new perspective on ",
                    "Out of nowhere, I'm thinking about "
                };
                
                content = serendipitousPrefixes[_random.Next(serendipitousPrefixes.Length)] + content.ToLowerInvariant();
            }
            
            // Calculate originality
            var originality = 0.4 + (0.6 * _random.NextDouble() * _randomThoughtLevel);
            
            // Calculate coherence (random thoughts have variable coherence)
            var coherence = 0.3 + (0.4 * _random.NextDouble());
            
            // Get concept category
            var category = _conceptCategories.FirstOrDefault(c => c.Value.Contains(concept)).Key ?? "General";
            
            // Create thought model
            var thought = new ThoughtModel
            {
                Id = Guid.NewGuid().ToString(),
                Content = content,
                Method = ThoughtGenerationMethod.RandomGeneration,
                Significance = significance,
                Timestamp = DateTime.UtcNow,
                Context = new Dictionary<string, object> 
                { 
                    { "Concept", concept },
                    { "Category", category },
                    { "IsSerendipitous", isSerendipitous }
                },
                Tags = [concept, category, "random", isSerendipitous ? "serendipitous" : "ordinary"],
                Source = "RandomThoughtGeneration",
                Category = category,
                Originality = originality,
                Coherence = coherence
            };
            
            _logger.LogInformation("Generated random thought: {Content} (Significance: {Significance:F2})", 
                thought.Content, thought.Significance);
            
            return thought;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating random thought");
            
            // Return basic thought
            return new ThoughtModel
            {
                Id = Guid.NewGuid().ToString(),
                Content = "I had a random thought but can't quite articulate it",
                Method = ThoughtGenerationMethod.RandomGeneration,
                Significance = 0.3,
                Timestamp = DateTime.UtcNow,
                Source = "RandomThoughtGeneration"
            };
        }
    }
    
    /// <summary>
    /// Generates multiple random thoughts
    /// </summary>
    /// <param name="count">The number of thoughts to generate</param>
    /// <param name="serendipityLevel">The serendipity level</param>
    /// <returns>The generated thoughts</returns>
    public List<ThoughtModel> GenerateRandomThoughts(int count, double serendipityLevel)
    {
        var thoughts = new List<ThoughtModel>();
        
        for (var i = 0; i < count; i++)
        {
            var thought = GenerateRandomThought(serendipityLevel);
            thoughts.Add(thought);
        }
        
        return thoughts;
    }
    
    /// <summary>
    /// Evaluates the quality of a random thought
    /// </summary>
    /// <param name="thought">The thought to evaluate</param>
    /// <returns>The evaluation score (0.0 to 1.0)</returns>
    public double EvaluateThought(ThoughtModel thought)
    {
        try
        {
            // Check if thought is from this source
            if (thought.Method != ThoughtGenerationMethod.RandomGeneration)
            {
                return 0.5; // Neutral score for thoughts from other sources
            }
            
            // Calculate novelty based on content length and complexity
            var novelty = Math.Min(1.0, thought.Content.Length / 100.0);
            
            // Calculate interestingness based on significance and originality
            var interestingness = (thought.Significance + thought.Originality) / 2.0;
            
            // Calculate potential based on serendipity
            var isSerendipitous = thought.Tags.Contains("serendipitous");
            var potential = isSerendipitous ? 0.8 : 0.5;
            
            // Calculate overall score
            var score = (novelty * 0.3) + (interestingness * 0.4) + (potential * 0.3);
            
            return score;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error evaluating thought");
            return 0.5; // Default score
        }
    }
}
