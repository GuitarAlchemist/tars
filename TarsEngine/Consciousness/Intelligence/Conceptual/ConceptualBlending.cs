using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Consciousness.Intelligence;

namespace TarsEngine.Consciousness.Intelligence.Conceptual;

/// <summary>
/// Implements conceptual blending capabilities for creative idea generation
/// </summary>
public class ConceptualBlending
{
    private readonly ILogger<ConceptualBlending> _logger;
    private readonly System.Random _random = new();
    private double _conceptualBlendingLevel = 0.5; // Starting with moderate conceptual blending
    private readonly Dictionary<string, ConceptModel> _conceptModels = new();
    private readonly List<BlendSpace> _blendSpaces = [];
    
    /// <summary>
    /// Gets the conceptual blending level (0.0 to 1.0)
    /// </summary>
    public double ConceptualBlendingLevel => _conceptualBlendingLevel;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="ConceptualBlending"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public ConceptualBlending(ILogger<ConceptualBlending> logger)
    {
        _logger = logger;
        InitializeConceptModels();
    }
    
    /// <summary>
    /// Initializes the concept models with seed concepts
    /// </summary>
    private void InitializeConceptModels()
    {
        // Add seed concepts with attributes
        AddConceptModel("algorithm", new Dictionary<string, double>
        {
            { "procedural", 0.9 },
            { "step-by-step", 0.8 },
            { "deterministic", 0.7 },
            { "computational", 0.9 },
            { "problem-solving", 0.8 }
        });
        
        AddConceptModel("pattern", new Dictionary<string, double>
        {
            { "repetitive", 0.8 },
            { "structured", 0.9 },
            { "recognizable", 0.7 },
            { "predictable", 0.6 },
            { "template", 0.8 }
        });
        
        AddConceptModel("abstraction", new Dictionary<string, double>
        {
            { "conceptual", 0.9 },
            { "simplified", 0.8 },
            { "generalized", 0.9 },
            { "high-level", 0.7 },
            { "essential", 0.6 }
        });
        
        AddConceptModel("modularity", new Dictionary<string, double>
        {
            { "component-based", 0.9 },
            { "reusable", 0.8 },
            { "encapsulated", 0.7 },
            { "independent", 0.6 },
            { "composable", 0.8 }
        });
        
        AddConceptModel("recursion", new Dictionary<string, double>
        {
            { "self-referential", 0.9 },
            { "nested", 0.8 },
            { "repetitive", 0.7 },
            { "hierarchical", 0.8 },
            { "elegant", 0.6 }
        });
        
        _logger.LogInformation("Initialized conceptual blending with {ConceptCount} concepts", _conceptModels.Count);
    }
    
    /// <summary>
    /// Updates the conceptual blending level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase conceptual blending level over time (very slowly)
            if (_conceptualBlendingLevel < 0.95)
            {
                _conceptualBlendingLevel += 0.0001 * _random.NextDouble();
                _conceptualBlendingLevel = Math.Min(_conceptualBlendingLevel, 1.0);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating conceptual blending");
            return false;
        }
    }
    
    /// <summary>
    /// Adds a concept model
    /// </summary>
    /// <param name="concept">The concept</param>
    /// <param name="attributes">The attributes</param>
    /// <returns>The created concept model</returns>
    public ConceptModel AddConceptModel(string concept, Dictionary<string, double> attributes)
    {
        if (_conceptModels.ContainsKey(concept))
        {
            return _conceptModels[concept];
        }
        
        // Create concept model
        var model = new ConceptModel
        {
            Name = concept,
            Attributes = attributes ?? new Dictionary<string, double>()
        };
        
        // Add to concept models
        _conceptModels[concept] = model;
        
        _logger.LogInformation("Added concept model: {Concept}", concept);
        
        return model;
    }
    
    /// <summary>
    /// Gets random concepts from the models
    /// </summary>
    /// <param name="count">The number of concepts to get</param>
    /// <returns>The random concepts</returns>
    public List<string> GetRandomConcepts(int count)
    {
        var concepts = new List<string>();
        
        // Ensure we don't try to get more concepts than exist
        count = Math.Min(count, _conceptModels.Count);
        
        // Get random concepts
        var shuffled = _conceptModels.Keys.OrderBy(c => _random.Next()).ToList();
        concepts.AddRange(shuffled.Take(count));
        
        return concepts;
    }
    
    /// <summary>
    /// Creates a blend space between concepts
    /// </summary>
    /// <param name="inputConcepts">The input concepts</param>
    /// <returns>The created blend space</returns>
    public BlendSpace CreateBlendSpace(List<string> inputConcepts)
    {
        try
        {
            _logger.LogInformation("Creating blend space for concepts: {Concepts}", string.Join(", ", inputConcepts));
            
            // Ensure all concepts exist in the model
            foreach (var concept in inputConcepts)
            {
                if (!_conceptModels.ContainsKey(concept))
                {
                    AddConceptModel(concept, null);
                }
            }
            
            // Create blend space
            var blendSpace = new BlendSpace
            {
                Id = Guid.NewGuid().ToString(),
                InputConcepts = inputConcepts.ToList(),
                CreatedAt = DateTime.UtcNow
            };
            
            // Create concept mappings
            for (int i = 0; i < inputConcepts.Count; i++)
            {
                for (int j = i + 1; j < inputConcepts.Count; j++)
                {
                    var mapping = CreateConceptMapping(inputConcepts[i], inputConcepts[j]);
                    blendSpace.ConceptMappings.Add(mapping);
                }
            }
            
            // Add to blend spaces
            _blendSpaces.Add(blendSpace);
            
            _logger.LogInformation("Created blend space: {BlendSpaceId} with {MappingCount} mappings", 
                blendSpace.Id, blendSpace.ConceptMappings.Count);
            
            return blendSpace;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating blend space");
            
            // Return empty blend space
            return new BlendSpace
            {
                Id = Guid.NewGuid().ToString(),
                InputConcepts = inputConcepts.ToList(),
                CreatedAt = DateTime.UtcNow
            };
        }
    }
    
    /// <summary>
    /// Creates a concept mapping between two concepts
    /// </summary>
    /// <param name="concept1">The first concept</param>
    /// <param name="concept2">The second concept</param>
    /// <returns>The created concept mapping</returns>
    private ConceptMapping CreateConceptMapping(string concept1, string concept2)
    {
        var mapping = new ConceptMapping
        {
            SourceConcept = concept1,
            TargetConcept = concept2
        };
        
        // Get concept models
        var model1 = _conceptModels[concept1];
        var model2 = _conceptModels[concept2];
        
        // Find common attributes
        foreach (var attr1 in model1.Attributes)
        {
            if (model2.Attributes.ContainsKey(attr1.Key))
            {
                // Add attribute mapping
                mapping.AttributeMappings.Add(new AttributeMapping
                {
                    SourceAttribute = attr1.Key,
                    TargetAttribute = attr1.Key,
                    Strength = (attr1.Value + model2.Attributes[attr1.Key]) / 2.0
                });
            }
        }
        
        return mapping;
    }
    
    /// <summary>
    /// Generates a conceptual blend idea
    /// </summary>
    /// <returns>The generated creative idea</returns>
    public CreativeIdea GenerateConceptualBlendIdea()
    {
        // Get random seed concepts
        var seedConcepts = GetRandomConcepts(3);
        
        // Create blend space
        var blendSpace = CreateBlendSpace(seedConcepts);
        
        // Create blend descriptions
        var blendDescriptions = new List<string>
        {
            $"A hybrid of {seedConcepts[0]} and {seedConcepts[1]} with aspects of {seedConcepts[2]}",
            $"A new approach that merges {seedConcepts[0]} with {seedConcepts[1]}, influenced by {seedConcepts[2]}",
            $"A {seedConcepts[0]}-{seedConcepts[1]} fusion system with {seedConcepts[2]} characteristics",
            $"A {seedConcepts[2]}-inspired blend of {seedConcepts[0]} and {seedConcepts[1]}"
        };
        
        // Choose a random description
        var description = blendDescriptions[_random.Next(blendDescriptions.Count)];
        
        // Calculate average association strength
        double totalAssociation = 0.0;
        int pairs = 0;
        
        foreach (var mapping in blendSpace.ConceptMappings)
        {
            totalAssociation += mapping.AttributeMappings.Count > 0 
                ? mapping.AttributeMappings.Average(am => am.Strength) 
                : 0.5;
            pairs++;
        }
        
        double avgAssociation = pairs > 0 ? totalAssociation / pairs : 0.5;
        
        // Calculate originality (lower average association = higher originality)
        double originality = 0.6 + (0.4 * (1.0 - avgAssociation)) * _conceptualBlendingLevel;
        
        // Calculate value (somewhat random but influenced by conceptual blending level)
        double value = 0.4 + (0.6 * _random.NextDouble() * _conceptualBlendingLevel);
        
        return new CreativeIdea
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Originality = originality,
            Value = value,
            Timestamp = DateTime.UtcNow,
            ProcessType = CreativeProcessType.ConceptualBlending,
            Concepts = seedConcepts.ToList(),
            Context = new Dictionary<string, object>
            {
                { "BlendSpaceId", blendSpace.Id }
            }
        };
    }
    
    /// <summary>
    /// Generates a blended solution for a problem
    /// </summary>
    /// <param name="problem">The problem description</param>
    /// <param name="constraints">The constraints</param>
    /// <returns>The blended solution</returns>
    public CreativeIdea GenerateBlendedSolution(string problem, List<string>? constraints = null)
    {
        try
        {
            _logger.LogInformation("Generating blended solution for problem: {Problem}", problem);
            
            // Extract concepts from problem
            var problemConcepts = ExtractConcepts(problem);
            
            // Get additional concepts
            var additionalConcepts = GetRandomConcepts(2);
            
            // Create all concepts list
            var allConcepts = new List<string>(problemConcepts);
            allConcepts.AddRange(additionalConcepts);
            
            // Create blend space
            var blendSpace = CreateBlendSpace(allConcepts);
            
            // Generate solution description
            string description = $"Solution approach: Create a hybrid solution that blends {problemConcepts[0]} with {additionalConcepts[0]}, " +
                                $"incorporating elements of {additionalConcepts[1]} to address the core challenge.";
            
            // Apply constraints if provided
            if (constraints != null && constraints.Count > 0)
            {
                description += $" While ensuring {String.Join(" and ", constraints)}.";
            }
            
            // Calculate originality
            double originality = 0.6 + (0.4 * _conceptualBlendingLevel);
            
            // Calculate value
            double value = 0.7 + (0.3 * _conceptualBlendingLevel);
            
            // Create implementation steps
            var implementationSteps = new List<string>
            {
                $"1. Analyze the core elements of {problemConcepts[0]}",
                $"2. Identify the key principles of {additionalConcepts[0]} that can be applied",
                $"3. Determine how {additionalConcepts[1]} can enhance the solution",
                $"4. Create a prototype that combines these elements",
                $"5. Test and refine the blended solution"
            };
            
            return new CreativeIdea
            {
                Id = Guid.NewGuid().ToString(),
                Description = description,
                Originality = originality,
                Value = value,
                Timestamp = DateTime.UtcNow,
                ProcessType = CreativeProcessType.ConceptualBlending,
                Concepts = allConcepts,
                Problem = problem,
                Constraints = constraints?.ToList() ?? [],
                ImplementationSteps = implementationSteps,
                Context = new Dictionary<string, object>
                {
                    { "BlendSpaceId", blendSpace.Id }
                }
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating blended solution");
            
            // Return basic idea
            return new CreativeIdea
            {
                Id = Guid.NewGuid().ToString(),
                Description = $"A blended approach to solving the problem by combining multiple perspectives",
                Originality = 0.5,
                Value = 0.5,
                Timestamp = DateTime.UtcNow,
                ProcessType = CreativeProcessType.ConceptualBlending,
                Problem = problem
            };
        }
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
    /// Evaluates the emergent structure of a blend
    /// </summary>
    /// <param name="blendSpaceId">The blend space ID</param>
    /// <returns>The evaluation score (0.0 to 1.0)</returns>
    public double EvaluateEmergentStructure(string blendSpaceId)
    {
        try
        {
            // Find blend space
            var blendSpace = _blendSpaces.FirstOrDefault(bs => bs.Id == blendSpaceId);
            if (blendSpace == null)
            {
                _logger.LogWarning("Blend space not found: {BlendSpaceId}", blendSpaceId);
                return 0.5;
            }
            
            // Calculate coherence
            double coherence = blendSpace.ConceptMappings.Count > 0
                ? blendSpace.ConceptMappings.Average(m => m.AttributeMappings.Count) / 5.0
                : 0.3;
            coherence = Math.Min(1.0, coherence);
            
            // Calculate integration
            double integration = blendSpace.InputConcepts.Count > 0
                ? Math.Min(1.0, blendSpace.ConceptMappings.Count / (double)(blendSpace.InputConcepts.Count * (blendSpace.InputConcepts.Count - 1) / 2))
                : 0.3;
            
            // Calculate emergent structure score
            double emergentScore = (coherence * 0.6) + (integration * 0.4);
            
            return emergentScore;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error evaluating emergent structure");
            return 0.5;
        }
    }
}

/// <summary>
/// Represents a concept model
/// </summary>
public class ConceptModel
{
    /// <summary>
    /// Gets or sets the concept name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the concept attributes
    /// </summary>
    public Dictionary<string, double> Attributes { get; set; } = new();
}

/// <summary>
/// Represents a blend space
/// </summary>
public class BlendSpace
{
    /// <summary>
    /// Gets or sets the blend space ID
    /// </summary>
    public string Id { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the input concepts
    /// </summary>
    public List<string> InputConcepts { get; set; } = [];
    
    /// <summary>
    /// Gets or sets the concept mappings
    /// </summary>
    public List<ConceptMapping> ConceptMappings { get; set; } = [];
    
    /// <summary>
    /// Gets or sets the creation timestamp
    /// </summary>
    public DateTime CreatedAt { get; set; }
}

/// <summary>
/// Represents a concept mapping
/// </summary>
public class ConceptMapping
{
    /// <summary>
    /// Gets or sets the source concept
    /// </summary>
    public string SourceConcept { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the target concept
    /// </summary>
    public string TargetConcept { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the attribute mappings
    /// </summary>
    public List<AttributeMapping> AttributeMappings { get; set; } = [];
}

/// <summary>
/// Represents an attribute mapping
/// </summary>
public class AttributeMapping
{
    /// <summary>
    /// Gets or sets the source attribute
    /// </summary>
    public string SourceAttribute { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the target attribute
    /// </summary>
    public string TargetAttribute { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the mapping strength (0.0 to 1.0)
    /// </summary>
    public double Strength { get; set; } = 0.5;
}

