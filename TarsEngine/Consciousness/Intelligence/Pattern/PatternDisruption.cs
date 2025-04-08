using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Consciousness.Intelligence;

namespace TarsEngine.Consciousness.Intelligence.Pattern;

/// <summary>
/// Implements pattern disruption capabilities for creative idea generation
/// </summary>
public class PatternDisruption
{
    private readonly ILogger<PatternDisruption> _logger;
    private readonly System.Random _random = new();
    private double _patternDisruptionLevel = 0.5; // Starting with moderate pattern disruption
    private readonly List<PatternModel> _patternModels = [];
    private readonly List<DisruptionStrategy> _disruptionStrategies = [];
    
    /// <summary>
    /// Gets the pattern disruption level (0.0 to 1.0)
    /// </summary>
    public double PatternDisruptionLevel => _patternDisruptionLevel;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="PatternDisruption"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public PatternDisruption(ILogger<PatternDisruption> logger)
    {
        _logger = logger;
        InitializePatternModels();
        InitializeDisruptionStrategies();
    }
    
    /// <summary>
    /// Initializes the pattern models
    /// </summary>
    private void InitializePatternModels()
    {
        // Add common software development patterns
        _patternModels.Add(new PatternModel
        {
            Name = "Singleton",
            Domain = "Software Design",
            Description = "Ensures a class has only one instance and provides a global point of access to it",
            Elements = ["single instance", "global access", "private constructor", "static instance"]
        });
        
        _patternModels.Add(new PatternModel
        {
            Name = "Observer",
            Domain = "Software Design",
            Description = "Defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified",
            Elements = ["subject", "observer", "notification", "subscription"]
        });
        
        _patternModels.Add(new PatternModel
        {
            Name = "Factory Method",
            Domain = "Software Design",
            Description = "Defines an interface for creating an object, but lets subclasses decide which class to instantiate",
            Elements = ["creator", "product", "interface", "subclass"]
        });
        
        _patternModels.Add(new PatternModel
        {
            Name = "Waterfall Development",
            Domain = "Software Process",
            Description = "Sequential software development process where progress flows downwards through phases",
            Elements = ["requirements", "design", "implementation", "verification", "maintenance"]
        });
        
        _patternModels.Add(new PatternModel
        {
            Name = "Agile Development",
            Domain = "Software Process",
            Description = "Iterative approach to software development emphasizing flexibility and customer satisfaction",
            Elements = ["iteration", "customer collaboration", "responding to change", "working software"]
        });
        
        _logger.LogInformation("Initialized pattern models: {Count}", _patternModels.Count);
    }
    
    /// <summary>
    /// Initializes the disruption strategies
    /// </summary>
    private void InitializeDisruptionStrategies()
    {
        _disruptionStrategies.Add(new DisruptionStrategy
        {
            Name = "Reversal",
            Description = "Reverse the relationship or flow between elements",
            DisruptionFunction = (pattern) => $"What if we reversed the relationship between {pattern.Elements[0]} and {pattern.Elements[1]}?"
        });
        
        _disruptionStrategies.Add(new DisruptionStrategy
        {
            Name = "Elimination",
            Description = "Remove a key element or constraint",
            DisruptionFunction = (pattern) => $"What if we eliminated {pattern.Elements[_random.Next(pattern.Elements.Count)]} from {pattern.Name}?"
        });
        
        _disruptionStrategies.Add(new DisruptionStrategy
        {
            Name = "Exaggeration",
            Description = "Exaggerate a key element to an extreme",
            DisruptionFunction = (pattern) => $"What if we maximized {pattern.Elements[_random.Next(pattern.Elements.Count)]} to an extreme in {pattern.Name}?"
        });
        
        _disruptionStrategies.Add(new DisruptionStrategy
        {
            Name = "Combination",
            Description = "Combine with an unrelated pattern",
            DisruptionFunction = (pattern) => 
            {
                var otherPattern = _patternModels.Where(p => p.Name != pattern.Name).OrderBy(_ => _random.Next()).FirstOrDefault();
                return otherPattern != null 
                    ? $"What if we combined {pattern.Name} with principles from {otherPattern.Name}?" 
                    : $"What if we combined {pattern.Name} with a completely unrelated concept?";
            }
        });
        
        _disruptionStrategies.Add(new DisruptionStrategy
        {
            Name = "Inversion",
            Description = "Invert the purpose or goal",
            DisruptionFunction = (pattern) => $"What if the goal of {pattern.Name} was the opposite of what it is now?"
        });
        
        _logger.LogInformation("Initialized disruption strategies: {Count}", _disruptionStrategies.Count);
    }
    
    /// <summary>
    /// Updates the pattern disruption level
    /// </summary>
    /// <returns>True if the update was successful, false otherwise</returns>
    public bool Update()
    {
        try
        {
            // Gradually increase pattern disruption level over time (very slowly)
            if (_patternDisruptionLevel < 0.95)
            {
                _patternDisruptionLevel += 0.0001 * _random.NextDouble();
                _patternDisruptionLevel = Math.Min(_patternDisruptionLevel, 1.0);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating pattern disruption");
            return false;
        }
    }
    
    /// <summary>
    /// Adds a pattern model
    /// </summary>
    /// <param name="pattern">The pattern model</param>
    public void AddPatternModel(PatternModel pattern)
    {
        if (!_patternModels.Any(p => p.Name == pattern.Name))
        {
            _patternModels.Add(pattern);
            _logger.LogInformation("Added pattern model: {PatternName}", pattern.Name);
        }
    }
    
    /// <summary>
    /// Adds a disruption strategy
    /// </summary>
    /// <param name="strategy">The disruption strategy</param>
    public void AddDisruptionStrategy(DisruptionStrategy strategy)
    {
        if (!_disruptionStrategies.Any(s => s.Name == strategy.Name))
        {
            _disruptionStrategies.Add(strategy);
            _logger.LogInformation("Added disruption strategy: {StrategyName}", strategy.Name);
        }
    }
    
    /// <summary>
    /// Gets a random pattern model
    /// </summary>
    /// <returns>The random pattern model</returns>
    public PatternModel GetRandomPatternModel()
    {
        if (_patternModels.Count == 0)
        {
            // Create a default pattern if none exist
            return new PatternModel
            {
                Name = "Default Pattern",
                Domain = "General",
                Description = "A generic pattern",
                Elements = ["element1", "element2"]
            };
        }
        
        return _patternModels[_random.Next(_patternModels.Count)];
    }
    
    /// <summary>
    /// Gets a random disruption strategy
    /// </summary>
    /// <returns>The random disruption strategy</returns>
    public DisruptionStrategy GetRandomDisruptionStrategy()
    {
        if (_disruptionStrategies.Count == 0)
        {
            // Create a default strategy if none exist
            return new DisruptionStrategy
            {
                Name = "Default Strategy",
                Description = "A generic disruption strategy",
                DisruptionFunction = (pattern) => $"What if we changed {pattern.Name} in an unexpected way?"
            };
        }
        
        return _disruptionStrategies[_random.Next(_disruptionStrategies.Count)];
    }
    
    /// <summary>
    /// Generates a pattern disruption idea
    /// </summary>
    /// <returns>The generated creative idea</returns>
    public CreativeIdea GeneratePatternDisruptionIdea()
    {
        // Get random pattern and strategy
        var pattern = GetRandomPatternModel();
        var strategy = GetRandomDisruptionStrategy();
        
        // Generate disruption description
        string description = strategy.DisruptionFunction(pattern);
        
        // Calculate originality (higher for pattern disruption)
        double originality = 0.7 + (0.3 * _random.NextDouble() * _patternDisruptionLevel);
        
        // Calculate value (more variable for pattern disruption)
        double value = 0.2 + (0.8 * _random.NextDouble() * _patternDisruptionLevel);
        
        // Extract concepts from pattern
        var concepts = new List<string> { pattern.Name, pattern.Domain };
        concepts.AddRange(pattern.Elements.Take(2)); // Add first two elements
        
        return new CreativeIdea
        {
            Id = Guid.NewGuid().ToString(),
            Description = description,
            Originality = originality,
            Value = value,
            Timestamp = DateTime.UtcNow,
            ProcessType = CreativeProcessType.PatternDisruption,
            Concepts = concepts,
            Context = new Dictionary<string, object>
            {
                { "PatternName", pattern.Name },
                { "StrategyName", strategy.Name }
            }
        };
    }
    
    /// <summary>
    /// Generates a disruptive solution for a problem
    /// </summary>
    /// <param name="problem">The problem description</param>
    /// <param name="constraints">The constraints</param>
    /// <returns>The disruptive solution</returns>
    public CreativeIdea GenerateDisruptiveSolution(string problem, List<string>? constraints = null)
    {
        try
        {
            _logger.LogInformation("Generating disruptive solution for problem: {Problem}", problem);
            
            // Extract concepts from problem
            var problemConcepts = ExtractConcepts(problem);
            
            // Create a temporary pattern from the problem
            var problemPattern = new PatternModel
            {
                Name = problemConcepts.FirstOrDefault() ?? "Problem",
                Domain = "Problem Domain",
                Description = problem,
                Elements = problemConcepts
            };
            
            // Get random additional concepts
            var additionalConcepts = _patternModels
                .SelectMany(p => p.Elements)
                .OrderBy(_ => _random.Next())
                .Take(2)
                .ToList();
            
            // Choose a random disruption strategy
            var strategy = GetRandomDisruptionStrategy();
            
            // Generate disruption description
            string disruptionDescription = strategy.DisruptionFunction(problemPattern);
            
            // Generate solution description
            string description = $"Solution approach: {disruptionDescription} " +
                                $"Challenge the fundamental assumptions about {problemConcepts.FirstOrDefault() ?? "the problem"}. " +
                                $"What if we reversed the relationship between {problemConcepts.FirstOrDefault() ?? "the problem"} and {additionalConcepts.FirstOrDefault() ?? "the solution"}?";
            
            // Apply constraints if provided, but with a disruptive twist
            if (constraints != null && constraints.Count > 0)
            {
                description += $" Consider eliminating {constraints.FirstOrDefault() ?? "constraints"} entirely or turning them into advantages.";
            }
            
            // Calculate originality
            double originality = 0.7 + (0.3 * _patternDisruptionLevel);
            
            // Calculate value
            double value = 0.5 + (0.5 * _patternDisruptionLevel);
            
            // Create all concepts list
            var allConcepts = new List<string>(problemConcepts);
            allConcepts.AddRange(additionalConcepts);
            
            // Create implementation steps with a disruptive approach
            var implementationSteps = new List<string>
            {
                $"1. Identify the core assumptions in the current approach to {problemConcepts.FirstOrDefault() ?? "the problem"}",
                $"2. Challenge each assumption by asking 'What if the opposite were true?'",
                $"3. Explore the implications of reversing the relationship between {problemConcepts.FirstOrDefault() ?? "the problem"} and {additionalConcepts.FirstOrDefault() ?? "the solution"}",
                $"4. Prototype a solution that deliberately breaks conventional patterns",
                $"5. Test the disruptive solution and refine based on results"
            };
            
            // Create potential impact
            string potentialImpact = $"This disruptive approach could lead to a breakthrough solution by challenging the fundamental assumptions that have limited previous approaches to {problemConcepts.FirstOrDefault() ?? "the problem"}.";
            
            // Create limitations
            var limitations = new List<string>
            {
                "May be difficult to implement within existing frameworks",
                "Could face resistance due to its unconventional nature",
                "Might require significant changes to current processes"
            };
            
            return new CreativeIdea
            {
                Id = Guid.NewGuid().ToString(),
                Description = description,
                Originality = originality,
                Value = value,
                Timestamp = DateTime.UtcNow,
                ProcessType = CreativeProcessType.PatternDisruption,
                Concepts = allConcepts,
                Problem = problem,
                Constraints = constraints?.ToList() ?? [],
                ImplementationSteps = implementationSteps,
                PotentialImpact = potentialImpact,
                Limitations = limitations,
                Context = new Dictionary<string, object>
                {
                    { "StrategyName", strategy.Name }
                }
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating disruptive solution");
            
            // Return basic idea
            return new CreativeIdea
            {
                Id = Guid.NewGuid().ToString(),
                Description = $"A disruptive approach that challenges the fundamental assumptions about the problem",
                Originality = 0.7,
                Value = 0.5,
                Timestamp = DateTime.UtcNow,
                ProcessType = CreativeProcessType.PatternDisruption,
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
    /// Identifies patterns in a problem description
    /// </summary>
    /// <param name="problem">The problem description</param>
    /// <returns>The identified patterns</returns>
    public List<PatternModel> IdentifyPatterns(string problem)
    {
        var identifiedPatterns = new List<PatternModel>();
        
        try
        {
            // Extract concepts from problem
            var problemConcepts = ExtractConcepts(problem);
            
            // Look for patterns that match the problem concepts
            foreach (var pattern in _patternModels)
            {
                // Check if any pattern elements match problem concepts
                var matchingElements = pattern.Elements.Intersect(problemConcepts, StringComparer.OrdinalIgnoreCase).ToList();
                
                if (matchingElements.Count > 0)
                {
                    identifiedPatterns.Add(pattern);
                }
            }
            
            _logger.LogInformation("Identified {Count} patterns in problem: {Problem}", identifiedPatterns.Count, problem);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error identifying patterns");
        }
        
        return identifiedPatterns;
    }
    
    /// <summary>
    /// Generates constraint-breaking ideas
    /// </summary>
    /// <param name="constraints">The constraints to break</param>
    /// <param name="count">The number of ideas to generate</param>
    /// <returns>The generated ideas</returns>
    public List<string> GenerateConstraintBreakingIdeas(List<string> constraints, int count)
    {
        var ideas = new List<string>();
        
        try
        {
            _logger.LogInformation("Generating {Count} constraint-breaking ideas", count);
            
            foreach (var constraint in constraints)
            {
                // Generate ideas for breaking each constraint
                ideas.Add($"What if {constraint} was not a limitation but an advantage?");
                ideas.Add($"What if we completely eliminated the need for {constraint}?");
                ideas.Add($"What if {constraint} was the opposite of what it is now?");
                
                if (ideas.Count >= count)
                {
                    break;
                }
            }
            
            // If we need more ideas, generate generic constraint-breaking ideas
            while (ideas.Count < count)
            {
                ideas.Add("What if we removed all constraints and started from first principles?");
                ideas.Add("What if the constraints are actually hiding the solution?");
                ideas.Add("What if we embraced the constraints and pushed them to their logical extreme?");
                
                if (ideas.Count >= count)
                {
                    break;
                }
            }
            
            _logger.LogInformation("Generated {Count} constraint-breaking ideas", ideas.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating constraint-breaking ideas");
        }
        
        return ideas.Take(count).ToList();
    }
}

/// <summary>
/// Represents a pattern model
/// </summary>
public class PatternModel
{
    /// <summary>
    /// Gets or sets the pattern name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the pattern domain
    /// </summary>
    public string Domain { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the pattern description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the pattern elements
    /// </summary>
    public List<string> Elements { get; set; } = [];
}

/// <summary>
/// Represents a disruption strategy
/// </summary>
public class DisruptionStrategy
{
    /// <summary>
    /// Gets or sets the strategy name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the strategy description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the disruption function
    /// </summary>
    public Func<PatternModel, string> DisruptionFunction { get; set; } = _ => string.Empty;
}

