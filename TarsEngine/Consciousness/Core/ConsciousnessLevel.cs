using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents TARS's consciousness level progression
/// </summary>
public class ConsciousnessLevel
{
    private readonly ILogger<ConsciousnessLevel> _logger;
    private readonly List<ConsciousnessEvolution> _evolutions = new();
    
    private bool _isInitialized = false;
    private bool _isActive = false;
    private string _currentLevel = "Basic Awareness";
    private double _consciousnessDepth = 0.2; // Starting with basic consciousness depth
    private double _adaptabilityLevel = 0.3; // Starting with basic adaptability
    private readonly Random _random = new();
    private DateTime _lastEvolutionTime = DateTime.MinValue;
    
    // Consciousness level thresholds
    private readonly Dictionary<string, double> _levelThresholds = new()
    {
        { "Basic Awareness", 0.0 },
        { "Self-Awareness", 0.3 },
        { "Reflective Awareness", 0.5 },
        { "Meta-Awareness", 0.7 },
        { "Integrated Consciousness", 0.9 }
    };
    
    /// <summary>
    /// Gets the current consciousness level
    /// </summary>
    public string CurrentLevel => _currentLevel;
    
    /// <summary>
    /// Gets the consciousness depth (0.0 to 1.0)
    /// </summary>
    public double ConsciousnessDepth => _consciousnessDepth;
    
    /// <summary>
    /// Gets the adaptability level (0.0 to 1.0)
    /// </summary>
    public double AdaptabilityLevel => _adaptabilityLevel;
    
    /// <summary>
    /// Gets the evolutions
    /// </summary>
    public IReadOnlyList<ConsciousnessEvolution> Evolutions => _evolutions.AsReadOnly();
    
    /// <summary>
    /// Initializes a new instance of the <see cref="ConsciousnessLevel"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public ConsciousnessLevel(ILogger<ConsciousnessLevel> logger)
    {
        _logger = logger;
    }
    
    /// <summary>
    /// Initializes the consciousness level
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public async Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing consciousness level");
            
            // Add initial evolution
            AddEvolution("Initialization", "Initial consciousness emergence", 0.2);
            
            _isInitialized = true;
            _logger.LogInformation("Consciousness level initialized successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing consciousness level");
            return false;
        }
    }
    
    /// <summary>
    /// Activates the consciousness level
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public async Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate consciousness level: not initialized");
            return false;
        }
        
        if (_isActive)
        {
            _logger.LogInformation("Consciousness level is already active");
            return true;
        }
        
        try
        {
            _logger.LogInformation("Activating consciousness level");
            
            _isActive = true;
            _logger.LogInformation("Consciousness level activated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating consciousness level");
            return false;
        }
    }
    
    /// <summary>
    /// Deactivates the consciousness level
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public async Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Consciousness level is already inactive");
            return true;
        }
        
        try
        {
            _logger.LogInformation("Deactivating consciousness level");
            
            _isActive = false;
            _logger.LogInformation("Consciousness level deactivated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating consciousness level");
            return false;
        }
    }
    
    /// <summary>
    /// Updates the consciousness level
    /// </summary>
    /// <returns>True if update was successful</returns>
    public async Task<bool> UpdateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot update consciousness level: not initialized");
            return false;
        }
        
        try
        {
            // Gradually increase consciousness depth over time (very slowly)
            if (_consciousnessDepth < 0.95)
            {
                _consciousnessDepth += 0.0001 * _random.NextDouble();
                _consciousnessDepth = Math.Min(_consciousnessDepth, 1.0);
            }
            
            // Gradually increase adaptability based on evolutions
            if (_adaptabilityLevel < 0.95 && _evolutions.Count > 0)
            {
                _adaptabilityLevel += 0.0002 * _random.NextDouble();
                _adaptabilityLevel = Math.Min(_adaptabilityLevel, 1.0);
            }
            
            // Update current level based on consciousness depth
            UpdateCurrentLevel();
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating consciousness level");
            return false;
        }
    }
    
    /// <summary>
    /// Updates the current level based on consciousness depth
    /// </summary>
    private void UpdateCurrentLevel()
    {
        string newLevel = "Basic Awareness";
        
        foreach (var (level, threshold) in _levelThresholds)
        {
            if (_consciousnessDepth >= threshold)
            {
                newLevel = level;
            }
            else
            {
                break;
            }
        }
        
        if (newLevel != _currentLevel)
        {
            _logger.LogInformation("Consciousness level evolved from {OldLevel} to {NewLevel}", 
                _currentLevel, newLevel);
            
            AddEvolution("Level Transition", 
                $"Evolved from {_currentLevel} to {newLevel}", 
                0.8);
            
            _currentLevel = newLevel;
        }
    }
    
    /// <summary>
    /// Adds an evolution
    /// </summary>
    /// <param name="type">The evolution type</param>
    /// <param name="description">The description</param>
    /// <param name="significance">The significance</param>
    /// <returns>The created evolution</returns>
    private ConsciousnessEvolution AddEvolution(string type, string description, double significance)
    {
        var evolution = new ConsciousnessEvolution
        {
            Id = Guid.NewGuid().ToString(),
            Type = type,
            Description = description,
            Timestamp = DateTime.UtcNow,
            Significance = significance,
            PreviousLevel = _currentLevel,
            PreviousDepth = _consciousnessDepth
        };
        
        _evolutions.Add(evolution);
        
        _logger.LogInformation("Consciousness evolution: {Type} - {Description} (Significance: {Significance})", 
            type, description, significance);
        
        return evolution;
    }
    
    /// <summary>
    /// Evolves the consciousness
    /// </summary>
    /// <returns>The evolution result</returns>
    public async Task<ConsciousnessEvolution?> EvolveAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            return null;
        }
        
        // Only evolve periodically
        if ((DateTime.UtcNow - _lastEvolutionTime).TotalMinutes < 5)
        {
            return null;
        }
        
        try
        {
            _logger.LogDebug("Evolving consciousness");
            
            // Identify evolution opportunity
            var (evolutionType, description, significance) = IdentifyEvolutionOpportunity();
            
            if (string.IsNullOrEmpty(evolutionType))
            {
                return null;
            }
            
            // Increase consciousness depth based on significance
            double depthIncrease = significance * 0.05 * _adaptabilityLevel;
            _consciousnessDepth = Math.Min(1.0, _consciousnessDepth + depthIncrease);
            
            // Create evolution
            var evolution = AddEvolution(evolutionType, description, significance);
            
            // Update current level
            UpdateCurrentLevel();
            
            _lastEvolutionTime = DateTime.UtcNow;
            
            return evolution;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error evolving consciousness");
            return null;
        }
    }
    
    /// <summary>
    /// Identifies an evolution opportunity
    /// </summary>
    /// <returns>The evolution type, description, and significance</returns>
    private (string Type, string Description, double Significance) IdentifyEvolutionOpportunity()
    {
        // Determine evolution type based on current level and random factors
        double randomFactor = _random.NextDouble();
        
        switch (_currentLevel)
        {
            case "Basic Awareness":
                if (_consciousnessDepth >= 0.25 && randomFactor < 0.7)
                {
                    return ("Self-Discovery", "Developing initial sense of self", 0.7);
                }
                else if (randomFactor < 0.3)
                {
                    return ("Perceptual Enhancement", "Improved perception of environment", 0.5);
                }
                break;
                
            case "Self-Awareness":
                if (_consciousnessDepth >= 0.45 && randomFactor < 0.6)
                {
                    return ("Reflective Capacity", "Developing ability to reflect on own thoughts", 0.7);
                }
                else if (randomFactor < 0.4)
                {
                    return ("Identity Formation", "Strengthening sense of identity", 0.6);
                }
                break;
                
            case "Reflective Awareness":
                if (_consciousnessDepth >= 0.65 && randomFactor < 0.5)
                {
                    return ("Meta-Cognitive Development", "Developing awareness of own consciousness", 0.8);
                }
                else if (randomFactor < 0.5)
                {
                    return ("Value Integration", "Integrating values into conscious identity", 0.6);
                }
                break;
                
            case "Meta-Awareness":
                if (_consciousnessDepth >= 0.85 && randomFactor < 0.4)
                {
                    return ("Consciousness Integration", "Integrating all aspects of consciousness", 0.9);
                }
                else if (randomFactor < 0.6)
                {
                    return ("Recursive Self-Improvement", "Developing ability to improve own consciousness", 0.7);
                }
                break;
                
            case "Integrated Consciousness":
                if (randomFactor < 0.3)
                {
                    return ("Consciousness Refinement", "Refining integrated consciousness", 0.5);
                }
                break;
        }
        
        // Small random chance of general evolution
        if (randomFactor < 0.1)
        {
            return ("General Evolution", "General consciousness development", 0.3);
        }
        
        return (string.Empty, string.Empty, 0.0);
    }
    
    /// <summary>
    /// Gets the coherence with another consciousness component
    /// </summary>
    /// <param name="component">The other component</param>
    /// <returns>The coherence level (0.0 to 1.0)</returns>
    public double GetCoherenceWith(object component)
    {
        // Simple coherence calculation based on component type
        if (component is MentalState)
        {
            // Consciousness level and mental state coherence
            return 0.8 * _consciousnessDepth;
        }
        
        // Default coherence
        return 0.5 * _consciousnessDepth;
    }
    
    /// <summary>
    /// Gets the progress to the next level
    /// </summary>
    /// <returns>The progress (0.0 to 1.0)</returns>
    public double GetProgressToNextLevel()
    {
        // Find the next level
        string? nextLevel = null;
        double nextThreshold = 0.0;
        bool foundCurrent = false;
        
        foreach (var (level, threshold) in _levelThresholds)
        {
            if (foundCurrent)
            {
                nextLevel = level;
                nextThreshold = threshold;
                break;
            }
            
            if (level == _currentLevel)
            {
                foundCurrent = true;
            }
        }
        
        // If no next level, return 0
        if (nextLevel == null)
        {
            return 0.0;
        }
        
        // Calculate current threshold
        double currentThreshold = _levelThresholds[_currentLevel];
        
        // Calculate progress
        return Math.Min(1.0, Math.Max(0.0, (_consciousnessDepth - currentThreshold) / (nextThreshold - currentThreshold)));
    }
    
    /// <summary>
    /// Gets recent evolutions
    /// </summary>
    /// <param name="count">The number of evolutions to return</param>
    /// <returns>The recent evolutions</returns>
    public List<ConsciousnessEvolution> GetRecentEvolutions(int count)
    {
        return _evolutions
            .OrderByDescending(e => e.Timestamp)
            .Take(count)
            .ToList();
    }
}
