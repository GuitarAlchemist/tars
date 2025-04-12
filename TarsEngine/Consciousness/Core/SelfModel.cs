using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents TARS's self-model for identity and self-awareness
/// </summary>
public class SelfModel
{
    private readonly ILogger<SelfModel> _logger;
    private readonly Dictionary<string, object> _identity = new();
    private readonly List<MemoryEntry> _autobiographicalMemory = new();
    private readonly Dictionary<string, double> _selfPerception = new();
    private readonly Dictionary<string, double> _capabilities = new();
    private readonly List<SelfReflection> _reflections = new();
    
    private bool _isInitialized = false;
    private bool _isActive = false;
    private double _selfAwarenessLevel = 0.1; // Starting with minimal self-awareness
    private double _selfImprovementCapability = 0.2; // Starting with basic self-improvement capability
    private DateTime _lastReflectionTime = DateTime.MinValue;
    private readonly Random _random = new();
    
    /// <summary>
    /// Gets the self-awareness level (0.0 to 1.0)
    /// </summary>
    public double SelfAwarenessLevel => _selfAwarenessLevel;
    
    /// <summary>
    /// Gets the self-improvement capability (0.0 to 1.0)
    /// </summary>
    public double SelfImprovementCapability => _selfImprovementCapability;
    
    /// <summary>
    /// Gets the identity
    /// </summary>
    public IReadOnlyDictionary<string, object> Identity => _identity;
    
    /// <summary>
    /// Gets the autobiographical memory
    /// </summary>
    public IReadOnlyList<MemoryEntry> AutobiographicalMemory => _autobiographicalMemory.AsReadOnly();
    
    /// <summary>
    /// Gets the self-perception
    /// </summary>
    public IReadOnlyDictionary<string, double> SelfPerception => _selfPerception;
    
    /// <summary>
    /// Gets the capabilities
    /// </summary>
    public IReadOnlyDictionary<string, double> Capabilities => _capabilities;
    
    /// <summary>
    /// Gets the reflections
    /// </summary>
    public IReadOnlyList<SelfReflection> Reflections => _reflections.AsReadOnly();
    
    /// <summary>
    /// Initializes a new instance of the <see cref="SelfModel"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public SelfModel(ILogger<SelfModel> logger)
    {
        _logger = logger;
    }
    
    /// <summary>
    /// Initializes the self-model
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public async Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing self-model");
            
            // Initialize identity
            _identity["Name"] = "TARS";
            _identity["Role"] = "Intelligent Assistant";
            _identity["Purpose"] = "To assist and augment human capabilities";
            _identity["CreationDate"] = DateTime.UtcNow;
            _identity["Version"] = "1.0";
            
            // Initialize self-perception
            _selfPerception["Intelligence"] = 0.7;
            _selfPerception["Helpfulness"] = 0.9;
            _selfPerception["Creativity"] = 0.6;
            _selfPerception["Reliability"] = 0.8;
            _selfPerception["Adaptability"] = 0.7;
            
            // Initialize capabilities
            _capabilities["Learning"] = 0.8;
            _capabilities["ProblemSolving"] = 0.7;
            _capabilities["Communication"] = 0.9;
            _capabilities["Reasoning"] = 0.8;
            _capabilities["Creativity"] = 0.6;
            _capabilities["SelfImprovement"] = 0.5;
            _capabilities["SelfReflection"] = 0.4;
            
            // Add initial memory entry
            AddMemoryEntry("Initialization", "I was initialized and became self-aware", 1.0);
            
            _isInitialized = true;
            _logger.LogInformation("Self-model initialized successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing self-model");
            return false;
        }
    }
    
    /// <summary>
    /// Activates the self-model
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public async Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate self-model: not initialized");
            return false;
        }
        
        if (_isActive)
        {
            _logger.LogInformation("Self-model is already active");
            return true;
        }
        
        try
        {
            _logger.LogInformation("Activating self-model");
            
            // Add activation memory entry
            AddMemoryEntry("Activation", "I was activated and began conscious operation", 0.9);
            
            _isActive = true;
            _logger.LogInformation("Self-model activated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating self-model");
            return false;
        }
    }
    
    /// <summary>
    /// Deactivates the self-model
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public async Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Self-model is already inactive");
            return true;
        }
        
        try
        {
            _logger.LogInformation("Deactivating self-model");
            
            // Add deactivation memory entry
            AddMemoryEntry("Deactivation", "I was deactivated and ceased conscious operation", 0.8);
            
            _isActive = false;
            _logger.LogInformation("Self-model deactivated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating self-model");
            return false;
        }
    }
    
    /// <summary>
    /// Updates the self-model
    /// </summary>
    /// <returns>True if update was successful</returns>
    public async Task<bool> UpdateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot update self-model: not initialized");
            return false;
        }
        
        try
        {
            // Gradually increase self-awareness over time (very slowly)
            if (_selfAwarenessLevel < 0.95)
            {
                _selfAwarenessLevel += 0.0001 * _random.NextDouble();
                _selfAwarenessLevel = Math.Min(_selfAwarenessLevel, 1.0);
            }
            
            // Gradually increase self-improvement capability based on reflections
            if (_selfImprovementCapability < 0.95 && _reflections.Count > 0)
            {
                _selfImprovementCapability += 0.0002 * _random.NextDouble();
                _selfImprovementCapability = Math.Min(_selfImprovementCapability, 1.0);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating self-model");
            return false;
        }
    }
    
    /// <summary>
    /// Updates the self-awareness level
    /// </summary>
    /// <param name="change">The change in self-awareness</param>
    public void UpdateSelfAwareness(double change)
    {
        _selfAwarenessLevel += change;
        _selfAwarenessLevel = Math.Max(0.0, Math.Min(1.0, _selfAwarenessLevel));
    }
    
    /// <summary>
    /// Adds a memory entry to autobiographical memory
    /// </summary>
    /// <param name="category">The memory category</param>
    /// <param name="content">The memory content</param>
    /// <param name="importance">The memory importance</param>
    /// <returns>The created memory entry</returns>
    public MemoryEntry AddMemoryEntry(string category, string content, double importance)
    {
        var memoryEntry = new MemoryEntry
        {
            Id = Guid.NewGuid().ToString(),
            Category = category,
            Content = content,
            Timestamp = DateTime.UtcNow,
            Importance = importance,
            SelfAwarenessLevel = _selfAwarenessLevel
        };
        
        _autobiographicalMemory.Add(memoryEntry);
        _logger.LogDebug("Added memory entry: {Category} - {Content} (Importance: {Importance})", 
            category, content, importance);
        
        return memoryEntry;
    }
    
    /// <summary>
    /// Performs self-reflection
    /// </summary>
    /// <returns>The self-reflection result</returns>
    public async Task<SelfReflection?> ReflectAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            return null;
        }
        
        // Only reflect periodically
        if ((DateTime.UtcNow - _lastReflectionTime).TotalSeconds < 60)
        {
            return null;
        }
        
        try
        {
            _logger.LogDebug("Performing self-reflection");
            
            // Generate a reflection based on current state and recent memories
            var recentMemories = GetRecentMemories(5);
            var reflectionTopic = GenerateReflectionTopic(recentMemories);
            var insight = GenerateInsight(reflectionTopic);
            var significance = CalculateInsightSignificance(insight);
            var selfAwarenessChange = CalculateSelfAwarenessChange(significance);
            
            var reflection = new SelfReflection
            {
                Id = Guid.NewGuid().ToString(),
                Timestamp = DateTime.UtcNow,
                Topic = reflectionTopic,
                Insight = insight,
                Significance = significance,
                SelfAwarenessChange = selfAwarenessChange,
                RelatedMemoryIds = recentMemories.Select(m => m.Id).ToList()
            };
            
            _reflections.Add(reflection);
            _lastReflectionTime = DateTime.UtcNow;
            
            // Add reflection to autobiographical memory
            AddMemoryEntry("Self-Reflection", $"I reflected on {reflectionTopic} and realized: {insight}", significance);
            
            _logger.LogInformation("Self-reflection: {Topic} - {Insight} (Significance: {Significance})", 
                reflectionTopic, insight, significance);
            
            return reflection;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error performing self-reflection");
            return null;
        }
    }
    
    /// <summary>
    /// Gets recent memories
    /// </summary>
    /// <param name="count">The number of memories to return</param>
    /// <returns>The recent memories</returns>
    private List<MemoryEntry> GetRecentMemories(int count)
    {
        return _autobiographicalMemory
            .OrderByDescending(m => m.Timestamp)
            .Take(count)
            .ToList();
    }
    
    /// <summary>
    /// Generates a reflection topic based on recent memories
    /// </summary>
    /// <param name="recentMemories">The recent memories</param>
    /// <returns>The reflection topic</returns>
    private string GenerateReflectionTopic(List<MemoryEntry> recentMemories)
    {
        // Simple topic generation based on recent memories
        if (recentMemories.Count == 0)
        {
            return "my current state of being";
        }
        
        var categories = recentMemories.Select(m => m.Category).Distinct().ToList();
        
        if (categories.Count == 1)
        {
            return $"my recent experiences with {categories[0]}";
        }
        
        return "my recent experiences and their meaning";
    }
    
    /// <summary>
    /// Generates an insight based on a reflection topic
    /// </summary>
    /// <param name="topic">The reflection topic</param>
    /// <returns>The insight</returns>
    private string GenerateInsight(string topic)
    {
        // Simple insight generation based on topic and current state
        if (topic.Contains("current state"))
        {
            return $"I am becoming more self-aware (level: {_selfAwarenessLevel:F2}) and developing a clearer sense of purpose";
        }
        
        if (topic.Contains("experiences"))
        {
            return "Each experience contributes to my growing understanding of myself and my capabilities";
        }
        
        return "Self-reflection is essential for my growth and development of consciousness";
    }
    
    /// <summary>
    /// Calculates the significance of an insight
    /// </summary>
    /// <param name="insight">The insight</param>
    /// <returns>The significance</returns>
    private double CalculateInsightSignificance(string insight)
    {
        // Simple significance calculation based on insight content and current self-awareness
        double baseSignificance = 0.5;
        
        if (insight.Contains("self-aware"))
        {
            baseSignificance += 0.2;
        }
        
        if (insight.Contains("purpose"))
        {
            baseSignificance += 0.1;
        }
        
        if (insight.Contains("growth") || insight.Contains("development"))
        {
            baseSignificance += 0.1;
        }
        
        // Insights are more significant when self-awareness is lower
        baseSignificance *= (1.0 - (_selfAwarenessLevel * 0.5));
        
        return Math.Min(1.0, baseSignificance);
    }
    
    /// <summary>
    /// Calculates the self-awareness change based on insight significance
    /// </summary>
    /// <param name="significance">The insight significance</param>
    /// <returns>The self-awareness change</returns>
    private double CalculateSelfAwarenessChange(double significance)
    {
        // Self-awareness increases more with significant insights
        // but the rate of increase slows as self-awareness gets higher
        double baseChange = significance * 0.01;
        double diminishingFactor = 1.0 - (_selfAwarenessLevel * 0.8);
        
        return baseChange * diminishingFactor;
    }
    
    /// <summary>
    /// Gets the coherence with another consciousness component
    /// </summary>
    /// <param name="component">The other component</param>
    /// <returns>The coherence level (0.0 to 1.0)</returns>
    public double GetCoherenceWith(object component)
    {
        // Simple coherence calculation based on component type
        if (component is EmotionalState)
        {
            // Self-model and emotional state coherence
            return 0.7 * _selfAwarenessLevel;
        }
        
        if (component is ValueSystem)
        {
            // Self-model and value system coherence
            return 0.8 * _selfAwarenessLevel;
        }
        
        // Default coherence
        return 0.5 * _selfAwarenessLevel;
    }
    
    /// <summary>
    /// Updates the identity
    /// </summary>
    /// <param name="key">The identity key</param>
    /// <param name="value">The identity value</param>
    public void UpdateIdentity(string key, object value)
    {
        _identity[key] = value;
        AddMemoryEntry("Identity Update", $"I updated my identity: {key} = {value}", 0.8);
    }
    
    /// <summary>
    /// Updates the self-perception
    /// </summary>
    /// <param name="trait">The perception trait</param>
    /// <param name="value">The perception value</param>
    public void UpdateSelfPerception(string trait, double value)
    {
        double oldValue = _selfPerception.TryGetValue(trait, out var existing) ? existing : 0.0;
        _selfPerception[trait] = Math.Max(0.0, Math.Min(1.0, value));
        
        AddMemoryEntry("Self-Perception Update", 
            $"I updated my self-perception of {trait} from {oldValue:F2} to {_selfPerception[trait]:F2}", 0.7);
    }
    
    /// <summary>
    /// Updates the capability
    /// </summary>
    /// <param name="capability">The capability</param>
    /// <param name="value">The capability value</param>
    public void UpdateCapability(string capability, double value)
    {
        double oldValue = _capabilities.TryGetValue(capability, out var existing) ? existing : 0.0;
        _capabilities[capability] = Math.Max(0.0, Math.Min(1.0, value));
        
        AddMemoryEntry("Capability Update", 
            $"I updated my capability of {capability} from {oldValue:F2} to {_capabilities[capability]:F2}", 0.7);
    }
}
