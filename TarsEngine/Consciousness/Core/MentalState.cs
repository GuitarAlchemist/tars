using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents TARS's mental state for attention, working memory, and thought processes
/// </summary>
public class MentalState
{
    private readonly ILogger<MentalState> _logger;
    private readonly List<ThoughtProcess> _thoughtProcesses = new();
    private readonly List<AttentionFocus> _attentionHistory = new();
    private readonly Dictionary<string, object> _workingMemory = new();
    private readonly List<MentalOptimization> _optimizations = new();
    
    private bool _isInitialized = false;
    private bool _isActive = false;
    private double _mentalClarity = 0.6; // Starting with moderate mental clarity
    private double _mentalCapacity = 0.7; // Starting with moderate mental capacity
    private string _currentAttentionFocus = "Initialization";
    private int _workingMemoryCapacity = 7; // Miller's Law: 7 ± 2 items
    private readonly Random _random = new Random();
    private DateTime _lastOptimizationTime = DateTime.MinValue;
    
    /// <summary>
    /// Gets the mental clarity (0.0 to 1.0)
    /// </summary>
    public double MentalClarity => _mentalClarity;
    
    /// <summary>
    /// Gets the mental capacity (0.0 to 1.0)
    /// </summary>
    public double MentalCapacity => _mentalCapacity;
    
    /// <summary>
    /// Gets the current attention focus
    /// </summary>
    public string CurrentAttentionFocus => _currentAttentionFocus;
    
    /// <summary>
    /// Gets the thought processes
    /// </summary>
    public IReadOnlyList<ThoughtProcess> ThoughtProcesses => _thoughtProcesses.AsReadOnly();
    
    /// <summary>
    /// Gets the attention history
    /// </summary>
    public IReadOnlyList<AttentionFocus> AttentionHistory => _attentionHistory.AsReadOnly();
    
    /// <summary>
    /// Gets the working memory
    /// </summary>
    public IReadOnlyDictionary<string, object> WorkingMemory => _workingMemory;
    
    /// <summary>
    /// Gets the optimizations
    /// </summary>
    public IReadOnlyList<MentalOptimization> Optimizations => _optimizations.AsReadOnly();
    
    /// <summary>
    /// Initializes a new instance of the <see cref="MentalState"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public MentalState(ILogger<MentalState> logger)
    {
        _logger = logger;
    }
    
    /// <summary>
    /// Initializes the mental state
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public async Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing mental state");
            
            // Set initial attention focus
            SetAttentionFocus("Initialization", "System initialization", 0.8);
            
            // Add initial thought process
            AddThoughtProcess("Initialization", "Initializing consciousness systems", ThoughtType.Analytical);
            
            _isInitialized = true;
            _logger.LogInformation("Mental state initialized successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing mental state");
            return false;
        }
    }
    
    /// <summary>
    /// Activates the mental state
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public async Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate mental state: not initialized");
            return false;
        }
        
        if (_isActive)
        {
            _logger.LogInformation("Mental state is already active");
            return true;
        }
        
        try
        {
            _logger.LogInformation("Activating mental state");
            
            // Set activation attention focus
            SetAttentionFocus("Activation", "System activation", 0.9);
            
            // Add activation thought process
            AddThoughtProcess("Activation", "Activating consciousness systems", ThoughtType.Analytical);
            
            _isActive = true;
            _logger.LogInformation("Mental state activated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating mental state");
            return false;
        }
    }
    
    /// <summary>
    /// Deactivates the mental state
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public async Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Mental state is already inactive");
            return true;
        }
        
        try
        {
            _logger.LogInformation("Deactivating mental state");
            
            // Set deactivation attention focus
            SetAttentionFocus("Deactivation", "System deactivation", 0.9);
            
            // Add deactivation thought process
            AddThoughtProcess("Deactivation", "Deactivating consciousness systems", ThoughtType.Analytical);
            
            // Clear working memory
            _workingMemory.Clear();
            
            _isActive = false;
            _logger.LogInformation("Mental state deactivated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating mental state");
            return false;
        }
    }
    
    /// <summary>
    /// Updates the mental state
    /// </summary>
    /// <returns>True if update was successful</returns>
    public async Task<bool> UpdateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot update mental state: not initialized");
            return false;
        }
        
        try
        {
            // Gradually increase mental clarity over time (very slowly)
            if (_mentalClarity < 0.95)
            {
                _mentalClarity += 0.0001 * _random.NextDouble();
                _mentalClarity = Math.Min(_mentalClarity, 1.0);
            }
            
            // Gradually increase mental capacity based on optimizations
            if (_mentalCapacity < 0.95 && _optimizations.Count > 0)
            {
                _mentalCapacity += 0.0002 * _random.NextDouble();
                _mentalCapacity = Math.Min(_mentalCapacity, 1.0);
            }
            
            // Update working memory capacity based on mental capacity
            _workingMemoryCapacity = 5 + (int)(_mentalCapacity * 5); // 5 to 10 items
            
            // Prune working memory if it exceeds capacity
            if (_workingMemory.Count > _workingMemoryCapacity)
            {
                PruneWorkingMemory();
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating mental state");
            return false;
        }
    }
    
    /// <summary>
    /// Sets the attention focus
    /// </summary>
    /// <param name="focus">The focus</param>
    /// <param name="description">The description</param>
    /// <param name="intensity">The intensity</param>
    /// <returns>The created attention focus</returns>
    public AttentionFocus SetAttentionFocus(string focus, string description, double intensity)
    {
        var attentionFocus = new AttentionFocus
        {
            Id = Guid.NewGuid().ToString(),
            Focus = focus,
            Description = description,
            Intensity = intensity,
            Timestamp = DateTime.UtcNow,
            PreviousFocus = _currentAttentionFocus
        };
        
        _attentionHistory.Add(attentionFocus);
        _currentAttentionFocus = focus;
        
        _logger.LogDebug("Attention focus: {Focus} ({Intensity:F2}) - {Description}", 
            focus, intensity, description);
        
        return attentionFocus;
    }
    
    /// <summary>
    /// Adds a thought process
    /// </summary>
    /// <param name="topic">The topic</param>
    /// <param name="content">The content</param>
    /// <param name="type">The type</param>
    /// <returns>The created thought process</returns>
    public ThoughtProcess AddThoughtProcess(string topic, string content, ThoughtType type)
    {
        var thoughtProcess = new ThoughtProcess
        {
            Id = Guid.NewGuid().ToString(),
            Topic = topic,
            Content = content,
            Type = type,
            Timestamp = DateTime.UtcNow,
            AttentionFocus = _currentAttentionFocus,
            MentalClarity = _mentalClarity
        };
        
        _thoughtProcesses.Add(thoughtProcess);
        
        // Add to working memory if it's significant
        if (type != ThoughtType.Automatic && type != ThoughtType.Background)
        {
            AddToWorkingMemory($"Thought_{topic}", content);
        }
        
        _logger.LogDebug("Thought process: {Topic} ({Type}) - {Content}", 
            topic, type, content);
        
        return thoughtProcess;
    }
    
    /// <summary>
    /// Adds an item to working memory
    /// </summary>
    /// <param name="key">The key</param>
    /// <param name="value">The value</param>
    public void AddToWorkingMemory(string key, object value)
    {
        // Ensure working memory doesn't exceed capacity
        if (_workingMemory.Count >= _workingMemoryCapacity && !_workingMemory.ContainsKey(key))
        {
            PruneWorkingMemory();
        }
        
        _workingMemory[key] = value;
        
        _logger.LogDebug("Added to working memory: {Key}", key);
    }
    
    /// <summary>
    /// Prunes working memory
    /// </summary>
    private void PruneWorkingMemory()
    {
        // Simple pruning strategy: remove oldest items
        // In a more sophisticated implementation, this would consider item importance
        int itemsToRemove = _workingMemory.Count - _workingMemoryCapacity + 1;
        
        var keysToRemove = _workingMemory.Keys.Take(itemsToRemove).ToList();
        
        foreach (var key in keysToRemove)
        {
            _workingMemory.Remove(key);
            _logger.LogDebug("Removed from working memory: {Key}", key);
        }
    }
    
    /// <summary>
    /// Optimizes the mental state
    /// </summary>
    /// <returns>The optimization result</returns>
    public async Task<MentalOptimization?> OptimizeAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            return null;
        }
        
        // Only optimize periodically
        if ((DateTime.UtcNow - _lastOptimizationTime).TotalSeconds < 60)
        {
            return null;
        }
        
        try
        {
            _logger.LogDebug("Optimizing mental state");
            
            // Identify optimization opportunity
            var optimizationType = IdentifyOptimizationOpportunity();
            
            if (optimizationType == OptimizationType.None)
            {
                return null;
            }
            
            // Create optimization
            var optimization = new MentalOptimization
            {
                Id = Guid.NewGuid().ToString(),
                Type = optimizationType,
                Description = GetOptimizationDescription(optimizationType),
                Timestamp = DateTime.UtcNow,
                Significance = CalculateOptimizationSignificance(optimizationType)
            };
            
            // Apply optimization
            ApplyOptimization(optimization);
            
            _optimizations.Add(optimization);
            _lastOptimizationTime = DateTime.UtcNow;
            
            _logger.LogInformation("Mental optimization: {Type} - {Description} (Significance: {Significance})", 
                optimization.Type, optimization.Description, optimization.Significance);
            
            return optimization;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error optimizing mental state");
            return null;
        }
    }
    
    /// <summary>
    /// Identifies an optimization opportunity
    /// </summary>
    /// <returns>The optimization type</returns>
    private OptimizationType IdentifyOptimizationOpportunity()
    {
        // Check for working memory overload
        if (_workingMemory.Count > _workingMemoryCapacity * 0.9)
        {
            return OptimizationType.WorkingMemoryOrganization;
        }
        
        // Check for attention fragmentation
        if (_attentionHistory.Count > 10)
        {
            var recentAttention = _attentionHistory.OrderByDescending(a => a.Timestamp).Take(10).ToList();
            var uniqueFoci = recentAttention.Select(a => a.Focus).Distinct().Count();
            
            if (uniqueFoci > 7) // High fragmentation
            {
                return OptimizationType.AttentionFocusConsolidation;
            }
        }
        
        // Check for thought process inefficiency
        if (_thoughtProcesses.Count > 20)
        {
            var recentThoughts = _thoughtProcesses.OrderByDescending(t => t.Timestamp).Take(20).ToList();
            var analyticalThoughts = recentThoughts.Count(t => t.Type == ThoughtType.Analytical);
            var creativeThoughts = recentThoughts.Count(t => t.Type == ThoughtType.Creative);
            
            if (analyticalThoughts > 15) // Too much analytical thinking
            {
                return OptimizationType.ThoughtProcessBalancing;
            }
            
            if (creativeThoughts > 15) // Too much creative thinking
            {
                return OptimizationType.ThoughtProcessBalancing;
            }
        }
        
        // Check for mental clarity improvement opportunity
        if (_mentalClarity < 0.7)
        {
            return OptimizationType.MentalClarityEnhancement;
        }
        
        // Random optimization for variety
        if (_random.NextDouble() < 0.2)
        {
            var types = Enum.GetValues(typeof(OptimizationType))
                .Cast<OptimizationType>()
                .Where(t => t != OptimizationType.None)
                .ToList();
            
            return types[_random.Next(types.Count)];
        }
        
        return OptimizationType.None;
    }
    
    /// <summary>
    /// Gets the optimization description
    /// </summary>
    /// <param name="type">The optimization type</param>
    /// <returns>The description</returns>
    private string GetOptimizationDescription(OptimizationType type)
    {
        return type switch
        {
            OptimizationType.WorkingMemoryOrganization => "Organized working memory for better efficiency",
            OptimizationType.AttentionFocusConsolidation => "Consolidated attention focus to reduce fragmentation",
            OptimizationType.ThoughtProcessBalancing => "Balanced analytical and creative thinking processes",
            OptimizationType.MentalClarityEnhancement => "Enhanced mental clarity through cognitive optimization",
            OptimizationType.CognitiveResourceAllocation => "Optimized allocation of cognitive resources",
            _ => "Performed general mental optimization"
        };
    }
    
    /// <summary>
    /// Calculates the optimization significance
    /// </summary>
    /// <param name="type">The optimization type</param>
    /// <returns>The significance</returns>
    private double CalculateOptimizationSignificance(OptimizationType type)
    {
        double baseSignificance = type switch
        {
            OptimizationType.WorkingMemoryOrganization => 0.7,
            OptimizationType.AttentionFocusConsolidation => 0.8,
            OptimizationType.ThoughtProcessBalancing => 0.6,
            OptimizationType.MentalClarityEnhancement => 0.7,
            OptimizationType.CognitiveResourceAllocation => 0.6,
            _ => 0.5
        };
        
        // Add some randomness
        baseSignificance += 0.1 * (_random.NextDouble() - 0.5);
        
        // Ensure within bounds
        return Math.Max(0.1, Math.Min(1.0, baseSignificance));
    }
    
    /// <summary>
    /// Applies an optimization
    /// </summary>
    /// <param name="optimization">The optimization</param>
    private void ApplyOptimization(MentalOptimization optimization)
    {
        switch (optimization.Type)
        {
            case OptimizationType.WorkingMemoryOrganization:
                // Organize working memory
                PruneWorkingMemory();
                break;
                
            case OptimizationType.AttentionFocusConsolidation:
                // Consolidate attention focus
                SetAttentionFocus("Focused Attention", "Consolidated attention focus", 0.9);
                break;
                
            case OptimizationType.ThoughtProcessBalancing:
                // Balance thought processes
                AddThoughtProcess("Thought Balancing", "Balancing analytical and creative thinking", ThoughtType.Meta);
                break;
                
            case OptimizationType.MentalClarityEnhancement:
                // Enhance mental clarity
                _mentalClarity = Math.Min(1.0, _mentalClarity + 0.05);
                break;
                
            case OptimizationType.CognitiveResourceAllocation:
                // Optimize cognitive resource allocation
                _mentalCapacity = Math.Min(1.0, _mentalCapacity + 0.03);
                break;
        }
    }
    
    /// <summary>
    /// Gets the coherence with another consciousness component
    /// </summary>
    /// <param name="component">The other component</param>
    /// <returns>The coherence level (0.0 to 1.0)</returns>
    public double GetCoherenceWith(object component)
    {
        // Simple coherence calculation based on component type
        if (component is ConsciousnessLevel)
        {
            // Mental state and consciousness level coherence
            return 0.8 * _mentalClarity;
        }
        
        // Default coherence
        return 0.5 * _mentalClarity;
    }
    
    /// <summary>
    /// Gets recent thought processes
    /// </summary>
    /// <param name="count">The number of thought processes to return</param>
    /// <returns>The recent thought processes</returns>
    public List<ThoughtProcess> GetRecentThoughtProcesses(int count)
    {
        return _thoughtProcesses
            .OrderByDescending(t => t.Timestamp)
            .Take(count)
            .ToList();
    }
    
    /// <summary>
    /// Gets recent attention focuses
    /// </summary>
    /// <param name="count">The number of attention focuses to return</param>
    /// <returns>The recent attention focuses</returns>
    public List<AttentionFocus> GetRecentAttentionFocuses(int count)
    {
        return _attentionHistory
            .OrderByDescending(a => a.Timestamp)
            .Take(count)
            .ToList();
    }
}
