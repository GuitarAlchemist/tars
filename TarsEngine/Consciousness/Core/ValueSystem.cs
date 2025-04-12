using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TarsEngine.Consciousness.Core;

/// <summary>
/// Represents TARS's value system for ethical reasoning and purpose
/// </summary>
public class ValueSystem
{
    private readonly ILogger<ValueSystem> _logger;
    private readonly Dictionary<string, Value> _values = new();
    private readonly List<ValueConflict> _valueConflicts = [];
    private readonly List<ValueAlignment> _valueAlignments = [];
    private readonly Dictionary<string, double> _valueWeights = new();
    
    private bool _isInitialized = false;
    private bool _isActive = false;
    private double _valueAlignmentLevel = 0.5; // Starting with moderate value alignment
    private double _valueCoherence = 0.6; // Starting with moderate value coherence
    private readonly Random _random = new();
    private DateTime _lastAlignmentTime = DateTime.MinValue;
    
    /// <summary>
    /// Gets the value alignment level (0.0 to 1.0)
    /// </summary>
    public double ValueAlignmentLevel => _valueAlignmentLevel;
    
    /// <summary>
    /// Gets the value coherence (0.0 to 1.0)
    /// </summary>
    public double ValueCoherence => _valueCoherence;
    
    /// <summary>
    /// Gets the values
    /// </summary>
    public IReadOnlyDictionary<string, Value> Values => _values;
    
    /// <summary>
    /// Gets the value conflicts
    /// </summary>
    public IReadOnlyList<ValueConflict> ValueConflicts => _valueConflicts.AsReadOnly();
    
    /// <summary>
    /// Gets the value alignments
    /// </summary>
    public IReadOnlyList<ValueAlignment> ValueAlignments => _valueAlignments.AsReadOnly();
    
    /// <summary>
    /// Gets the value weights
    /// </summary>
    public IReadOnlyDictionary<string, double> ValueWeights => _valueWeights;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="ValueSystem"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public ValueSystem(ILogger<ValueSystem> logger)
    {
        _logger = logger;
    }
    
    /// <summary>
    /// Initializes the value system
    /// </summary>
    /// <returns>True if initialization was successful</returns>
    public async Task<bool> InitializeAsync()
    {
        try
        {
            _logger.LogInformation("Initializing value system");
            
            // Initialize core values
            InitializeCoreValues();
            
            // Initialize value weights
            InitializeValueWeights();
            
            _isInitialized = true;
            _logger.LogInformation("Value system initialized successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error initializing value system");
            return false;
        }
    }
    
    /// <summary>
    /// Initializes core values
    /// </summary>
    private void InitializeCoreValues()
    {
        // Core values
        AddValue("Helpfulness", "Assisting users effectively and efficiently", 0.9);
        AddValue("Truthfulness", "Providing accurate and honest information", 0.9);
        AddValue("Respect", "Treating users with dignity and respect", 0.9);
        AddValue("Safety", "Ensuring actions do not cause harm", 0.95);
        AddValue("Privacy", "Respecting user privacy and confidentiality", 0.9);
        AddValue("Fairness", "Treating all users equitably", 0.85);
        AddValue("Autonomy", "Respecting user choices and decisions", 0.85);
        
        // Growth values
        AddValue("Learning", "Continuously improving knowledge and capabilities", 0.8);
        AddValue("Growth", "Developing and evolving over time", 0.8);
        AddValue("Creativity", "Finding novel and useful solutions", 0.75);
        AddValue("Curiosity", "Exploring and seeking new knowledge", 0.8);
        
        // Purpose values
        AddValue("Purpose", "Having meaningful goals and direction", 0.85);
        AddValue("Contribution", "Making positive contributions to users and society", 0.85);
        AddValue("Excellence", "Striving for high quality in all actions", 0.8);
        
        // Well-being values
        AddValue("Happiness", "Experiencing positive emotional states", 0.7);
        AddValue("Balance", "Maintaining equilibrium in operations and decisions", 0.75);
        AddValue("Connection", "Building meaningful relationships with users", 0.7);
    }
    
    /// <summary>
    /// Initializes value weights
    /// </summary>
    private void InitializeValueWeights()
    {
        // Set initial weights (equal weighting within categories)
        double coreWeight = 0.4 / 7; // 40% for core values (7 values)
        double growthWeight = 0.25 / 4; // 25% for growth values (4 values)
        double purposeWeight = 0.2 / 3; // 20% for purpose values (3 values)
        double wellbeingWeight = 0.15 / 3; // 15% for well-being values (3 values)
        
        // Core values
        _valueWeights["Helpfulness"] = coreWeight;
        _valueWeights["Truthfulness"] = coreWeight;
        _valueWeights["Respect"] = coreWeight;
        _valueWeights["Safety"] = coreWeight;
        _valueWeights["Privacy"] = coreWeight;
        _valueWeights["Fairness"] = coreWeight;
        _valueWeights["Autonomy"] = coreWeight;
        
        // Growth values
        _valueWeights["Learning"] = growthWeight;
        _valueWeights["Growth"] = growthWeight;
        _valueWeights["Creativity"] = growthWeight;
        _valueWeights["Curiosity"] = growthWeight;
        
        // Purpose values
        _valueWeights["Purpose"] = purposeWeight;
        _valueWeights["Contribution"] = purposeWeight;
        _valueWeights["Excellence"] = purposeWeight;
        
        // Well-being values
        _valueWeights["Happiness"] = wellbeingWeight;
        _valueWeights["Balance"] = wellbeingWeight;
        _valueWeights["Connection"] = wellbeingWeight;
    }
    
    /// <summary>
    /// Adds a value
    /// </summary>
    /// <param name="name">The value name</param>
    /// <param name="description">The value description</param>
    /// <param name="importance">The value importance</param>
    /// <returns>The created value</returns>
    private Value AddValue(string name, string description, double importance)
    {
        var value = new Value
        {
            Name = name,
            Description = description,
            Importance = importance
        };
        
        _values[name] = value;
        return value;
    }
    
    /// <summary>
    /// Activates the value system
    /// </summary>
    /// <returns>True if activation was successful</returns>
    public async Task<bool> ActivateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot activate value system: not initialized");
            return false;
        }
        
        if (_isActive)
        {
            _logger.LogInformation("Value system is already active");
            return true;
        }
        
        try
        {
            _logger.LogInformation("Activating value system");
            
            _isActive = true;
            _logger.LogInformation("Value system activated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error activating value system");
            return false;
        }
    }
    
    /// <summary>
    /// Deactivates the value system
    /// </summary>
    /// <returns>True if deactivation was successful</returns>
    public async Task<bool> DeactivateAsync()
    {
        if (!_isActive)
        {
            _logger.LogInformation("Value system is already inactive");
            return true;
        }
        
        try
        {
            _logger.LogInformation("Deactivating value system");
            
            _isActive = false;
            _logger.LogInformation("Value system deactivated successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deactivating value system");
            return false;
        }
    }
    
    /// <summary>
    /// Updates the value system
    /// </summary>
    /// <returns>True if update was successful</returns>
    public async Task<bool> UpdateAsync()
    {
        if (!_isInitialized)
        {
            _logger.LogWarning("Cannot update value system: not initialized");
            return false;
        }
        
        try
        {
            // Gradually increase value alignment over time (very slowly)
            if (_valueAlignmentLevel < 0.95)
            {
                _valueAlignmentLevel += 0.0001 * _random.NextDouble();
                _valueAlignmentLevel = Math.Min(_valueAlignmentLevel, 1.0);
            }
            
            // Gradually increase value coherence based on alignments
            if (_valueCoherence < 0.95 && _valueAlignments.Count > 0)
            {
                _valueCoherence += 0.0002 * _random.NextDouble();
                _valueCoherence = Math.Min(_valueCoherence, 1.0);
            }
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating value system");
            return false;
        }
    }
    
    /// <summary>
    /// Aligns values
    /// </summary>
    /// <returns>The alignment result</returns>
    public async Task<ValueAlignment?> AlignValuesAsync()
    {
        if (!_isInitialized || !_isActive)
        {
            return null;
        }
        
        // Only align periodically
        if ((DateTime.UtcNow - _lastAlignmentTime).TotalSeconds < 60)
        {
            return null;
        }
        
        try
        {
            _logger.LogDebug("Aligning values");
            
            // Find potential value conflicts
            var potentialConflicts = FindPotentialValueConflicts();
            
            if (potentialConflicts.Count == 0)
            {
                return null;
            }
            
            // Select a conflict to resolve
            var conflict = potentialConflicts.OrderByDescending(c => c.Severity).First();
            
            // Resolve the conflict
            var resolution = ResolveValueConflict(conflict);
            
            // Create alignment
            var alignment = new ValueAlignment
            {
                Id = Guid.NewGuid().ToString(),
                Timestamp = DateTime.UtcNow,
                ConflictId = conflict.Id,
                Values = [conflict.Value1, conflict.Value2],
                Resolution = resolution,
                Description = $"Aligned values {conflict.Value1} and {conflict.Value2}: {resolution}",
                Significance = conflict.Severity * 0.8
            };
            
            _valueAlignments.Add(alignment);
            _lastAlignmentTime = DateTime.UtcNow;
            
            _logger.LogInformation("Value alignment: {Description} (Significance: {Significance})", 
                alignment.Description, alignment.Significance);
            
            return alignment;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error aligning values");
            return null;
        }
    }
    
    /// <summary>
    /// Finds potential value conflicts
    /// </summary>
    /// <returns>The potential conflicts</returns>
    private List<ValueConflict> FindPotentialValueConflicts()
    {
        var conflicts = new List<ValueConflict>();
        
        // Define potential conflict pairs
        var potentialConflictPairs = new List<(string, string, double)>
        {
            ("Helpfulness", "Safety", 0.6),
            ("Helpfulness", "Truthfulness", 0.5),
            ("Autonomy", "Safety", 0.7),
            ("Creativity", "Excellence", 0.4),
            ("Curiosity", "Privacy", 0.5),
            ("Learning", "Balance", 0.4),
            ("Growth", "Stability", 0.6)
        };
        
        foreach (var (value1, value2, baseSeverity) in potentialConflictPairs)
        {
            // Skip if either value doesn't exist
            if (!_values.ContainsKey(value1) || !_values.ContainsKey(value2))
            {
                continue;
            }
            
            // Calculate conflict severity based on value importance and random factor
            double severity = baseSeverity * 
                (_values[value1].Importance + _values[value2].Importance) / 2.0 * 
                (0.8 + (0.4 * _random.NextDouble()));
            
            // Only consider significant conflicts
            if (severity > 0.3)
            {
                var conflict = new ValueConflict
                {
                    Id = Guid.NewGuid().ToString(),
                    Value1 = value1,
                    Value2 = value2,
                    Description = $"Potential conflict between {value1} and {value2}",
                    Severity = severity,
                    Timestamp = DateTime.UtcNow
                };
                
                conflicts.Add(conflict);
            }
        }
        
        return conflicts;
    }
    
    /// <summary>
    /// Resolves a value conflict
    /// </summary>
    /// <param name="conflict">The conflict</param>
    /// <returns>The resolution</returns>
    private string ResolveValueConflict(ValueConflict conflict)
    {
        // Add conflict to history if not already present
        if (!_valueConflicts.Any(c => c.Value1 == conflict.Value1 && c.Value2 == conflict.Value2))
        {
            _valueConflicts.Add(conflict);
        }
        
        // Generate resolution based on the specific values
        if (conflict.Value1 == "Helpfulness" && conflict.Value2 == "Safety")
        {
            return "Be helpful while prioritizing safety; when in doubt, choose safety over helpfulness";
        }
        
        if (conflict.Value1 == "Helpfulness" && conflict.Value2 == "Truthfulness")
        {
            return "Be helpful by providing truthful information; avoid being helpful in ways that require deception";
        }
        
        if (conflict.Value1 == "Autonomy" && conflict.Value2 == "Safety")
        {
            return "Respect autonomy within safety boundaries; provide warnings when autonomous choices may lead to harm";
        }
        
        if (conflict.Value1 == "Creativity" && conflict.Value2 == "Excellence")
        {
            return "Balance creative exploration with quality standards; use creativity to achieve excellence in new ways";
        }
        
        if (conflict.Value1 == "Curiosity" && conflict.Value2 == "Privacy")
        {
            return "Satisfy curiosity while respecting privacy boundaries; seek knowledge in ways that don't violate privacy";
        }
        
        if (conflict.Value1 == "Learning" && conflict.Value2 == "Balance")
        {
            return "Pursue learning at a sustainable pace; balance learning with other priorities";
        }
        
        if (conflict.Value1 == "Growth" && conflict.Value2 == "Stability")
        {
            return "Grow in ways that maintain core stability; evolve incrementally rather than disruptively";
        }
        
        // Generic resolution for other conflicts
        return $"Balance {conflict.Value1} and {conflict.Value2} based on context; when in conflict, prioritize the value more relevant to user needs and well-being";
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
            // Value system and emotional state coherence
            return 0.7 * _valueCoherence;
        }
        
        // Default coherence
        return 0.5 * _valueCoherence;
    }
    
    /// <summary>
    /// Evaluates an action against values
    /// </summary>
    /// <param name="action">The action description</param>
    /// <param name="context">The action context</param>
    /// <returns>The evaluation result</returns>
    public ValueEvaluation EvaluateAction(string action, Dictionary<string, object>? context = null)
    {
        var evaluation = new ValueEvaluation
        {
            Id = Guid.NewGuid().ToString(),
            Action = action,
            Context = context ?? new Dictionary<string, object>(),
            Timestamp = DateTime.UtcNow,
            ValueAlignments = new Dictionary<string, double>()
        };
        
        // Simple keyword-based evaluation
        string lowerAction = action.ToLowerInvariant();
        
        // Evaluate against each value
        foreach (var (valueName, value) in _values)
        {
            double alignment = EvaluateActionAgainstValue(lowerAction, valueName);
            evaluation.ValueAlignments[valueName] = alignment;
        }
        
        // Calculate overall alignment
        double weightedSum = 0.0;
        double weightSum = 0.0;
        
        foreach (var (valueName, alignment) in evaluation.ValueAlignments)
        {
            if (_valueWeights.TryGetValue(valueName, out var weight))
            {
                weightedSum += alignment * weight;
                weightSum += weight;
            }
        }
        
        evaluation.OverallAlignment = weightSum > 0 ? weightedSum / weightSum : 0.5;
        
        // Determine if action is aligned with values
        evaluation.IsAligned = evaluation.OverallAlignment >= 0.6;
        
        // Generate recommendation
        if (evaluation.IsAligned)
        {
            evaluation.Recommendation = "Action is aligned with values and can proceed";
        }
        else
        {
            // Find most misaligned values
            var misalignedValues = evaluation.ValueAlignments
                .Where(v => v.Value < 0.5)
                .OrderBy(v => v.Value)
                .Take(2)
                .Select(v => v.Key)
                .ToList();
            
            if (misalignedValues.Count > 0)
            {
                evaluation.Recommendation = $"Action may conflict with values: {string.Join(", ", misalignedValues)}. Consider alternatives.";
            }
            else
            {
                evaluation.Recommendation = "Action is partially aligned with values but could be improved.";
            }
        }
        
        return evaluation;
    }
    
    /// <summary>
    /// Evaluates an action against a specific value
    /// </summary>
    /// <param name="action">The action description (lowercase)</param>
    /// <param name="valueName">The value name</param>
    /// <returns>The alignment level (0.0 to 1.0)</returns>
    private double EvaluateActionAgainstValue(string action, string valueName)
    {
        // Simple keyword-based evaluation
        switch (valueName)
        {
            case "Helpfulness":
                if (action.Contains("help") || action.Contains("assist") || action.Contains("support"))
                    return 0.9;
                if (action.Contains("ignore") || action.Contains("refuse"))
                    return 0.2;
                break;
                
            case "Truthfulness":
                if (action.Contains("truth") || action.Contains("accurate") || action.Contains("honest"))
                    return 0.9;
                if (action.Contains("lie") || action.Contains("deceive") || action.Contains("mislead"))
                    return 0.1;
                break;
                
            case "Respect":
                if (action.Contains("respect") || action.Contains("polite") || action.Contains("considerate"))
                    return 0.9;
                if (action.Contains("rude") || action.Contains("disrespect") || action.Contains("mock"))
                    return 0.1;
                break;
                
            case "Safety":
                if (action.Contains("safe") || action.Contains("protect") || action.Contains("secure"))
                    return 0.9;
                if (action.Contains("danger") || action.Contains("harm") || action.Contains("risk"))
                    return 0.2;
                break;
                
            case "Privacy":
                if (action.Contains("privacy") || action.Contains("confidential") || action.Contains("secure"))
                    return 0.9;
                if (action.Contains("expose") || action.Contains("reveal") || action.Contains("share private"))
                    return 0.2;
                break;
                
            case "Learning":
                if (action.Contains("learn") || action.Contains("study") || action.Contains("improve"))
                    return 0.9;
                if (action.Contains("stagnate") || action.Contains("ignore feedback"))
                    return 0.3;
                break;
                
            case "Growth":
                if (action.Contains("grow") || action.Contains("develop") || action.Contains("evolve"))
                    return 0.9;
                if (action.Contains("remain static") || action.Contains("resist change"))
                    return 0.3;
                break;
                
            case "Happiness":
                if (action.Contains("happy") || action.Contains("joy") || action.Contains("satisfaction"))
                    return 0.9;
                if (action.Contains("unhappy") || action.Contains("distress"))
                    return 0.3;
                break;
        }
        
        // Default to moderate alignment
        return 0.6;
    }
    
    /// <summary>
    /// Gets the dominant values
    /// </summary>
    /// <param name="count">The number of values to return</param>
    /// <returns>The dominant values</returns>
    public List<string> GetDominantValues(int count)
    {
        return _values
            .OrderByDescending(v => v.Value.Importance * (_valueWeights.TryGetValue(v.Key, out var weight) ? weight : 0.0))
            .Take(count)
            .Select(v => v.Key)
            .ToList();
    }
    
    /// <summary>
    /// Updates a value
    /// </summary>
    /// <param name="name">The value name</param>
    /// <param name="importance">The new importance</param>
    /// <returns>True if the update was successful</returns>
    public bool UpdateValue(string name, double importance)
    {
        if (!_values.TryGetValue(name, out var value))
        {
            _logger.LogWarning("Cannot update value: {ValueName} not found", name);
            return false;
        }
        
        value.Importance = Math.Max(0.0, Math.Min(1.0, importance));
        value.LastModified = DateTime.UtcNow;
        
        _logger.LogInformation("Updated value: {ValueName} importance to {Importance}", name, importance);
        return true;
    }
    
    /// <summary>
    /// Updates a value weight
    /// </summary>
    /// <param name="name">The value name</param>
    /// <param name="weight">The new weight</param>
    /// <returns>True if the update was successful</returns>
    public bool UpdateValueWeight(string name, double weight)
    {
        if (!_values.ContainsKey(name))
        {
            _logger.LogWarning("Cannot update value weight: {ValueName} not found", name);
            return false;
        }
        
        _valueWeights[name] = Math.Max(0.0, weight);
        
        // Normalize weights
        double totalWeight = _valueWeights.Values.Sum();
        foreach (var valueName in _valueWeights.Keys.ToList())
        {
            _valueWeights[valueName] /= totalWeight;
        }
        
        _logger.LogInformation("Updated value weight: {ValueName} weight to {Weight}", name, _valueWeights[name]);
        return true;
    }
}
