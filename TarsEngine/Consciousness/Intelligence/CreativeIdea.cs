using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents a creative idea
/// </summary>
public class CreativeIdea
{
    /// <summary>
    /// Gets or sets the idea ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the idea description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the idea originality (0.0 to 1.0)
    /// </summary>
    public double Originality { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the idea value (0.0 to 1.0)
    /// </summary>
    public double Value { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the idea timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the creative process type
    /// </summary>
    public CreativeProcessType ProcessType { get; set; } = CreativeProcessType.DivergentThinking;
    
    /// <summary>
    /// Gets or sets the concepts involved in the idea
    /// </summary>
    public List<string> Concepts { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the idea tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the idea context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the problem the idea addresses
    /// </summary>
    public string Problem { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the constraints on the idea
    /// </summary>
    public List<string> Constraints { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the idea implementation steps
    /// </summary>
    public List<string> ImplementationSteps { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the idea potential impact
    /// </summary>
    public string PotentialImpact { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the idea limitations
    /// </summary>
    public List<string> Limitations { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the idea evaluation score (0.0 to 1.0)
    /// </summary>
    public double EvaluationScore { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets whether the idea has been implemented
    /// </summary>
    public bool IsImplemented { get; set; } = false;
    
    /// <summary>
    /// Gets or sets the idea implementation timestamp
    /// </summary>
    public DateTime? ImplementationTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the idea implementation outcome
    /// </summary>
    public string ImplementationOutcome { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets the idea quality score (combination of originality and value)
    /// </summary>
    public double QualityScore => (Originality + Value) / 2.0;
    
    /// <summary>
    /// Evaluates the idea
    /// </summary>
    /// <param name="criteria">The evaluation criteria</param>
    /// <returns>The evaluation score</returns>
    public double Evaluate(Dictionary<string, double> criteria)
    {
        double score = 0.0;
        double totalWeight = 0.0;
        
        // Default criteria if none provided
        if (criteria == null || criteria.Count == 0)
        {
            criteria = new Dictionary<string, double>
            {
                { "Originality", 0.4 },
                { "Value", 0.4 },
                { "Feasibility", 0.2 }
            };
        }
        
        // Evaluate based on criteria
        foreach (var (criterion, weight) in criteria)
        {
            double criterionScore = criterion.ToLowerInvariant() switch
            {
                "originality" => Originality,
                "value" => Value,
                "feasibility" => 0.8 - (0.3 * Originality), // More original ideas might be less feasible
                "impact" => Value * 1.2, // Impact related to value but potentially higher
                "novelty" => Originality * 1.1, // Novelty related to originality but slightly different
                "relevance" => Problem.Length > 0 ? 0.8 : 0.5, // More relevant if addressing a specific problem
                "completeness" => ImplementationSteps.Count > 0 ? 0.7 : 0.3, // More complete if implementation steps provided
                _ => 0.5 // Default score for unknown criteria
            };
            
            score += criterionScore * weight;
            totalWeight += weight;
        }
        
        // Calculate final score
        EvaluationScore = totalWeight > 0 ? score / totalWeight : 0.5;
        
        return EvaluationScore;
    }
    
    /// <summary>
    /// Adds an implementation step
    /// </summary>
    /// <param name="step">The implementation step</param>
    public void AddImplementationStep(string step)
    {
        ImplementationSteps.Add(step);
    }
    
    /// <summary>
    /// Adds a limitation
    /// </summary>
    /// <param name="limitation">The limitation</param>
    public void AddLimitation(string limitation)
    {
        Limitations.Add(limitation);
    }
    
    /// <summary>
    /// Marks the idea as implemented
    /// </summary>
    /// <param name="outcome">The implementation outcome</param>
    public void MarkAsImplemented(string outcome)
    {
        IsImplemented = true;
        ImplementationTimestamp = DateTime.UtcNow;
        ImplementationOutcome = outcome;
    }
}
