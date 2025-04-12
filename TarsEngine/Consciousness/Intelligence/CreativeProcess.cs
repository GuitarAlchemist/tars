using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents a creative process
/// </summary>
public class CreativeProcess
{
    /// <summary>
    /// Gets or sets the process ID
    /// </summary>
    public string Id { get; set; } = Guid.NewGuid().ToString();
    
    /// <summary>
    /// Gets or sets the process type
    /// </summary>
    public CreativeProcessType Type { get; set; } = CreativeProcessType.DivergentThinking;
    
    /// <summary>
    /// Gets or sets the process description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the process timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Gets or sets the idea ID
    /// </summary>
    public string IdeaId { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the process effectiveness (0.0 to 1.0)
    /// </summary>
    public double Effectiveness { get; set; } = 0.5;
    
    /// <summary>
    /// Gets or sets the process duration in seconds
    /// </summary>
    public double DurationSeconds { get; set; } = 0.0;
    
    /// <summary>
    /// Gets or sets the process steps
    /// </summary>
    public List<string> Steps { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the process inputs
    /// </summary>
    public List<string> Inputs { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the process outputs
    /// </summary>
    public List<string> Outputs { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the process context
    /// </summary>
    public Dictionary<string, object> Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the process tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the process learning
    /// </summary>
    public string Learning { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the process challenges
    /// </summary>
    public List<string> Challenges { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the process insights
    /// </summary>
    public List<string> Insights { get; set; } = new();
    
    /// <summary>
    /// Adds a process step
    /// </summary>
    /// <param name="step">The process step</param>
    public void AddStep(string step)
    {
        Steps.Add(step);
    }
    
    /// <summary>
    /// Adds a process input
    /// </summary>
    /// <param name="input">The process input</param>
    public void AddInput(string input)
    {
        Inputs.Add(input);
    }
    
    /// <summary>
    /// Adds a process output
    /// </summary>
    /// <param name="output">The process output</param>
    public void AddOutput(string output)
    {
        Outputs.Add(output);
    }
    
    /// <summary>
    /// Adds a process challenge
    /// </summary>
    /// <param name="challenge">The process challenge</param>
    public void AddChallenge(string challenge)
    {
        Challenges.Add(challenge);
    }
    
    /// <summary>
    /// Adds a process insight
    /// </summary>
    /// <param name="insight">The process insight</param>
    public void AddInsight(string insight)
    {
        Insights.Add(insight);
    }
}
