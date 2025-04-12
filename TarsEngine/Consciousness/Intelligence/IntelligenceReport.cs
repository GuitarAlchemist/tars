using System;
using System.Collections.Generic;

namespace TarsEngine.Consciousness.Intelligence;

/// <summary>
/// Represents a comprehensive intelligence report
/// </summary>
public class IntelligenceReport
{
    /// <summary>
    /// Gets or sets the report timestamp
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets or sets whether the intelligence spark is initialized
    /// </summary>
    public bool IsInitialized { get; set; }

    /// <summary>
    /// Gets or sets whether the intelligence spark is active
    /// </summary>
    public bool IsActive { get; set; }

    /// <summary>
    /// Gets or sets the intelligence level
    /// </summary>
    public double IntelligenceLevel { get; set; }

    /// <summary>
    /// Gets or sets the creativity level
    /// </summary>
    public double CreativityLevel { get; set; }

    /// <summary>
    /// Gets or sets the intuition level
    /// </summary>
    public double IntuitionLevel { get; set; }

    /// <summary>
    /// Gets or sets the curiosity level
    /// </summary>
    public double CuriosityLevel { get; set; }

    /// <summary>
    /// Gets or sets the insight level
    /// </summary>
    public double InsightLevel { get; set; }

    /// <summary>
    /// Gets or sets the recent events
    /// </summary>
    public List<IntelligenceEvent> RecentEvents { get; set; } = new();

    /// <summary>
    /// Gets or sets the creative ideas
    /// </summary>
    public List<CreativeIdea> CreativeIdeas { get; set; } = new();

    /// <summary>
    /// Gets or sets the intuitions
    /// </summary>
    public List<Intuition> Intuitions { get; set; } = new();

    /// <summary>
    /// Gets or sets the spontaneous thoughts
    /// </summary>
    public List<ThoughtModel> SpontaneousThoughts { get; set; } = new();

    /// <summary>
    /// Gets or sets the curiosity questions
    /// </summary>
    public List<CuriosityQuestion> CuriosityQuestions { get; set; } = new();

    /// <summary>
    /// Gets or sets the insights
    /// </summary>
    public List<InsightLegacy> Insights { get; set; } = new();

    /// <summary>
    /// Gets or sets the report summary
    /// </summary>
    public string Summary { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the report insights
    /// </summary>
    public List<string> ReportInsights { get; set; } = new();

    /// <summary>
    /// Gets or sets the report recommendations
    /// </summary>
    public List<string> Recommendations { get; set; } = new();

    /// <summary>
    /// Gets or sets the intelligence growth rate
    /// </summary>
    public double IntelligenceGrowthRate { get; set; }

    /// <summary>
    /// Gets or sets the creativity growth rate
    /// </summary>
    public double CreativityGrowthRate { get; set; }

    /// <summary>
    /// Gets or sets the intuition growth rate
    /// </summary>
    public double IntuitionGrowthRate { get; set; }

    /// <summary>
    /// Gets or sets the curiosity growth rate
    /// </summary>
    public double CuriosityGrowthRate { get; set; }

    /// <summary>
    /// Gets or sets the insight growth rate
    /// </summary>
    public double InsightGrowthRate { get; set; }

    /// <summary>
    /// Gets or sets the intelligence potential
    /// </summary>
    public double IntelligencePotential { get; set; }

    /// <summary>
    /// Gets or sets the intelligence development stage
    /// </summary>
    public string DevelopmentStage { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the intelligence evolution progress
    /// </summary>
    public double EvolutionProgress { get; set; }
}
