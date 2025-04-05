using System;

namespace TarsCli.Models;

/// <summary>
/// Status of the autonomous improvement process
/// </summary>
public class AutonomousImprovementStatus
{
    /// <summary>
    /// Whether the autonomous improvement process is running
    /// </summary>
    public bool IsRunning { get; set; }
    
    /// <summary>
    /// Start time of the autonomous improvement process
    /// </summary>
    public DateTime StartTime { get; set; }
    
    /// <summary>
    /// End time of the autonomous improvement process
    /// </summary>
    public DateTime EndTime { get; set; }
    
    /// <summary>
    /// Elapsed time of the autonomous improvement process
    /// </summary>
    public TimeSpan ElapsedTime { get; set; }
    
    /// <summary>
    /// Remaining time of the autonomous improvement process
    /// </summary>
    public TimeSpan RemainingTime { get; set; }
}
