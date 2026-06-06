namespace TarsSupervisor.Cli.Models;

public sealed class IterationRecord
{
    public string Id { get; set; } = DateTimeOffset.UtcNow.ToString("yyyyMMdd_HHmmss");
    public DateTimeOffset StartedAt { get; set; } = DateTimeOffset.UtcNow;
    public DateTimeOffset? CompletedAt { get; set; }
    public bool Success { get; set; }
    public string PlanPath { get; set; } = string.Empty;
    public string TrsxPath { get; set; } = string.Empty;
    public string LogPath { get; set; } = string.Empty;
    public string MetricsPath { get; set; } = string.Empty;
    public string? FailureReason { get; set; }
}
