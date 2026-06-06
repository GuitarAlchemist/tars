using System.Text.Json;
using TarsSupervisor.Cli.Models;
using TarsSupervisor.Cli.Utils;

namespace TarsSupervisor.Cli.Services;

public sealed class MetricsService
{
    private readonly SupervisorConfig _cfg;
    public MetricsService(SupervisorConfig cfg) { _cfg = cfg; }

    public async Task WriteAsync(IterationRecord rec, Dictionary<string, object> extra)
    {
        var path = Path.Combine(_cfg.MetricsDir, "metrics.jsonl");
        var payload = new Dictionary<string, object>(extra)
        {
            ["id"] = rec.Id,
            ["startedAt"] = rec.StartedAt,
            ["completedAt"] = rec.CompletedAt,
            ["success"] = rec.Success,
            ["failureReason"] = rec.FailureReason ?? ""
        };
        var line = JsonSerializer.Serialize(payload);
        await Io.AppendAllTextAsync(path, line + "\n");
    }
}
