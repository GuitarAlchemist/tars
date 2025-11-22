using TarsSupervisor.Cli.Models;
using TarsSupervisor.Cli.Utils;

namespace TarsSupervisor.Cli.Services;

public sealed class VersionStore
{
    private readonly SupervisorConfig _cfg;

    public VersionStore(SupervisorConfig cfg) { _cfg = cfg; }

    public string CreateIterationFolder(string? id = null)
    {
        var iterId = id ?? Io.SlugTimeUtc();
        var dir = Path.Combine(_cfg.VersionDir, iterId);
        Directory.CreateDirectory(dir);
        return dir;
    }

    public string LastSuccessfulVersion()
    {
        var root = _cfg.VersionDir;
        if (!Directory.Exists(root)) return string.Empty;
        var dirs = Directory.GetDirectories(root).OrderByDescending(d => d).ToList();
        foreach (var d in dirs)
        {
            var flag = Path.Combine(d, "SUCCESS");
            if (File.Exists(flag)) return d;
        }
        return string.Empty;
    }
}
