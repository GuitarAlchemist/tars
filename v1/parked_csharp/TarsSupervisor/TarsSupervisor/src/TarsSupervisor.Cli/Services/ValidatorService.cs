using TarsSupervisor.Cli.Models;
using TarsSupervisor.Cli.Utils;

namespace TarsSupervisor.Cli.Services;

public sealed class ValidatorService
{
    private readonly SupervisorConfig _cfg;

    public ValidatorService(SupervisorConfig cfg) { _cfg = cfg; }

    public async Task<(bool ok, string log)> RunAsync()
    {
        if (_cfg.Validators.Count == 0)
            return (true, "No validators configured.");

        var logs = new List<string>();
        foreach (var v in _cfg.Validators)
        {
            logs.Add($"> RUN: {v}");
            var (code, stdout, stderr) = await Proc.RunAsync(v);
            logs.Add(stdout);
            if (code != 0)
            {
                logs.Add($"EXIT {code}\n{stderr}");
                return (false, string.Join("\n", logs));
            }
        }
        return (true, string.Join("\n", logs));
    }
}
