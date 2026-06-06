using System.Text.Json;
using TarsSupervisor.Cli.Models;
using TarsSupervisor.Cli.Utils;

namespace TarsSupervisor.Cli.Services;

public sealed class ConfigService
{
    private const string ConfigFile = "tars.supervisor.json";
    public SupervisorConfig Config { get; private set; }

    public ConfigService()
    {
        if (File.Exists(ConfigFile))
        {
            var json = File.ReadAllText(ConfigFile);
            Config = JsonSerializer.Deserialize<SupervisorConfig>(json, new JsonSerializerOptions { PropertyNameCaseInsensitive = true }) ?? new SupervisorConfig();
        }
        else
        {
            Config = new SupervisorConfig();
        }
    }

    public async Task SaveAsync()
    {
        var json = JsonSerializer.Serialize(Config, new JsonSerializerOptions { WriteIndented = true });
        await Io.WriteAllTextAsync(ConfigFile, json);
    }
}
