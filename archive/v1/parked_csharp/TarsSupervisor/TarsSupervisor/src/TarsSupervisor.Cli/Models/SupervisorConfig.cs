using System.Text.Json.Serialization;

namespace TarsSupervisor.Cli.Models;

public sealed class OllamaConfig
{
    public string Endpoint { get; set; } = "http://localhost:11434";
    public string Model { get; set; } = "qwen2.5-coder:latest";
    public double Temperature { get; set; } = 0.2;
    public int NumCtx { get; set; } = 8192;
}

public sealed class SupervisorConfig
{
    public string ProjectRoot { get; set; } = Directory.GetCurrentDirectory();
    public string VersionDir { get; set; } = Path.Combine("output", "versions");
    public string MetricsDir { get; set; } = Path.Combine("output", "metrics");
    public OllamaConfig Ollama { get; set; } = new();
    public List<string> Validators { get; set; } = new();
    public List<string> PostPlanHooks { get; set; } = new();
    public List<string> PostValidateHooks { get; set; } = new();
}
