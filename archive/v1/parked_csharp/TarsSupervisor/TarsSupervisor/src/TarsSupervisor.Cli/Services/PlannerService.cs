using System.Text.RegularExpressions;
using TarsSupervisor.Cli.Utils;

namespace TarsSupervisor.Cli.Services;

public sealed class PlannerService
{
    private readonly OllamaClient _llm;
    private readonly string _projectRoot;

    public PlannerService(OllamaClient llm, string projectRoot)
    {
        _llm = llm;
        _projectRoot = projectRoot;
    }

    public async Task<(string planMd, string trsx)> MakePlanAsync(string? trsxPath = null, string? promptPath = null)
    {
        var trsx = trsxPath != null && File.Exists(trsxPath)
            ? await File.ReadAllTextAsync(trsxPath)
            : await File.ReadAllTextAsync(Path.Combine("samples", "next_steps.trsx"));

        var prompt = promptPath != null && File.Exists(promptPath)
            ? await File.ReadAllTextAsync(promptPath)
            : await File.ReadAllTextAsync(Path.Combine("samples", "planning_prompt.txt"));

        var metricsSummary = ""; // hook up if you want to condition planning

        var fullPrompt = $"""{prompt}

---
# INPUT_TRSX
{trsx}

# METRICS_SUMMARY
{metricsSummary}
""";

        var raw = await _llm.GenerateAsync(fullPrompt);
        string Extract(string name)
        {
            var rx = new Regex($@"<<<{name}\s*(.*?)\s*{name}>>>", RegexOptions.Singleline);
            var m = rx.Match(raw);
            return m.Success ? m.Groups[1].Value.Trim() : raw;
        }
        var plan = Extract("PLAN.md");
        var nextTrsx = Extract("next_steps.trsx");
        return (plan, nextTrsx);
    }
}
