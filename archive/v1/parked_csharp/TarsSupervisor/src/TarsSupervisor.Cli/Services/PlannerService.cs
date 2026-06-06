using System.Text.RegularExpressions;

namespace TarsSupervisor.Cli.Services
{
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
            // Resolve sample files robustly (works from src/, bin/, or repo root)
            string SamplePath(string fileName)
            {
                var start = Environment.CurrentDirectory;

                // 1) walk up to find a "samples" directory
                string? dir = start;
                for (int i = 0; i < 6 && dir != null; i++)
                {
                    var candidateDir = Path.Combine(dir, "samples");
                    var candidate = Path.Combine(candidateDir, fileName);
                    if (File.Exists(candidate)) return candidate;
                    dir = Directory.GetParent(dir)?.FullName;
                }

                // 2) check AppContext.BaseDirectory (bin/…)
                var fromBase = Path.Combine(AppContext.BaseDirectory, "samples", fileName);
                if (File.Exists(fromBase)) return fromBase;

                // 3) last resort: relative to repo-style layout (bin/…/net8.0 → up 3–4)
                var deep = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "samples", fileName));
                if (File.Exists(deep)) return deep;

                throw new FileNotFoundException($"Could not find samples/{fileName}");
            }

            // Prefer explicit paths if provided; otherwise load bundled samples
            var trsx = trsxPath != null && File.Exists(trsxPath)
                ? await File.ReadAllTextAsync(trsxPath)
                : await File.ReadAllTextAsync(SamplePath("next_steps.trsx"));

            var prompt = promptPath != null && File.Exists(promptPath)
                ? await File.ReadAllTextAsync(promptPath)
                : await File.ReadAllTextAsync(SamplePath("planning_prompt.txt"));

            var metricsSummary = ""; // hook stub if you want to condition planning

            var fullPrompt =
                prompt + "\n\n" +
                "---\n" +
                "# INPUT_TRSX\n" +
                trsx + "\n\n" +
                "# METRICS_SUMMARY\n" +
                metricsSummary + "\n";

            var raw = await _llm.GenerateAsync(fullPrompt);

            static string Extract(string name, string text)
            {
                var rx = new Regex(@"<<<" + name + @"\s*(.*?)\s*" + name + @">>>", RegexOptions.Singleline);
                var m = rx.Match(text);
                return m.Success ? m.Groups[1].Value.Trim() : text;
            }

            var plan = Extract("PLAN.md", raw);
            var nextTrsx = Extract("next_steps.trsx", raw);
            return (plan, nextTrsx);
        }
    }
}
