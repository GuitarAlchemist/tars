
using System.Text.Json;
using TarsSupervisor.Cli.Models;
using TarsSupervisor.Cli.Services;
using TarsSupervisor.Cli.Utils;

var cfgSvc = new ConfigService();
var cfg = cfgSvc.Config;

static void Head(string t)
{
    Console.WriteLine(new string('=', Math.Max(6, t.Length)));
    Console.WriteLine(t);
    Console.WriteLine(new string('=', Math.Max(6, t.Length)));
}

switch (args.FirstOrDefault())
{
    case "init":
        {
            var rootIdx = Array.IndexOf(args, "--project-root");
            if (rootIdx >= 0 && rootIdx + 1 < args.Length)
                cfg.ProjectRoot = args[rootIdx + 1];

            Io.EnsureDir(cfg.VersionDir);
            Io.EnsureDir(cfg.MetricsDir);
            await cfgSvc.SaveAsync();
            Console.WriteLine($"Initialized. ProjectRoot={cfg.ProjectRoot}");
            break;
        }
    case "plan":
        {
            Head("PLAN");
            var llm = new OllamaClient(cfg.Ollama.Endpoint, cfg.Ollama.Model, cfg.Ollama.Temperature, cfg.Ollama.NumCtx);
            var planner = new PlannerService(llm, cfg.ProjectRoot);
            var (plan, trsx) = await planner.MakePlanAsync();
            var iterDir = new VersionStore(cfg).CreateIterationFolder();
            var planPath = Path.Combine(iterDir, "PLAN.md");
            var trsxPath = Path.Combine(iterDir, "next_steps.trsx");
            await Io.WriteAllTextAsync(planPath, plan);
            await Io.WriteAllTextAsync(trsxPath, trsx);
            Console.WriteLine($"Wrote:\n  {planPath}\n  {trsxPath}");
            break;
        }
    case "iterate":
        {
            Head("ITERATE");
            var vs = new VersionStore(cfg);
            var iterDir = vs.CreateIterationFolder();
            var rec = new IterationRecord { Id = Path.GetFileName(iterDir) };

            var llm = new OllamaClient(cfg.Ollama.Endpoint, cfg.Ollama.Model, cfg.Ollama.Temperature, cfg.Ollama.NumCtx);
            var planner = new PlannerService(llm, cfg.ProjectRoot);
            var (plan, trsx) = await planner.MakePlanAsync();
            rec.PlanPath = Path.Combine(iterDir, "PLAN.md");
            rec.TrsxPath = Path.Combine(iterDir, "next_steps.trsx");
            await Io.WriteAllTextAsync(rec.PlanPath, plan);
            await Io.WriteAllTextAsync(rec.TrsxPath, trsx);

            foreach (var hook in cfg.PostPlanHooks)
            {
                Console.WriteLine($"> PostPlanHook: {hook}");
                var (c, o, e) = await Proc.RunAsync(hook);
                if (c != 0) Console.WriteLine($"Hook failed: {e}");
            }

            var validator = new ValidatorService(cfg);
            var (ok, log) = await validator.RunAsync();
            rec.LogPath = Path.Combine(iterDir, "validate.log");
            await Io.WriteAllTextAsync(rec.LogPath, log);

            if (ok)
            {
                rec.Success = true;
                await Io.WriteAllTextAsync(Path.Combine(iterDir, "SUCCESS"), "ok");
                foreach (var hook in cfg.PostValidateHooks)
                {
                    Console.WriteLine($"> PostValidateHook: {hook}");
                    var (c, o, e) = await Proc.RunAsync(hook);
                    if (c != 0) Console.WriteLine($"Hook failed: {e}");
                }
            }
            else
            {
                rec.Success = false;
                rec.FailureReason = "Validation failed";
                await Io.WriteAllTextAsync(Path.Combine(iterDir, "FAILED"), "validation");
            }

            rec.CompletedAt = DateTimeOffset.UtcNow;
            var metrics = new MetricsService(cfg);
            await metrics.WriteAsync(rec, new Dictionary<string, object>
            {
                ["model"] = cfg.Ollama.Model,
                ["projectRoot"] = cfg.ProjectRoot
            });

            Console.WriteLine($"Iteration {rec.Id} => Success={rec.Success}");
            break;
        }
    case "validate":
        {
            Head("VALIDATE");
            var validator = new ValidatorService(cfg);
            var (ok, log) = await validator.RunAsync();
            Console.WriteLine(log);
            Environment.ExitCode = ok ? 0 : 1;
            break;
        }
    case "rollback":
        {
            Head("ROLLBACK");
            var vs = new VersionStore(cfg);
            var last = vs.LastSuccessfulVersion();
            if (string.IsNullOrEmpty(last))
            {
                Console.WriteLine("No successful versions found.");
                break;
            }
            var marker = Path.Combine(cfg.VersionDir, "CURRENT");
            await Io.WriteAllTextAsync(marker, last);
            Console.WriteLine($"Rolled back pointer to: {last}");
            break;
        }
    case "report":
        {
            Head("REPORT");
            var metricsPath = Path.Combine(cfg.MetricsDir, "metrics.jsonl");
            if (!File.Exists(metricsPath))
            {
                Console.WriteLine("No metrics found yet.");
                break;
            }
            var lines = await File.ReadAllLinesAsync(metricsPath);
            var total = lines.Length;
            var successes = lines.Count(l => l.Contains("\"success\":true"));
            var failures = total - successes;
            Console.WriteLine($"Total runs: {total}\nSuccesses: {successes}\nFailures: {failures}");
            break;
        }
    default:
        {
            Console.WriteLine(
@"TarsSupervisor CLI

USAGE
  dotnet run -- init --project-root ""<path>""
  dotnet run -- plan
  dotnet run -- iterate
  dotnet run -- validate
  dotnet run -- rollback
  dotnet run -- report
");
            break;
        }
}
