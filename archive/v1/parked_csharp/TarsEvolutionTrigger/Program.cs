using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;
using Newtonsoft.Json;

/// <summary>
/// TARS Evolution Trigger - Runs a real evolution session
/// </summary>
class Program
{
    static async Task<int> Main(string[] args)
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║              🧬 TARS EVOLUTION TRIGGER                      ║");
        Console.WriteLine("║                Real Auto-Improvement Session                ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        try
        {
            // Create logger
            using var loggerFactory = LoggerFactory.Create(builder => 
                builder.AddConsole().SetMinimumLevel(LogLevel.Information));
            var logger = loggerFactory.CreateLogger<Program>();

            logger.LogInformation("🚀 Starting TARS Real Evolution Session...");

            // Determine start path
            string startPath = args.Length > 0 ? args[0] : "/app";
            Console.WriteLine($"🔍 Evolution start path: {startPath}");

            // Create evolution engine
            var evolutionEngine = new TarsEngine.FSharp.Core.TarsEvolutionEngine.TarsEvolutionEngineService(
                loggerFactory.CreateLogger<TarsEngine.FSharp.Core.TarsEvolutionEngine.TarsEvolutionEngineService>());

            Console.WriteLine("\n🧬 Running Full Evolution Session:");
            Console.WriteLine("===================================");

            // Run the evolution session with default config
            var evolutionResult = await evolutionEngine.RunEvolutionSession(startPath, null);

            // Display results
            Console.WriteLine("\n📊 EVOLUTION SESSION RESULTS:");
            Console.WriteLine("==============================");
            Console.WriteLine($"Session ID: {evolutionResult.SessionId}");
            Console.WriteLine($"Duration: {evolutionResult.TotalDurationMs}ms");
            Console.WriteLine($"Overall Success: {(evolutionResult.OverallSuccess ? "✅ YES" : "❌ NO")}");
            Console.WriteLine($"Projects Analyzed: {evolutionResult.ProjectsAnalyzed}");
            Console.WriteLine($"Improvements Applied: {evolutionResult.ImprovementsApplied}");
            
            if (evolutionResult.PerformanceGain.HasValue)
            {
                Console.WriteLine($"Performance Gain: {evolutionResult.PerformanceGain.Value:F2}%");
            }

            Console.WriteLine("\n🔄 Evolution Steps:");
            foreach (var step in evolutionResult.Steps)
            {
                string status = step.Success ? "✅" : "❌";
                Console.WriteLine($"  {status} {step.StepName} ({step.ExecutionTimeMs}ms)");
                if (!step.Success && step.ErrorMessage != null)
                {
                    Console.WriteLine($"      Error: {step.ErrorMessage}");
                }
            }

            Console.WriteLine("\n🎯 Recommended Next Steps:");
            foreach (var recommendation in evolutionResult.RecommendedNextSteps)
            {
                Console.WriteLine($"  • {recommendation}");
            }

            // Save results to file for monitoring
            var resultsJson = JsonConvert.SerializeObject(evolutionResult, Formatting.Indented);
            var resultsPath = $"/app/data/evolution/evolution-result-{evolutionResult.SessionId}.json";
            
            try
            {
                var directory = System.IO.Path.GetDirectoryName(resultsPath);
                if (!string.IsNullOrEmpty(directory))
                {
                    System.IO.Directory.CreateDirectory(directory);
                }
                await System.IO.File.WriteAllTextAsync(resultsPath, resultsJson);
                Console.WriteLine($"\n💾 Results saved to: {resultsPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ Could not save results: {ex.Message}");
            }

            Console.WriteLine("\n🎉 TARS Evolution Session Completed!");
            
            return evolutionResult.OverallSuccess ? 0 : 1;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n❌ TARS Evolution Failed: {ex.Message}");
            Console.WriteLine($"Stack Trace: {ex.StackTrace}");
            return 1;
        }
    }
}
