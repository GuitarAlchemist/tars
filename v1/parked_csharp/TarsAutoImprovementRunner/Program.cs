using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;

/// <summary>
/// TARS Auto-Improvement Runner - Docker Entry Point
/// Runs the TARS auto-improvement system in a containerized environment
/// </summary>
class Program
{
    static async Task<int> Main(string[] args)
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║              🚀 TARS AUTO-IMPROVEMENT RUNNER                ║");
        Console.WriteLine("║                  Fresh Baseline Deployment                  ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        try
        {
            // Create logger
            using var loggerFactory = LoggerFactory.Create(builder => 
                builder.AddConsole().SetMinimumLevel(LogLevel.Information));
            var logger = loggerFactory.CreateLogger<Program>();

            logger.LogInformation("🔍 Starting TARS Auto-Improvement System...");

            // Test the F# auto-improvement modules
            Console.WriteLine("🧬 Testing Auto-Improvement Modules:");
            Console.WriteLine("=====================================");
            
            // Call the F# test function
            TarsEngine.FSharp.Core.TestTarsEvolution.runTest();
            
            Console.WriteLine("\n✅ TARS Auto-Improvement System Running Successfully!");
            Console.WriteLine("🌐 System is ready for Blue-Green evolution testing");
            
            // Keep the container running
            Console.WriteLine("\n⏳ Keeping container alive for monitoring...");
            Console.WriteLine("Press Ctrl+C to stop");
            
            // Simple HTTP server simulation for health checks
            await RunHealthServer();
            
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n❌ TARS Auto-Improvement Runner Failed: {ex.Message}");
            Console.WriteLine($"Stack Trace: {ex.StackTrace}");
            return 1;
        }
    }

    static async Task RunHealthServer()
    {
        // Simple health check loop
        while (true)
        {
            await Task.Delay(30000); // Wait 30 seconds
            Console.WriteLine($"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] 💓 Health check - System running");
        }
    }
}
