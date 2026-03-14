
module TarsEvolution.CliHelpers

open System

/// Enhanced CLI utilities with better user experience
module EnhancedCli =
    
    let printBanner() =
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗")
        Console.WriteLine("║                    TARS CLI (Evolved)                       ║")
        Console.WriteLine("║              Enhanced Performance & Reliability             ║")
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝")
    
    let printPerformanceInfo (startTime: DateTime) (elapsedMs: int64) =
        Console.WriteLine($"⏱️  Execution time: {elapsedMs}ms")
        Console.WriteLine($"📅 Started: {startTime:HH:mm:ss}")
        Console.WriteLine($"✅ Completed: {DateTime.Now:HH:mm:ss}")
