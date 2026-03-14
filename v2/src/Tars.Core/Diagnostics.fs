namespace Tars.Core

open System

/// <summary>
/// Structured diagnostics for a reasoning run.
/// Captures the "vitals" of the agent's cognition.
/// </summary>
type RunDiagnostics =
    { 
      /// <summary>Correlation ID for the run.</summary>
      TraceId: string
      
      /// <summary>The goal of the run.</summary>
      Goal: string
      
      /// <summary>When the run started.</summary>
      StartTime: DateTimeOffset
      
      /// <summary>When the run ended.</summary>
      EndTime: DateTimeOffset
      
      /// <summary>Total duration of the run.</summary>
      Duration: TimeSpan
      
      /// <summary>Outcome label (Success, Failure, Partial).</summary>
      Outcome: string
      
      /// <summary>Number of reasoning steps taken.</summary>
      StepCount: int
      
      /// <summary>Full execution trace/log of the run.</summary>
      ExecutionTrace: string list
      
      /// <summary>Key metrics (Entropy, Efficiency, numeric scores).</summary>
      Metrics: Map<string, float>
      
      /// <summary>Cost incurred (in Tokens/USD).</summary>
      Cost: Cost
    }

/// <summary>
/// Service for managing diagnostics.
/// </summary>
module Diagnostics =
    open System.IO
    open System.Text.Json

    let save (path: string) (diagnostics: RunDiagnostics) =
        let options = JsonSerializerOptions(WriteIndented = true)
        let json = JsonSerializer.Serialize(diagnostics, options)
        File.WriteAllText(path, json)
