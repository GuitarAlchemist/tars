namespace Tars.Interface.Cli.Reasoning

open System
open System.Diagnostics
open System.Text.RegularExpressions
open Serilog

/// GPU metrics captured from nvidia-smi
type GpuMetrics = 
    { VramUsedMiB: int option
      VramTotalMiB: int option
      GpuUtilPercent: int option
      MemoryUtilPercent: int option
      PowerDrawW: float option
      Timestamp: DateTime }
    
    static member Empty = 
        { VramUsedMiB = None
          VramTotalMiB = None
          GpuUtilPercent = None
          MemoryUtilPercent = None
          PowerDrawW = None
          Timestamp = DateTime.UtcNow }

/// Metrics for a single LLM call
type LlmCallMetrics =
    { Before: GpuMetrics
      After: GpuMetrics
      DurationMs: int64
      VramDeltaMiB: int option
      ModelFullyOnGpu: bool option }

module GpuMonitor =
    
    let private parseNvidiaSmiOutput (output: string) : GpuMetrics =
        try
            // Expected format: "3627, 16303, 0, 22, 33.08"
            let parts = output.Split(',') |> Array.map (fun s -> s.Trim())
            if parts.Length >= 5 then
                { VramUsedMiB = Int32.TryParse(parts.[0]) |> function | true, v -> Some v | _ -> None
                  VramTotalMiB = Int32.TryParse(parts.[1]) |> function | true, v -> Some v | _ -> None
                  GpuUtilPercent = Int32.TryParse(parts.[2]) |> function | true, v -> Some v | _ -> None
                  MemoryUtilPercent = Int32.TryParse(parts.[3]) |> function | true, v -> Some v | _ -> None
                  PowerDrawW = Double.TryParse(parts.[4]) |> function | true, v -> Some v | _ -> None
                  Timestamp = DateTime.UtcNow }
            else
                GpuMetrics.Empty
        with _ ->
            GpuMetrics.Empty

    /// Query nvidia-smi for current GPU stats
    let queryGpuMetrics (logger: ILogger) : GpuMetrics =
        try
            let psi = ProcessStartInfo(
                FileName = "nvidia-smi",
                Arguments = "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            )
            use proc = Process.Start(psi)
            let output = proc.StandardOutput.ReadToEnd()
            proc.WaitForExit(3000) |> ignore
            
            if proc.ExitCode = 0 then
                parseNvidiaSmiOutput output
            else
                logger.Warning("nvidia-smi returned non-zero exit code: {ExitCode}", proc.ExitCode)
                GpuMetrics.Empty
        with ex ->
            logger.Debug("nvidia-smi not available: {Message}", ex.Message)
            GpuMetrics.Empty

    /// Query Ollama for model processor info (CPU vs GPU)
    let queryOllamaProcessor (logger: ILogger) : (string * int) option =
        try
            let psi = ProcessStartInfo(
                FileName = "ollama",
                Arguments = "ps",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            )
            use proc = Process.Start(psi)
            let output = proc.StandardOutput.ReadToEnd()
            proc.WaitForExit(3000) |> ignore
            
            if proc.ExitCode = 0 then
                // Parse: "100% GPU" or "50% GPU/50% CPU"
                let lines = output.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
                if lines.Length > 1 then
                    let dataLine = lines.[1]
                    // Look for GPU percentage
                    let gpuMatch = Regex.Match(dataLine, @"(\d+)%\s*GPU")
                    if gpuMatch.Success then
                        let pct = Int32.Parse(gpuMatch.Groups.[1].Value)
                        Some ("GPU", pct)
                    else
                        Some ("CPU", 0)
                else
                    None
            else
                None
        with ex ->
            logger.Debug("ollama ps failed: {Message}", ex.Message)
            None

    /// Capture metrics around an async operation
    let withMetrics (logger: ILogger) (operation: Async<'T>) : Async<'T * LlmCallMetrics> =
        async {
            let before = queryGpuMetrics logger
            let sw = Stopwatch.StartNew()
            
            let! result = operation
            
            sw.Stop()
            let after = queryGpuMetrics logger
            
            let ollamaInfo = queryOllamaProcessor logger
            
            let metrics = 
                { Before = before
                  After = after
                  DurationMs = sw.ElapsedMilliseconds
                  VramDeltaMiB = 
                      match before.VramUsedMiB, after.VramUsedMiB with
                      | Some b, Some a -> Some (a - b)
                      | _ -> None
                  ModelFullyOnGpu = 
                      ollamaInfo |> Option.map (fun (_, pct) -> pct >= 100) }
            
            return (result, metrics)
        }
