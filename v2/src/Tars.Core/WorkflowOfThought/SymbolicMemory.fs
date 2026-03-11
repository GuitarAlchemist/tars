namespace Tars.Core.WorkflowOfThought

open System
open System.IO
open System.Text.Json

/// <summary>
/// A single entry in the symbolic memory log (Phase 15.2).
/// Items are logged as NDJSON to enable efficient append and stream processing.
/// </summary>
type MemoryEntry =
    { Id: Guid
      Timestamp: DateTime
      RunId: Guid
      StepId: string option
      Type: string // "fact" | "failure" | "strategy" | "observation"
      Content: string
      Confidence: float
      Metadata: Map<string, string> }

module SymbolicMemory =

    let private options = JsonSerializerOptions(WriteIndented = false)

    /// <summary>
    /// Gets the path to the memory directory.
    /// </summary>
    let getMemoryDir () =
        // Path matches roadmap: .tars/memory/
        let baseDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ".tars", "memory")

        if not (Directory.Exists(baseDir)) then
            Directory.CreateDirectory(baseDir) |> ignore

        baseDir

    /// <summary>
    /// Append an entry to a memory log.
    /// </summary>
    let append (logName: string) (entry: MemoryEntry) : Async<unit> =
        async {
            let path = Path.Combine(getMemoryDir (), logName + ".ndjson")
            let json = JsonSerializer.Serialize(entry, options)
            do! File.AppendAllLinesAsync(path, [ json ]) |> Async.AwaitTask
        }

    /// <summary>
    /// Read all entries from a memory log.
    /// </summary>
    let readAll (logName: string) : Async<MemoryEntry list> =
        async {
            let path = Path.Combine(getMemoryDir (), logName + ".ndjson")

            if not (File.Exists(path)) then
                return []
            else
                let! lines = File.ReadAllLinesAsync(path) |> Async.AwaitTask

                return
                    lines
                    |> Seq.choose (fun line ->
                        if String.IsNullOrWhiteSpace line then
                            None
                        else
                            try
                                Some(JsonSerializer.Deserialize<MemoryEntry>(line, options))
                            with _ ->
                                None)
                    |> Seq.toList
        }

    /// <summary>
    /// Log a fact to the symbolic memory.
    /// </summary>
    let logFact (runId: Guid) (stepId: string option) (fact: string) (confidence: float) =
        let entry =
            { Id = Guid.NewGuid()
              Timestamp = DateTime.UtcNow
              RunId = runId
              StepId = stepId
              Type = "fact"
              Content = fact
              Confidence = confidence
              Metadata = Map.empty }

        append "facts" entry

    /// <summary>
    /// Log a failure to the symbolic memory.
    /// </summary>
    let logFailure (runId: Guid) (stepId: string option) (error: string) (metadata: Map<string, string>) =
        let entry =
            { Id = Guid.NewGuid()
              Timestamp = DateTime.UtcNow
              RunId = runId
              StepId = stepId
              Type = "failure"
              Content = error
              Confidence = 1.0
              Metadata = metadata }

        append "failures" entry

    /// <summary>
    /// Log a strategy/insight to the symbolic memory.
    /// </summary>
    let logStrategy (runId: Guid) (insight: string) (metadata: Map<string, string>) =
        let entry =
            { Id = Guid.NewGuid()
              Timestamp = DateTime.UtcNow
              RunId = runId
              StepId = None
              Type = "strategy"
              Content = insight
              Confidence = 0.9
              Metadata = metadata }

        append "strategies" entry
