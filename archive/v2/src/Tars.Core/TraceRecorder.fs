namespace Tars.Core

open System
open System.Collections.Concurrent
open System.Text.Json
open System.Text.Json.Serialization

/// In-memory implementation of ITraceRecorder
type TraceRecorder() =
    let events = ConcurrentQueue<TraceEvent>()
    let mutable startTime = DateTime.UtcNow
    let mutable traceId = Guid.NewGuid()

    interface ITraceRecorder with
        member _.StartTraceAsync() =
            async {
                events.Clear()
                startTime <- DateTime.UtcNow
                traceId <- Guid.NewGuid()
                return traceId
            }

        member _.RecordEventAsync
            (eventType: TraceEventType)
            (input: string)
            (output: string)
            (metadata: Map<string, string>)
            =
            async {
                let evt =
                    { Id = Guid.NewGuid()
                      Timestamp = DateTime.UtcNow
                      Type = eventType
                      AgentId = None // Can be added to metadata if needed
                      Input = input
                      Output = output
                      Metadata = metadata }

                events.Enqueue(evt)
            }

        member _.EndTraceAsync() =
            async {
                // Could persist here
                ()
            }

        member _.GetTraceAsync() =
            async {
                let sortedEvents =
                    events.ToArray() |> Array.sortBy (fun e -> e.Timestamp) |> Array.toList

                return
                    Some
                        { Id = traceId
                          StartTime = startTime
                          EndTime = Some DateTime.UtcNow
                          Events = sortedEvents
                          Tags = Map.empty }
            }

    member this.SaveToFileAsync(path: string) =
        async {
            let! traceOpt = (this :> ITraceRecorder).GetTraceAsync()

            match traceOpt with
            | Some trace ->
                let options = JsonSerializerOptions(WriteIndented = true)
                options.Converters.Add(JsonStringEnumConverter())
                let json = JsonSerializer.Serialize(trace, options)

                do! System.IO.File.WriteAllTextAsync(path, json) |> Async.AwaitTask
            | None -> ()
        }
