namespace Tars.Core

open System

/// Represents the type of a recorded event
type TraceEventType =
    | LlmCall = 0
    | ToolExecution = 1
    | AgentStateChange = 2
    | MessageReceived = 3

/// A single recorded event in the trace
type TraceEvent =
    { Id: Guid
      Timestamp: DateTime
      Type: TraceEventType
      AgentId: Guid option
      Input: string
      Output: string
      Metadata: Map<string, string> }

/// A complete execution trace
type Trace =
    { Id: Guid
      StartTime: DateTime
      EndTime: DateTime option
      Events: TraceEvent list
      Tags: Map<string, string> }

/// Interface for recording traces
type ITraceRecorder =
    abstract member RecordEventAsync: TraceEventType -> string -> string -> Map<string, string> -> Async<unit>
    abstract member StartTraceAsync: unit -> Async<Guid>
    abstract member EndTraceAsync: unit -> Async<unit>
    abstract member GetTraceAsync: unit -> Async<Trace option>

/// Interface for replaying traces (Mocking)
type ITraceReplayer =
    abstract member GetNextEventAsync: TraceEventType -> Async<TraceEvent option>
    abstract member Reset: unit -> unit
