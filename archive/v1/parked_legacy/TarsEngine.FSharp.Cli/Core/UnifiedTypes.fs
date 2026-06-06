namespace TarsEngine.FSharp.Cli.Core

open System
open System.Text.Json
open TarsEngine.FSharp.Cli.Core.UnifiedCore

/// TARS Unified Types - Common data structures and interfaces for cross-module communication
module UnifiedTypes =
    
    /// Agent state enumeration
    type AgentState =
        | Initializing
        | Ready
        | Busy
        | Error of message: string
        | Shutdown

    /// Agent status information
    type AgentStatus = {
        AgentId: string
        State: AgentState
        LastActivity: DateTime
        ProcessedMessages: int64
        ErrorCount: int64
        AverageResponseTime: TimeSpan
        CurrentLoad: float
    }

    /// Unified agent interface for all TARS agents
    type ITarsAgent =
        inherit ITarsComponent
        abstract member AgentId: string
        abstract member AgentType: string
        abstract member Capabilities: string list
        abstract member ProcessMessage: context: TarsOperationContext * message: obj -> Async<TarsResult<obj>>
        abstract member GetStatus: unit -> TarsResult<AgentStatus>

    /// Unified data source interface
    type ITarsDataSource =
        abstract member SourceId: string
        abstract member SourceType: string
        abstract member IsAvailable: unit -> Async<bool>
        abstract member FetchData: context: TarsOperationContext * query: obj -> Async<TarsResult<obj>>
        abstract member GetSchema: unit -> TarsResult<Map<string, obj>>

    /// Computation complexity levels
    type ComputationComplexity =
        | Trivial
        | Low
        | Medium
        | High
        | Extreme

    /// Component status enumeration
    type ComponentStatus =
        | Stopped
        | Starting
        | Running
        | Stopping
        | Error
        | Maintenance

    /// Unified computation interface
    type ITarsComputation<'TInput, 'TOutput> =
        inherit ITarsOperation<'TInput, 'TOutput>
        abstract member RequiresCuda: bool
        abstract member EstimatedComplexity: 'TInput -> ComputationComplexity
        abstract member CanBatch: bool

    /// Message priority levels
    type MessagePriority =
        | Low = 1
        | Normal = 2
        | High = 3
        | Critical = 4
        | Emergency = 5

    /// Unified message type for inter-component communication
    type TarsMessage = {
        MessageId: string
        MessageType: string
        Sender: string
        Recipient: string option
        Payload: obj
        Priority: MessagePriority
        CreatedAt: DateTime
        ExpiresAt: DateTime option
        CorrelationId: string
        ReplyTo: string option
    }

    /// Event severity levels
    type EventSeverity =
        | Trace
        | Debug
        | Information
        | Warning
        | Error
        | Critical

    /// Unified event type for system events
    type TarsEvent = {
        EventId: string
        EventType: string
        Source: string
        Data: Map<string, obj>
        Timestamp: DateTime
        CorrelationId: string
        Severity: EventSeverity
    }

    /// Unified proof type for cryptographic verification
    type TarsProof = {
        ProofId: string
        ProofType: string
        Subject: string
        Claims: Map<string, obj>
        Evidence: byte[]
        Signature: byte[]
        Timestamp: DateTime
        ValidUntil: DateTime option
        ChainPosition: int64
        PreviousProofHash: string option
    }

    /// Unified cache entry type
    type TarsCacheEntry<'T> = {
        Key: string
        Value: 'T
        CreatedAt: DateTime
        LastAccessed: DateTime
        ExpiresAt: DateTime option
        AccessCount: int64
        Size: int64
        Tags: string list
    }

    /// Unified query interface
    type ITarsQuery<'TResult> =
        abstract member QueryId: string
        abstract member QueryType: string
        abstract member Parameters: Map<string, obj>
        abstract member Execute: context: TarsOperationContext -> Async<TarsResult<'TResult>>
        abstract member Validate: unit -> TarsResult<unit>
        abstract member EstimateResultSize: unit -> int64

    /// Unified storage interface
    type ITarsStorage<'TKey, 'TValue> =
        abstract member StorageId: string
        abstract member StorageType: string
        abstract member Get: key: 'TKey -> Async<TarsResult<'TValue option>>
        abstract member Set: key: 'TKey * value: 'TValue -> Async<TarsResult<unit>>
        abstract member Delete: key: 'TKey -> Async<TarsResult<bool>>
        abstract member Exists: key: 'TKey -> Async<TarsResult<bool>>
        abstract member GetKeys: unit -> Async<TarsResult<'TKey list>>
        abstract member Clear: unit -> Async<TarsResult<unit>>

    /// Unified serialization interface
    type ITarsSerializer<'T> =
        abstract member Serialize: value: 'T -> TarsResult<byte[]>
        abstract member Deserialize: data: byte[] -> TarsResult<'T>
        abstract member GetContentType: unit -> string

    /// JSON serializer implementation
    type JsonTarsSerializer<'T>() =
        interface ITarsSerializer<'T> with
            member _.Serialize(value: 'T) : TarsResult<byte[]> =
                try
                    let json = JsonSerializer.Serialize(value)
                    let bytes = System.Text.Encoding.UTF8.GetBytes(json)
                    Success (bytes, Map [("format", "json"); ("size", bytes.Length)])
                with
                | ex -> Failure (ExecutionError ("JSON serialization failed", Some ex), generateCorrelationId())

            member _.Deserialize(data: byte[]) : TarsResult<'T> =
                try
                    let json = System.Text.Encoding.UTF8.GetString(data)
                    let value = JsonSerializer.Deserialize<'T>(json)
                    Success (value, Map [("format", "json"); ("size", data.Length)])
                with
                | ex -> Failure (ExecutionError ("JSON deserialization failed", Some ex), generateCorrelationId())

            member _.GetContentType() : string = "application/json"

    /// Unified validation interface
    type ITarsValidator<'T> =
        abstract member Validate: value: 'T -> TarsResult<unit>
        abstract member GetValidationRules: unit -> string list

    /// Validation severity levels
    type ValidationSeverity =
        | Info
        | Warning
        | Error
        | Critical

    /// Validation rule type
    type ValidationRule<'T> = {
        RuleName: string
        Description: string
        Validator: 'T -> TarsResult<unit>
        Severity: ValidationSeverity
    }

    /// Unified metrics interface
    type ITarsMetrics =
        abstract member RecordCounter: name: string * value: int64 * ?tags: Map<string, string> -> unit
        abstract member RecordGauge: name: string * value: float * ?tags: Map<string, string> -> unit
        abstract member RecordHistogram: name: string * value: float * ?tags: Map<string, string> -> unit
        abstract member RecordTimer: name: string * duration: TimeSpan * ?tags: Map<string, string> -> unit
        abstract member GetMetrics: unit -> Map<string, obj>

    /// Unified configuration section interface
    type ITarsConfigSection =
        abstract member SectionName: string
        abstract member GetValue: key: string -> obj option
        abstract member SetValue: key: string * value: obj -> TarsResult<unit>
        abstract member GetAllValues: unit -> Map<string, obj>
        abstract member Validate: unit -> TarsResult<unit>

    /// Helper functions for working with unified types
    module TarsMessage =
        let create messageType sender payload correlationId =
            {
                MessageId = Guid.NewGuid().ToString()
                MessageType = messageType
                Sender = sender
                Recipient = None
                Payload = payload
                Priority = MessagePriority.Normal
                CreatedAt = DateTime.Now
                ExpiresAt = None
                CorrelationId = correlationId
                ReplyTo = None
            }

        let withRecipient recipient message =
            { message with Recipient = Some recipient }

        let withPriority priority message =
            { message with Priority = priority }

        let withExpiration expiration message =
            { message with ExpiresAt = Some expiration }

        let withReplyTo replyTo message =
            { message with ReplyTo = Some replyTo }

        let isExpired message =
            match message.ExpiresAt with
            | Some expiry -> DateTime.Now > expiry
            | None -> false

    module TarsEvent =
        let create eventType source data correlationId =
            {
                EventId = Guid.NewGuid().ToString()
                EventType = eventType
                Source = source
                Data = data
                Timestamp = DateTime.Now
                CorrelationId = correlationId
                Severity = EventSeverity.Information
            }

        let withSeverity severity event =
            { event with Severity = severity }

        let addData key value event =
            { event with Data = Map.add key value event.Data }

    module TarsProof =
        let create proofType subject claims evidence signature =
            {
                ProofId = Guid.NewGuid().ToString()
                ProofType = proofType
                Subject = subject
                Claims = claims
                Evidence = evidence
                Signature = signature
                Timestamp = DateTime.Now
                ValidUntil = None
                ChainPosition = 0L
                PreviousProofHash = None
            }

        let withValidUntil validUntil proof =
            { proof with ValidUntil = Some validUntil }

        let withChainPosition position previousHash proof =
            { proof with ChainPosition = position; PreviousProofHash = previousHash }

        let isValid proof =
            match proof.ValidUntil with
            | Some expiry -> DateTime.Now <= expiry
            | None -> true

    module TarsCacheEntry =
        let create key value =
            {
                Key = key
                Value = value
                CreatedAt = DateTime.Now
                LastAccessed = DateTime.Now
                ExpiresAt = None
                AccessCount = 0L
                Size = 0L
                Tags = []
            }

        let withExpiration expiration entry =
            { entry with ExpiresAt = Some expiration }

        let withTags tags entry =
            { entry with Tags = tags }

        let access entry =
            { entry with LastAccessed = DateTime.Now; AccessCount = entry.AccessCount + 1L }

        let isExpired entry =
            match entry.ExpiresAt with
            | Some expiry -> DateTime.Now > expiry
            | None -> false

    /// Type conversion utilities
    module TypeConversion =
        let tryConvert<'T> (value: obj) : 'T option =
            try
                match value with
                | :? 'T as typed -> Some typed
                | _ -> 
                    let converted = Convert.ChangeType(value, typeof<'T>)
                    Some (unbox<'T> converted)
            with
            | _ -> None

        let convertWithDefault<'T> (defaultValue: 'T) (value: obj) : 'T =
            tryConvert<'T> value |> Option.defaultValue defaultValue

        let convertOrError<'T> (value: obj) (correlationId: string) : TarsResult<'T> =
            match tryConvert<'T> value with
            | Some converted -> Success (converted, Map.empty)
            | None -> Failure (ValidationError ($"Cannot convert {value.GetType().Name} to {typeof<'T>.Name}", Map.empty), correlationId)
