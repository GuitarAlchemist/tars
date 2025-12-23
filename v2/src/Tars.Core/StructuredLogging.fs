/// Structured Logging for Production
/// Provides JSON-formatted logs with correlation tracking
namespace Tars.Core

open System
open System.Text.Json
open System.Collections.Generic

/// Log severity levels
type LogLevel =
    | Trace = 0
    | Debug = 1
    | Info = 2
    | Warn = 3
    | Error = 4
    | Critical = 5

/// Structured log entry
type LogEntry =
    { Timestamp: DateTime
      Level: LogLevel
      Message: string
      Category: string
      CorrelationId: Guid option
      TraceId: string option
      SpanId: string option
      Properties: Map<string, obj>
      Exception: exn option }

/// Structured logger interface
type IStructuredLogger =
    abstract member Log: LogEntry -> unit
    abstract member Trace: message: string * ?properties: (string * obj) list -> unit
    abstract member Debug: message: string * ?properties: (string * obj) list -> unit
    abstract member Info: message: string * ?properties: (string * obj) list -> unit
    abstract member Warn: message: string * ?properties: (string * obj) list -> unit
    abstract member Error: message: string * ?ex: exn * ?properties: (string * obj) list -> unit
    abstract member Critical: message: string * ?ex: exn * ?properties: (string * obj) list -> unit
    abstract member WithCorrelation: correlationId: Guid -> IStructuredLogger
    abstract member WithCategory: category: string -> IStructuredLogger
    abstract member WithProperty: key: string * value: obj -> IStructuredLogger

/// Logging configuration
type LoggingConfig =
    { MinLevel: LogLevel
      OutputJson: bool
      IncludeTimestamp: bool
      IncludeCategory: bool
      IncludeCorrelation: bool }

    static member Default =
        { MinLevel = LogLevel.Info
          OutputJson = true
          IncludeTimestamp = true
          IncludeCategory = true
          IncludeCorrelation = true }

    static member Development =
        { MinLevel = LogLevel.Debug
          OutputJson = false
          IncludeTimestamp = true
          IncludeCategory = true
          IncludeCorrelation = true }

/// Console structured logger implementation
type StructuredLogger(config: LoggingConfig, ?category: string, ?correlationId: Guid, ?properties: Map<string, obj>) =

    let mutable currentCategory = category |> Option.defaultValue "Tars"
    let mutable currentCorrelation = correlationId
    let mutable extraProperties = properties |> Option.defaultValue Map.empty

    let levelToString level =
        match level with
        | LogLevel.Trace -> "TRC"
        | LogLevel.Debug -> "DBG"
        | LogLevel.Info -> "INF"
        | LogLevel.Warn -> "WRN"
        | LogLevel.Error -> "ERR"
        | LogLevel.Critical -> "CRT"
        | _ -> "???"

    let levelToColor level =
        match level with
        | LogLevel.Trace -> ConsoleColor.DarkGray
        | LogLevel.Debug -> ConsoleColor.Gray
        | LogLevel.Info -> ConsoleColor.White
        | LogLevel.Warn -> ConsoleColor.Yellow
        | LogLevel.Error -> ConsoleColor.Red
        | LogLevel.Critical -> ConsoleColor.Magenta
        | _ -> ConsoleColor.White

    let formatJson (entry: LogEntry) =
        let dict = Dictionary<string, obj>()

        if config.IncludeTimestamp then
            dict.["timestamp"] <- entry.Timestamp.ToString("o")

        dict.["level"] <- levelToString entry.Level
        dict.["message"] <- entry.Message

        if config.IncludeCategory then
            dict.["category"] <- entry.Category

        if config.IncludeCorrelation then
            entry.CorrelationId
            |> Option.iter (fun c -> dict.["correlationId"] <- c.ToString())

            entry.TraceId |> Option.iter (fun t -> dict.["traceId"] <- t)
            entry.SpanId |> Option.iter (fun s -> dict.["spanId"] <- s)

        // Add custom properties
        for kvp in entry.Properties do
            dict.[kvp.Key] <- kvp.Value

        // Add exception if present
        entry.Exception
        |> Option.iter (fun ex ->
            dict.["exception"] <- ex.GetType().Name
            dict.["exceptionMessage"] <- ex.Message
            dict.["stackTrace"] <- ex.StackTrace)

        JsonSerializer.Serialize(dict)

    let formatConsole (entry: LogEntry) =
        let timestamp =
            if config.IncludeTimestamp then
                sprintf "[%s] " (entry.Timestamp.ToString("HH:mm:ss"))
            else
                ""

        let category =
            if config.IncludeCategory then
                sprintf "[%s] " entry.Category
            else
                ""

        let correlation =
            match entry.CorrelationId, config.IncludeCorrelation with
            | Some c, true -> sprintf "(corr:%s) " (c.ToString().Substring(0, 8))
            | _ -> ""

        let level = sprintf "[%s]" (levelToString entry.Level)

        sprintf "%s%s%s%s %s" timestamp level category correlation entry.Message

    let writeLog (entry: LogEntry) =
        if entry.Level >= config.MinLevel then
            let output =
                if config.OutputJson then
                    formatJson entry
                else
                    formatConsole entry

            let originalColor = Console.ForegroundColor
            Console.ForegroundColor <- levelToColor entry.Level
            Console.WriteLine(output)
            Console.ForegroundColor <- originalColor

            // Also write exception details for non-JSON format
            if not config.OutputJson then
                entry.Exception
                |> Option.iter (fun ex ->
                    Console.ForegroundColor <- ConsoleColor.DarkRed
                    Console.WriteLine($"  Exception: {ex.GetType().Name}: {ex.Message}")
                    Console.ForegroundColor <- originalColor)

    let createEntry level message props ex =
        { Timestamp = DateTime.UtcNow
          Level = level
          Message = message
          Category = currentCategory
          CorrelationId = currentCorrelation
          TraceId = None
          SpanId = None
          Properties =
            props
            |> Option.defaultValue []
            |> List.fold (fun m (k, v) -> Map.add k v m) extraProperties
          Exception = ex }

    interface IStructuredLogger with
        member _.Log(entry) = writeLog entry

        member _.Trace(message, ?properties) =
            createEntry LogLevel.Trace message properties None |> writeLog

        member _.Debug(message, ?properties) =
            createEntry LogLevel.Debug message properties None |> writeLog

        member _.Info(message, ?properties) =
            createEntry LogLevel.Info message properties None |> writeLog

        member _.Warn(message, ?properties) =
            createEntry LogLevel.Warn message properties None |> writeLog

        member _.Error(message, ?ex, ?properties) =
            createEntry LogLevel.Error message properties ex |> writeLog

        member _.Critical(message, ?ex, ?properties) =
            createEntry LogLevel.Critical message properties ex |> writeLog

        member _.WithCorrelation(correlationId) =
            StructuredLogger(config, currentCategory, correlationId, extraProperties) :> IStructuredLogger

        member _.WithCategory(cat) =
            StructuredLogger(config, cat, ?correlationId = currentCorrelation, ?properties = Some extraProperties)
            :> IStructuredLogger

        member _.WithProperty(key, value) =
            let newProps = Map.add key value extraProperties

            StructuredLogger(config, currentCategory, ?correlationId = currentCorrelation, ?properties = Some newProps)
            :> IStructuredLogger

/// Logging module for easy access
module Logging =

    let mutable private defaultLogger: IStructuredLogger option = None

    /// Initialize the default logger
    let init config =
        defaultLogger <- Some(StructuredLogger(config) :> IStructuredLogger)

    /// Get the current logger
    let logger () =
        match defaultLogger with
        | Some l -> l
        | None ->
            init LoggingConfig.Default
            defaultLogger.Value

    /// Quick log functions
    let trace msg = (logger ()).Trace(msg)
    let debug msg = (logger ()).Debug(msg)
    let info msg = (logger ()).Info(msg)
    let warn msg = (logger ()).Warn(msg)
    let error msg ex = (logger ()).Error(msg, ex)
    let critical msg ex = (logger ()).Critical(msg, ex)

    /// Create a scoped logger with correlation
    let withCorrelation (correlationId: Guid) =
        (logger ()).WithCorrelation(correlationId)

    /// Create a scoped logger with category
    let withCategory (category: string) = (logger ()).WithCategory(category)
