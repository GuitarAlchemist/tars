namespace TarsEngine.FSharp.Cli.Core

open System
open System.IO
open System.Text.Json
open System.Collections.Concurrent
open System.Threading
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core.UnifiedCore

/// TARS Unified Logger - Centralized logging with structured data and correlation tracking
module UnifiedLogger =
    
    /// Log entry structure
    type TarsLogEntry = {
        EntryId: string
        Timestamp: DateTime
        Level: LogLevel
        CorrelationId: string
        Message: string
        Parameters: obj[] option
        Exception: Exception option
        Source: string
        Category: string
        Properties: Map<string, obj>
    }

    /// Log output destination
    type LogDestination =
        | Console
        | File of path: string
        | Database of connectionString: string
        | Remote of endpoint: string

    /// Log configuration
    type LogConfiguration = {
        MinimumLevel: LogLevel
        Destinations: LogDestination list
        EnableStructuredLogging: bool
        EnableCorrelationTracking: bool
        MaxLogFileSize: int64
        MaxLogFiles: int
        LogRetentionDays: int
        BufferSize: int
        FlushIntervalMs: int
    }

    /// Thread-safe unified logger implementation
    type TarsUnifiedLogger(config: LogConfiguration, componentName: string) =
        let logBuffer = ConcurrentQueue<TarsLogEntry>()
        let mutable correlationContext = ""
        let mutable isDisposed = false
        
        /// Generate unique entry ID
        let generateEntryId() =
            let timestamp = DateTime.Now.ToString("yyyyMMddHHmmss")
            let guid = Guid.NewGuid().ToString("N").Substring(0, 8)
            $"log-{timestamp}-{guid}"
        
        /// Check if log level is enabled
        let isLevelEnabled(level: LogLevel) = level >= config.MinimumLevel
        
        /// Create log entry
        let createLogEntry (level: LogLevel) (correlationId: string) (message: string) (parameters: obj[] option) (ex: Exception option) =
            {
                EntryId = generateEntryId()
                Timestamp = DateTime.Now
                Level = level
                CorrelationId = correlationId
                Message = message
                Parameters = parameters
                Exception = ex
                Source = componentName
                Category = "TARS"
                Properties = Map.empty
            }
        
        /// Format log message with parameters
        let formatMessage (message: string) (parameters: obj[] option) =
            match parameters with
            | Some pars when pars.Length > 0 ->
                try
                    String.Format(message, pars)
                with
                | _ ->
                    let paramsStr = String.Join(", ", pars)
                    $"{message} [Params: {paramsStr}]"
            | _ -> message
        
        /// Write to console with color coding
        let writeToConsole(entry: TarsLogEntry) =
            let color = 
                match entry.Level with
                | LogLevel.Trace -> ConsoleColor.Gray
                | LogLevel.Debug -> ConsoleColor.DarkGray
                | LogLevel.Information -> ConsoleColor.White
                | LogLevel.Warning -> ConsoleColor.Yellow
                | LogLevel.Error -> ConsoleColor.Red
                | LogLevel.Critical -> ConsoleColor.Magenta
                | _ -> ConsoleColor.White
            
            let originalColor = Console.ForegroundColor
            Console.ForegroundColor <- color
            
            let formattedMessage = formatMessage entry.Message entry.Parameters
            let timestamp = entry.Timestamp.ToString("yyyy-MM-dd HH:mm:ss.fff")
            let logLine = $"[{timestamp}] [{entry.Level}] [{entry.CorrelationId}] {formattedMessage}"
            
            Console.WriteLine(logLine)
            
            if entry.Exception.IsSome then
                Console.WriteLine($"Exception: {entry.Exception.Value}")
            
            Console.ForegroundColor <- originalColor
        
        /// Write to file with rotation
        let writeToFile(entry: TarsLogEntry, filePath: string) =
            let directory = Path.GetDirectoryName(filePath)
            if not (Directory.Exists(directory)) then
                Directory.CreateDirectory(directory) |> ignore
            
            let formattedMessage = formatMessage entry.Message entry.Parameters
            let logLine =
                if config.EnableStructuredLogging then
                    JsonSerializer.Serialize(entry)
                else
                    let timestamp = entry.Timestamp.ToString("yyyy-MM-dd HH:mm:ss.fff")
                    $"[{timestamp}] [{entry.Level}] [{entry.CorrelationId}] {formattedMessage}"
            
            File.AppendAllText(filePath, logLine + Environment.NewLine)
        
        /// Write log entry to destinations
        let writeToDestinations (entry: TarsLogEntry) =
            for destination in config.Destinations do
                try
                    match destination with
                    | Console -> writeToConsole(entry)
                    | File path -> writeToFile(entry, path)
                    | Database _ -> () // TODO: Implement database logging
                    | Remote _ -> () // TODO: Implement remote logging
                with
                | ex -> 
                    // Fallback to console if destination fails
                    Console.WriteLine($"[ERROR] Failed to write to log destination: {ex.Message}")
        
        /// Process log buffer
        member private this.ProcessLogBuffer() =
            while not logBuffer.IsEmpty do
                match logBuffer.TryDequeue() with
                | true, entry -> writeToDestinations entry
                | false, _ -> ()
        
        /// Set correlation context
        member this.SetCorrelationId(correlationId: string) =
            correlationContext <- correlationId
        
        /// Get current correlation ID
        member this.GetCorrelationId() =
            if String.IsNullOrEmpty(correlationContext) then
                generateCorrelationId()
            else
                correlationContext

        interface ITarsLogger with
            member this.LogTrace(correlationId: string, message: string, ?parameters: obj[]) =
                if isLevelEnabled LogLevel.Trace then
                    let entry = createLogEntry LogLevel.Trace correlationId message parameters None
                    logBuffer.Enqueue(entry)
                    this.ProcessLogBuffer()

            member this.LogDebug(correlationId: string, message: string, ?parameters: obj[]) =
                if isLevelEnabled LogLevel.Debug then
                    let entry = createLogEntry LogLevel.Debug correlationId message parameters None
                    logBuffer.Enqueue(entry)
                    this.ProcessLogBuffer()

            member this.LogInformation(correlationId: string, message: string, ?parameters: obj[]) =
                if isLevelEnabled LogLevel.Information then
                    let entry = createLogEntry LogLevel.Information correlationId message parameters None
                    logBuffer.Enqueue(entry)
                    this.ProcessLogBuffer()

            member this.LogWarning(correlationId: string, message: string, ?parameters: obj[]) =
                if isLevelEnabled LogLevel.Warning then
                    let entry = createLogEntry LogLevel.Warning correlationId message parameters None
                    logBuffer.Enqueue(entry)
                    this.ProcessLogBuffer()

            member this.LogError(correlationId: string, error: TarsError, ?ex: Exception) =
                if isLevelEnabled LogLevel.Error then
                    let message = TarsError.toString error
                    let entry = createLogEntry LogLevel.Error correlationId message None ex
                    logBuffer.Enqueue(entry)
                    this.ProcessLogBuffer()

            member this.LogCritical(correlationId: string, message: string, ?ex: Exception) =
                if isLevelEnabled LogLevel.Critical then
                    let entry = createLogEntry LogLevel.Critical correlationId message None ex
                    logBuffer.Enqueue(entry)
                    this.ProcessLogBuffer()

        interface IDisposable with
            member this.Dispose() =
                if not isDisposed then
                    this.ProcessLogBuffer() // Flush remaining logs
                    isDisposed <- true

    /// Default log configuration
    let defaultLogConfiguration = {
        MinimumLevel = LogLevel.Information
        Destinations = [Console; File (Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "logs", "tars.log"))]
        EnableStructuredLogging = true
        EnableCorrelationTracking = true
        MaxLogFileSize = 10L * 1024L * 1024L // 10MB
        MaxLogFiles = 5
        LogRetentionDays = 30
        BufferSize = 1000
        FlushIntervalMs = 5000
    }

    /// Create unified logger with default configuration
    let createLogger (componentName: string) =
        new TarsUnifiedLogger(defaultLogConfiguration, componentName)

    /// Logger factory for creating component-specific loggers
    type TarsLoggerFactory(config: LogConfiguration) =
        let loggers = ConcurrentDictionary<string, TarsUnifiedLogger>()
        
        member this.CreateLogger(componentName: string) : ITarsLogger =
            loggers.GetOrAdd(componentName, fun name -> new TarsUnifiedLogger(config, name)) :> ITarsLogger
        
        interface IDisposable with
            member this.Dispose() =
                for logger in loggers.Values do
                    (logger :> IDisposable).Dispose()
                loggers.Clear()

    /// Create logger factory with default configuration
    let createLoggerFactory() = new TarsLoggerFactory(defaultLogConfiguration)

    /// Create logger factory with custom configuration
    let createLoggerFactoryWithConfig(config: LogConfiguration) = new TarsLoggerFactory(config)
