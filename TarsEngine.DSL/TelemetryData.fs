namespace TarsEngine.DSL

open System

/// <summary>
/// Represents telemetry data for parser usage.
/// </summary>
type ParserUsageTelemetry = {
    /// <summary>
    /// The parser type used (Original, FParsec, Incremental).
    /// </summary>
    ParserType: string
    
    /// <summary>
    /// The file size in bytes.
    /// </summary>
    FileSizeBytes: int64
    
    /// <summary>
    /// The number of lines in the file.
    /// </summary>
    LineCount: int
    
    /// <summary>
    /// The number of blocks in the parsed program.
    /// </summary>
    BlockCount: int
    
    /// <summary>
    /// The number of properties in the parsed program.
    /// </summary>
    PropertyCount: int
    
    /// <summary>
    /// The number of nested blocks in the parsed program.
    /// </summary>
    NestedBlockCount: int
    
    /// <summary>
    /// The timestamp when the parsing started.
    /// </summary>
    StartTimestamp: DateTime
    
    /// <summary>
    /// The timestamp when the parsing completed.
    /// </summary>
    EndTimestamp: DateTime
    
    /// <summary>
    /// The total parsing time in milliseconds.
    /// </summary>
    TotalParseTimeMs: int64
}

/// <summary>
/// Represents telemetry data for parsing performance.
/// </summary>
type ParsingPerformanceTelemetry = {
    /// <summary>
    /// The parser type used (Original, FParsec, Incremental).
    /// </summary>
    ParserType: string
    
    /// <summary>
    /// The file size in bytes.
    /// </summary>
    FileSizeBytes: int64
    
    /// <summary>
    /// The number of lines in the file.
    /// </summary>
    LineCount: int
    
    /// <summary>
    /// The total parsing time in milliseconds.
    /// </summary>
    TotalParseTimeMs: int64
    
    /// <summary>
    /// The time spent tokenizing in milliseconds.
    /// </summary>
    TokenizingTimeMs: int64 option
    
    /// <summary>
    /// The time spent parsing blocks in milliseconds.
    /// </summary>
    BlockParsingTimeMs: int64 option
    
    /// <summary>
    /// The time spent parsing properties in milliseconds.
    /// </summary>
    PropertyParsingTimeMs: int64 option
    
    /// <summary>
    /// The time spent parsing nested blocks in milliseconds.
    /// </summary>
    NestedBlockParsingTimeMs: int64 option
    
    /// <summary>
    /// The time spent chunking in milliseconds (for incremental parsing).
    /// </summary>
    ChunkingTimeMs: int64 option
    
    /// <summary>
    /// The time spent parsing chunks in milliseconds (for incremental parsing).
    /// </summary>
    ChunkParsingTimeMs: int64 option
    
    /// <summary>
    /// The time spent combining chunks in milliseconds (for incremental parsing).
    /// </summary>
    ChunkCombiningTimeMs: int64 option
    
    /// <summary>
    /// The number of chunks (for incremental parsing).
    /// </summary>
    ChunkCount: int option
    
    /// <summary>
    /// The number of cached chunks (for incremental parsing).
    /// </summary>
    CachedChunkCount: int option
    
    /// <summary>
    /// The peak memory usage in bytes.
    /// </summary>
    PeakMemoryUsageBytes: int64 option
}

/// <summary>
/// Represents telemetry data for errors and warnings.
/// </summary>
type ErrorWarningTelemetry = {
    /// <summary>
    /// The parser type used (Original, FParsec, Incremental).
    /// </summary>
    ParserType: string
    
    /// <summary>
    /// The file size in bytes.
    /// </summary>
    FileSizeBytes: int64
    
    /// <summary>
    /// The number of lines in the file.
    /// </summary>
    LineCount: int
    
    /// <summary>
    /// The number of errors.
    /// </summary>
    ErrorCount: int
    
    /// <summary>
    /// The number of warnings.
    /// </summary>
    WarningCount: int
    
    /// <summary>
    /// The number of informational messages.
    /// </summary>
    InfoCount: int
    
    /// <summary>
    /// The number of hints.
    /// </summary>
    HintCount: int
    
    /// <summary>
    /// The error codes and their counts.
    /// </summary>
    ErrorCodes: Map<string, int>
    
    /// <summary>
    /// The warning codes and their counts.
    /// </summary>
    WarningCodes: Map<string, int>
    
    /// <summary>
    /// The number of suppressed warnings.
    /// </summary>
    SuppressedWarningCount: int
    
    /// <summary>
    /// The suppressed warning codes and their counts.
    /// </summary>
    SuppressedWarningCodes: Map<string, int>
}

/// <summary>
/// Represents telemetry data for the TARS DSL parser.
/// </summary>
type TelemetryData = {
    /// <summary>
    /// The unique identifier for this telemetry data.
    /// </summary>
    Id: Guid
    
    /// <summary>
    /// The timestamp when the telemetry data was collected.
    /// </summary>
    Timestamp: DateTime
    
    /// <summary>
    /// The version of the TARS DSL parser.
    /// </summary>
    ParserVersion: string
    
    /// <summary>
    /// The operating system.
    /// </summary>
    OperatingSystem: string
    
    /// <summary>
    /// The .NET runtime version.
    /// </summary>
    RuntimeVersion: string
    
    /// <summary>
    /// The parser usage telemetry.
    /// </summary>
    UsageTelemetry: ParserUsageTelemetry
    
    /// <summary>
    /// The parsing performance telemetry.
    /// </summary>
    PerformanceTelemetry: ParsingPerformanceTelemetry
    
    /// <summary>
    /// The error and warning telemetry.
    /// </summary>
    ErrorWarningTelemetry: ErrorWarningTelemetry option
}
