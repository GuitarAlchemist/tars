/// Error Handling Module for Production
/// Provides typed error categories and graceful degradation
namespace Tars.Core

open System

/// Error severity level
type ErrorSeverity =
    | Low // Minor issue, operation can continue
    | Medium // Notable issue, may affect results
    | High // Significant issue, operation impacted
    | Critical // System-level failure, requires attention

/// Error categories for classification
type ErrorCategory =
    | Configuration of message: string
    | Network of message: string * endpoint: string option
    | Timeout of operation: string * duration: TimeSpan
    | Resource of resource: string * reason: string
    | Validation of field: string * reason: string
    | Authorization of reason: string
    | NotFound of resourceType: string * id: string
    | RateLimited of service: string * retryAfter: TimeSpan option
    | LlmError of model: string * message: string
    | ToolError of toolName: string * message: string
    | InternalError of message: string
    | Unknown of ex: exn

/// Structured error with context
type TarsError =
    { Id: Guid
      Category: ErrorCategory
      Severity: ErrorSeverity
      Timestamp: DateTime
      CorrelationId: Guid option
      Context: Map<string, string>
      InnerException: exn option
      Recoverable: bool }

/// Error result type
type TarsResult<'T> =
    | TarsOk of 'T
    | TarsFail of TarsError

/// Error handling module
module Errors =

    /// Create a new error
    let create category severity recoverable =
        { Id = Guid.NewGuid()
          Category = category
          Severity = severity
          Timestamp = DateTime.UtcNow
          CorrelationId = None
          Context = Map.empty
          InnerException = None
          Recoverable = recoverable }

    /// Create error from exception
    let fromException (ex: exn) =
        let category =
            match ex with
            | :? TimeoutException as te -> Timeout("Unknown", TimeSpan.Zero)
            | :? UnauthorizedAccessException -> Authorization "Access denied"
            | :? System.IO.FileNotFoundException as fnf -> NotFound("File", fnf.FileName)
            | :? ArgumentException as ae -> Validation(ae.ParamName, ae.Message)
            | _ -> Unknown ex

        { Id = Guid.NewGuid()
          Category = category
          Severity = High
          Timestamp = DateTime.UtcNow
          CorrelationId = None
          Context = Map.empty
          InnerException = Some ex
          Recoverable = false }

    /// Add context to error
    let withContext key value (error: TarsError) : TarsError =
        { error with
            Context = Map.add key value error.Context }

    /// Add correlation ID
    let withCorrelation correlationId (error: TarsError) : TarsError =
        { error with
            CorrelationId = Some correlationId }

    /// Get error message
    let message error =
        match error.Category with
        | Configuration msg -> $"Configuration error: {msg}"
        | Network(msg, Some endpoint) -> $"Network error ({endpoint}): {msg}"
        | Network(msg, None) -> $"Network error: {msg}"
        | Timeout(op, duration) -> sprintf "Timeout: %s exceeded %.1fs" op duration.TotalSeconds
        | Resource(res, reason) -> $"Resource error ({res}): {reason}"
        | Validation(field, reason) -> $"Validation error on '{field}': {reason}"
        | Authorization reason -> $"Authorization denied: {reason}"
        | NotFound(resType, id) -> $"{resType} not found: {id}"
        | RateLimited(svc, Some retry) -> sprintf "Rate limited by %s, retry after %.0fs" svc retry.TotalSeconds
        | RateLimited(svc, None) -> $"Rate limited by {svc}"
        | LlmError(model, msg) -> $"LLM error ({model}): {msg}"
        | ToolError(tool, msg) -> $"Tool error ({tool}): {msg}"
        | InternalError msg -> $"Internal error: {msg}"
        | Unknown ex -> $"Unknown error: {ex.Message}"

    /// Get error code for logging/monitoring
    let code error =
        match error.Category with
        | Configuration _ -> "ERR_CONFIG"
        | Network _ -> "ERR_NETWORK"
        | Timeout _ -> "ERR_TIMEOUT"
        | Resource _ -> "ERR_RESOURCE"
        | Validation _ -> "ERR_VALIDATION"
        | Authorization _ -> "ERR_AUTH"
        | NotFound _ -> "ERR_NOT_FOUND"
        | RateLimited _ -> "ERR_RATE_LIMIT"
        | LlmError _ -> "ERR_LLM"
        | ToolError _ -> "ERR_TOOL"
        | InternalError _ -> "ERR_INTERNAL"
        | Unknown _ -> "ERR_UNKNOWN"

    /// Check if error is transient (worth retrying)
    let isTransient error =
        match error.Category with
        | Network _ -> true
        | Timeout _ -> true
        | RateLimited _ -> true
        | _ -> false

    /// Suggest retry delay for transient errors
    let retryDelay error =
        match error.Category with
        | RateLimited(_, Some delay) -> delay
        | RateLimited(_, None) -> TimeSpan.FromSeconds(30.0)
        | Timeout _ -> TimeSpan.FromSeconds(5.0)
        | Network _ -> TimeSpan.FromSeconds(2.0)
        | _ -> TimeSpan.Zero

    /// Format error for logging
    let format error =
        let contextStr =
            if error.Context.IsEmpty then
                ""
            else
                let pairs = error.Context |> Map.toList |> List.map (fun (k, v) -> $"{k}={v}")
                sprintf " [%s]" (String.concat ", " pairs)

        sprintf
            "[%s] %s%s (id=%s, recoverable=%b)"
            (code error)
            (message error)
            contextStr
            (error.Id.ToString().Substring(0, 8))
            error.Recoverable

/// Result computation expression for TarsResult
type TarsResultBuilder() =
    member _.Bind(result: TarsResult<'a>, f: 'a -> TarsResult<'b>) : TarsResult<'b> =
        match result with
        | TarsOk v -> f v
        | TarsFail e -> TarsFail e

    member _.Return(value: 'a) : TarsResult<'a> = TarsOk value

    member _.ReturnFrom(result: TarsResult<'a>) : TarsResult<'a> = result

    member _.Zero() : TarsResult<unit> = TarsOk()

[<AutoOpen>]
module TarsResultBuilderExtensions =
    let tarsResult = TarsResultBuilder()

/// Helper functions for TarsResult
module TarsResultHelpers =

    /// Create a success result
    let succeed<'T> (value: 'T) : TarsResult<'T> = TarsOk value

    /// Create a failure result
    let fail<'T> (error: TarsError) : TarsResult<'T> = TarsFail error

    /// Check if result is success
    let isSuccess (result: TarsResult<'T>) =
        match result with
        | TarsOk _ -> true
        | TarsFail _ -> false

    /// Get value or default
    let defaultValue (defaultVal: 'T) (result: TarsResult<'T>) =
        match result with
        | TarsOk v -> v
        | TarsFail _ -> defaultVal
