/// Practical examples of using TARS functional patterns
/// This file demonstrates real-world usage patterns for functional programming in TARS
module Tars.Examples.FunctionalPatterns

open Tars.Core
open System

// ============================================================================
// EXAMPLE 1: Configuration Validation with Error Accumulation
// ============================================================================

module ConfigValidation =
    type ConfigError =
        | ModelNameEmpty
        | ModelNameInvalid
        | PortOutOfRange
        | TimeoutNegative

    type TarsConfig =
        { ModelName: string
          Port: int
          TimeoutMs: int }

    /// Validate configuration with ALL errors reported
    let validateConfig modelName port timeoutMs =
        let validModel =
            Validators.notEmpty ModelNameEmpty modelName
            |> Validation.map (fun m ->
                if m.StartsWith("tars-") then
                    Validation.valid m
                else
                    Validation.invalid ModelNameInvalid)
            |> function
                | Valid v -> v
                | Invalid es -> Invalid es

        let validPort = Validators.inRange 1 65535 PortOutOfRange port
        let validTimeout = Validators.satisfies (fun t -> t >= 0) TimeoutNegative timeoutMs

        match validModel, validPort, validTimeout with
        | Valid m, Valid p, Valid t ->
            Validation.valid
                { ModelName = m
                  Port = p
                  TimeoutMs = t }
        | _ ->
            let errors =
                [ validModel; validPort; validTimeout ]
                |> List.choose (function
                    | Invalid es -> Some es
                    | _ -> None)
                |> List.concat

            Validation.invalidMany errors

    // Usage example
    let example () =
        match validateConfig "" 99999 -100 with
        | Valid config -> printfn "Config valid: %A" config
        | Invalid errors ->
            printfn "Config has %d errors:" errors.Length
            errors |> List.iter (printfn "  - %A")
// Output: Config has 3 errors:
//   - ModelNameEmpty
//   - PortOutOfRange
//   - TimeoutNegative

// ============================================================================
// EXAMPLE 2: Async with Error Handling using AsyncResult
// ============================================================================

module LlmCallExample =
    open AsyncResult

    type LlmError =
        | InvalidPrompt of string
        | RateLimitExceeded
        | ApiTimeout
        | ResponseTooLarge

    /// Simulate LLM call
    let callLlm (prompt: string) : AsyncResult<string, LlmError> =
        asyncResult {
            if String.IsNullOrWhiteSpace prompt then
                return! AsyncResult.ofResult (Error(InvalidPrompt "Empty prompt"))
            else
                // Simulate async call
                do! Async.Sleep 10 |> AsyncResult.ofAsync
                return $"Response to: {prompt}"
        }

    /// Process LLM response
    let processResponse (response: string) : AsyncResult<string, LlmError> =
        asyncResult {
            if response.Length > 1000 then
                return! AsyncResult.ofResult (Error ResponseTooLarge)
            else
                return response.ToUpper()
        }

    /// Complete pipeline
    let generateAndProcess (prompt: string) : AsyncResult<string, LlmError> =
        asyncResult {
            let! response = callLlm prompt
            let! processed = processResponse response
            return processed
        }

    // Usage example
    let example () =
        async {
            let! result = generateAndProcess "What is TARS?"

            match result with
            | Ok answer -> printfn "Success: %s" answer
            | Error err -> printfn "Error: %A" err
        }

// ============================================================================
// EXAMPLE 3: Reader Monad for Dependency Injection
// ============================================================================

module DependencyInjection =
    open Reader

    type AppContext =
        { LlmEndpoint: string
          Database: string
          LogLevel: string }

    /// Get LLM endpoint from context
    let getLlmEndpoint: Reader<AppContext, string> =
        reader {
            let! ctx = Reader.ask
            return ctx.LlmEndpoint
        }

    /// Get database connection from context
    let getDatabase: Reader<AppContext, string> =
        reader {
            let! ctx = Reader.ask
            return ctx.Database
        }

    /// Business logic that needs dependencies
    let processRequest (input: string) : Reader<AppContext, string> =
        reader {
            let! endpoint = getLlmEndpoint
            let! db = getDatabase
            return $"Processing '{input}' with LLM at {endpoint} and DB {db}"
        }

    // Usage example
    let example () =
        let ctx =
            { LlmEndpoint = "http://localhost:11434"
              Database = "postgres://localhost"
              LogLevel = "Info" }

        let result = processRequest "Hello TARS" |> Reader.run <| ctx
        printfn "%s" result
// Output: Processing 'Hello TARS' with LLM at http://localhost:11434 and DB postgres://localhost

// ============================================================================
// EXAMPLE 4: Writer Monad for Structured Logging
// ============================================================================

module StructuredLogging =
    open Writer

    type LogEntry =
        { Timestamp: DateTime
          Level: string
          Message: string }

    let log level message : Writer<LogEntry, unit> =
        Writer.tell
            { Timestamp = DateTime.UtcNow
              Level = level
              Message = message }

    /// Workflow with logging
    let processWithLogging (input: int) : Writer<LogEntry, int> =
        writer {
            do! log "Info" $"Starting processing of {input}"
            let result = input * 2
            do! log "Debug" $"Intermediate result: {result}"
            let final = result + 10
            do! log "Info" $"Processing complete: {final}"
            return final
        }

    // Usage example
    let example () =
        let (result, logs) = processWithLogging 21 |> Writer.run
        printfn "Result: %d" result // 52
        printfn "Logs:"

        logs
        |> List.iter (fun log -> printfn "[%s] %s: %s" (log.Timestamp.ToString("HH:mm:ss")) log.Level log.Message)

// ============================================================================
// EXAMPLE 5: NonEmptyList for Guaranteed Safety
// ============================================================================

module SafeCollections =
    open NonEmptyList

    /// Process list of agents (guaranteed at least one)
    let processAgents (agents: NonEmptyList<string>) : string =
        let primary = NonEmptyList.head agents
        let count = NonEmptyList.length agents
        $"Processing {count} agents, primary: {primary}"

    /// Try to create from list
    let tryProcessAgentList (agents: string list) : string option =
        match NonEmptyList.ofList agents with
        | Some nel -> Some(processAgents nel)
        | None -> None

    // Usage example
    let example () =
        // Safe - guaranteed non-empty
        let agents = NonEmptyList.singleton "TARS-Alpha" |> NonEmptyList.cons "TARS-Beta"
        printfn "%s" (processAgents agents)

        // Unsafe list - might be empty
        match tryProcessAgentList [] with
        | Some result -> printfn "%s" result
        | None -> printfn "No agents to process!"

// ==============================================================================
// EXAMPLE 6: Combining Multiple Patterns
// ============================================================================

module CompleteExample =
    open AsyncResult
    open Reader
    open Writer

    type AppConfig = { ApiUrl: string; MaxRetries: int }

    type ProcessError =
        | ValidationFailed of string
        | ApiError of string
        | TooManyRetries

    type LogMessage = string

    /// Complete workflow combining AsyncResult, Reader, and logging
    let processData (input: string) : Reader<AppConfig, AsyncResult<string, ProcessError>> =
        reader {
            let! config = Reader.ask

            return
                asyncResult {
                    // Validate input
                    if String.IsNullOrWhiteSpace input then
                        return! AsyncResult.ofResult (Error(ValidationFailed "Empty input"))
                    else
                        // Simulate API call with config
                        do! Async.Sleep 10 |> AsyncResult.ofAsync
                        let result = $"Processed '{input}' via {config.ApiUrl}"
                        return result
                }
        }

    // Usage example
    let example () =
        async {
            let config =
                { ApiUrl = "http://localhost:11434"
                  MaxRetries = 3 }

            let workflow = processData "Hello TARS"
            let asyncResult = Reader.run workflow config

            let! result = asyncResult

            match result with
            | Ok value -> printfn "Success: %s" value
            | Error err -> printfn "Error: %A" err
        }

// ============================================================================
// EXAMPLE 7: Practical Operators Usage
// ============================================================================

module OperatorExamples =
    open FunctionalOps

    /// Parse and validate with operators
    let parseAndValidate (input: string) : Result<int, string> =
        // Parse string to int
        let parse s =
            try
                Ok(int s)
            with _ ->
                Error "Parse error"

        // Validate positive
        let validatePositive n =
            if n > 0 then Ok n else Error "Must be positive"

        // Kleisli composition
        let parseAndValidateOp = parse >=> validatePositive

        parseAndValidateOp input

    /// Process option with operators
    let processOption (opt: int option) : string option =
        opt <!> (fun x -> x * 2) // Map: multiply by 2
        >>= (fun x -> Some(x + 10)) // Bind: add 10
        <!> (fun x -> $"Result: {x}") // Map: to string

    // Usage examples
    let example () =
        // Kleisli example
        match parseAndValidate "42" with
        | Ok n -> printf "Parsed: %d" n // 42
        | Error e -> printfn "Error: %s" e

        match parseAndValidate "-5" with
        | Ok n -> printfn "Parsed: %d" n
        | Error e -> printfn "Error: %s" e // "Must be positive"

        // Option operators
        match processOption (Some 5) with
        | Some s -> printfn "%s" s // "Result: 20"
        | None -> printfn "None"

// ============================================================================
// EXAMPLE 8: Real TARS Integration - Evolution Task Validation
// ============================================================================

module EvolutionTaskValidation =
    type TaskValidationError =
        | GoalEmpty
        | DescriptionTooShort
        | ComplexityInvalid

    type EvolutionTask =
        { Goal: string
          Description: string
          Complexity: int }

    /// Validate evolution task
    let validateTask goal description complexity =
        let validGoal = Validators.notEmpty GoalEmpty goal

        let validDesc =
            Validators.satisfies (fun d -> d.Length >= 10) DescriptionTooShort description

        let validComplexity = Validators.inRange 1 10 ComplexityInvalid complexity

        match validGoal, validDesc, validComplexity with
        | Valid g, Valid d, Valid c ->
            Validation.valid
                { Goal = g
                  Description = d
                  Complexity = c }
        | _ ->
            let errors =
                [ validGoal; validDesc; validComplexity ]
                |> List.choose (function
                    | Invalid es -> Some es
                    | _ -> None)
                |> List.concat

            Validation.invalidMany errors

    // Usage example
    let example () =
        // Valid task
        match validateTask "Implement feature X" "A detailed description of the feature" 5 with
        | Valid task -> printfn "Task created: %A" task
        | Invalid errors -> printfn "Validation errors: %A" errors

        // Invalid task - get ALL errors
        match validateTask "" "short" 15 with
        | Valid task -> printfn "Task created: %A" task
        | Invalid errors ->
            printfn "Validation failed with %d errors:" errors.Length
            errors |> List.iter (printfn "  - %A")
// Output:
//   - GoalEmpty
//   - DescriptionTooShort
//   - ComplexityInvalid

// ============================================================================
// RUNNING ALL EXAMPLES
// ============================================================================

/// Run all examples
let runAll () =
    printfn "=== TARS Functional Patterns Examples ===%"

    printfn "\n1. Configuration Validation:"
    ConfigValidation.example ()

    printfn "\n2. LLM Call with AsyncResult:"
    LlmCallExample.example () |> Async.RunSynchronously

    printfn "\n3. Dependency Injection with Reader:"
    DependencyInjection.example ()

    printfn "\n4. Structured Logging with Writer:"
    StructuredLogging.example ()

    printfn "\n5. Safe Collections with NonEmptyList:"
    SafeCollections.example ()

    printfn "\n6. Complete Combined Example:"
    CompleteExample.example () |> Async.RunSynchronously

    printfn "\n7. Operator Examples:"
    OperatorExamples.example ()

    printfn "\n8. Evolution Task Validation:"
    EvolutionTaskValidation.example ()

    printfn "\n=== All examples complete! ==="
