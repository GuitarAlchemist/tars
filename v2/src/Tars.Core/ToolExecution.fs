namespace Tars.Core

open System
open System.Collections.Concurrent

/// Records the outcome of a tool execution. The production recorder writes to the
/// global ToolLedger (which also feeds Metrics); tests can supply a stub.
type IToolRecorder =
    abstract member Record:
        toolName: string *
        input: string *
        output: string *
        durationMs: float *
        success: bool *
        category: ToolFailureCategory ->
            unit

/// Production recorder — writes to the global ToolLedger.
type LedgerToolRecorder() =
    interface IToolRecorder with
        member _.Record(toolName, input, output, durationMs, success, category) =
            ToolLedger.record toolName input output durationMs success category None

/// Resilient tool execution: a per-tool circuit breaker plus recording, shared by
/// every tool caller. The registry now stores raw tools; resilience lives here so
/// the behaviour is identical whether a tool is invoked via IToolInvoker, the MAF
/// adapter, or directly. This is the single home of the logic that used to be
/// wrapped onto each tool at registration.
module ToolExecution =

    let private defaultRecorder = LedgerToolRecorder() :> IToolRecorder

    // One circuit breaker per tool name, process-wide.
    let private breakers = ConcurrentDictionary<string, Resilience.CircuitBreaker>()

    let private breakerFor (name: string) =
        breakers.GetOrAdd(name, (fun _ -> Resilience.CircuitBreaker(3, TimeSpan.FromMinutes(1.0))))

    /// Classify a tool error message into a failure category.
    let classifyFailure (err: string) : ToolFailureCategory =
        if err.Contains("timeout") || err.Contains("timed out") then
            Timeout
        elif err.Contains("connection") || err.Contains("http") || err.Contains("network") then
            DependencyFailure
        else
            Unknown

    /// Run a tool through its circuit breaker, recording the outcome.
    let run (recorder: IToolRecorder) (tool: Tool) (input: string) : Async<Result<string, string>> =
        async {
            let cb = breakerFor tool.Name
            let sw = System.Diagnostics.Stopwatch.StartNew()
            let taskOp () = Async.StartAsTask(tool.Execute input)

            try
                let! (result: Result<string, string>) = Async.AwaitTask(cb.ExecuteAsync(taskOp))
                sw.Stop()
                let duration = sw.Elapsed.TotalMilliseconds

                match result with
                | Result.Ok output -> recorder.Record(tool.Name, input, output, duration, true, NoFailure)
                | Result.Error err -> recorder.Record(tool.Name, input, err, duration, false, classifyFailure err)

                return result
            with
            | :? InvalidOperationException as ex when ex.Message.Contains("CircuitBreaker is OPEN") ->
                sw.Stop()

                recorder.Record(
                    tool.Name,
                    input,
                    "Circuit Breaker is OPEN",
                    sw.Elapsed.TotalMilliseconds,
                    false,
                    CircuitBreakerOpen
                )

                return Result.Error "Circuit Breaker is OPEN"
            | ex ->
                sw.Stop()
                let cat = if ex.Message.Contains("timeout") then Timeout else Unknown
                recorder.Record(tool.Name, input, ex.Message, sw.Elapsed.TotalMilliseconds, false, cat)
                return Result.Error ex.Message
        }

    /// Run a tool through its circuit breaker using the default ledger recorder.
    let runDefault (tool: Tool) (input: string) : Async<Result<string, string>> = run defaultRecorder tool input
