namespace Tars.Tools

open System.Text.Json
open Tars.Core
open Tars.Core.WorkflowOfThought

/// Invokes registry tools with resilience + recording (delegating to
/// Tars.Core.ToolExecution) and surfaces the distinct failure modes as a
/// ToolOutcome. This is the deep seam the WoT executor (and any other caller
/// wanting typed outcomes) uses instead of touching tools directly. The recorder
/// is injectable so tests can observe recording without the global ledger.
type ToolInvoker(registry: IToolRegistry, recorder: IToolRecorder) =

    /// Default invoker writing to the global ToolLedger.
    static member create(registry: IToolRegistry) =
        ToolInvoker(registry, LedgerToolRecorder() :> IToolRecorder)

    interface IToolInvoker with
        member _.Invoke(toolName, args) =
            async {
                match registry.Get toolName with
                | None -> return ToolOutcome.NotFound
                | Some tool ->
                    let input = JsonSerializer.Serialize(args)
                    let! result = ToolExecution.run recorder tool input

                    return
                        match result with
                        | Result.Ok output -> ToolOutcome.Succeeded output
                        | Result.Error "Circuit Breaker is OPEN" -> ToolOutcome.CircuitOpen
                        | Result.Error msg -> ToolOutcome.Failed(string (ToolExecution.classifyFailure msg), msg)
            }
