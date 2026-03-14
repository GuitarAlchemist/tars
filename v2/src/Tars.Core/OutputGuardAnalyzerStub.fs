namespace Tars.Core

/// Stub analyzer that can be swapped with a real LLM-based cargo-cult analysis.
/// Returns None by default (non-blocking).
type NoopOutputGuardAnalyzer() =
    interface IOutputGuardAnalyzer with
        member _.Analyze (_input: GuardInput) : Async<GuardResult option> =
            async { return None }

/// Delegate-based analyzer for tests or adapters.
type DelegateOutputGuardAnalyzer(analyze: GuardInput -> Async<GuardResult option>) =
    interface IOutputGuardAnalyzer with
        member _.Analyze input = analyze input

/// Example of composing default guard with a (currently noop) analyzer.
module OutputGuardComposer =
    let defaultWithNoopAnalyzer : IOutputGuard =
        OutputGuard.withAnalyzer OutputGuard.defaultGuard (Some (NoopOutputGuardAnalyzer() :> IOutputGuardAnalyzer))
