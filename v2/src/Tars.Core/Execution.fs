namespace Tars.Core

[<AutoOpen>]
module Execution =

    type ExecutionBuilder() =
        member _.Bind(x: ExecutionOutcome<'T>, f: 'T -> ExecutionOutcome<'U>) =
            match x with
            | Success v -> f v
            | PartialSuccess(v, warnings) ->
                match f v with
                | Success u -> PartialSuccess(u, warnings)
                | PartialSuccess(u, newWarnings) -> PartialSuccess(u, warnings @ newWarnings)
                | Failure err -> Failure err
            | Failure err -> Failure err

        member _.Return(x: 'T) = Success x
        member _.ReturnFrom(x: ExecutionOutcome<'T>) = x
        member _.Zero() = Success()

    let execution = ExecutionBuilder()
