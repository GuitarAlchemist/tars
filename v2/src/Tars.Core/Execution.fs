namespace Tars.Core

[<AutoOpen>]
module Execution =

    type ExecutionBuilder() =
        member _.Bind(x: ExecutionOutcome<'T>, f: 'T -> ExecutionOutcome<'U>) =
            match x with
            | ExecutionOutcome.Success v -> f v
            | ExecutionOutcome.PartialSuccess(v, warnings) ->
                match f v with
                | ExecutionOutcome.Success u -> ExecutionOutcome.PartialSuccess(u, warnings)
                | ExecutionOutcome.PartialSuccess(u, newWarnings) -> ExecutionOutcome.PartialSuccess(u, warnings @ newWarnings)
                | ExecutionOutcome.Failure err -> ExecutionOutcome.Failure err
            | ExecutionOutcome.Failure err -> ExecutionOutcome.Failure err

        member _.Return(x: 'T) = ExecutionOutcome.Success x
        member _.ReturnFrom(x: ExecutionOutcome<'T>) = x
        member _.Zero() = ExecutionOutcome.Success()

    let execution = ExecutionBuilder()
