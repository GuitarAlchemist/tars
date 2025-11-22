namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Text
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.SelfImprovement
open TarsEngine.FSharp.SelfImprovement.Tier2Runner

type AutoLoopCommand(logger: ILogger<AutoLoopCommand>, service: ISelfImprovementService, loggerFactory: ILoggerFactory) =

    let actionLabel = function
        | Tier2Action.PolicyTightened -> "policy tightened"
        | Tier2Action.PolicyRelaxed -> "policy relaxed"
        | Tier2Action.RemediationEnqueued -> "remediation enqueued"
        | Tier2Action.NoChange -> "no change"

    let describeOutcome index outcome =
        match outcome with
        | NoPendingWork -> sprintf "Iteration %d: no pending work discovered." index
        | Success (_, actions) ->
            let actionsText =
                match actions with
                | [] -> "no policy change"
                | list -> list |> List.map actionLabel |> String.concat ", "
            sprintf "Iteration %d: success (%s)." index actionsText
        | Failure (_, actions) ->
            let actionsText =
                actions
                |> List.map actionLabel
                |> String.concat ", "
            sprintf "Iteration %d: failure (%s)." index actionsText

    interface ICommand with
        member _.Name = "auto-loop"
        member _.Description = "Execute a Tier 2 autonomous improvement iteration and report results."
        member _.Usage = "tars auto-loop [iterations]"
        member _.Examples =
            [ "tars auto-loop"
              "tars auto-loop 3" ]

        member _.ValidateOptions(_options: CommandOptions) = true

        member _.ExecuteAsync(options: CommandOptions) =
            task {
                let iterations =
                    match options.Arguments with
                    | head :: _ ->
                        match Int32.TryParse head with
                        | true, value when value > 0 -> value
                        | _ -> 1
                    | [] -> 1

                let sb = StringBuilder()
                let mutable successes = 0
                let mutable failures = 0
                let mutable noWorkEncountered = false

                let mutable i = 1
                while i <= iterations && not noWorkEncountered do
                    let! outcome = Tier2Runner.runIterationAsync service loggerFactory None None
                    let description = describeOutcome i outcome
                    sb.AppendLine(description) |> ignore

                    match outcome with
                    | NoPendingWork ->
                        noWorkEncountered <- true
                    | Success _ ->
                        successes <- successes + 1
                    | Failure _ ->
                        failures <- failures + 1

                    i <- i + 1

                let summary =
                    if failures = 0 then
                        sprintf "Tier 2 loop complete. Successes: %d. Failures: %d.%s%s"
                            successes
                            failures
                            (if noWorkEncountered then " No pending work." else "")
                            (if iterations > 1 && not noWorkEncountered then " Iterations exhausted." else "")
                    else
                        sprintf "Tier 2 loop completed with failures. Successes: %d. Failures: %d."
                            successes
                            failures

                sb.AppendLine() |> ignore
                sb.AppendLine(summary) |> ignore

                logger.LogInformation("{Summary}", summary)

                return
                    if failures = 0 then
                        CommandResult.success (sb.ToString())
                    else
                        CommandResult.failure (sb.ToString())
            }
