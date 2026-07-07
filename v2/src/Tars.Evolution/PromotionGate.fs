namespace Tars.Evolution

open System

type DefaultPromotionGate(validator: IRoundtripValidator) =
    interface IPromotionGate with
        member _.Decide(existing: RecurrenceRecord list, candidate: PromotionCandidate) : GovernanceDecision * RoundtripResult option =
            let rawDecision = GrammarGovernor.evaluate existing candidate

            match rawDecision with
            | Approve _ ->
                let rtResult = validator.Validate candidate
                if rtResult.Passed then
                    rawDecision, Some rtResult
                else
                    let reason =
                        sprintf "Round-trip validation failed (semantic match: %.2f). %s"
                            rtResult.SemanticMatch
                            (rtResult.Issues |> String.concat "; ")
                    Reject reason, Some rtResult
            | _ ->
                rawDecision, None

module PromotionGate =
    let create validator = DefaultPromotionGate(validator) :> IPromotionGate
    let createDefault (llm: Tars.Llm.ILlmService option) =
        let validator = RoundtripValidation.DefaultRoundtripValidator(llm) :> IRoundtripValidator
        create validator
