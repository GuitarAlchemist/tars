namespace Tars.Core

open System
open System.Text.Json

/// Guard actions the caller can take after evaluating an LLM response
type GuardAction =
    | Accept
    | RetryWithHint of string
    | AskForEvidence of string
    | Reject of string
    | Fallback of string

/// Guard result with a normalized risk score (0.0 low risk, 1.0 high risk)
type GuardResult =
    { Risk: float
      Action: GuardAction
      Messages: string list }

/// Input to the output guard
type GuardInput =
    { ResponseText: string
      Grammar: string option
      ExpectedJsonFields: string list option
      RequireCitations: bool
      Citations: string list option
      AllowExtraFields: bool
      Metadata: Map<string, string> }

/// Pluggable interface for LLM-based cargo-cult analysis (optional sidecar)
type IOutputGuardAnalyzer =
    abstract member Analyze : GuardInput -> Async<GuardResult option>

/// Pluggable interface for output guards
type IOutputGuard =
    abstract member Evaluate : GuardInput -> Async<GuardResult>

module private Json =
    let tryParseFields (text: string) =
        try
            use doc = JsonDocument.Parse(text)
            let root = doc.RootElement
            if root.ValueKind = JsonValueKind.Object then
                root.EnumerateObject() |> Seq.map (fun p -> p.Name) |> Set.ofSeq |> Some
            else
                None
        with _ ->
            None

/// Basic output guard (shape + citations)
type BasicOutputGuard() =
    interface IOutputGuard with
        member _.Evaluate(input: GuardInput) = async {
            let messages = ResizeArray<string>()
            let mutable risk = 0.0

            // Shape: grammar check if provided
            match input.Grammar with
            | Some g when not (String.IsNullOrWhiteSpace g) ->
                if not (GrammarValidation.matches g input.ResponseText) then
                    risk <- max risk 0.7
                    messages.Add("Output failed grammar/shape validation.")
            | _ -> ()

            // Shape: expected JSON fields
            match input.ExpectedJsonFields with
            | Some expected when expected.Length > 0 ->
                match Json.tryParseFields input.ResponseText with
                | Some fields ->
                    let missing = expected |> List.filter (fun f -> not (fields.Contains f))
                    let extras =
                        if input.AllowExtraFields then []
                        else fields |> Set.filter (fun f -> not (expected |> List.contains f)) |> Set.toList
                    if missing.Length > 0 then
                        risk <- max risk 0.6
                        let missingStr = String.Join(", ", missing)
                        messages.Add($"Missing fields: {missingStr}")
                    if extras.Length > 0 then
                        risk <- max risk 0.4
                        let extrasStr = String.Join(", ", extras)
                        messages.Add($"Unexpected fields: {extrasStr}")
                | None ->
                    risk <- max risk 0.6
                    messages.Add("Expected JSON object but parsing failed.")
            | _ -> ()

            // Citations / provenance
            if input.RequireCitations then
                match input.Citations with
                | None
                | Some [] ->
                    risk <- max risk 0.5
                    messages.Add("Citations required but none provided.")
                | Some _ -> ()

            // Choose action based on risk
            let action =
                if risk >= 0.7 then
                    Reject "Output failed validation and appears untrustworthy."
                elif risk >= 0.5 then
                    RetryWithHint "Regenerate with citations and exact schema; do not invent content."
                elif risk >= 0.3 then
                    AskForEvidence "Provide supporting citations or tool outputs to verify claims."
                else
                    Accept

            return
                { Risk = risk
                  Action = action
                  Messages = messages |> List.ofSeq }
        }

/// Default guard instance
module OutputGuard =
    /// Compose a primary guard with an optional analyzer sidecar.
    /// The analyzer can raise the risk (never lower it) and append messages.
    let withAnalyzer (guard: IOutputGuard) (analyzer: IOutputGuardAnalyzer option) : IOutputGuard =
        { new IOutputGuard with
            member _.Evaluate(input) = async {
                // First, run the base guard
                let! baseResult = guard.Evaluate input

                match analyzer with
                | None -> return baseResult
                | Some a ->
                    try
                        let! analysis = a.Analyze input
                        match analysis with
                        | None -> return baseResult
                        | Some ar ->
                            let risk = max baseResult.Risk ar.Risk
                            let messages = baseResult.Messages @ ar.Messages
                            let action =
                                // Choose the stricter action
                                match baseResult.Action, ar.Action with
                                | Reject _, _ -> baseResult.Action
                                | _, Reject _ -> ar.Action
                                | RetryWithHint _, _ -> baseResult.Action
                                | _, RetryWithHint _ -> ar.Action
                                | AskForEvidence _, _ -> baseResult.Action
                                | _, AskForEvidence _ -> ar.Action
                                | Fallback _, _ -> baseResult.Action
                                | _, Fallback _ -> ar.Action
                                | Accept, Accept -> Accept
                            return { Risk = risk; Action = action; Messages = messages }
                    with _ ->
                        // Fail open: return base result if analyzer fails/times out
                        return baseResult
            } }

    let defaultGuard : IOutputGuard = BasicOutputGuard() :> IOutputGuard
