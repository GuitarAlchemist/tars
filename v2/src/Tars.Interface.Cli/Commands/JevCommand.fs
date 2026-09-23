module Tars.Interface.Cli.Commands.JevCommand

open System
open System.IO
open Tars.Llm

/// `tars jev` — the System One (Jev) wiring, and a probe that is allowed to spend
/// exactly one request.
///
/// Nothing here runs on its own: the dry run is the default, and a live call needs
/// `--live` on top of `TARS_JEV` and `TYPESAFE_API_KEY`. The reply is written out as
/// a fixture so the offline tests can replay a real answer instead of a guessed one.
/// One small question set that exercises all three primitives at once, so a single
/// call tells us whether every shape we parse is the shape Jev actually sends.
let private probeQuestions =
    [ "next_step",
      SystemOne.Choice(
          "A change is proposed to an agent's decision code. Which next step does the state support?",
          [ "design_review", "The evidence or the authority is incomplete."
            "bounded_implementation", "The change is small, reversible and explicitly authorised."
            "reject", "The change contradicts a stated constraint." ]
      )
      "reversibility",
      SystemOne.Score(
          "How hard would this change be to reverse?",
          [ "Low and reversible"; "Moderate or compensatable"; "High or hard to reverse" ]
      )
      "authority_present", SystemOne.Noul "Does the state carry explicit authority for the change?" ]

let private probeState =
    [ "change", SystemOne.Text "Add a typed decider to one Decide node, behind an opt-in flag."
      "tests", SystemOne.Text "Ten offline tests replay a saved reply; nothing calls the network by default."
      "authority", SystemOne.Text "The repository owner asked for this integration." ]

let private describe (id: string) (answer: SystemOne.Answer) =
    match answer with
    | SystemOne.Chose(choice, confidence, probabilities) ->
        let spread =
            probabilities
            |> Map.toList
            |> List.sortByDescending snd
            |> List.map (fun (option, p) -> $"%s{option} %.2f{p}")
            |> String.concat ", "

        printfn $"  %-18s{id} chose %s{choice} (confidence %.2f{confidence}) [%s{spread}]"
    | SystemOne.Scored(score, confidence, _) -> printfn $"  %-18s{id} scored %.2f{score} (confidence %.2f{confidence})"
    | SystemOne.Nouled noul -> printfn $"  %-18s{id} noul %.2f{noul}"

let private report (raw: string) (save: string option) =
    match SystemOne.parseReply probeQuestions raw with
    | Error message ->
        // The contract refusing a live reply is the single most useful thing a probe
        // can find: it means our parse and Jev disagree, and no seam should trust it.
        printfn $"Jev replied, and the contract rejected it: %s{message}"
        save |> Option.iter (fun path -> File.WriteAllText(path, raw))
        printfn "The raw reply was saved; that mismatch is what to fix first."
        1
    | Ok reply ->
        printfn $"Model: %s{reply.Model}"

        if reply.Model <> SystemOne.PinnedModel then
            printfn $"  (not the pinned %s{SystemOne.PinnedModel}: thresholds tuned here do not transfer)"

        for KeyValue(id, answer) in reply.Answers do
            describe id answer

        printfn $"Usage: %d{reply.Usage.InputTokens} in, %d{reply.Usage.OutputTokens} out"

        match save with
        | Some path ->
            File.WriteAllText(path, raw)
            printfn $"Saved the reply to %s{path} — replay it with SystemOne.ReplaySystemOne.FromFile."
        | None -> ()

        0

let private dryRun () =
    let body = SystemOne.payload probeState probeQuestions
    let bytes = SystemOne.payloadBytes body
    let limits = SystemOne.JevLimits.Default

    printfn $"Endpoint: %s{limits.Endpoint} (model %s{SystemOne.PinnedModel})"
    printfn $"Payload:  %d{bytes} bytes of %d{limits.MaxPayloadBytes} allowed"
    printfn $"Cost:     $%.6f{SystemOne.inputCostProxyUsd bytes} at worst, one request"
    printfn $"Digest:   %s{SystemOne.digest body}"
    printfn ""
    printfn "%s" body
    printfn ""
    printfn "Nothing was sent. Add --live to make the call (needs TARS_JEV and TYPESAFE_API_KEY)."
    0

let private live (save: string option) =
    match Environment.GetEnvironmentVariable "TARS_JEV", Environment.GetEnvironmentVariable "TYPESAFE_API_KEY" with
    | null, _
    | "", _ ->
        printfn "TARS_JEV is not set. A live call is opt-in: set TARS_JEV=1 and try again."
        2
    | _, key when String.IsNullOrWhiteSpace key ->
        printfn "TYPESAFE_API_KEY is not set in this process. The key is read from the environment and never printed."
        2
    | _, key ->
        let client = new SystemOne.JevClient(key.Trim())

        try
            let bytes = SystemOne.payloadBytes (SystemOne.payload probeState probeQuestions)
            printfn $"Sending %d{bytes} bytes to Jev (at most $%.6f{SystemOne.inputCostProxyUsd bytes})..."

            match client.EvaluateRaw(probeState, probeQuestions) |> Async.RunSynchronously with
            | Error message ->
                printfn $"No reply: %s{message}"
                1
            | Ok raw -> report raw save
        finally
            (client :> IDisposable).Dispose()

let private usage () =
    printfn "tars jev probe [--live] [--save <path>]"
    printfn ""
    printfn "  Sends one System One question set: a Choice, a Score and a Noul."
    printfn "  Without --live it prints the exact request and sends nothing."
    printfn "  --save writes the reply as a fixture for the offline tests."
    0

let run (args: string list) =
    match args with
    | "probe" :: rest ->
        let save =
            rest
            |> List.pairwise
            |> List.tryPick (fun (flag, value) -> if flag = "--save" then Some value else None)

        if rest |> List.contains "--live" then
            live save
        else
            dryRun ()
    | _ -> usage ()
