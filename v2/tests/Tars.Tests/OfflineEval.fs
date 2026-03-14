namespace Tars.Tests

open System
open System.Diagnostics
open Xunit
open Tars.Core
open Tars.Core.Metrics
open Tars.Cortex.Chunking
open Tars.Graph
open Tars.Core.GrammarDistill

module OfflineEval =

    type EvalResult = { Name: string; Status: string; Detail: string }

    let private cases : (string * (unit -> bool * string)) list =
        [
            // Tool parsing sanity
            ("response-parser:tool-call",
             fun () ->
                 match ResponseParser.parse "TOOL:Calc:1+1" with
                 | ResponseParser.ToolCall ("Calc", "1+1") -> true, ""
                 | other -> false, $"Unexpected parse: {other}")
            // Chunking overlap correctness
            ("chunking:sliding-overlap",
             fun () ->
                 let cfg = { defaultConfig with ChunkSize = 16; ChunkOverlap = 4; MinChunkSize = 4; Strategy = SlidingWindow }
                 let text = String.replicate 10 "abcd "
                 let chunks = chunk cfg "doc" text
                 let overlaps =
                     if chunks.Length >= 2 then
                         chunks.[1].Metadata.StartChar < chunks.[0].Metadata.EndChar
                     else
                         true
                 overlaps, $"count={chunks.Length}")
            // Budget enforcement
            ("budget:prevents-overrun",
             fun () ->
                 let g =
                     BudgetGovernor(
                         { Budget.Infinite with
                             MaxTokens = Some 4<token>
                             MaxCalls = Some 2<requests> }
                     )

                 let ok = g.TryConsume { Cost.Zero with Tokens = 2<token>; CallCount = 1<requests> }
                 let over = g.TryConsume { Cost.Zero with Tokens = 10<token>; CallCount = 1<requests> }
                 ok.IsOk && over.IsError, $"ok={ok.IsOk}; overErr={over.IsError}")
            // Grammar distillation sanity
            ("grammar:json-distill",
             fun () ->
                 let spec =
                     fromJsonExamples
                         [ """{"goal":"g","constraints":[],"validation_criteria":"c"}"""
                           """{"goal":"x","constraints":["a"],"validation_criteria":"y"}""" ]

                 let sample = """{"goal":"g","constraints":[],"validation_criteria":"c"}"""
                 let ok = spec.Validator sample
                 let fieldList = String.concat "," spec.Fields
                 ok, $"fields={fieldList}")
        ]

    let runAll () =
        Metrics.clear()

        let results =
            cases
            |> List.map (fun (name, test) ->
                let sw = Stopwatch.StartNew()
                let ok, detail = test ()
                sw.Stop()
                let status = if ok then "pass" else "fail"
                Metrics.record "eval.case" status sw.Elapsed.TotalMilliseconds None (Map [ "name", name; "detail", detail ])
                { Name = name; Status = status; Detail = detail })

        // Optional dump if caller sets path
        match Environment.GetEnvironmentVariable "TARS_EVAL_OUT" with
        | null
        | "" -> ()
        | path -> Metrics.dumpCsv path

        results

type OfflineEvalTests() =

    [<Fact>]
    member _.``Offline eval golden cases pass``() =
        let results = OfflineEval.runAll()
        let failures = results |> List.filter (fun r -> r.Status = "fail")
        if not failures.IsEmpty then
            let msg =
                failures
                |> List.map (fun f -> $"{f.Name}: {f.Detail}")
                |> String.concat "; "
            Assert.True(false, msg)
