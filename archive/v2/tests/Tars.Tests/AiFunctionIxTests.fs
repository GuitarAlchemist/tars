namespace Tars.Tests

open System.Threading.Tasks
open Xunit
open Tars.Llm

/// Test types for ix convolution tests.
type GenreResult = { Genre: string; Confidence: float }
type IxClassification = { Label: string; Probability: float }
type TopoResult = { Interpretation: string; BettiNumbers: int list }
type HyperParams = { LearningRate: float; Epochs: int; BatchSize: int }

module AiFunctionIxTests =

    // ── Stub ix caller ──────────────────────────────────────────────────────

    /// Create a stub IxCaller that returns a fixed response for any tool.
    let stubIx (response: string) : IxCaller =
        fun _tool _args -> async { return Ok response }

    /// Create a stub IxCaller that fails.
    let failingIx (error: string) : IxCaller =
        fun _tool _args -> async { return Error error }

    /// Create a stub IxCaller that returns different responses per tool name.
    let routingIx (responses: (string * string) list) : IxCaller =
        let map = Map.ofList responses
        fun tool _args ->
            async {
                match Map.tryFind tool map with
                | Some r -> return Ok r
                | None -> return Error $"Unknown tool: {tool}"
            }

    // ── Pattern 1: ix post-condition ─────────────────────────────────────────

    [<Fact>]
    let ``ix post-condition passes when ix agrees`` () = async {
        let ix = stubIx """{"Label": "jazz", "Probability": 0.92}"""

        let cfg =
            AiFunction.create<string, GenreResult> "genre" "Classify genre. Return JSON." id
            |> AiFunctionIx.withIxPostCondition<string, GenreResult, IxClassification>
                "ix-agrees"
                ix
                "ix_supervised"
                (fun output -> $"""[{{"features": [1,2,3]}}]""")
                (fun output ixResult ->
                    if output.Genre.ToLower() = ixResult.Label.ToLower() then Ok ()
                    else Error $"LLM said '{output.Genre}' but ix said '{ixResult.Label}'")

        let stub = StubLlmService(["""{"Genre": "Jazz", "Confidence": 0.85}"""])
        let! result = AiFunction.executeAsync stub cfg "some features"

        match result with
        | Ok r -> Assert.Equal("Jazz", r.Genre)
        | Error e -> failwith $"Expected Ok but got: {e}"
    }

    [<Fact>]
    let ``ix post-condition fails when ix disagrees and retries`` () = async {
        let ix = stubIx """{"Label": "blues", "Probability": 0.88}"""

        let cfg =
            AiFunction.create<string, GenreResult> "genre" "Classify genre. Return JSON." id
            |> AiFunctionIx.withIxPostCondition<string, GenreResult, IxClassification>
                "ix-agrees"
                ix
                "ix_supervised"
                (fun _ -> """[]""")
                (fun output ixResult ->
                    if output.Genre.ToLower() = ixResult.Label.ToLower() then Ok ()
                    else Error $"LLM said '{output.Genre}' but ix said '{ixResult.Label}'")
            |> AiFunction.withMaxAttempts 2

        let stub = StubLlmService([
            """{"Genre": "Jazz", "Confidence": 0.7}"""    // disagrees with ix
            """{"Genre": "Blues", "Confidence": 0.9}"""   // agrees after feedback
        ])
        let! result = AiFunction.executeAsync stub cfg "features"

        match result with
        | Ok r -> Assert.Equal("Blues", r.Genre)
        | Error e -> failwith $"Expected Ok on retry but got: {e}"
    }

    [<Fact>]
    let ``ix post-condition handles ix unavailable`` () = async {
        let ix = failingIx "ix not installed"

        let cfg =
            AiFunction.create<string, GenreResult> "genre" "Classify genre. Return JSON." id
            |> AiFunctionIx.withIxPostCondition<string, GenreResult, IxClassification>
                "ix-check"
                ix
                "ix_supervised"
                (fun _ -> "[]")
                (fun _ _ -> Ok ())
            |> AiFunction.withMaxAttempts 1

        let stub = StubLlmService(["""{"Genre": "Rock", "Confidence": 0.9}"""])
        let! result = AiFunction.executeAsync stub cfg "features"

        match result with
        | Error (MaxAttemptsExhausted(_, reason)) ->
            Assert.Contains("ix", reason)
        | _ -> failwith "Expected error when ix is unavailable"
    }

    // ── Pattern 2: ix enrichment ─────────────────────────────────────────────

    [<Fact>]
    let ``ix enrichment injects computed data into LLM prompt`` () = async {
        let ix = stubIx """{"betti_0": 1, "betti_1": 2, "betti_2": 0}"""

        let cfg =
            AiFunction.create<string, TopoResult> "topo-explain"
                "Explain topological features. Return JSON with Interpretation and BettiNumbers."
                (fun text -> text)
            |> AiFunctionIx.withIxEnrichment
                ix
                "ix_topo"
                (fun input -> $"""{{ "points": {input} }}""")
                (fun _input ixJson -> $"ix computed topology: {ixJson}\nExplain these Betti numbers.")

        let stub = StubLlmService(["""{"Interpretation": "One connected component, two loops, Betti=[1,2,0]", "BettiNumbers": [1, 2, 0]}"""])
        let! result = AiFunction.executeAsync stub cfg "[[1,2],[3,4],[5,6]]"

        match result with
        | Ok r ->
            Assert.Contains("loop", r.Interpretation)
            Assert.Equal(3, r.BettiNumbers.Length)
        | Error e -> failwith $"Expected Ok but got: {e}"

        // Verify the LLM prompt was enriched with ix data
        let req = stub.Requests |> List.head
        Assert.Contains("ix computed topology", req.Messages.Head.Content)
    }

    [<Fact>]
    let ``ix enrichment degrades gracefully when ix unavailable`` () = async {
        let ix = failingIx "connection refused"

        let cfg =
            AiFunction.create<string, TopoResult> "topo-explain"
                "Explain topological features. Return JSON."
                (fun text -> text)
            |> AiFunctionIx.withIxEnrichment
                ix
                "ix_topo"
                (fun _ -> "{}")
                (fun _input ixJson -> $"Context: {ixJson}")

        let stub = StubLlmService(["""{"Interpretation": "No data available", "BettiNumbers": []}"""])
        let! result = AiFunction.executeAsync stub cfg "data"

        // Should still succeed — ix failure is injected as context, not a hard error
        Assert.True(Result.isOk result)
        let req = stub.Requests |> List.head
        Assert.Contains("ix unavailable", req.Messages.Head.Content)
    }

    // ── Pattern 3: ix threshold ──────────────────────────────────────────────

    [<Fact>]
    let ``ix threshold passes when metric exceeds threshold`` () = async {
        let ix = stubIx """{"accuracy": 0.92, "f1": 0.89}"""

        let cfg =
            AiFunction.create<string, HyperParams> "tune"
                "Propose hyperparameters. Return JSON."
                (fun ds -> $"Dataset: {ds}")
            |> AiFunctionIx.withIxThreshold
                "accuracy >= 0.85"
                ix
                "ix_ml_pipeline"
                (fun hp -> $"""{{ "learning_rate": {hp.LearningRate}, "epochs": {hp.Epochs} }}""")
                (fun json ->
                    match JsonParsing.tryParseElement json with
                    | Error e -> Error e
                    | Ok elem ->
                        try Ok (elem.GetProperty("accuracy").GetDouble())
                        with ex -> Error ex.Message)
                0.85

        let stub = StubLlmService(["""{"LearningRate": 0.001, "Epochs": 50, "BatchSize": 32}"""])
        let! result = AiFunction.executeAsync stub cfg "100 rows, 5 features"

        Assert.True(Result.isOk result)
    }

    [<Fact>]
    let ``ix threshold fails and retries when metric below threshold`` () = async {
        let mutable callCount = 0
        let ix: IxCaller = fun _tool _args ->
            async {
                callCount <- callCount + 1
                if callCount <= 1 then
                    return Ok """{"accuracy": 0.60}"""
                else
                    return Ok """{"accuracy": 0.90}"""
            }

        let cfg =
            AiFunction.create<string, HyperParams> "tune"
                "Propose hyperparameters. Return JSON."
                (fun ds -> ds)
            |> AiFunctionIx.withIxThreshold
                "accuracy >= 0.85"
                ix
                "ix_ml_pipeline"
                (fun hp -> "{}")
                (fun json ->
                    match JsonParsing.tryParseElement json with
                    | Ok elem -> Ok (elem.GetProperty("accuracy").GetDouble())
                    | Error e -> Error e)
                0.85
            |> AiFunction.withMaxAttempts 3

        let stub = StubLlmService([
            """{"LearningRate": 0.1, "Epochs": 10, "BatchSize": 128}"""
            """{"LearningRate": 0.001, "Epochs": 100, "BatchSize": 32}"""
        ])
        let! result = AiFunction.executeAsync stub cfg "dataset"

        match result with
        | Ok hp -> Assert.Equal(0.001, hp.LearningRate)
        | Error e -> failwith $"Expected Ok on retry but got: {e}"
    }

    // ── Pattern 4: Bidirectional ix scoring ──────────────────────────────────

    [<Fact>]
    let ``ix scoring accepts output when score passes`` () = async {
        let ix = stubIx """{"score": 0.95, "verdict": "excellent"}"""

        let cfg =
            AiFunction.create<string, GenreResult> "genre" "Classify genre." id

        let stub = StubLlmService(["""{"Genre": "Jazz", "Confidence": 0.88}"""])
        let! result =
            AiFunctionIx.executeWithIxScoringAsync
                stub ix "ix_supervised"
                (fun output -> $"""{{ "genre": "{output.Genre}" }}""")
                (fun ixJson -> "PASS")  // score passes
                cfg "features"

        match result with
        | Ok (output, ixJson) ->
            Assert.Equal("Jazz", output.Genre)
            Assert.Contains("score", ixJson)
        | Error e -> failwith $"Expected Ok but got: {e}"
    }

    [<Fact>]
    let ``ix scoring retries when score fails`` () = async {
        let mutable callCount = 0
        let ix: IxCaller = fun _tool _args ->
            async {
                callCount <- callCount + 1
                if callCount <= 1 then
                    return Ok """{"score": 0.3}"""
                else
                    return Ok """{"score": 0.95}"""
            }

        let cfg =
            AiFunction.create<string, GenreResult> "genre" "Classify genre." id
            |> AiFunction.withMaxAttempts 3

        let stub = StubLlmService([
            """{"Genre": "Unknown", "Confidence": 0.2}"""
            """{"Genre": "Jazz", "Confidence": 0.9}"""
        ])
        let! result =
            AiFunctionIx.executeWithIxScoringAsync
                stub ix "ix_supervised"
                (fun output -> "{}")
                (fun ixJson ->
                    match JsonParsing.tryParseElement ixJson with
                    | Ok elem when elem.GetProperty("score").GetDouble() >= 0.8 -> "PASS"
                    | Ok elem ->
                        let score = elem.GetProperty("score").GetDouble()
                        $"Score too low: {score}"
                    | Error e -> e)
                cfg "features"

        match result with
        | Ok (output, _) -> Assert.Equal("Jazz", output.Genre)
        | Error e -> failwith $"Expected Ok on retry but got: {e}"
    }

    [<Fact>]
    let ``ix scoring gracefully accepts when ix unavailable`` () = async {
        let ix = failingIx "ix not found"

        let cfg =
            AiFunction.create<string, GenreResult> "genre" "Classify." id

        let stub = StubLlmService(["""{"Genre": "Rock", "Confidence": 0.7}"""])
        let! result =
            AiFunctionIx.executeWithIxScoringAsync
                stub ix "ix_supervised"
                (fun _ -> "{}")
                (fun _ -> "PASS")
                cfg "input"

        match result with
        | Ok (output, msg) ->
            Assert.Equal("Rock", output.Genre)
            Assert.Contains("ix unavailable", msg)
        | Error e -> failwith $"Expected graceful degradation but got: {e}"
    }
