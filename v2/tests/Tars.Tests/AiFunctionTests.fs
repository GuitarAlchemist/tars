namespace Tars.Tests

open System.Threading.Tasks
open Xunit
open Tars.Llm

/// Test output type — a simple summary record.
type TestSummary = { Summary: string; WordCount: int }

/// Stub ILlmService that returns canned responses in sequence.
type StubLlmService(responses: string list) =
    let mutable idx = 0
    let mutable requests: LlmRequest list = []

    member _.Requests = requests

    interface ILlmService with
        member _.CompleteAsync(req) =
            requests <- requests @ [req]
            let text = responses.[min idx (responses.Length - 1)]
            idx <- idx + 1
            Task.FromResult({ Text = text; FinishReason = Some "stop"; Usage = None; Raw = None })

        member _.EmbedAsync(_) = Task.FromResult([||]: float32[])
        member _.CompleteStreamAsync(_, _) =
            Task.FromResult({ Text = ""; FinishReason = None; Usage = None; Raw = None })
        member _.RouteAsync(_) =
            Task.FromResult({ Backend = Ollama "test"; Endpoint = System.Uri("http://localhost"); ApiKey = None })

module AiFunctionTests =

    let baseCfg: AiFunctionConfig<string, TestSummary> =
        AiFunction.create "test-summarize"
            "You are a summarizer. Return JSON: {\"Summary\": \"...\", \"WordCount\": N}"
            (fun text -> $"Summarize:\n{text}")

    // ── Happy path ───────────────────────────────────────────────────────────

    [<Fact>]
    let ``happy path - valid JSON returned and deserialized`` () = async {
        let stub = StubLlmService(["""{"Summary": "All good", "WordCount": 2}"""])
        let! result = AiFunction.executeAsync stub baseCfg "some text"

        match result with
        | Ok s ->
            Assert.Equal("All good", s.Summary)
            Assert.Equal(2, s.WordCount)
        | Error e -> failwith $"Expected Ok but got Error: {e}"
    }

    // ── Post-condition passes ────────────────────────────────────────────────

    [<Fact>]
    let ``post-condition passes on valid output`` () = async {
        let cfg =
            baseCfg
            |> AiFunction.withPostCondition "concise" (fun s -> s.WordCount <= 10)

        let stub = StubLlmService(["""{"Summary": "Brief", "WordCount": 1}"""])
        let! result = AiFunction.executeAsync stub cfg "some text"

        Assert.True(Result.isOk result)
    }

    // ── Post-condition retry then succeed ────────────────────────────────────

    [<Fact>]
    let ``post-condition retry - fails first then succeeds`` () = async {
        let cfg =
            baseCfg
            |> AiFunction.withPostCondition "concise" (fun s -> s.WordCount <= 5)
            |> AiFunction.withMaxAttempts 3

        let stub = StubLlmService([
            """{"Summary": "This is way too long of a summary", "WordCount": 50}"""
            """{"Summary": "Short", "WordCount": 1}"""
        ])
        let! result = AiFunction.executeAsync stub cfg "some text"

        match result with
        | Ok s -> Assert.Equal(1, s.WordCount)
        | Error e -> failwith $"Expected Ok on retry but got: {e}"
    }

    // ── Max attempts exhausted ───────────────────────────────────────────────

    [<Fact>]
    let ``max attempts exhausted when post-condition always fails`` () = async {
        let cfg =
            baseCfg
            |> AiFunction.withPostCondition "impossible" (fun _ -> false)
            |> AiFunction.withMaxAttempts 2

        let stub = StubLlmService([
            """{"Summary": "A", "WordCount": 1}"""
            """{"Summary": "B", "WordCount": 1}"""
        ])
        let! result = AiFunction.executeAsync stub cfg "text"

        match result with
        | Error (MaxAttemptsExhausted(attempts, _)) -> Assert.Equal(2, attempts)
        | other -> failwith $"Expected MaxAttemptsExhausted but got: {other}"
    }

    // ── Deserialization failure ──────────────────────────────────────────────

    [<Fact>]
    let ``deserialization failure on non-JSON`` () = async {
        let cfg = baseCfg |> AiFunction.withMaxAttempts 1

        let stub = StubLlmService(["This is not JSON at all"])
        let! result = AiFunction.executeAsync stub cfg "text"

        match result with
        | Error (MaxAttemptsExhausted _) -> () // deserialization failure wrapped
        | Error (DeserializationFailed _) -> ()
        | other -> failwith $"Expected deserialization error but got: {other}"
    }

    // ── Fenced JSON recovery ─────────────────────────────────────────────────

    [<Fact>]
    let ``fenced JSON code block is parsed correctly`` () = async {
        let fencedJson = "Here's the result:\n```json\n{\"Summary\": \"Fenced\", \"WordCount\": 1}\n```"
        let stub = StubLlmService([fencedJson])
        let! result = AiFunction.executeAsync stub baseCfg "text"

        match result with
        | Ok s -> Assert.Equal("Fenced", s.Summary)
        | Error e -> failwith $"Expected Ok but got: {e}"
    }

    // ── No post-conditions - any valid deser is Ok ──────────────────────────

    [<Fact>]
    let ``no post-conditions - any valid deserialization succeeds`` () = async {
        let cfg = baseCfg // no post-conditions added
        let stub = StubLlmService(["""{"Summary": "Anything", "WordCount": 999}"""])
        let! result = AiFunction.executeAsync stub cfg "text"

        Assert.True(Result.isOk result)
    }

    // ── JSON schema wiring ──────────────────────────────────────────────────

    [<Fact>]
    let ``json schema is set on LLM request when configured`` () = async {
        let schema = """{"type":"object","properties":{"Summary":{"type":"string"}}}"""
        let cfg = baseCfg |> AiFunction.withSchema schema

        let stub = StubLlmService(["""{"Summary": "Ok", "WordCount": 0}"""])
        let! _ = AiFunction.executeAsync stub cfg "text"

        let req = stub.Requests |> List.head
        match req.ResponseFormat with
        | Some (Constrained (JsonSchema s)) -> Assert.Equal(schema, s)
        | other -> failwith $"Expected Constrained JsonSchema but got: {other}"
    }

    // ── Feedback accumulates across retries ──────────────────────────────────

    [<Fact>]
    let ``feedback messages accumulate across retry attempts`` () = async {
        let cfg =
            baseCfg
            |> AiFunction.withPostCondition "positive" (fun s -> s.WordCount > 0)
            |> AiFunction.withMaxAttempts 3

        let stub = StubLlmService([
            """{"Summary": "Zero", "WordCount": 0}"""
            """{"Summary": "Still zero", "WordCount": 0}"""
            """{"Summary": "Fixed", "WordCount": 5}"""
        ])
        let! result = AiFunction.executeAsync stub cfg "text"

        Assert.True(Result.isOk result)
        // Third request should have 2 feedback messages (1 user + 2 feedback = 3 messages)
        let lastReq = stub.Requests |> List.last
        Assert.Equal(3, lastReq.Messages.Length)
    }

    // ── Pipeline-style config ────────────────────────────────────────────────

    [<Fact>]
    let ``pipeline config API works`` () =
        let cfg =
            AiFunction.create<string, TestSummary> "test" "prompt" id
            |> AiFunction.withPostCondition "not empty" (fun s -> s.Summary.Length > 0)
            |> AiFunction.withMaxAttempts 5
            |> AiFunction.withSchema """{"type":"object"}"""
            |> AiFunction.withModel "reasoning"
            |> AiFunction.withTemperature 0.7

        Assert.Equal("test", cfg.Name)
        Assert.Equal(5, cfg.MaxAttempts)
        Assert.Equal(Some """{"type":"object"}""", cfg.JsonSchema)
        Assert.Equal(Some "reasoning", cfg.ModelHint)
        Assert.Equal(Some 0.7, cfg.Temperature)
        Assert.Equal(1, cfg.PostConditions.Length)
