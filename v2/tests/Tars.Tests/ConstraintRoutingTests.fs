namespace Tars.Tests

open System
open System.Text.Json
open Xunit
open Tars.Llm
open Tars.Llm.Routing

/// Item 4 slice B — routing must know whether the backend it picked can actually
/// enforce the requested constraint, and must say so out loud when it cannot.
/// Before this, `Routing.fs` never inspected `ResponseFormat` at all: an EBNF
/// grammar routed to Ollama was discarded server-side with nothing logged.
module ConstraintRoutingTests =

    let private req (fmt: ResponseFormat option) : LlmRequest =
        { ModelHint = None
          Model = None
          SystemPrompt = None
          MaxTokens = None
          Temperature = None
          Stop = []
          Messages = [ { Role = Role.User; Content = "hi" } ]
          Tools = []
          ToolChoice = None
          ResponseFormat = fmt
          Stream = false
          JsonMode = true
          Seed = None
          ContextWindow = None }

    /// Ollama-only: PreferredProvider forces localRoute onto Ollama regardless of
    /// the vLLM base URI, which is the shape of a default dev deployment.
    let private ollamaOnly =
        { RoutingConfig.Default with PreferredProvider = "Ollama" }

    // ── ConstraintNeed classification ─────────────────────────────────────────

    [<Fact>]
    let ``ofRequest classifies each grammar kind`` () =
        Assert.Equal(NeedsJsonSchema, ConstraintNeed.ofRequest (req (Some(Constrained(JsonSchema "{}")))))
        Assert.Equal(NeedsGrammar, ConstraintNeed.ofRequest (req (Some(Constrained(Ebnf "root ::= digit")))))
        // Regex is its own need: vLLM enforces it, llama.cpp does not.
        Assert.Equal(NeedsRegex, ConstraintNeed.ofRequest (req (Some(Constrained(Regex "[0-9]+")))))
        Assert.Equal(NoNeed, ConstraintNeed.ofRequest (req (Some ResponseFormat.Json)))
        Assert.Equal(NoNeed, ConstraintNeed.ofRequest (req None))

    [<Fact>]
    let ``supports routes by capability not provider name`` () =
        // JSON schema: everything on the OpenAI wire plus Ollama's schema-aware format.
        Assert.True(ConstraintNeed.supports (Ollama "m") NeedsJsonSchema)
        Assert.True(ConstraintNeed.supports (Vllm "m") NeedsJsonSchema)
        Assert.True(ConstraintNeed.supports (LlamaCpp("m", None)) NeedsJsonSchema)
        // Anthropic/Gemini degrade to prompt hints — they enforce nothing.
        Assert.False(ConstraintNeed.supports (Anthropic "m") NeedsJsonSchema)
        Assert.False(ConstraintNeed.supports (GoogleGemini "m") NeedsJsonSchema)

        // Raw grammars need a real grammar engine.
        Assert.True(ConstraintNeed.supports (Vllm "m") NeedsGrammar)
        Assert.True(ConstraintNeed.supports (LlamaCpp("m", None)) NeedsGrammar)
        Assert.False(ConstraintNeed.supports (Ollama "m") NeedsGrammar)
        Assert.False(ConstraintNeed.supports (OpenAI "m") NeedsGrammar)

        // Regex is NOT the same capability: LlamaCppClient maps Regex to nothing, so
        // folding it in with grammars would claim support it lacks and, worse, suppress
        // the downgrade warning for a constraint that silently vanishes.
        Assert.True(ConstraintNeed.supports (Vllm "m") NeedsRegex)
        Assert.False(ConstraintNeed.supports (LlamaCpp("m", None)) NeedsRegex)

        // NoNeed is always satisfiable, including on backends that enforce nothing.
        Assert.True(ConstraintNeed.supports (Anthropic "m") NoNeed)

    // ── Downgrade reporting ───────────────────────────────────────────────────

    [<Fact>]
    let ``Constrained JsonSchema on Ollama does not downgrade`` () =
        let chosen = chooseBackendWithConstraints ollamaOnly (req (Some(Constrained(JsonSchema "{}"))))
        Assert.Equal(None, chosen.Downgrade)

    [<Fact>]
    let ``Constrained Ebnf on Ollama downgrades and names the grammar`` () =
        let chosen =
            chooseBackendWithConstraints ollamaOnly (req (Some(Constrained(Ebnf "root ::= 'x'"))))

        match chosen.Downgrade with
        | Some d ->
            Assert.Equal("ebnf", d.RequestedGrammar)
            Assert.Equal("Ollama", d.Backend)
        | None -> failwith "expected a downgrade — Ollama has no raw GBNF API"

    [<Fact>]
    let ``unconstrained requests never downgrade`` () =
        Assert.Equal(None, (chooseBackendWithConstraints ollamaOnly (req None)).Downgrade)
        Assert.Equal(None, (chooseBackendWithConstraints ollamaOnly (req (Some ResponseFormat.Json))).Downgrade)

    [<Fact>]
    let ``downgrade is logged loudly, every time`` () =
        let captured = ResizeArray<string>()

        try
            ConstraintDowngradeLog.setSink captured.Add

            // Same request twice: both must warn. Silent-after-first would be the
            // same defect in a new costume.
            for _ in 1..2 do
                ConstraintDowngradeLog.routeAndWarn ollamaOnly (req (Some(Constrained(Ebnf "root ::= 'x'"))))
                |> ignore

            Assert.Equal(2, captured.Count)

            for msg in captured do
                Assert.Contains("CONSTRAINT DOWNGRADE", msg)
                Assert.Contains("ebnf", msg)
                Assert.Contains("Ollama", msg)
        finally
            ConstraintDowngradeLog.resetSink ()

    [<Fact>]
    let ``no warning is emitted when the backend can enforce the constraint`` () =
        let captured = ResizeArray<string>()

        try
            ConstraintDowngradeLog.setSink captured.Add

            ConstraintDowngradeLog.routeAndWarn ollamaOnly (req (Some(Constrained(JsonSchema "{}"))))
            |> ignore

            Assert.Empty(captured)
        finally
            ConstraintDowngradeLog.resetSink ()

    // ── Wire format ───────────────────────────────────────────────────────────
    // Nothing in the suite asserted the serialized constraint shape before this.
    // The old vLLM payload was a nested `{"extra_body":{"guided_decoding":...}}`,
    // which vLLM's server never read — it routed correctly and enforced nothing.

    let private serializedBody (vllmExtensions: bool) (fmt: ResponseFormat option) =
        let dto =
            OpenAiCompatibleClient.buildRequestDto vllmExtensions "test-model" false (req fmt)

        JsonSerializer.Serialize(dto, OpenAiCompatibleClient.jsonOptions)

    [<Fact>]
    let ``vLLM emits top-level structured_outputs and never extra_body`` () =
        // Quote-free grammar on purpose: System.Text.Json's default encoder escapes
        // apostrophes to ', so a GBNF literal like root ::= 'x' is correct on the
        // wire but unsearchable in the raw string.
        let body = serializedBody true (Some(Constrained(Ebnf "root ::= digit")))
        // CamelCase policy only lowercases the first character, so a snake_case
        // record field reaches the wire unchanged.
        Assert.Contains("structured_outputs", body)
        Assert.Contains("grammar", body)
        Assert.Contains("root ::= digit", body)
        Assert.DoesNotContain("extra_body", body)
        Assert.DoesNotContain("guided_decoding", body)

    [<Fact>]
    let ``OpenAI-targeted requests carry no vLLM-only parameters`` () =
        // OpenAI proper 400s on unknown top-level params, so the gate must hold.
        let body = serializedBody false (Some(Constrained(Ebnf "root ::= 'x'")))
        Assert.DoesNotContain("structured_outputs", body)
        Assert.DoesNotContain("extra_body", body)
        Assert.DoesNotContain("guided_decoding", body)

    [<Fact>]
    let ``JsonSchema still travels as response_format json_schema`` () =
        let body = serializedBody false (Some(Constrained(JsonSchema """{"type":"object"}""")))
        Assert.Contains("json_schema", body)
