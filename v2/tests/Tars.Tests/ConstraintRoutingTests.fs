namespace Tars.Tests

open System
open System.Text.Json
open System.Threading.Tasks
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

    /// One constraint, one spelling. The first cut sent the schema twice on the
    /// vLLM path — once as `structured_outputs.json`, once as `response_format`.
    /// That was harmless only because response_format takes precedence, which is
    /// a server-side detail no caller should depend on.
    [<Fact>]
    let ``vLLM sends a JSON schema once, via response_format only`` () =
        let body = serializedBody true (Some(Constrained(JsonSchema """{"type":"object"}""")))
        Assert.Contains("json_schema", body)
        Assert.DoesNotContain("structured_outputs", body)

    /// Grammars and regexes have no response_format spelling, so for those
    /// structured_outputs must still be the carrier.
    [<Fact>]
    let ``vLLM still sends grammar and regex via structured_outputs`` () =
        let ebnf = serializedBody true (Some(Constrained(Ebnf "root ::= digit")))
        Assert.Contains("structured_outputs", ebnf)
        Assert.Contains("grammar", ebnf)

        let rx = serializedBody true (Some(Constrained(Regex "[0-9]+")))
        Assert.Contains("structured_outputs", rx)
        Assert.Contains("regex", rx)

    // ── sink scoping and query-vs-execution ─────────────────────────────────

    /// The sink used to be a process-global `mutable`. xUnit runs distinct test
    /// collections in parallel, so one test's redirect could swallow another's
    /// warnings and lose its own. AsyncLocal confines the override to the context
    /// that set it: a nested context may reassign it without the parent seeing
    /// either the reassignment or the warnings it captures.
    [<Fact>]
    let ``a sink redirect does not leak out of the context that set it`` () =
        task {
            let outer = ResizeArray<string>()
            let inner = ResizeArray<string>()

            try
                ConstraintDowngradeLog.setSink outer.Add

                do!
                    Task.Run(fun () ->
                        ConstraintDowngradeLog.setSink inner.Add

                        ConstraintDowngradeLog.routeAndWarn ollamaOnly (req (Some(Constrained(Ebnf "root ::= digit"))))
                        |> ignore)

                ConstraintDowngradeLog.routeAndWarn ollamaOnly (req (Some(Constrained(Ebnf "root ::= digit"))))
                |> ignore

                // With a global mutable these would read 2 and 0: the nested
                // assignment would still be in force for the outer warning.
                Assert.Equal(1, inner.Count)
                Assert.Equal(1, outer.Count)
            finally
                ConstraintDowngradeLog.resetSink ()
        }

    /// RouteAsync answers "which backend would serve this?" without sending
    /// anything. Warning there means a caller that asks and then completes —
    /// CliReasoner does exactly that — gets two warnings for one request.
    [<Fact>]
    let ``RouteAsync does not warn because it executes nothing`` () =
        task {
            let captured = ResizeArray<string>()

            try
                ConstraintDowngradeLog.setSink captured.Add
                use http = new System.Net.Http.HttpClient()
                let svc = LlmService.DefaultLlmService(http, { Routing = ollamaOnly }) :> ILlmService
                let! _ = svc.RouteAsync(req (Some(Constrained(Ebnf "root ::= digit"))))
                Assert.Empty(captured)
            finally
                ConstraintDowngradeLog.resetSink ()
        }
