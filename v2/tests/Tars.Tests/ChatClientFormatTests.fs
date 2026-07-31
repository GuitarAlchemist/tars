module Tars.Tests.ChatClientFormatTests

open System
open System.Collections.Generic
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.AI
open Xunit
open Tars.Llm
open Tars.Llm.Routing

/// `LlmServiceChatClient` exposes a TARS backend as an M.E.AI `IChatClient`: a
/// caller hands it a `ChatOptions`, the adapter turns it back into an `LlmRequest`,
/// and the backend serialises whatever it finds there.
///
/// Scope, so these tests are not read as more than they are: the adapter pair has
/// no production callers today. `LlmServiceChatClient` is reached only through
/// `ToolAwareChatClient`, whose sole reference outside its own file is a comment in
/// `Agent.fs` — the live `agent run` path routes tools through `WoTExecutor`
/// instead. So this is protective coverage on infrastructure that is wired but not
/// yet used, not a guard on a live request path.
///
/// It still matters, because the mapping was wrong in a way nothing could notice at
/// runtime: `ChatResponseFormat.ForJsonSchema` returns a *new*
/// `ChatResponseFormatJson` rather than the `ChatResponseFormat.Json` singleton, so
/// the old `obj.ReferenceEquals(options.ResponseFormat, ChatResponseFormat.Json)`
/// evaluated to false for every schema-constrained request. Anything wiring this
/// adapter up would have gotten silently unconstrained requests.

/// Captures the LlmRequest the adapter builds, so the ChatOptions -> LlmRequest
/// direction can be asserted without a backend.
type private CapturingLlmService() =
    member val Captured: LlmRequest option = None with get, set

    interface ILlmService with
        member this.CompleteAsync(req: LlmRequest) : Task<LlmResponse> =
            this.Captured <- Some req

            Task.FromResult
                { Text = "{}"
                  FinishReason = Some "stop"
                  Usage = None
                  Raw = None }

        member this.CompleteStreamAsync(req: LlmRequest, onToken: string -> unit) : Task<LlmResponse> =
            this.Captured <- Some req
            onToken "{}"

            Task.FromResult
                { Text = "{}"
                  FinishReason = Some "stop"
                  Usage = None
                  Raw = None }

        member _.EmbedAsync(_: string) : Task<float32[]> = Task.FromResult [||]

        member _.RouteAsync(_: LlmRequest) : Task<RoutedBackend> =
            Task.FromResult
                { Backend = Ollama "stub"
                  Endpoint = Uri("http://localhost:11434")
                  ApiKey = None }

let private schemaText = """{"type":"object","properties":{"answer":{"type":"string"}}}"""

let private schemaElement () =
    JsonDocument.Parse(schemaText).RootElement

/// `GetRawText()` returns the original span verbatim — whitespace included — so
/// comparing two schemas by raw text is a whitespace-sensitive string compare that
/// happens to pass only while both sides travel an identical path. Re-serializing
/// normalizes, which is what a schema comparison should have been doing.
let private normalizeJson (json: string) =
    JsonSerializer.Serialize(JsonDocument.Parse(json).RootElement)

let private captureWith (options: ChatOptions) =
    let inner = CapturingLlmService()
    let client = new LlmServiceChatClient(inner) :> IChatClient
    let messages = [ ChatMessage(ChatRole.User, "hi") ] :> IEnumerable<ChatMessage>
    client.GetResponseAsync(messages, options, CancellationToken.None).GetAwaiter().GetResult() |> ignore
    Assert.True(inner.Captured.IsSome, "the adapter never reached the inner service")
    inner.Captured.Value

/// Same, through the streaming entry point. The streaming path builds its request
/// separately from the non-streaming one, so it can drift independently — and it
/// maps the options lazily, inside GetAsyncEnumerator, so nothing is captured until
/// the sequence is actually enumerated.
let private captureStreaming (options: ChatOptions) =
    let inner = CapturingLlmService()
    let client = new LlmServiceChatClient(inner) :> IChatClient
    let messages = [ ChatMessage(ChatRole.User, "hi") ] :> IEnumerable<ChatMessage>
    let updates = client.GetStreamingResponseAsync(messages, options, CancellationToken.None)
    let e = updates.GetAsyncEnumerator(CancellationToken.None)

    try
        e.MoveNextAsync().AsTask().GetAwaiter().GetResult() |> ignore
    finally
        e.DisposeAsync().AsTask().GetAwaiter().GetResult()

    Assert.True(inner.Captured.IsSome, "the streaming adapter never reached the inner service")
    inner.Captured.Value

// ── ChatOptions -> LlmRequest (the MAF direction) ───────────────────────────

[<Fact>]
let ``a schema-constrained ChatOptions survives as a constrained LlmRequest`` () =
    let options = ChatOptions()
    options.ResponseFormat <- ChatResponseFormat.ForJsonSchema(schemaElement (), "s", "d")

    let req = captureWith options

    match req.ResponseFormat with
    | Some (ResponseFormat.Constrained (Grammar.JsonSchema recovered)) ->
        Assert.Equal(normalizeJson schemaText, normalizeJson recovered)
    | other -> failwith $"expected a JsonSchema constraint, got %A{other}"

    // The legacy flag must agree with the format, or backends reading it disagree
    // with backends reading ResponseFormat.
    Assert.True(req.JsonMode, "a schema-constrained request was not marked as JSON mode")

[<Fact>]
let ``plain JSON mode survives without inventing a schema`` () =
    let options = ChatOptions()
    options.ResponseFormat <- ChatResponseFormat.Json

    let req = captureWith options

    Assert.Equal(Some ResponseFormat.Json, req.ResponseFormat)
    Assert.True(req.JsonMode)

[<Fact>]
let ``text format does not turn into JSON mode`` () =
    let options = ChatOptions()
    options.ResponseFormat <- ChatResponseFormat.Text

    let req = captureWith options

    Assert.Equal(Some ResponseFormat.Text, req.ResponseFormat)
    Assert.False(req.JsonMode, "a plain-text request was marked as JSON mode")

[<Fact>]
let ``an EBNF grammar carried in AdditionalProperties is recovered`` () =
    let options = ChatOptions()
    let dict = Dictionary<string, obj>()
    // The literal, not the constant: using `ChatClientMapping.GrammarKey` here would
    // make the test tautological — rename the constant and the wire shape changes
    // while the test still passes.
    dict.["structured_outputs_grammar"] <- box "root ::= \"yes\" | \"no\""
    options.AdditionalProperties <- AdditionalPropertiesDictionary(dict)

    let req = captureWith options

    match req.ResponseFormat with
    | Some (ResponseFormat.Constrained (Grammar.Ebnf g)) -> Assert.Contains("root ::=", g)
    | other -> failwith $"expected an Ebnf constraint, got %A{other}"

    // EBNF is not JSON. This does NOT stop a backend emitting JSON mode for a
    // grammar — Ollama coerces `Constrained _` to "json" from ResponseFormat alone
    // (OllamaClient.fs:186), which is the declared downgrade policy in Routing.fs.
    // It only pins that the two fields never state different things.
    Assert.False(req.JsonMode, "an EBNF-constrained request was marked as JSON mode")

[<Fact>]
let ``a regex constraint carried in AdditionalProperties is recovered`` () =
    let options = ChatOptions()
    let dict = Dictionary<string, obj>()
    dict.["structured_outputs_regex"] <- box "^[0-9]{4}$"
    options.AdditionalProperties <- AdditionalPropertiesDictionary(dict)

    let req = captureWith options

    match req.ResponseFormat with
    | Some (ResponseFormat.Constrained (Grammar.Regex p)) -> Assert.Equal("^[0-9]{4}$", p)
    | other -> failwith $"expected a Regex constraint, got %A{other}"

[<Fact>]
let ``options with no format leave the request untouched`` () =
    let req = captureWith (ChatOptions())

    Assert.True(req.ResponseFormat.IsNone)
    Assert.False(req.JsonMode)

// ── LlmRequest -> ChatOptions (the provider direction) ──────────────────────

[<Fact>]
let ``a JSON schema goes out on the channel providers enforce`` () =
    let req =
        { LlmRequest.Default with
            ResponseFormat = Some(ResponseFormat.Constrained(Grammar.JsonSchema schemaText)) }

    let opts = ChatClientMapping.toChatOptions req

    // The point of the fix: not the bare `Json` singleton, but a format actually
    // carrying the schema. AdditionalProperties is not a channel stock M.E.AI
    // providers forward, so a schema hidden there is a schema never applied.
    match box opts.ResponseFormat with
    | :? ChatResponseFormatJson as json ->
        Assert.True(json.Schema.HasValue, "the schema was dropped on the way to the provider")
        // The whole schema, not merely a substring of it: `Contains` would pass on a
        // wrapped or mangled schema that still mentioned the property name.
        Assert.Equal(normalizeJson schemaText, normalizeJson (json.Schema.Value.GetRawText()))
    | other -> failwith $"expected ChatResponseFormatJson, got %A{other}"

[<Fact>]
let ``an unparseable schema degrades to JSON mode instead of throwing`` () =
    let req =
        { LlmRequest.Default with
            ResponseFormat = Some(ResponseFormat.Constrained(Grammar.JsonSchema "{ not json")) }

    let opts = ChatClientMapping.toChatOptions req

    match box opts.ResponseFormat with
    | :? ChatResponseFormatJson as json -> Assert.False(json.Schema.HasValue)
    | other -> failwith $"expected ChatResponseFormatJson, got %A{other}"

// ── round trip ──────────────────────────────────────────────────────────────

/// The two adapters are inverses in principle; nothing checked that they were in
/// practice, and they were not.
///
/// `jsonMode` is varied deliberately. `toChatOptions` consults `req.JsonMode` in its
/// fall-through arm, so a format with no arm of its own is silently rewritten by it
/// — which is exactly how `Text` used to invert into `Json`. A round trip that only
/// ever passes `JsonMode = false` cannot see that class of bug.
[<Theory>]
[<InlineData("json", true)>]
[<InlineData("json", false)>]
[<InlineData("text", true)>]
[<InlineData("text", false)>]
[<InlineData("schema", true)>]
[<InlineData("schema", false)>]
[<InlineData("ebnf", true)>]
[<InlineData("ebnf", false)>]
[<InlineData("regex", true)>]
[<InlineData("regex", false)>]
let ``every constraint survives a round trip regardless of JsonMode`` (kind: string, jsonMode: bool) =
    let original =
        match kind with
        | "json" -> ResponseFormat.Json
        | "text" -> ResponseFormat.Text
        | "schema" -> ResponseFormat.Constrained(Grammar.JsonSchema schemaText)
        | "ebnf" -> ResponseFormat.Constrained(Grammar.Ebnf "root ::= \"a\"")
        | _ -> ResponseFormat.Constrained(Grammar.Regex "^a$")

    let recovered =
        { LlmRequest.Default with
            ResponseFormat = Some original
            JsonMode = jsonMode }
        |> ChatClientMapping.toChatOptions
        |> ChatClientMapping.fromChatOptions

    match original, recovered with
    | ResponseFormat.Constrained (Grammar.JsonSchema a),
      Some (ResponseFormat.Constrained (Grammar.JsonSchema b)) ->
        Assert.Equal(normalizeJson a, normalizeJson b)
    | a, b -> Assert.Equal(Some a, b)

/// The specific inversion: an explicit request for prose, with the legacy flag also
/// set, must not come back demanding JSON.
[<Fact>]
let ``Text with JsonMode set does not invert into JSON`` () =
    let opts =
        { LlmRequest.Default with
            ResponseFormat = Some ResponseFormat.Text
            JsonMode = true }
        |> ChatClientMapping.toChatOptions

    Assert.Equal(Some ResponseFormat.Text, ChatClientMapping.fromChatOptions opts)

    // And it must reach the wire as text: OpenAiCompatibleClient maps Text -> no
    // response_format, but Json -> {"type":"json_object"}.
    match box opts.ResponseFormat with
    | :? ChatResponseFormatText -> ()
    | other -> failwith $"expected ChatResponseFormatText, got %A{other}"

/// JsonMode with no ResponseFormat is the one case where the legacy flag is still
/// load-bearing — every backend consults it in its `None` branch.
[<Fact>]
let ``JsonMode alone still produces JSON mode`` () =
    let opts =
        { LlmRequest.Default with
            ResponseFormat = None
            JsonMode = true }
        |> ChatClientMapping.toChatOptions

    Assert.Equal(Some ResponseFormat.Json, ChatClientMapping.fromChatOptions opts)

// ── the streaming path maps formats independently ───────────────────────────

/// Streaming builds its own request, so "constraints survive streaming" needs its
/// own guard — gutting that mapping is invisible to every non-streaming test.
[<Theory>]
[<InlineData("schema")>]
[<InlineData("json")>]
[<InlineData("text")>]
[<InlineData("ebnf")>]
let ``constraints survive the streaming path too`` (kind: string) =
    let options = ChatOptions()

    match kind with
    | "schema" -> options.ResponseFormat <- ChatResponseFormat.ForJsonSchema(schemaElement (), "s", "d")
    | "json" -> options.ResponseFormat <- ChatResponseFormat.Json
    | "text" -> options.ResponseFormat <- ChatResponseFormat.Text
    | _ ->
        let dict = Dictionary<string, obj>()
        dict.["structured_outputs_grammar"] <- box "root ::= \"a\""
        options.AdditionalProperties <- AdditionalPropertiesDictionary(dict)

    let streamed = (captureStreaming options).ResponseFormat
    let direct = (captureWith options).ResponseFormat

    Assert.True(streamed.IsSome, "the streaming path dropped the format entirely")

    // Pinned against the non-streaming path rather than a literal, so the two
    // cannot drift apart without a test noticing.
    Assert.Equal(direct, streamed)

/// Documents what streaming still drops, so a future reader can tell "deliberate"
/// from "forgotten". If these ever start surviving, this test should be deleted.
[<Fact>]
let ``streaming still drops stop sequences and seed`` () =
    let options = ChatOptions()
    options.StopSequences <- ResizeArray [ "STOP" ]
    options.Seed <- Nullable 42L

    let req = captureStreaming options

    Assert.Empty(req.Stop)
    Assert.True(req.Seed.IsNone)

    // The non-streaming path does carry them — that asymmetry is the point.
    let direct = captureWith options
    Assert.True(([ "STOP" ] = direct.Stop), "the non-streaming path dropped stop sequences too")
    Assert.Equal(Some 42, direct.Seed)

// ── precedence when both channels are populated ─────────────────────────────

/// `toChatOptions` never emits both, so this only arises from an external producer
/// — e.g. a caller using M.E.AI's own `GetResponseAsync<T>` (which sets
/// ForJsonSchema) on options that already carry a grammar key. The explicit typed
/// schema must win; silently discarding it is how a caller ends up unconstrained
/// without being told.
[<Fact>]
let ``an explicit schema outranks a grammar carried alongside it`` () =
    let options = ChatOptions()
    options.ResponseFormat <- ChatResponseFormat.ForJsonSchema(schemaElement (), "s", "d")
    let dict = Dictionary<string, obj>()
    dict.["structured_outputs_grammar"] <- box "root ::= \"a\""
    options.AdditionalProperties <- AdditionalPropertiesDictionary(dict)

    match ChatClientMapping.fromChatOptions options with
    | Some (ResponseFormat.Constrained (Grammar.JsonSchema _)) -> ()
    | other -> failwith $"the caller's explicit schema was discarded, got %A{other}"

/// But a bare `Json` is weaker than a grammar — it says "some JSON", the grammar
/// says exactly what. Letting bare JSON win would discard the stronger constraint.
[<Fact>]
let ``a grammar outranks a bare JSON response format`` () =
    let options = ChatOptions()
    options.ResponseFormat <- ChatResponseFormat.Json
    let dict = Dictionary<string, obj>()
    dict.["structured_outputs_grammar"] <- box "root ::= \"a\""
    options.AdditionalProperties <- AdditionalPropertiesDictionary(dict)

    match ChatClientMapping.fromChatOptions options with
    | Some (ResponseFormat.Constrained (Grammar.Ebnf _)) -> ()
    | other -> failwith $"expected the grammar to win, got %A{other}"

// ── malformed input must not escape as an exception ─────────────────────────

/// `carried` only accepts a non-blank string. A grammar key holding anything else
/// must fall through to the typed channel rather than being read as a constraint —
/// AdditionalProperties is `obj`-valued, and a JSON round trip through MCP or A2A
/// delivers a JsonElement under that key, not a string.
[<Theory>]
[<InlineData(0)>]
[<InlineData(1)>]
[<InlineData(2)>]
let ``a non-string or blank grammar key is ignored`` (variant: int) =
    let options = ChatOptions()
    options.ResponseFormat <- ChatResponseFormat.Json
    let dict = Dictionary<string, obj>()

    dict.["structured_outputs_grammar"] <-
        match variant with
        | 0 -> box 42
        | 1 -> box (JsonDocument.Parse("\"root ::= \\\"a\\\"\"").RootElement)
        | _ -> box "   "

    options.AdditionalProperties <- AdditionalPropertiesDictionary(dict)

    Assert.Equal(Some ResponseFormat.Json, ChatClientMapping.fromChatOptions options)

/// M.E.AI accepts `ForJsonSchema(default)` with no validation, yielding
/// `Schema.HasValue = true` with `ValueKind = Undefined`, on which `GetRawText()`
/// throws. An adapter must not turn that into an exception out of GetResponseAsync.
[<Fact>]
let ``an Undefined schema degrades instead of throwing`` () =
    let options = ChatOptions()
    options.ResponseFormat <- ChatResponseFormat.ForJsonSchema(Unchecked.defaultof<JsonElement>, "s", "d")

    Assert.Equal(Some ResponseFormat.Json, ChatClientMapping.fromChatOptions options)

/// A JSON schema must be an object. These all parse, so a parse-only guard lets
/// them through and the provider 400s instead of degrading.
[<Theory>]
[<InlineData("null")>]
[<InlineData("42")>]
[<InlineData("[1,2]")>]
[<InlineData("\"hello\"")>]
let ``a non-object schema degrades to plain JSON mode`` (schema: string) =
    let opts =
        { LlmRequest.Default with
            ResponseFormat = Some(ResponseFormat.Constrained(Grammar.JsonSchema schema)) }
        |> ChatClientMapping.toChatOptions

    match box opts.ResponseFormat with
    | :? ChatResponseFormatJson as json ->
        Assert.False(json.Schema.HasValue, $"'{schema}' was accepted as a JSON schema")
    | other -> failwith $"expected ChatResponseFormatJson, got %A{other}"
