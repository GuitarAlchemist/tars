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

/// The MAF path runs TARS backends through `LlmServiceChatClient`: an agent hands
/// M.E.AI a `ChatOptions`, the adapter turns it back into an `LlmRequest`, and the
/// backend serialises whatever it finds there.
///
/// That reverse mapping had no tests, and was wrong in a way nothing could notice
/// at runtime: `ChatResponseFormat.ForJsonSchema` returns a *new*
/// `ChatResponseFormatJson` rather than the `ChatResponseFormat.Json` singleton, so
/// the old `obj.ReferenceEquals(options.ResponseFormat, ChatResponseFormat.Json)`
/// evaluated to false for every schema-constrained request. The agent asked for
/// structured output; the request went out with `JsonMode = false`, no schema, and
/// no error anywhere. These tests pin the mapping in both directions.

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

let private captureWith (options: ChatOptions) =
    let inner = CapturingLlmService()
    let client = new LlmServiceChatClient(inner) :> IChatClient
    let messages = [ ChatMessage(ChatRole.User, "hi") ] :> IEnumerable<ChatMessage>
    client.GetResponseAsync(messages, options, CancellationToken.None).GetAwaiter().GetResult() |> ignore
    Assert.True(inner.Captured.IsSome, "the adapter never reached the inner service")
    inner.Captured.Value

// ── ChatOptions -> LlmRequest (the MAF direction) ───────────────────────────

[<Fact>]
let ``a schema-constrained ChatOptions survives as a constrained LlmRequest`` () =
    let options = ChatOptions()
    options.ResponseFormat <- ChatResponseFormat.ForJsonSchema(schemaElement (), "s", "d")

    let req = captureWith options

    match req.ResponseFormat with
    | Some (ResponseFormat.Constrained (Grammar.JsonSchema recovered)) ->
        // Compare parsed, not textual: JsonElement round-trips without whitespace.
        Assert.Equal(
            JsonDocument.Parse(schemaText).RootElement.GetRawText(),
            JsonDocument.Parse(recovered).RootElement.GetRawText()
        )
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
    dict.[ChatClientMapping.GrammarKey] <- box "root ::= \"yes\" | \"no\""
    options.AdditionalProperties <- AdditionalPropertiesDictionary(dict)

    let req = captureWith options

    match req.ResponseFormat with
    | Some (ResponseFormat.Constrained (Grammar.Ebnf g)) -> Assert.Contains("root ::=", g)
    | other -> failwith $"expected an Ebnf constraint, got %A{other}"

    // EBNF is not JSON — flagging JsonMode here would make a backend request
    // `response_format: json_object` alongside a grammar that emits "yes".
    Assert.False(req.JsonMode, "an EBNF-constrained request was marked as JSON mode")

[<Fact>]
let ``a regex constraint carried in AdditionalProperties is recovered`` () =
    let options = ChatOptions()
    let dict = Dictionary<string, obj>()
    dict.[ChatClientMapping.RegexKey] <- box "^[0-9]{4}$"
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
        Assert.Contains("answer", json.Schema.Value.GetRawText())
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
[<Theory>]
[<InlineData("json")>]
[<InlineData("schema")>]
[<InlineData("ebnf")>]
[<InlineData("regex")>]
let ``every constraint survives a round trip through both adapters`` (kind: string) =
    let original =
        match kind with
        | "json" -> ResponseFormat.Json
        | "schema" -> ResponseFormat.Constrained(Grammar.JsonSchema schemaText)
        | "ebnf" -> ResponseFormat.Constrained(Grammar.Ebnf "root ::= \"a\"")
        | _ -> ResponseFormat.Constrained(Grammar.Regex "^a$")

    let recovered =
        { LlmRequest.Default with ResponseFormat = Some original }
        |> ChatClientMapping.toChatOptions
        |> ChatClientMapping.fromChatOptions

    match original, recovered with
    | ResponseFormat.Constrained (Grammar.JsonSchema a),
      Some (ResponseFormat.Constrained (Grammar.JsonSchema b)) ->
        Assert.Equal(
            JsonDocument.Parse(a).RootElement.GetRawText(),
            JsonDocument.Parse(b).RootElement.GetRawText()
        )
    | a, b -> Assert.Equal(Some a, b)
