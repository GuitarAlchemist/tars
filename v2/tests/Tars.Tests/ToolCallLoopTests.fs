module Tars.Tests.ToolCallLoopTests

open System
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.AI
open Xunit
open Tars.Llm
open Tars.Cortex

// #305: the tool pipeline was inert in both directions — `ChatOptions.Tools` never
// reached the provider, and a provider's `tool_calls` never reached the invoking
// client. These tests drive the whole loop offline, through a recorded Ollama reply.

/// The tool the model is offered, built exactly as `MafToolAdapter` builds one.
let private calls = ResizeArray<string>()

let private weatherTool () =
    let run =
        Func<string, string>(fun (city: string) ->
            calls.Add city
            $"It is sunny in {city}.")

    AIFunctionFactory.Create(run, "GetWeather", "Report the weather in a city.")

/// Answers with a recorded tool call first, then with prose, and keeps every request
/// it was given so the return leg can be inspected.
type private ScriptedService(replies: LlmResponse list) =
    let mutable remaining = replies
    let seen = ResizeArray<LlmRequest>()

    member _.Requests = List.ofSeq seen

    interface ILlmService with
        member _.CompleteAsync(req) =
            task {
                seen.Add req

                match remaining with
                | next :: rest ->
                    remaining <- rest
                    return next
                | [] ->
                    return
                        { Text = "nothing left to say"
                          FinishReason = Some "stop"
                          Usage = None
                          Raw = None }
            }

        member _.CompleteStreamAsync(_req, _onToken) = raise (NotImplementedException())
        member _.EmbedAsync(_text) = Task.FromResult(Array.empty<float32>)

        member _.RouteAsync(_) =
            task {
                return
                    { Backend = Ollama "mock"
                      Endpoint = Uri "http://localhost:11434"
                      ApiKey = None }
            }

/// An Ollama reply that asks for a tool: no call id, arguments as an object.
let private ollamaToolCall =
    """{"model":"qwen3-coder:30b","message":{"role":"assistant","content":"",
       "tool_calls":[{"function":{"name":"GetWeather","arguments":{"city":"Montreal"}}}]},"done":true}"""

/// An OpenAI-shaped reply that asks for the same tool: call id, arguments as a string.
let private openAiToolCall =
    """{"choices":[{"index":0,"message":{"role":"assistant","content":null,
       "tool_calls":[{"id":"call_abc","type":"function",
                      "function":{"name":"GetWeather","arguments":"{\"city\":\"Montreal\"}"}}]},
       "finish_reason":"tool_calls"}]}"""

/// Serialize as the client does, where an absent field is absent rather than null.
let private onTheWire (dto: obj) =
    Text.Json.JsonSerializer.Serialize(
        dto,
        Text.Json.JsonSerializerOptions(
            DefaultIgnoreCondition = Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        )
    )

let private asking (raw: string) =
    { Text = ""
      FinishReason = Some "stop"
      Usage = None
      Raw = Some raw }

let private answering (text: string) =
    { Text = text
      FinishReason = Some "stop"
      Usage = None
      Raw = Some """{"message":{"role":"assistant","content":"done"},"done":true}""" }

[<Fact>]
let ``both wire shapes are read as the same call`` () =
    for raw in [ ollamaToolCall; openAiToolCall ] do
        match ChatClientMapping.toolCallsOf (Some raw) with
        | [ call ] ->
            Assert.Equal("GetWeather", call.Name)
            Assert.Equal(box "Montreal", call.Arguments.["city"])
            Assert.False(String.IsNullOrWhiteSpace call.CallId)
        | other -> failwith $"expected one call, got {List.length other}"

[<Fact>]
let ``a reply with no calls, or no raw body, asks for nothing`` () =
    Assert.Empty(ChatClientMapping.toolCallsOf None)
    Assert.Empty(ChatClientMapping.toolCallsOf (Some """{"message":{"content":"hello"}}"""))
    Assert.Empty(ChatClientMapping.toolCallsOf (Some "not json at all"))

[<Fact>]
let ``the offered tools reach the request`` () =
    let options = ToolAwareChatClient.optionsWithTools [ weatherTool () ]

    match ChatClientMapping.toolsOf options with
    | [ tool ] ->
        let serialized = Text.Json.JsonSerializer.Serialize tool
        Assert.Contains("\"type\":\"function\"", serialized)
        Assert.Contains("GetWeather", serialized)
        Assert.Contains("city", serialized) // the schema the provider needs, not just a name
    | other -> failwith $"expected one tool, got {List.length other}"

[<Fact>]
let ``a model can ask for a tool, have it run, and answer with the result`` () =
    calls.Clear()

    let service = ScriptedService([ asking ollamaToolCall; answering "It is sunny in Montreal." ])
    let client = ToolAwareChatClient.build (service :> ILlmService)

    let options =
        ToolAwareChatClient.optionsWithTools [ weatherTool () ]

    let messages = List<ChatMessage>([ ChatMessage(ChatRole.User, "What is the weather in Montreal?") ])

    let response =
        client.GetResponseAsync(messages, options)
        |> Async.AwaitTask
        |> Async.RunSynchronously

    // The tool actually ran, with the argument the model named.
    Assert.Equal<string>("Montreal", String.concat "," calls)
    Assert.Contains("sunny in Montreal", response.Text)

    // And the second request carried the result back as a tool turn, answering the
    // call by id — the leg that had nowhere to go before.
    match service.Requests with
    | [ first; second ] ->
        Assert.NotEmpty(first.Tools)

        // The assistant turn that asked for the tool travels back too: a result that
        // answers nothing is rejected by OpenAI-shaped endpoints.
        let asked =
            second.Messages
            |> List.tryPick (fun m ->
                match m.Role with
                | Role.AssistantCalling calls -> Some calls
                | _ -> None)

        match asked with
        | Some [ call ] ->
            Assert.Equal("GetWeather", call.Name)
            Assert.Contains("Montreal", call.ArgumentsJson)
        | _ -> failwith "the assistant's own call never made it into the next request"

        let toolTurn =
            second.Messages
            |> List.tryFind (fun m ->
                match m.Role with
                | Role.Tool _ -> true
                | _ -> false)

        match toolTurn with
        | Some turn ->
            Assert.Contains("sunny in Montreal", turn.Content)

            match turn.Role with
            | Role.Tool callId -> Assert.False(String.IsNullOrWhiteSpace callId)
            | _ -> failwith "unreachable"
        | None -> failwith "the tool result never made it into the next request"
    | other -> failwith $"expected two requests, got {List.length other}"

[<Fact>]
let ``an OpenAI-shaped request offers the tools and keeps the exchange answerable`` () =
    // The return leg alone is not enough: without `tools` on the way out, no
    // OpenAI-shaped endpoint would ever produce the calls this adapter parses.
    let call =
        { Id = "call_abc"
          Name = "GetWeather"
          ArgumentsJson = """{"city":"Montreal"}""" }

    let request =
        { LlmRequest.Default with
            Messages =
                [ { Role = Role.User; Content = "weather?" }
                  { Role = Role.AssistantCalling [ call ]; Content = "" }
                  { Role = Role.Tool "call_abc"; Content = "It is sunny in Montreal." } ]
            Tools = ChatClientMapping.toolsOf (ToolAwareChatClient.optionsWithTools [ weatherTool () ]) }

    let dto = OpenAiCompatibleClient.buildRequestDto false "gpt-test" false request
    let json = onTheWire dto

    Assert.Contains("\"tools\":", json)
    Assert.Contains("GetWeather", json)
    Assert.Contains("\"tool_calls\":", json)
    Assert.Contains("\"tool_call_id\":\"call_abc\"", json)

[<Fact>]
let ``a request with no tools offers none, rather than an empty list`` () =
    let dto =
        OpenAiCompatibleClient.buildRequestDto false "gpt-test" false
            { LlmRequest.Default with Messages = [ { Role = Role.User; Content = "hello" } ] }

    Assert.True(dto.tools.IsNone)
    Assert.DoesNotContain("tools", onTheWire dto)

// --------------------------------------------------------------- the streaming leg

/// Streams the tokens it was given, then finishes with the reply it was given.
type private StreamingService(tokens: string list, final: LlmResponse, ?failWith: exn) =
    let seen = ResizeArray<LlmRequest>()

    member _.Requests = List.ofSeq seen

    interface ILlmService with
        member _.CompleteAsync(req) =
            seen.Add req

            match failWith with
            | Some error -> Task.FromException<LlmResponse>(error)
            | None -> Task.FromResult final

        member _.CompleteStreamAsync(req, onToken) =
            seen.Add req

            task {
                match failWith with
                | Some error -> return raise error
                | None ->
                    for token in tokens do
                        onToken token

                    return final
            }

        member _.EmbedAsync(_text) = Task.FromResult(Array.empty<float32>)

        member _.RouteAsync(_) =
            task {
                return
                    { Backend = Ollama "mock"
                      Endpoint = Uri "http://localhost:11434"
                      ApiKey = None }
            }

let private drain (client: IChatClient) (options: ChatOptions) =
    task {
        let updates = ResizeArray<ChatResponseUpdate>()
        let messages = List<ChatMessage>([ ChatMessage(ChatRole.User, "go") ])
        let stream = client.GetStreamingResponseAsync(messages, options, Threading.CancellationToken.None)
        let enumerator = stream.GetAsyncEnumerator(Threading.CancellationToken.None)

        let mutable go = true

        while go do
            let! moved = enumerator.MoveNextAsync()

            if moved then
                // Reading Current twice must hand back the same update, not eat one.
                Assert.Same(enumerator.Current, enumerator.Current)
                updates.Add enumerator.Current
            else
                go <- false

        return List.ofSeq updates
    }

[<Fact>]
let ``streamed tokens all arrive, and the reply closes the stream`` () =
    let service =
        StreamingService([ "he"; "ll"; "o" ], answering "hello") :> ILlmService

    let updates =
        drain (new LlmServiceChatClient(service) :> IChatClient) (ChatOptions())
        |> Async.AwaitTask
        |> Async.RunSynchronously

    let text = updates |> List.collect (fun u -> List.ofSeq u.Contents) |> List.choose (fun c ->
        match c with
        | :? TextContent as t -> Some t.Text
        | _ -> None)

    Assert.Equal("hello", String.concat "" text)
    Assert.Contains(updates, fun u -> u.FinishReason.HasValue)

[<Fact>]
let ``a streaming caller offering tools still gets the call`` () =
    // Providers put tool calls in the final message, not in the token stream, so this
    // path takes the buffered reply rather than losing them.
    let service = StreamingService([], asking ollamaToolCall) :> ILlmService

    let options = ToolAwareChatClient.optionsWithTools [ weatherTool () ]

    let updates =
        drain (new LlmServiceChatClient(service) :> IChatClient) options
        |> Async.AwaitTask
        |> Async.RunSynchronously

    let calls =
        updates
        |> List.collect (fun u -> List.ofSeq u.Contents)
        |> List.choose (fun c ->
            match c with
            | :? FunctionCallContent as call -> Some call.Name
            | _ -> None)

    Assert.Equal<string>("GetWeather", String.concat "," calls)

[<Fact>]
let ``a provider failure ends the stream loudly, not quietly`` () =
    // It used to end the stream as though the provider had simply finished, so a
    // caller read a successful, empty answer out of a failed call.
    let service =
        StreamingService([], answering "unused", InvalidOperationException "provider exploded") :> ILlmService

    let failure =
        Assert.ThrowsAny<exn>(fun () ->
            drain (new LlmServiceChatClient(service) :> IChatClient) (ChatOptions())
            |> Async.AwaitTask
            |> Async.RunSynchronously
            |> ignore)

    // However it is wrapped on the way out, the provider's own failure is in there.
    let rec messages (error: exn) =
        match error with
        | null -> []
        | :? AggregateException as aggregate ->
            error.Message :: (aggregate.InnerExceptions |> Seq.collect messages |> List.ofSeq)
        | _ -> error.Message :: messages error.InnerException

    Assert.Contains(messages failure, fun (m: string) -> m.Contains "provider exploded")
