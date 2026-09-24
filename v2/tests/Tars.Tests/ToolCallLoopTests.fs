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
