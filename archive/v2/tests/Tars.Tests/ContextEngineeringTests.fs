module Tars.Tests.ContextEngineeringTests

open Xunit
open FsUnit
open Tars.Cortex
open Tars.Llm
open Tars.Core
open System.Threading.Tasks

type StubTokenCounter() =
    interface ITokenCounter with
        member _.Count text = text.Length / 4

        member _.CountMessages msgs =
            msgs |> List.sumBy (fun m -> (m.Content.Length / 4) + 3)

type StubLlmService() =
    interface ILlmService with
        member _.CompleteAsync req =
            Task.FromResult
                { Text = "Short summary."
                  FinishReason = Some "stop"
                  Usage = None
                  Raw = None }

        member _.EmbedAsync _ = Task.FromResult [||]

        member _.CompleteStreamAsync(_, _) =
            Task.FromResult
                { Text = ""
                  FinishReason = None
                  Usage = None
                  Raw = None }

        member _.RouteAsync _ =
            Task.FromResult
                { Backend = LlmBackend.OpenAI "test"
                  Endpoint = System.Uri("http://localhost")
                  ApiKey = None }

[<Fact>]
let ``SlidingWindow keeps most recent messages fit within limit`` () =
    let tokenCounter = StubTokenCounter()
    let manager = ContextManager(tokenCounter, Unchecked.defaultof<ContextCompressor>)

    // "Message X" is 9 chars -> 2 tokens. +3 overhead = 5 tokens per msg.
    let msgs =
        [ for i in 1..10 do
              yield
                  { Role = Role.User
                    Content = $"Message {i}" } ]

    // Max 15 tokens -> should keep 3 messages (5 * 3 = 15)
    let strategy = ContextStrategy.SlidingWindow(15, false)
    let result = manager.FitMessages(msgs, strategy).Result

    result.Length |> should equal 3
    result.[0].Content |> should equal "Message 8"
    result.[2].Content |> should equal "Message 10"

[<Fact>]
let ``SlidingWindow preserves System prompt`` () =
    let tokenCounter = StubTokenCounter()
    let manager = ContextManager(tokenCounter, Unchecked.defaultof<ContextCompressor>)

    let msgs =
        [ yield
              { Role = Role.System
                Content = "System" } // 6 chars -> 1 token + 3 = 4 tokens
          for i in 1..10 do
              yield
                  { Role = Role.User
                    Content = $"Message {i}" } ] // 5 tokens each

    // Max 14 tokens.
    // System takes 4. Available 10.
    // Should keep 2 user messages.

    let strategy = ContextStrategy.SlidingWindow(14, true)
    let result = manager.FitMessages(msgs, strategy).Result

    result.Length |> should equal 3 // System + 2 user
    result.[0].Role |> should equal Role.System
    result.[2].Content |> should equal "Message 10"

[<Fact>]
let ``SlidingWindow returns empty if max tokens too small`` () =
    let tokenCounter = StubTokenCounter()
    let manager = ContextManager(tokenCounter, Unchecked.defaultof<ContextCompressor>)

    let msgs =
        [ { Role = Role.User
            Content = "Long Message" } ] // 12 chars -> 3 + 3 = 6 tokens

    let strategy = ContextStrategy.SlidingWindow(5, false)
    let result = manager.FitMessages(msgs, strategy).Result

    result.IsEmpty |> should be True

[<Fact>]
let ``Summarization strategy compresses older messages`` () =
    let tokenCounter = StubTokenCounter()
    let llm = StubLlmService()
    let entropyMonitor = EntropyMonitor()
    let compressor = ContextCompressor(llm, entropyMonitor)
    let manager = ContextManager(tokenCounter, compressor)

    // Setup: System (4 tokens) + 10 messages (50 tokens) = 54 tokens total.
    let msgs =
        [ yield
              { Role = Role.System
                Content = "System" }
          for i in 1..10 do
              yield
                  { Role = Role.User
                    Content = $"Message {i}" } ]

    // Strategy: Max 20 tokens. Keep last 2 raw (10 tokens).
    // System (4) + Summary + kept(2).
    // To summarize: Message 1..8.
    // StubLLM returns "Summary: " + prompt.
    // Prompt is "Summarize... \n\nSystem: ..."
    // The summary might be long, so checking strict content is hard, but we check structure.

    let strategy = ContextStrategy.Summarization(30, 2)
    let result = manager.FitMessages(msgs, strategy).Result

    // With max 20, we expect:
    // System (4)
    // Summary (?)
    // Message 9 (5)
    // Message 10 (5)
    // If Summary is small enough.

    // Check minimal structure
    (result.Length > 2) |> should equal true
    result.[0].Role |> should equal Role.System
    // Second message should be summary (Role System)
    result.[1].Role |> should equal Role.System
    result.[1].Content |> should startWith "Previous conversation summary"

    // Last ones should be preserved
    result.[result.Length - 1].Content |> should equal "Message 10"
