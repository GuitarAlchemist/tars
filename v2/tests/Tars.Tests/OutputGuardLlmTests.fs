module Tars.Tests.OutputGuardLlmTests

open Xunit
open Tars.Core
open Tars.Cortex
open Tars.Llm
open Tars.Llm.LlmService

// Fake LLM service for testing analyzer behavior
type FakeLlmService(responseText: string) =
    interface ILlmServiceFunctional with
        member _.CompleteAsync(_req: LlmRequest) =
            async {
                return Ok { Text = responseText; FinishReason = Some "stop"; Usage = None; Raw = None }
            }
        member _.EmbedAsync(_text: string) =
            async {
                return Ok [||]
            }
        member _.RouteAsync(_req) =
            async {
                return Ok ({ Backend = Tars.Llm.LlmBackend.Ollama "mock"; Endpoint = System.Uri "http://localhost:11434"; ApiKey = None } : Tars.Llm.Routing.RoutedBackend)
            }

[<Fact>]
let ``LlmOutputGuardAnalyzer parses risk/action from JSON`` () =
    let fakeResponse = """{"risk":0.9,"action":"reject","reasons":["duplicate kernel layer","no citations"]}"""
    let service = FakeLlmService(fakeResponse) :> ILlmServiceFunctional
    let analyzer = LlmOutputGuardAnalyzer(service, modelHint = "cheap", temperature = 0.2)
    let guard = OutputGuard.withAnalyzer OutputGuard.defaultGuard (Some (analyzer :> IOutputGuardAnalyzer))

    let input : GuardInput =
        { ResponseText = """{"foo":1,"bar":2}"""
          Grammar = None
          ExpectedJsonFields = Some [ "foo"; "bar" ]
          RequireCitations = true
          Citations = Some []
          AllowExtraFields = false
          Metadata = Map.empty }

    let result = guard.Evaluate input |> Async.RunSynchronously

    match result.Action with
    | GuardAction.Reject _ ->
        Assert.True(result.Risk >= 0.9)
        Assert.Contains("duplicate kernel layer", result.Messages)
    | _ -> Assert.Fail($"Expected Reject, got {result.Action}")

[<Fact>]
let ``LlmOutputGuardAnalyzer returns None on bad JSON and falls back`` () =
    let fakeResponse = "not json"
    let service = FakeLlmService(fakeResponse) :> ILlmServiceFunctional
    let analyzer = LlmOutputGuardAnalyzer(service)
    let guard = OutputGuard.withAnalyzer OutputGuard.defaultGuard (Some (analyzer :> IOutputGuardAnalyzer))

    let input : GuardInput =
        { ResponseText = """{"foo":1,"bar":2}"""
          Grammar = None
          ExpectedJsonFields = Some [ "foo"; "bar" ]
          RequireCitations = false
          Citations = None
          AllowExtraFields = false
          Metadata = Map.empty }

    let result = guard.Evaluate input |> Async.RunSynchronously

    // Base guard should accept (no missing fields, no citations required)
    Assert.Equal(GuardAction.Accept, result.Action)
    Assert.True(result.Risk < 0.3)

[<Fact>]
let ``Factory createFromEnv returns analyzer`` () =
    // Set env for the duration of the test
    System.Environment.SetEnvironmentVariable("OLLAMA_BASE_URL", "http://localhost:11434/")
    System.Environment.SetEnvironmentVariable("DEFAULT_OLLAMA_MODEL", "llama3.1:latest")

    let analyzer = OutputGuardAnalyzerFactory.createFromEnv ()
    Assert.NotNull(analyzer)
