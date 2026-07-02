module Tars.Tests.AskCommandTests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Llm
open Tars.Interface.Cli

// These tests exercise the `ask` command's logic through the ITarsRuntime seam.
// Before TarsRuntime, a command built its own LlmFactory/ToolRegistry inline and
// could only be tested by running the CLI; the stub runtime below is the second,
// real adapter that makes the seam worth its keep.

/// Minimal ILlmService stub returning a fixed completion.
type private StubLlm(text: string, finishReason: string option) =
    let response: LlmResponse =
        { Text = text
          FinishReason = finishReason
          Usage = None
          Raw = None }

    interface ILlmService with
        member _.CompleteAsync(_) = Task.FromResult response
        member _.CompleteStreamAsync(_, _) = Task.FromResult response
        member _.EmbedAsync(_) = Task.FromResult([| 0.0f |])

        member _.RouteAsync(_) =
            Task.FromResult(
                ({ Backend = Tars.Llm.LlmBackend.Ollama "stub"
                   Endpoint = Uri "http://localhost:11434"
                   ApiKey = None }
                : Tars.Llm.Routing.RoutedBackend)
            )

/// Empty tool registry stub — the `ask` command does not use tools.
type private StubTools() =
    interface IToolRegistry with
        member _.Register(_) = ()
        member _.Get(_) = None
        member _.GetAll() = []

type private StubSkills() =
    interface ISkillRegistry with
        member _.GetSkill(_) = None
        member _.GetSkillsByDomain(_) = []
        member _.GetAllSkills() = []

/// Test runtime: stub LLM + stub tools, real config defaults.
let private stubRuntime (llm: ILlmService) : ITarsRuntime =
    { new ITarsRuntime with
        member _.Config = ConfigurationLoader.load ()
        member _.Tools = StubTools() :> IToolRegistry
        member _.Skills = StubSkills() :> ISkillRegistry
        member _.Llm _ = llm }

[<Fact>]
let ``ask returns 0 on a successful completion`` () =
    let rt = stubRuntime (StubLlm("the answer", Some "stop"))
    let code = (Commands.Ask.run rt "what is 2+2?").Result
    Assert.Equal(0, code)

[<Fact>]
let ``ask returns 1 when the response is a parse error`` () =
    let rt = stubRuntime (StubLlm("", Some "parse_error"))
    let code = (Commands.Ask.run rt "bad input").Result
    Assert.Equal(1, code)
