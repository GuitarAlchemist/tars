namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Cortex
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService

module PreLlmPipelineTests =

    type StubLlm() =
        interface ILlmService with
            member _.CompleteAsync(_) =
                Task.FromResult(
                    { Text = "compressed"
                      FinishReason = None
                      Usage = None
                      Raw = None }
                )

            member _.CompleteStreamAsync(_, _) =
                Task.FromResult(
                    { Text = "compressed"
                      FinishReason = None
                      Usage = None
                      Raw = None }
                )

            member _.EmbedAsync(_) = Task.FromResult([| 0.1f |])
            member _.RouteAsync(_req) = Task.FromResult(({ Backend = Tars.Llm.LlmBackend.Ollama "mock"; Endpoint = Uri "http://localhost:11434"; ApiKey = None } : Tars.Llm.Routing.RoutedBackend))

    type StubIntentClassifier(result: AgentDomain option) =
        interface IIntentClassifier with
            member _.ClassifyAsync(_) = Task.FromResult(result)

    [<Fact>]
    let ``SafetyFilter blocks dangerous keywords`` () =
        task {
            let stage = SafetyFilterStage() :> IPreLlmStage
            let ctx = PreLlmContext.Create("I want to run rm -rf /")

            let! result = stage.ExecuteAsync(ctx)

            Assert.False(result.IsSafe)
            Assert.Contains("Destructive command", result.BlockReason.Value)
        }

    [<Fact>]
    let ``IntentClassifier detects coding intent`` () =
        task {
            let stage =
                IntentClassifierStage(StubIntentClassifier(Some AgentDomain.Coding)) :> IPreLlmStage
            let ctx = PreLlmContext.Create("Please write a python script to sort a list")

            let! result = stage.ExecuteAsync(ctx)

            Assert.Equal(Some AgentDomain.Coding, result.Intent)
        }

    [<Fact>]
    let ``ContextSummarizer compresses a prompt too long to send as is`` () =
        task {
            let llm = StubLlm()
            let monitor = EntropyMonitor()
            let compressor = ContextCompressor(llm, monitor)
            let stage = ContextSummarizerStage(compressor, 1000) :> IPreLlmStage

            let longPrompt = String.replicate 200 "repeat " // 1,400 bytes, over the 1,000 allowed
            let ctx = PreLlmContext.Create(longPrompt)

            let! result = stage.ExecuteAsync(ctx)

            Assert.Equal("compressed", result.CurrentPrompt)
        }

    [<Fact>]
    let ``ContextSummarizer sends a prompt that fits unchanged`` () =
        task {
            // In evolve, every task prompt over 500 characters was summarized, and the
            // summary of one came back as "Understood. I will follow the instructions".
            let llm = StubLlm()
            let compressor = ContextCompressor(llm, EntropyMonitor())
            let stage = ContextSummarizerStage(compressor, 24576) :> IPreLlmStage

            let taskPrompt = String.replicate 200 "repeat "

            let! result = stage.ExecuteAsync(PreLlmContext.Create(taskPrompt))

            Assert.Equal(taskPrompt, result.CurrentPrompt)
        }

    [<Fact>]
    let ``ContextSummarizer measures the prompt in bytes, not characters`` () =
        task {
            // 600 characters, 1,200 UTF-8 bytes: it may need more tokens than 1,000.
            let llm = StubLlm()
            let compressor = ContextCompressor(llm, EntropyMonitor())
            let stage = ContextSummarizerStage(compressor, 1000) :> IPreLlmStage

            let! result = stage.ExecuteAsync(PreLlmContext.Create(String.replicate 300 "語 "))

            Assert.Equal("compressed", result.CurrentPrompt)
        }

    [<Fact>]
    let ``Pipeline runs stages sequentially`` () =
        task {
            let safety = SafetyFilterStage() :> IPreLlmStage
            let classifier =
                IntentClassifierStage(StubIntentClassifier(Some AgentDomain.Coding)) :> IPreLlmStage
            let pipeline = PreLlmPipeline([ safety; classifier ])

            let ctx = PreLlmContext.Create("write a safe function")
            let! result = pipeline.ExecuteAsync("write a safe function")

            Assert.True(result.IsSafe)
            Assert.Equal(Some AgentDomain.Coding, result.Intent)
        }
