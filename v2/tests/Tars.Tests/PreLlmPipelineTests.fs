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

    [<Fact>]
    let ``SafetyFilter blocks dangerous keywords`` () =
        task {
            let stage = SafetyFilterStage() :> IPreLlmStage
            let ctx = PreLlmContext.Create("I want to run rm -rf /")

            let! result = stage.ExecuteAsync(ctx)

            Assert.False(result.IsSafe)
            Assert.Contains("dangerous keyword", result.BlockReason.Value)
        }

    [<Fact>]
    let ``IntentClassifier detects coding intent`` () =
        task {
            let stage = IntentClassifierStage() :> IPreLlmStage
            let ctx = PreLlmContext.Create("Please write a python script to sort a list")

            let! result = stage.ExecuteAsync(ctx)

            Assert.Equal(Some AgentIntent.Coding, result.Intent)
        }

    [<Fact>]
    let ``ContextSummarizer compresses long prompts`` () =
        task {
            let llm = StubLlm()
            let monitor = EntropyMonitor()
            let compressor = ContextCompressor(llm, monitor)
            let stage = ContextSummarizerStage(compressor) :> IPreLlmStage

            let longPrompt = String.replicate 200 "repeat " // > 1000 chars
            let ctx = PreLlmContext.Create(longPrompt)

            let! result = stage.ExecuteAsync(ctx)

            Assert.Equal("compressed", result.CurrentPrompt)
        }

    [<Fact>]
    let ``Pipeline runs stages sequentially`` () =
        task {
            let safety = SafetyFilterStage() :> IPreLlmStage
            let classifier = IntentClassifierStage() :> IPreLlmStage
            let pipeline = PreLlmPipeline([ safety; classifier ])

            let ctx = PreLlmContext.Create("write a safe function")
            let! result = pipeline.ExecuteAsync("write a safe function")

            Assert.True(result.IsSafe)
            Assert.Equal(Some AgentIntent.Coding, result.Intent)
        }
