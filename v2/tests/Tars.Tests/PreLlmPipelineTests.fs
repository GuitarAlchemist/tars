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
    let ``ContextSummarizer blocks a summary still too long to send`` () =
        task {
            // The stub's summary is 10 bytes; 5 are allowed.
            let compressor = ContextCompressor(StubLlm(), EntropyMonitor())
            let stage = ContextSummarizerStage(compressor, 5) :> IPreLlmStage

            let! result = stage.ExecuteAsync(PreLlmContext.Create(String.replicate 200 "repeat "))

            Assert.False(result.IsSafe)
            Assert.Contains("does not fit", result.BlockReason |> Option.defaultValue "")
        }

    [<Fact>]
    let ``ContextSummarizer blocks a prompt the compressor leaves too long`` () =
        task {
            // Every word distinct, under 2,000 characters: the compressor returns it unchanged.
            let prompt = String.Join(" ", [ for i in 1..150 -> $"word{i}" ])
            let compressor = ContextCompressor(StubLlm(), EntropyMonitor())
            let stage = ContextSummarizerStage(compressor, 500) :> IPreLlmStage

            let! result = stage.ExecuteAsync(PreLlmContext.Create(prompt))

            Assert.True(prompt.Length > 500 && prompt.Length < 2000)
            Assert.False(result.IsSafe)
        }

    [<Fact>]
    let ``PromptLimit reserves the agent's preamble and answer`` () =
        // The evolve executor's preamble measured 10,447 bytes, and its largest task
        // prompt 4,830 bytes. At 16k such a prompt is sent as is.
        let limit = ContextSummarizerStage.PromptLimit(16384, 10447, 1024)
        Assert.True(limit >= 4830, $"limit {limit}")
        Assert.True(limit + 10447 / 3 + 1024 <= 16384)

        // At 4096, the preamble and the answer leave no room for a task prompt.
        Assert.Equal(0, ContextSummarizerStage.PromptLimit(4096, 10447, 1024))

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
