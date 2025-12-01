namespace Tars.Tests

open Xunit
open Tars.Cortex
open Tars.Core
open Tars.Llm.LlmService

type CompressionStubLlm(responseText: string) =
    interface ILlmService with
        member _.CompleteAsync(_req) =
            task {
                return
                    { Text = responseText
                      FinishReason = Some "stop"
                      Usage = None
                      Raw = None }
            }

        member this.CompleteStreamAsync(req, _onToken) = (this :> ILlmService).CompleteAsync(req)

        member _.EmbedAsync(_text) = task { return [| 0.1f |] }

type ContextCompressionTests() =

    [<Fact>]
    member _.``Compress returns LLM output``() =
        let llm = CompressionStubLlm("compressed") :> ILlmService
        let compressor = ContextCompressor(llm, EntropyMonitor())
        let result =
            compressor.Compress("some long text", CompressionStrategy.Summarization)
            |> Async.AwaitTask
            |> Async.RunSynchronously
        Assert.Equal("compressed", result)

    [<Fact>]
    member _.``AutoCompress only compresses when entropy low``() =
        let llm = CompressionStubLlm("shorter") :> ILlmService
        let monitor = EntropyMonitor()
        let compressor = ContextCompressor(llm, monitor)
        let repetitive = "repeat repeat repeat repeat"
        let varied = "alpha beta gamma delta epsilon"

        let lowEntropy =
            compressor.AutoCompress(repetitive, 0.9)
            |> Async.AwaitTask
            |> Async.RunSynchronously
        Assert.Equal("shorter", lowEntropy)

        let highEntropy =
            compressor.AutoCompress(varied, 0.1)
            |> Async.AwaitTask
            |> Async.RunSynchronously
        Assert.Equal(varied, highEntropy)
