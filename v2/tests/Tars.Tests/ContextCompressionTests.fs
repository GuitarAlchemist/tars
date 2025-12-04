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

        member this.CompleteStreamAsync(req, _onToken) =
            (this :> ILlmService).CompleteAsync(req)

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
        let repetitive = String.replicate 100 "repeat " // > 500 chars, low entropy
        // Generate high entropy string with unique tokens
        let varied = [ 1..100 ] |> List.map (fun i -> $"word{i}") |> String.concat " " // 100 unique words, entropy = 1.0

        let lowEntropy =
            compressor.AutoCompress(repetitive) |> Async.AwaitTask |> Async.RunSynchronously

        Assert.Equal("shorter", lowEntropy)

        let highEntropy =
            compressor.AutoCompress(varied) |> Async.AwaitTask |> Async.RunSynchronously

        Assert.Equal(varied, highEntropy)
