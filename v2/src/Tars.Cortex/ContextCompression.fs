namespace Tars.Cortex

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService

/// <summary>
/// Strategies for compressing context.
/// </summary>
type CompressionStrategy =
    /// <summary>Summarize the text, retaining key details.</summary>
    | Summarization
    /// <summary>Extract bullet points of key information.</summary>
    | KeyPointExtraction
    /// <summary>Rewrite to be more concise.</summary>
    | RemoveRedundancy

/// <summary>
/// Compresses context to manage token usage and entropy.
/// </summary>
type ContextCompressor(llm: ILlmService, entropyMonitor: EntropyMonitor) =

    /// <summary>
    /// Compresses the given context using the specified strategy.
    /// </summary>
    member this.Compress(context: string, strategy: CompressionStrategy) =
        task {
            // Check entropy to see if compression is worthwhile
            let entropy = entropyMonitor.Measure(context)

            // If entropy is very high (already dense), aggressive compression might lose info.
            // If entropy is low (repetitive), compression is highly effective.

            // For now, we proceed with compression regardless, but log the entropy if we had a logger.

            let prompt =
                match strategy with
                | Summarization ->
                    $"Summarize the following text, retaining all critical technical details and decision points:\n\n{context}"
                | KeyPointExtraction ->
                    $"Extract the key points from the following text as a bulleted list:\n\n{context}"
                | RemoveRedundancy ->
                    $"Rewrite the following text to be more concise, removing redundant information but keeping the meaning:\n\n{context}"

            let request =
                { ModelHint = Some "fast" // Prefer a fast model for compression
                  MaxTokens = None
                  Temperature = Some 0.3 // Low temperature for deterministic summary
                  Messages = [ { Role = Role.User; Content = prompt } ] }

            let! response = llm.CompleteAsync(request)
            return response.Text
        }

    /// <summary>
    /// Auto-compresses context if entropy is below a threshold (indicating redundancy).
    /// Returns the original context if no compression is needed.
    /// </summary>
    member this.AutoCompress(context: string, entropyThreshold: float) =
        task {
            let entropy = entropyMonitor.Measure(context)

            if entropy < entropyThreshold then
                return! this.Compress(context, RemoveRedundancy)
            else
                return context
        }
