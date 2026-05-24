namespace Tars.Cortex

open Tars.Llm

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
/// Result of a compression operation.
/// </summary>
type CompressionResult =
    { OriginalText: string
      CompressedText: string
      Strategy: CompressionStrategy
      OriginalLength: int
      CompressedLength: int
      CompressionRatio: float
      Entropy: float }

/// <summary>
/// Compresses context to manage token usage and entropy.
/// </summary>
type ContextCompressor(llm: ILlmService, entropyMonitor: EntropyMonitor) =

    /// <summary>
    /// Compresses the given context using the specified strategy.
    /// </summary>
    member this.Compress(context: string, strategy: CompressionStrategy) =
        task {
            let! result = this.CompressWithMetrics(context, strategy)
            return result.CompressedText
        }

    /// <summary>
    /// Compresses the context and returns detailed metrics.
    /// </summary>
    member this.CompressWithMetrics(context: string, strategy: CompressionStrategy) =
        task {
            let entropy = entropyMonitor.Measure(context)

            let prompt =
                match strategy with
                | Summarization ->
                    $"Summarize the following text, retaining all critical technical details and decision points:\n\n{context}"
                | KeyPointExtraction ->
                    $"Extract the key points from the following text as a bulleted list:\n\n{context}"
                | RemoveRedundancy ->
                    $"Rewrite the following text to be more concise, removing redundant information but keeping the meaning:\n\n{context}"

            let request =
                { ModelHint = Some "fast"
                  Model = None
                  SystemPrompt = None
                  MaxTokens = None
                  Temperature = Some 0.3
                  Stop = []
                  Messages = [ { Role = Role.User; Content = prompt } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None
                  ContextWindow = None }

            let! response = llm.CompleteAsync(request)
            let compressed = response.Text.Trim()

            return
                { OriginalText = context
                  CompressedText = compressed
                  Strategy = strategy
                  OriginalLength = context.Length
                  CompressedLength = compressed.Length
                  CompressionRatio = float compressed.Length / float (max 1 context.Length)
                  Entropy = entropy }
        }

    /// <summary>
    /// Auto-compresses context based on entropy and length.
    /// </summary>
    member this.AutoCompress(context: string) =
        task {
            let entropy = entropyMonitor.Measure(context)
            let length = context.Length

            // Heuristics for strategy selection
            // Low entropy (< 0.4) -> Repetitive -> Remove Redundancy
            // Medium entropy (0.4 - 0.7) -> Normal -> Summarize
            // High entropy (> 0.7) -> Dense -> Key Points (if long) or Keep (if short)

            if length < 500 then
                // Too short to compress meaningfully
                return context
            elif entropy < 0.4 then
                return! this.Compress(context, RemoveRedundancy)
            elif entropy < 0.7 then
                return! this.Compress(context, Summarization)
            else if
                // High entropy, only compress if very long
                length > 2000
            then
                return! this.Compress(context, KeyPointExtraction)
            else
                return context
        }
