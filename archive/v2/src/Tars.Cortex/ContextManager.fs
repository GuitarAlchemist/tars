namespace Tars.Cortex

open System
open System.Threading.Tasks
open Tars.Llm

/// <summary>
/// Strategy for context management.
/// </summary>
type ContextStrategy =
    /// <summary>
    /// Basic sliding window: keep most recent messages that fit.
    /// Can preserve System prompt.
    /// </summary>
    | SlidingWindow of MaxTokens: int * PreserveSystem: bool

    /// <summary>
    /// Summarize older messages when limit is reached.
    /// Keeps last N messages raw, summarizes previous ones.
    /// </summary>
    | Summarization of MaxTokens: int * PreserveLastMessages: int

/// <summary>
/// Service to manage context window fitting.
/// </summary>
type ContextManager(tokenCounter: ITokenCounter, compressor: ContextCompressor) =

    /// <summary>
    /// Fit messages to the strategy.
    /// </summary>
    member this.FitMessages(messages: LlmMessage list, strategy: ContextStrategy) =
        task {
            match strategy with
            | SlidingWindow(max, preserve) -> return this.ApplySlidingWindow(messages, max, preserve)
            | Summarization(max, preserveLast) -> return! this.ApplySummarization(messages, max, preserveLast)
        }

    member private this.ApplySlidingWindow(messages: LlmMessage list, maxTokens: int, preserveSystem: bool) =
        let total = tokenCounter.CountMessages(messages)

        if total <= maxTokens then
            messages
        else
            // Filter Logic
            // Always keep System if preserveSystem is true
            let systemMsg, others =
                if preserveSystem then
                    match messages with
                    | head :: tail when head.Role = Role.System -> Some head, tail
                    | _ -> None, messages
                else
                    None, messages

            let systemTokens =
                systemMsg
                |> Option.map (fun m -> tokenCounter.CountMessages([ m ]))
                |> Option.defaultValue 0

            let available = maxTokens - systemTokens

            if available <= 0 then
                // Corner case: System prompt consumes all context. Return just system or nothing.
                systemMsg |> Option.toList
            else
                // Take from end backwards
                let reversed = List.rev others
                let mutable currentTokens = 0
                let mutable kept = []

                for msg in reversed do
                    let count = tokenCounter.CountMessages([ msg ])

                    if currentTokens + count <= available then
                        kept <- msg :: kept
                        currentTokens <- currentTokens + count
                // else stop (drop older)

                match systemMsg with
                | Some sys -> sys :: kept
                | None -> kept

    member private this.ApplySummarization(messages: LlmMessage list, maxTokens: int, preserveLast: int) =
        task {
            let total = tokenCounter.CountMessages(messages)

            if total <= maxTokens then
                return messages
            else
                // Identify messages to summarize
                // Keep System
                let systemMsg, contentMessages =
                    match messages with
                    | head :: tail when head.Role = Role.System -> Some head, tail
                    | _ -> None, messages

                // Keep last N
                let splitIndex = max 0 (contentMessages.Length - preserveLast)
                let toSummarize = contentMessages |> List.take splitIndex
                let toKeep = contentMessages |> List.skip splitIndex

                if toSummarize.IsEmpty then
                    // Fallback to sliding window if nothing to summarize (everything is in preserveLast)
                    return this.ApplySlidingWindow(messages, maxTokens, true)
                else
                    // Summarize
                    let textToSummarize =
                        toSummarize
                        |> List.map (fun m -> $"{m.Role}: {m.Content}")
                        |> String.concat "\n"

                    // Compress using existing tool
                    let! summary = compressor.Compress(textToSummarize, CompressionStrategy.Summarization)

                    let summaryMsg =
                        { Role = Role.System
                          Content = $"Previous conversation summary:\n{summary}" }

                    let newMessages =
                        match systemMsg with
                        | Some sys -> sys :: summaryMsg :: toKeep
                        | None -> summaryMsg :: toKeep

                    // Recursively check if it fits?
                    // For now assume summary fits. If not, apply sliding window on result.
                    let resultCount = tokenCounter.CountMessages(newMessages)

                    if resultCount > maxTokens then
                        return this.ApplySlidingWindow(newMessages, maxTokens, true)
                    else
                        return newMessages
        }
