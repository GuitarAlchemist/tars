namespace Tars.Cortex

open System
open Microsoft.ML.Tokenizers
open Tars.Llm

/// <summary>
/// Interface for token counting services.
/// </summary>
type ITokenCounter =
    /// <summary>
    /// Count tokens in a raw string.
    /// </summary>
    abstract member Count: text: string -> int

    /// <summary>
    /// Count tokens in a list of messages, accounting for protocol overhead.
    /// </summary>
    abstract member CountMessages: messages: LlmMessage list -> int

/// <summary>
/// A token counter using Microsoft.ML.Tokenizers (Tiktoken).
/// Default to cl100k_base (GPT-4) behavior.
/// </summary>
type TiktokenCounter(?modelName: string) =
    let tokenizer =
        try
            match modelName with
            | Some m -> TiktokenTokenizer.CreateForModel(m)
            | None -> TiktokenTokenizer.CreateForModel("gpt-4")
        with _ ->
            // Fallback if model not found, usually shouldn't happen for standard models
            TiktokenTokenizer.CreateForModel("gpt-3.5-turbo")

    interface ITokenCounter with
        member this.Count(text: string) =
            if String.IsNullOrEmpty(text) then
                0
            else
                tokenizer.EncodeToIds(text).Count

        member this.CountMessages(messages: LlmMessage list) =
            // OpenAI approximation:
            // ~3 tokens per message for overhead (<|start|>, role, <|end|>)
            // + tokens in content
            // + 3 tokens for the assistant reply prime

            let mutable count = 0

            for msg in messages do
                count <- count + 3 // Per message overhead
                // Role handling (users/assistant are 1 token usually)
                // Content
                count <- count + (this :> ITokenCounter).Count(msg.Content)

            count + 3 // Reply prime
