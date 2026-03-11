namespace Tars.Cortex

/// Builds an IChatClient pipeline that wraps an ILlmService with automatic
/// tool invocation via FunctionInvokingChatClient.
///
/// This module is dependency-free of Tars.Tools (avoids circular reference).
/// Callers provide AITool instances directly — use MafToolAdapter.toAITools
/// at the call site to convert from the tool registry.
///
/// Usage:
///   let tools = MafToolAdapter.toAITools registry
///   let client = ToolAwareChatClient.build llmService
///   let opts = ToolAwareChatClient.optionsWithTools tools
///   let resp = client.GetResponseAsync(messages, opts) |> Async.AwaitTask

open Microsoft.Extensions.AI
open Tars.Llm

module ToolAwareChatClient =

    /// Build a tool-aware IChatClient from an ILlmService.
    /// The returned client handles tool call loops automatically.
    let build (llmService: ILlmService) : IChatClient =
        let innerClient = new LlmServiceChatClient(llmService) :> IChatClient
        ChatClientBuilder(innerClient)
            .UseFunctionInvocation()
            .Build()

    /// Build a tool-aware IChatClient with custom max iterations per request.
    let buildWithOptions (llmService: ILlmService) (maxIterations: int) : IChatClient =
        let innerClient = new LlmServiceChatClient(llmService) :> IChatClient
        let client = new FunctionInvokingChatClient(innerClient)
        client.MaximumIterationsPerRequest <- maxIterations
        client :> IChatClient

    /// Create ChatOptions populated with the given AITool list.
    let optionsWithTools (tools: AITool list) : ChatOptions =
        let opts = ChatOptions()
        opts.Tools <- System.Collections.Generic.List<AITool>(tools |> List.toSeq)
        opts

    /// Convenience: send a single prompt through the tool-aware pipeline.
    let completeWithTools
        (llmService: ILlmService)
        (tools: AITool list)
        (systemPrompt: string option)
        (userPrompt: string)
        : System.Threading.Tasks.Task<ChatResponse> =
        let client = build llmService
        let messages = System.Collections.Generic.List<ChatMessage>()
        systemPrompt |> Option.iter (fun sp ->
            messages.Add(ChatMessage(ChatRole.System, sp)))
        messages.Add(ChatMessage(ChatRole.User, userPrompt))
        let opts = optionsWithTools tools
        client.GetResponseAsync(messages, opts)
