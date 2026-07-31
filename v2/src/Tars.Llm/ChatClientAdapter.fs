namespace Tars.Llm

// Bidirectional adapters between TARS ILlmService and Microsoft.Extensions.AI IChatClient.
// - LlmServiceChatClient: wraps ILlmService as an IChatClient (for new MAF code)
// - ChatClientLlmService: wraps IChatClient as an ILlmService (for existing TARS code)

open System
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.AI
open Tars.Llm
open Tars.Llm.Routing

// ─────────────────────────────────────────────────────────────────────
// Mapping helpers
// ─────────────────────────────────────────────────────────────────────
module internal ChatClientMapping =

    /// Map TARS Role to M.E.AI ChatRole
    let toAIRole (role: Role) : ChatRole =
        match role with
        | Role.System -> ChatRole.System
        | Role.User -> ChatRole.User
        | Role.Assistant -> ChatRole.Assistant

    /// Map M.E.AI ChatRole to TARS Role
    let fromAIRole (role: ChatRole) : Role =
        if role = ChatRole.System then Role.System
        elif role = ChatRole.Assistant then Role.Assistant
        else Role.User

    /// Convert a TARS LlmMessage to an M.E.AI ChatMessage
    let toAIChatMessage (msg: LlmMessage) : ChatMessage =
        ChatMessage(toAIRole msg.Role, msg.Content)

    /// Convert an M.E.AI ChatMessage to a TARS LlmMessage
    let fromAIChatMessage (msg: ChatMessage) : LlmMessage =
        { Role = fromAIRole msg.Role
          Content = msg.Text |> Option.ofObj |> Option.defaultValue "" }

    /// Wire names for the constraints M.E.AI has no first-class channel for.
    /// EBNF and regex have no `ChatResponseFormat` equivalent, so they ride in
    /// `AdditionalProperties` under the same names `OpenAiCompatibleClient` uses.
    [<Literal>]
    let GrammarKey = "structured_outputs_grammar"

    [<Literal>]
    let RegexKey = "structured_outputs_regex"

    let private tryParseSchema (schema: string) =
        try
            Some(System.Text.Json.JsonDocument.Parse(schema).RootElement)
        with _ ->
            None

    /// Build M.E.AI ChatOptions from an LlmRequest
    let toChatOptions (req: LlmRequest) : ChatOptions =
        let opts = ChatOptions()
        req.Temperature |> Option.iter (fun t -> opts.Temperature <- Nullable(float32 t))
        req.MaxTokens |> Option.iter (fun m -> opts.MaxOutputTokens <- Nullable m)
        req.Model |> Option.iter (fun m -> opts.ModelId <- m)
        req.Seed |> Option.iter (fun s -> opts.Seed <- Nullable(int64 s))

        if not req.Stop.IsEmpty then
            opts.StopSequences <- req.Stop |> ResizeArray

        let carry key (value: string) =
            let dict = Dictionary<string, obj>()
            dict.[key] <- box value
            opts.AdditionalProperties <- AdditionalPropertiesDictionary(dict)

        match req.ResponseFormat with
        | Some ResponseFormat.Json -> opts.ResponseFormat <- ChatResponseFormat.Json
        | Some (ResponseFormat.Constrained (Grammar.JsonSchema schema)) ->
            // `ForJsonSchema` is the channel providers actually enforce. This used
            // to set bare JSON mode and hide the schema in AdditionalProperties,
            // where no stock M.E.AI provider looks — so the schema was requested
            // and never applied. Unparseable schemas degrade to plain JSON mode
            // rather than throwing out of a format mapping.
            match tryParseSchema schema with
            | Some element ->
                opts.ResponseFormat <-
                    ChatResponseFormat.ForJsonSchema(element, "tars_structured_output", "TARS constrained response schema")
            | None -> opts.ResponseFormat <- ChatResponseFormat.Json
        | Some (ResponseFormat.Constrained (Grammar.Ebnf grammar)) ->
            // No first-class M.E.AI channel for EBNF; travels as an additional
            // property, named in lockstep with OpenAiCompatibleClient so the two
            // pipelines never emit different wire shapes for one Grammar.
            // Backend selection is the server's job, so no backend key here.
            carry GrammarKey grammar
        | Some (ResponseFormat.Constrained (Grammar.Regex pattern)) -> carry RegexKey pattern
        | _ ->
            if req.JsonMode then
                opts.ResponseFormat <- ChatResponseFormat.Json

        opts

    /// Recover a TARS ResponseFormat from M.E.AI ChatOptions — the inverse of
    /// `toChatOptions`, and the half that was missing.
    ///
    /// The trap: `ChatResponseFormat.ForJsonSchema` returns a NEW
    /// `ChatResponseFormatJson`, not the `ChatResponseFormat.Json` singleton, so a
    /// reference comparison against that singleton is false in precisely the case
    /// that carries a schema. The old mapping did exactly that, and concluded "not
    /// JSON at all" for every schema-constrained request MAF issued.
    let fromChatOptions (options: ChatOptions) : ResponseFormat option =
        let carried key =
            match options.AdditionalProperties with
            | null -> None
            | props ->
                match props.TryGetValue key with
                | true, (:? string as s) when not (String.IsNullOrWhiteSpace s) -> Some s
                | _ -> None

        carried GrammarKey
        |> Option.map (Grammar.Ebnf >> ResponseFormat.Constrained)
        |> Option.orElseWith (fun () ->
            carried RegexKey |> Option.map (Grammar.Regex >> ResponseFormat.Constrained))
        |> Option.orElseWith (fun () ->
            match box options.ResponseFormat with
            | null -> None
            | :? ChatResponseFormatJson as json ->
                match Option.ofNullable json.Schema with
                | Some schema -> Some(ResponseFormat.Constrained(Grammar.JsonSchema(schema.GetRawText())))
                | None -> Some ResponseFormat.Json
            | _ -> Some ResponseFormat.Text)

    /// Whether a recovered format also implies the legacy `JsonMode` flag. Both
    /// exist on LlmRequest; backends read one or the other, so they must agree.
    let impliesJsonMode (format: ResponseFormat) =
        match format with
        | ResponseFormat.Json
        | ResponseFormat.Constrained (Grammar.JsonSchema _) -> true
        | _ -> false

    /// Convert an M.E.AI ChatResponse to a TARS LlmResponse
    let toLlmResponse (resp: ChatResponse) : LlmResponse =
        let text = resp.Text |> Option.ofObj |> Option.defaultValue ""

        let usage =
            resp.Usage
            |> Option.ofObj
            |> Option.map (fun u ->
                { PromptTokens = u.InputTokenCount |> Option.ofNullable |> Option.map int |> Option.defaultValue 0
                  CompletionTokens = u.OutputTokenCount |> Option.ofNullable |> Option.map int |> Option.defaultValue 0
                  TotalTokens = u.TotalTokenCount |> Option.ofNullable |> Option.map int |> Option.defaultValue 0 })

        let finishReason =
            resp.FinishReason
            |> Option.ofNullable
            |> Option.map (fun fr ->
                if fr = ChatFinishReason.Stop then "stop"
                elif fr = ChatFinishReason.Length then "length"
                elif fr = ChatFinishReason.ContentFilter then "content_filter"
                elif fr = ChatFinishReason.ToolCalls then "tool_calls"
                else "unknown")

        { Text = text
          FinishReason = finishReason
          Usage = usage
          Raw = None }

// ─────────────────────────────────────────────────────────────────────
// Adapter 1: ILlmService -> IChatClient
// Allows existing ILlmService implementations to be consumed as IChatClient.
// ─────────────────────────────────────────────────────────────────────
type LlmServiceChatClient(inner: ILlmService) =

    interface IChatClient with

        member this.GetService(serviceType: Type, serviceKey: obj) : obj =
            if serviceType = typeof<IChatClient> && isNull serviceKey then
                box this
            else
                null

        member _.GetResponseAsync(messages: IEnumerable<ChatMessage>, options: ChatOptions, cancellationToken: CancellationToken) : Task<ChatResponse> =
            task {
                let tarsMessages =
                    messages
                    |> Seq.map ChatClientMapping.fromAIChatMessage
                    |> Seq.toList

                let mutable req = Prompt.ofMessages tarsMessages

                if not (isNull options) then
                    let format = ChatClientMapping.fromChatOptions options

                    req <-
                        { req with
                            Temperature =
                                options.Temperature
                                |> Option.ofNullable
                                |> Option.map float
                                |> Option.orElse req.Temperature
                            MaxTokens =
                                options.MaxOutputTokens
                                |> Option.ofNullable
                                |> Option.orElse req.MaxTokens
                            Model =
                                options.ModelId
                                |> Option.ofObj
                                |> Option.orElse req.Model
                            Seed =
                                options.Seed
                                |> Option.ofNullable
                                |> Option.map int
                                |> Option.orElse req.Seed
                            Stop =
                                if isNull options.StopSequences then req.Stop
                                else options.StopSequences |> Seq.toList
                            ResponseFormat = format |> Option.orElse req.ResponseFormat
                            JsonMode =
                                match format with
                                | Some f -> ChatClientMapping.impliesJsonMode f
                                | None -> req.JsonMode }

                let! llmResp = inner.CompleteAsync(req)

                let responseMsg = ChatMessage(ChatRole.Assistant, llmResp.Text)
                let chatResp = ChatResponse(responseMsg)
                chatResp.ModelId <- req.Model |> Option.defaultValue null

                llmResp.FinishReason
                |> Option.iter (fun fr ->
                    chatResp.FinishReason <-
                        Nullable(
                            match fr with
                            | "stop" -> ChatFinishReason.Stop
                            | "length" -> ChatFinishReason.Length
                            | "content_filter" -> ChatFinishReason.ContentFilter
                            | "tool_calls" -> ChatFinishReason.ToolCalls
                            | _ -> ChatFinishReason.Stop))

                llmResp.Usage
                |> Option.iter (fun u ->
                    let usage = UsageDetails()
                    usage.InputTokenCount <- Nullable(int64 u.PromptTokens)
                    usage.OutputTokenCount <- Nullable(int64 u.CompletionTokens)
                    usage.TotalTokenCount <- Nullable(int64 u.TotalTokens)
                    chatResp.Usage <- usage)

                return chatResp
            }

        member _.GetStreamingResponseAsync(messages: IEnumerable<ChatMessage>, options: ChatOptions, cancellationToken: CancellationToken) : IAsyncEnumerable<ChatResponseUpdate> =
            let inner = inner
            { new IAsyncEnumerable<ChatResponseUpdate> with
                member _.GetAsyncEnumerator(ct) =
                    let tarsMessages =
                        messages
                        |> Seq.map ChatClientMapping.fromAIChatMessage
                        |> Seq.toList

                    let mutable req =
                        Prompt.ofMessages tarsMessages
                        |> Prompt.withStream true

                    if not (isNull options) then
                        // Constraints must survive streaming too: a MAF agent that
                        // streams is the common case, not the exception.
                        let format = ChatClientMapping.fromChatOptions options

                        req <-
                            { req with
                                Temperature =
                                    options.Temperature
                                    |> Option.ofNullable
                                    |> Option.map float
                                    |> Option.orElse req.Temperature
                                MaxTokens =
                                    options.MaxOutputTokens
                                    |> Option.ofNullable
                                    |> Option.orElse req.MaxTokens
                                Model =
                                    options.ModelId
                                    |> Option.ofObj
                                    |> Option.orElse req.Model
                                ResponseFormat = format |> Option.orElse req.ResponseFormat
                                JsonMode =
                                    match format with
                                    | Some f -> ChatClientMapping.impliesJsonMode f
                                    | None -> req.JsonMode }

                    let buffer = System.Collections.Concurrent.ConcurrentQueue<string>()
                    let mutable finished = false
                    let mutable started = false
                    let mutable completionTask: Task<LlmResponse> = null

                    { new IAsyncEnumerator<ChatResponseUpdate> with
                        member _.Current =
                            let mutable token = ""
                            buffer.TryDequeue(&token) |> ignore
                            let update = ChatResponseUpdate()
                            update.Role <- Nullable ChatRole.Assistant
                            update.Contents.Add(TextContent(token))
                            update

                        member _.MoveNextAsync() =
                            if not started then
                                started <- true
                                completionTask <- inner.CompleteStreamAsync(req, fun token -> buffer.Enqueue(token))

                            if not buffer.IsEmpty then
                                ValueTask<bool>(true)
                            elif finished then
                                ValueTask<bool>(false)
                            else
                                let waitTask = task {
                                    while buffer.IsEmpty && not completionTask.IsCompleted do
                                        do! Task.Delay(10, ct)
                                    if not buffer.IsEmpty then return true
                                    else
                                        finished <- true
                                        return not buffer.IsEmpty
                                }
                                ValueTask<bool>(waitTask)

                        member _.DisposeAsync() = ValueTask()
                    }
            }

        member _.Dispose() = ()

// ─────────────────────────────────────────────────────────────────────
// Adapter 2: IChatClient -> ILlmService
// Allows any M.E.AI provider to be used through the TARS ILlmService interface.
// ─────────────────────────────────────────────────────────────────────
type ChatClientLlmService(chatClient: IChatClient) =

    interface ILlmService with

        member _.CompleteAsync(req: LlmRequest) : Task<LlmResponse> =
            task {
                let messages = ResizeArray<ChatMessage>()

                req.SystemPrompt
                |> Option.iter (fun sp ->
                    messages.Add(ChatMessage(ChatRole.System, sp)))

                for msg in req.Messages do
                    messages.Add(ChatClientMapping.toAIChatMessage msg)

                let options = ChatClientMapping.toChatOptions req

                let! resp = chatClient.GetResponseAsync(messages, options, CancellationToken.None)

                return ChatClientMapping.toLlmResponse resp
            }

        member _.CompleteStreamAsync(req: LlmRequest, onToken: string -> unit) : Task<LlmResponse> =
            task {
                let messages = ResizeArray<ChatMessage>()

                req.SystemPrompt
                |> Option.iter (fun sp ->
                    messages.Add(ChatMessage(ChatRole.System, sp)))

                for msg in req.Messages do
                    messages.Add(ChatClientMapping.toAIChatMessage msg)

                let options = ChatClientMapping.toChatOptions req

                let mutable fullText = ""

                let updates = chatClient.GetStreamingResponseAsync(messages, options, CancellationToken.None)
                let enumerator = updates.GetAsyncEnumerator(CancellationToken.None)

                try
                    let mutable hasMore = true
                    while hasMore do
                        let! next = enumerator.MoveNextAsync()
                        hasMore <- next
                        if hasMore then
                            let update = enumerator.Current
                            let token = update.Text |> Option.ofObj |> Option.defaultValue ""
                            if token <> "" then
                                fullText <- fullText + token
                                onToken token
                finally
                    enumerator.DisposeAsync().AsTask().Wait()

                return
                    { Text = fullText
                      FinishReason = Some "stop"
                      Usage = None
                      Raw = None }
            }

        member _.EmbedAsync(_text: string) : Task<float32[]> =
            raise (NotSupportedException("IChatClient does not support embeddings. Use IEmbeddingGenerator instead."))

        member _.RouteAsync(_req: LlmRequest) : Task<RoutedBackend> =
            task {
                return
                    { Backend = Ollama "unknown"
                      Endpoint = Uri("http://localhost:11434")
                      ApiKey = None }
            }
