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
    /// Private to this adapter pair: no provider reads them, and they are NOT the
    /// shape `OpenAiCompatibleClient` puts on the wire (that is a nested
    /// `structured_outputs: { grammar = ... }` object). They exist only so the
    /// write and read halves here cannot drift apart.
    [<Literal>]
    let GrammarKey = "structured_outputs_grammar"

    [<Literal>]
    let RegexKey = "structured_outputs_regex"

    /// A JSON schema must be a JSON *object*. `"null"`, `"42"` and `"[1,2]"` all
    /// parse happily and would sail through as a schema, producing
    /// `json_schema: { schema: null }` and a 400 at the provider rather than the
    /// intended degrade to plain JSON mode.
    let private tryParseSchema (schema: string) =
        try
            let root = System.Text.Json.JsonDocument.Parse(schema).RootElement

            if root.ValueKind = System.Text.Json.JsonValueKind.Object then
                Some root
            else
                None
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
        // Text must be stated, not left to the catch-all: falling through would let
        // `req.JsonMode` overwrite it, turning an explicit request for prose into
        // one demanding JSON.
        | Some ResponseFormat.Text -> opts.ResponseFormat <- ChatResponseFormat.Text
        | Some (ResponseFormat.Constrained (Grammar.JsonSchema schema)) ->
            // `ForJsonSchema` is the channel providers actually enforce;
            // AdditionalProperties is not, so a schema left there is never applied.
            // A schema we cannot use degrades to plain JSON mode rather than
            // throwing out of a format mapping.
            match tryParseSchema schema with
            | Some element ->
                opts.ResponseFormat <-
                    ChatResponseFormat.ForJsonSchema(element, "tars_structured_output", "TARS constrained response schema")
            | None -> opts.ResponseFormat <- ChatResponseFormat.Json
        | Some (ResponseFormat.Constrained (Grammar.Ebnf grammar)) ->
            // Backend selection is the server's job, so no backend key here.
            carry GrammarKey grammar
        | Some (ResponseFormat.Constrained (Grammar.Regex pattern)) -> carry RegexKey pattern
        | None ->
            if req.JsonMode then
                opts.ResponseFormat <- ChatResponseFormat.Json

        opts

    /// Recover a TARS ResponseFormat from M.E.AI ChatOptions — the inverse of
    /// `toChatOptions`.
    ///
    /// The trap this exists to avoid: `ChatResponseFormat.ForJsonSchema` returns a
    /// NEW `ChatResponseFormatJson`, not the `ChatResponseFormat.Json` singleton, so
    /// a reference comparison against that singleton is false in precisely the case
    /// that carries a schema. Distinguish on `.Schema`, never on identity.
    ///
    /// Precedence, for options that carry both channels — `toChatOptions` never
    /// emits both, so this only arises from an external producer:
    ///   1. a schema-bearing ResponseFormat — the most specific constraint, and one
    ///      the caller set through the typed API on purpose
    ///   2. the grammar/regex side channel — used only because M.E.AI has no slot
    ///      for them, so it should not beat an explicit schema
    ///   3. a bare Json/Text ResponseFormat — weaker than a grammar, so it loses
    let fromChatOptions (options: ChatOptions) : ResponseFormat option =
        let carried key =
            match options.AdditionalProperties with
            | null -> None
            | props ->
                match props.TryGetValue key with
                | true, (:? string as s) when not (String.IsNullOrWhiteSpace s) -> Some s
                | _ -> None

        // `Undefined` is reachable: M.E.AI accepts `ForJsonSchema(default)` without
        // validation, and GetRawText() throws on it. A schema we cannot read is no
        // schema, not an exception escaping the adapter.
        let schemaOf (json: ChatResponseFormatJson) =
            Option.ofNullable json.Schema
            |> Option.filter (fun s -> s.ValueKind <> System.Text.Json.JsonValueKind.Undefined)

        let typed =
            match box options.ResponseFormat with
            | null -> None
            | :? ChatResponseFormatJson as json ->
                match schemaOf json with
                | Some schema -> Some(ResponseFormat.Constrained(Grammar.JsonSchema(schema.GetRawText())))
                | None -> Some ResponseFormat.Json
            | _ -> Some ResponseFormat.Text

        match typed with
        | Some (ResponseFormat.Constrained (Grammar.JsonSchema _)) -> typed
        | _ ->
            carried GrammarKey
            |> Option.map (Grammar.Ebnf >> ResponseFormat.Constrained)
            |> Option.orElseWith (fun () ->
                carried RegexKey |> Option.map (Grammar.Regex >> ResponseFormat.Constrained))
            |> Option.orElse typed

    /// Apply a recovered format to a request, setting `ResponseFormat` and the
    /// legacy `JsonMode` flag together.
    ///
    /// Both fields live on LlmRequest and every backend matches `ResponseFormat`
    /// first, consulting `JsonMode` only in the `None` branch — so `JsonMode` is
    /// dead whenever `ResponseFormat` is set. It is maintained here anyway so the
    /// two never state different things to a future reader or backend.
    let applyFormat (options: ChatOptions) (req: LlmRequest) =
        match fromChatOptions options with
        | None -> req
        | Some format ->
            { req with
                ResponseFormat = Some format
                JsonMode =
                    match format with
                    | ResponseFormat.Json
                    | ResponseFormat.Constrained (Grammar.JsonSchema _) -> true
                    | _ -> false }

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
                                else options.StopSequences |> Seq.toList }
                        |> ChatClientMapping.applyFormat options

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
                                    |> Option.orElse req.Model }
                            |> ChatClientMapping.applyFormat options

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
