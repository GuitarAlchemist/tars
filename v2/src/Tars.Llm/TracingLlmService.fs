namespace Tars.Llm

open Tars.Core

/// Decorator for ILlmService that records all calls to a trace recorder
type TracingLlmService(inner: ILlmService, recorder: ITraceRecorder) =
    interface ILlmService with
        member _.CompleteAsync(request: LlmRequest) =
            task {
                let input =
                    try
                        System.Text.Json.JsonSerializer.Serialize(request)
                    with ex ->
                        $"Serialization failed: {ex.Message}"

                let! response = inner.CompleteAsync(request)

                let output =
                    try
                        System.Text.Json.JsonSerializer.Serialize(response)
                    with ex ->
                        $"Serialization failed: {ex.Message}"

                let metadata =
                    Map
                        [ "model", request.ModelHint |> Option.defaultValue "unknown"
                          "tokens_in", string (request.Messages |> List.sumBy (fun m -> m.Content.Length / 4)) // approx
                          "tokens_out", string response.Text.Length ]

                try
                    // Fire and forget with error handling
                    let recordOp = async {
                        try
                            do! recorder.RecordEventAsync TraceEventType.LlmCall input output metadata
                        with ex ->
                            printfn $"DEBUG: Async trace recording failed: %s{ex.Message}"
                    }
                    Async.Start(recordOp)
                with ex ->
                    printfn $"DEBUG: Trace recording setup failed: %s{ex.Message}"

                return response
            }

        member _.CompleteStreamAsync(request: LlmRequest, onToken: string -> unit) =
            task {
                let input =
                    try
                        System.Text.Json.JsonSerializer.Serialize(request)
                    with ex ->
                        $"Serialization failed: {ex.Message}"

                let fullResponse = System.Text.StringBuilder()

                let wrappedOnToken (token: string) =
                    fullResponse.Append(token) |> ignore
                    onToken token

                let! response = inner.CompleteStreamAsync(request, wrappedOnToken)

                let output = fullResponse.ToString()

                let metadata =
                    Map
                        [ "model", request.ModelHint |> Option.defaultValue "unknown"
                          "stream", "true" ]

                try
                    // Fire and forget with error handling
                    let recordOp = async {
                        try
                            do! recorder.RecordEventAsync TraceEventType.LlmCall input output metadata
                        with ex ->
                            printfn $"DEBUG: Async trace recording failed: %s{ex.Message}"
                    }
                    Async.Start(recordOp)
                with ex ->
                    printfn $"DEBUG: Trace recording setup failed: %s{ex.Message}"

                return response
            }

        member _.EmbedAsync(text: string) =
            task {
                let! embedding = inner.EmbedAsync(text)
                // Optional: record embeddings? Might be too verbose.
                return embedding
            }

        member _.RouteAsync(request: LlmRequest) =
            inner.RouteAsync(request)

    interface ICancellableLlmService with
        member _.CompleteAsync(request, cancellationToken) =
            task {
                let input =
                    try
                        System.Text.Json.JsonSerializer.Serialize(request)
                    with ex ->
                        $"Serialization failed: {ex.Message}"

                let! response =
                    match inner with
                    | :? ICancellableLlmService as cancellable ->
                        cancellable.CompleteAsync(request, cancellationToken)
                    | _ -> inner.CompleteAsync(request)

                let output =
                    try
                        System.Text.Json.JsonSerializer.Serialize(response)
                    with ex ->
                        $"Serialization failed: {ex.Message}"

                let metadata =
                    Map
                        [ "model", request.ModelHint |> Option.defaultValue "unknown"
                          "tokens_in", string (request.Messages |> List.sumBy (fun m -> m.Content.Length / 4))
                          "tokens_out", string response.Text.Length ]

                try
                    let recordOp = async {
                        try
                            do! recorder.RecordEventAsync TraceEventType.LlmCall input output metadata
                        with ex ->
                            printfn $"DEBUG: Async trace recording failed: %s{ex.Message}"
                    }
                    Async.Start(recordOp)
                with ex ->
                    printfn $"DEBUG: Trace recording setup failed: %s{ex.Message}"

                return response
            }

        member _.CompleteStreamAsync(request, onToken, cancellationToken) =
            task {
                let input =
                    try
                        System.Text.Json.JsonSerializer.Serialize(request)
                    with ex ->
                        $"Serialization failed: {ex.Message}"

                let fullResponse = System.Text.StringBuilder()

                let wrappedOnToken (token: string) =
                    fullResponse.Append(token) |> ignore
                    onToken token

                let! response =
                    match inner with
                    | :? ICancellableLlmService as cancellable ->
                        cancellable.CompleteStreamAsync(request, wrappedOnToken, cancellationToken)
                    | _ -> inner.CompleteStreamAsync(request, wrappedOnToken)

                let output = fullResponse.ToString()

                let metadata =
                    Map
                        [ "model", request.ModelHint |> Option.defaultValue "unknown"
                          "stream", "true" ]

                try
                    let recordOp = async {
                        try
                            do! recorder.RecordEventAsync TraceEventType.LlmCall input output metadata
                        with ex ->
                            printfn $"DEBUG: Async trace recording failed: %s{ex.Message}"
                    }
                    Async.Start(recordOp)
                with ex ->
                    printfn $"DEBUG: Trace recording setup failed: %s{ex.Message}"

                return response
            }

        member _.EmbedAsync(text, cancellationToken) =
            match inner with
            | :? ICancellableLlmService as cancellable -> cancellable.EmbedAsync(text, cancellationToken)
            | _ -> inner.EmbedAsync(text)

        member _.RouteAsync(request, cancellationToken) =
            match inner with
            | :? ICancellableLlmService as cancellable -> cancellable.RouteAsync(request, cancellationToken)
            | _ -> inner.RouteAsync(request)
