namespace MyProject.Clients

open System
open System.IO
open System.Net.Http
open System.Text
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open FSharp.Control

/// Represents a request to the Ollama chat endpoint.
type ChatRequest = {
    model: string
    messages: obj list // replace with concrete message type as needed
    stream: bool
}

/// Ollama client that supports async streaming via Server‑Sent Events (SSE).
type OllamaClient(httpClient: HttpClient) =
    /// Sends a chat request and returns an IAsyncEnumerable<string> that yields each SSE data payload.
    member _.StreamChatAsync(request: ChatRequest, ?cancellationToken: CancellationToken) : IAsyncEnumerable<string> =
        let ct = defaultArg cancellationToken CancellationToken.None
        let uri = Uri("https://api.ollama.com/v1/chat")
        let json = JsonSerializer.Serialize(request)
        let content = new StringContent(json, Encoding.UTF8, "application/json")
        // Request streaming response
        let response =
            httpClient.PostAsync(uri, content, HttpCompletionOption.ResponseHeadersRead, ct)
                .GetAwaiter().GetResult()
        response.EnsureSuccessStatusCode() |> ignore
        let stream = response.Content.ReadAsStreamAsync(ct).GetAwaiter().GetResult()
        let reader = new StreamReader(stream)
        // Use taskSeq to produce IAsyncEnumerable<string>
        taskSeq {
            while not ct.IsCancellationRequested do
                let! line = reader.ReadLineAsync() |> Async.AwaitTask
                if isNull line then
                    // End of stream
                    ()
                elif line.StartsWith("data:") then
                    // Strip "data:" prefix and any leading whitespace
                    let payload = line.Substring(5).TrimStart()
                    // Skip empty heartbeat lines
                    if not (String.IsNullOrWhiteSpace payload) then
                        yield payload
                elif line.StartsWith(":") then
                    // Comment / keep‑alive line – ignore
                    ()
                else
                    // Unexpected line – ignore
                    ()
        }
