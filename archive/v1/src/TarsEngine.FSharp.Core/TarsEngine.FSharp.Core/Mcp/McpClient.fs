namespace TarsEngine.FSharp.Core.Mcp

open System
open System.Net.Http
open System.Text
open System.Threading.Tasks
open Microsoft.Extensions.Logging

module McpClient =

    let private httpClient = new HttpClient()

    let private tryAddHeader (request: HttpRequestMessage) (key: string) (value: string) =
        if String.IsNullOrWhiteSpace key || isNull value then
            ()
        else
            if not (request.Headers.TryAddWithoutValidation(key, value)) then
                request.Content.Headers.TryAddWithoutValidation(key, value) |> ignore

    let sendRequest (logger: ILogger) (server: McpServer) (request: McpRequest) : Task<Result<string, string>> =
        task {
            use message = new HttpRequestMessage(HttpMethod.Post, server.Url)
            message.Content <- new StringContent(request.Body, Encoding.UTF8, "application/json")

            server.Headers |> Map.iter (fun key value -> tryAddHeader message key value)
            request.Headers |> Map.iter (fun key value -> tryAddHeader message key value)

            try
                let! response = httpClient.SendAsync(message)
                let! responseText = response.Content.ReadAsStringAsync()

                if response.IsSuccessStatusCode then
                    logger.LogInformation("MCP request to {Url} succeeded with status {StatusCode}", server.Url, response.StatusCode)
                    return Ok responseText
                else
                    logger.LogWarning("MCP request to {Url} failed with status {StatusCode}", server.Url, response.StatusCode)
                    let error = $"HTTP {(int response.StatusCode)} {response.ReasonPhrase}. Body: {responseText}"
                    return Error error
            with ex ->
                logger.LogError(ex, "MCP request to {Url} failed", server.Url)
                return Error $"MCP request error: {ex.Message}"
        }
