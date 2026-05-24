namespace Tars.Connectors

open System.Net.Http

module Ollama =
    open System.Text
    open System.Text.Json
    open System.Text.Json.Serialization

    let private client = new HttpClient()
    let private baseUrl = "http://localhost:11434/api/generate"

    type OllamaRequest =
        { [<JsonPropertyName("model")>]
          Model: string
          [<JsonPropertyName("prompt")>]
          Prompt: string
          [<JsonPropertyName("stream")>]
          Stream: bool }

    type OllamaResponse =
        { [<JsonPropertyName("response")>]
          Response: string }

    let generate (model: string) (prompt: string) =
        task {
            try
                let req =
                    { Model = model
                      Prompt = prompt
                      Stream = false }

                let json = JsonSerializer.Serialize(req)
                let content = new StringContent(json, Encoding.UTF8, "application/json")

                let! response = client.PostAsync(baseUrl, content)
                response.EnsureSuccessStatusCode() |> ignore

                let! responseString = response.Content.ReadAsStringAsync()
                let respObj = JsonSerializer.Deserialize<OllamaResponse>(responseString)
                return Ok respObj.Response
            with ex ->
                return Error ex.Message
        }
