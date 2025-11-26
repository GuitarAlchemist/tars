namespace Tars.Connectors

open System.Net.Http
open System.Text
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks

module OpenAiCompatible =
    let private client = new HttpClient()

    type ChatMessage =
        { [<JsonPropertyName("role")>]
          Role: string
          [<JsonPropertyName("content")>]
          Content: string }

    type ChatRequest =
        { [<JsonPropertyName("model")>]
          Model: string
          [<JsonPropertyName("messages")>]
          Messages: ChatMessage list
          [<JsonPropertyName("stream")>]
          Stream: bool }

    type ChatChoice =
        { [<JsonPropertyName("message")>]
          Message: ChatMessage }

    type ChatResponse =
        { [<JsonPropertyName("choices")>]
          Choices: ChatChoice[] }

    let generate (baseUrl: string) (model: string) (prompt: string) =
        task {
            try
                let req =
                    { Model = model
                      Messages = [ { Role = "user"; Content = prompt } ]
                      Stream = false }

                let json = JsonSerializer.Serialize(req)
                let content = new StringContent(json, Encoding.UTF8, "application/json")

                let url = 
                    if baseUrl.EndsWith("/") then baseUrl + "v1/chat/completions"
                    else baseUrl + "/v1/chat/completions"

                let! response = client.PostAsync(url, content)
                
                if response.IsSuccessStatusCode then
                    let! responseString = response.Content.ReadAsStringAsync()
                    let options = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
                    let respObj = JsonSerializer.Deserialize<ChatResponse>(responseString, options)
                    if respObj.Choices <> null && respObj.Choices.Length > 0 then
                        return Ok respObj.Choices.[0].Message.Content
                    else
                        return Error "No choices returned from LLM"
                else
                    let! errorBody = response.Content.ReadAsStringAsync()
                    return Error $"HTTP Error {response.StatusCode}: {errorBody}"

            with ex ->
                return Error $"Exception: {ex.Message}"
        }
