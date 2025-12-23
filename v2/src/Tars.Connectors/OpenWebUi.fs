namespace Tars.Connectors

open System
open System.Net.Http
open System.Net.Http.Headers
open System.Text
open System.Text.Json
open System.Text.Json.Serialization
open System.Threading.Tasks
open Tars.Security

module OpenWebUi =
    let private client = new HttpClient()
    // Simple token cache
    let mutable private cachedToken: string option = None

    type AuthResponse =
        { [<JsonPropertyName("token")>]
          Token: string }

    type ModelInfo =
        { id: string
          name: string
          object: string
          created: int64
          owned_by: string }

    type ModelListResponse = { object: string; data: ModelInfo[] }

    type OpenWebUiMessage = { Role: string; Content: string }

    let private login (baseUrl: string) =
        task {
            // Try to get secrets
            let emailResult = CredentialVault.getSecret "OPENWEBUI_EMAIL"
            let passResult = CredentialVault.getSecret "OPENWEBUI_PASSWORD"

            match emailResult, passResult with
            | Ok email, Ok password ->
                try
                    let req = {| email = email; password = password |}
                    let json = JsonSerializer.Serialize(req)
                    let content = new StringContent(json, Encoding.UTF8, "application/json")

                    let url =
                        if baseUrl.EndsWith("/") then
                            baseUrl + "api/v1/auths/signin"
                        else
                            baseUrl + "/api/v1/auths/signin"

                    let! response = client.PostAsync(url, content)

                    if response.IsSuccessStatusCode then
                        let! respString = response.Content.ReadAsStringAsync()
                        let options = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
                        let respObj = JsonSerializer.Deserialize<AuthResponse>(respString, options)
                        // Check if token is present
                        if String.IsNullOrWhiteSpace(respObj.Token) then
                            return Error "Authentication succeeded but token is empty"
                        else
                            return Ok respObj.Token
                    else
                        let! err = response.Content.ReadAsStringAsync()
                        return Error $"Auth failed: {response.StatusCode} - {err}"
                with ex ->
                    return Error $"Auth Exception: {ex.Message}"
            | Error e, _ -> return Error $"Missing Credential: {e}"
            | _, Error e -> return Error $"Missing Credential: {e}"
        }

    let listModels (baseUrl: string) =
        task {
            // 1. Ensure we have a token
            let! tokenResult =
                match cachedToken with
                | Some t -> Task.FromResult(Ok t)
                | None -> login baseUrl

            match tokenResult with
            | Error e -> return Error e
            | Ok token ->
                cachedToken <- Some token

                try
                    let url =
                        if baseUrl.EndsWith("/") then
                            baseUrl + "api/models"
                        else
                            baseUrl + "/api/models"

                    use requestMessage = new HttpRequestMessage(HttpMethod.Get, url)
                    requestMessage.Headers.Authorization <- AuthenticationHeaderValue("Bearer", token)

                    let! response = client.SendAsync(requestMessage)

                    if response.IsSuccessStatusCode then
                        let! json = response.Content.ReadAsStringAsync()
                        let options = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
                        let respObj = JsonSerializer.Deserialize<ModelListResponse>(json, options)
                        return Ok respObj.data
                    else
                        let! err = response.Content.ReadAsStringAsync()
                        return Error $"HTTP Error {response.StatusCode}: {err}"
                with ex ->
                    return Error $"Exception during listModels: {ex.Message}"
        }

    let generate (baseUrl: string) (model: string) (messages: OpenWebUiMessage list) =
        task {
            // 1. Ensure we have a token
            let! tokenResult =
                match cachedToken with
                | Some t -> Task.FromResult(Ok t)
                | None -> login baseUrl

            match tokenResult with
            | Error e -> return Error e
            | Ok token ->
                cachedToken <- Some token

                try
                    // 2. Prepare Chat Request
                    let req =
                        {| model = model
                           messages = messages |> List.map (fun m -> {| role = m.Role; content = m.Content |})
                           stream = false |}

                    let json = JsonSerializer.Serialize(req)
                    let content = new StringContent(json, Encoding.UTF8, "application/json")

                    let url =
                        if baseUrl.EndsWith("/") then
                            baseUrl + "api/chat/completions"
                        else
                            baseUrl + "/api/chat/completions"

                    use requestMessage = new HttpRequestMessage(HttpMethod.Post, url)
                    requestMessage.Content <- content
                    requestMessage.Headers.Authorization <- AuthenticationHeaderValue("Bearer", token)

                    let! response = client.SendAsync(requestMessage)

                    if response.IsSuccessStatusCode then
                        let! responseString = response.Content.ReadAsStringAsync()
                        let options = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
                        // Reuse OpenAiCompatible types
                        let respObj =
                            JsonSerializer.Deserialize<OpenAiCompatible.ChatResponse>(responseString, options)

                        if respObj.Choices <> null && respObj.Choices.Length > 0 then
                            return Ok respObj.Choices[0].Message.Content
                        else
                            return Error "No choices returned from Open WebUI"
                    else
                        let! errorBody = response.Content.ReadAsStringAsync()
                        return Error $"HTTP Error {response.StatusCode}: {errorBody}"

                with ex ->
                    return Error $"Exception during generation: {ex.Message}"
        }
