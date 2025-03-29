module WebSearch

open System
open System.Net.Http
open System.Threading.Tasks
open Newtonsoft.Json.Linq
open Microsoft.FSharp.Control

// Types
type SearchResult = {
    Title: string
    Url: string
    Description: string
    Source: string
    Relevance: float
    Timestamp: DateTime
}

type SearchProvider = {
    Name: string
    ApiKey: string
    BaseUrl: string
    RateLimit: int // Requests per minute
}

type SearchError =
    | RateLimitExceeded
    | ApiError of string
    | NetworkError of string

// Configuration
let private providers = [
    { Name = "Google"
      ApiKey = Environment.GetEnvironmentVariable("GOOGLE_API_KEY")
      BaseUrl = "https://www.googleapis.com/customsearch/v1"
      RateLimit = 100 }
    { Name = "StackOverflow"
      ApiKey = Environment.GetEnvironmentVariable("STACKOVERFLOW_API_KEY")
      BaseUrl = "https://api.stackexchange.com/2.3/search"
      RateLimit = 300 }
    { Name = "GitHub"
      ApiKey = Environment.GetEnvironmentVariable("GITHUB_API_KEY")
      BaseUrl = "https://api.github.com/search/repositories"
      RateLimit = 60 }
]

// Rate limiting implementation
type RateLimiter(requestsPerMinute: int) =
    let mutable lastRequests = []
    
    member _.CheckLimit() =
        let now = DateTime.UtcNow
        lastRequests <- lastRequests 
                       |> List.filter (fun time -> (now - time).TotalMinutes < 1.0)
        
        if lastRequests.Length < requestsPerMinute then
            lastRequests <- now :: lastRequests
            true
        else
            false

// HTTP client
let private httpClient = new HttpClient()

// Search implementation
let private searchWithProvider (provider: SearchProvider) (query: string) = 
    async {
        let rateLimiter = RateLimiter(provider.RateLimit)
        
        if not (rateLimiter.CheckLimit()) then
            return Error RateLimitExceeded
        else
            try
                let queryParams = 
                    match provider.Name with
                    | "Google" -> 
                        sprintf "?key=%s&cx=YOUR_SEARCH_ENGINE_ID&q=%s" 
                            provider.ApiKey (Uri.EscapeDataString(query))
                    | "StackOverflow" -> 
                        sprintf "?site=stackoverflow&intitle=%s&key=%s" 
                            (Uri.EscapeDataString(query)) provider.ApiKey
                    | "GitHub" -> 
                        sprintf "?q=%s" (Uri.EscapeDataString(query))
                    | _ -> ""

                let requestUrl = provider.BaseUrl + queryParams
                use request = new HttpRequestMessage(HttpMethod.Get, requestUrl)
                
                if provider.Name = "GitHub" then
                    request.Headers.Add("Authorization", sprintf "token %s" provider.ApiKey)
                    request.Headers.Add("User-Agent", "TARS-Search-Agent")

                let! response = httpClient.SendAsync(request) |> Async.AwaitTask
                let! content = response.Content.ReadAsStringAsync() |> Async.AwaitTask

                if response.IsSuccessStatusCode then
                    let jsonResponse = JObject.Parse(content)
                    let results = 
                        match provider.Name with
                        | "Google" ->
                            jsonResponse.["items"]
                            |> Seq.map (fun item -> 
                                { Title = item.["title"].ToString()
                                  Url = item.["link"].ToString()
                                  Description = item.["snippet"].ToString()
                                  Source = provider.Name
                                  Relevance = 1.0
                                  Timestamp = DateTime.UtcNow })
                        | "StackOverflow" ->
                            jsonResponse.["items"]
                            |> Seq.map (fun item ->
                                { Title = item.["title"].ToString()
                                  Url = item.["link"].ToString()
                                  Description = item.["excerpt"].ToString()
                                  Source = provider.Name
                                  Relevance = float(item.["score"].ToString())
                                  Timestamp = DateTime.UtcNow })
                        | "GitHub" ->
                            jsonResponse.["items"]
                            |> Seq.map (fun item ->
                                { Title = item.["full_name"].ToString()
                                  Url = item.["html_url"].ToString()
                                  Description = item.["description"].ToString()
                                  Source = provider.Name
                                  Relevance = float(item.["stargazers_count"].ToString())
                                  Timestamp = DateTime.UtcNow })
                        | _ -> Seq.empty
                    
                    return Ok(results |> Seq.toList)
                else
                    return Error(ApiError(sprintf "HTTP %d: %s" 
                        (int response.StatusCode) content))
            with
            | ex -> return Error(NetworkError ex.Message)
    }

// Public interface
let searchAll (query: string) =
    async {
        let! results = 
            providers
            |> List.map (fun provider -> searchWithProvider provider query)
            |> Async.Parallel
        
        return results
               |> Array.choose (function 
                   | Ok results -> Some results 
                   | Error _ -> None)
               |> Array.concat
               |> Array.toList
               |> List.sortByDescending (fun r -> r.Relevance)
    }

let summarizeResults (results: SearchResult list) =
    let summary = 
        results
        |> List.groupBy (fun r -> r.Source)
        |> List.map (fun (source, items) ->
            sprintf "=== %s Results ===\n%s" 
                source
                (items 
                 |> List.take (min 5 items.Length)
                 |> List.map (fun r -> 
                     sprintf "- %s\n  %s\n  %s" 
                         r.Title r.Url r.Description)
                 |> String.concat "\n"))
        |> String.concat "\n\n"
    
    sprintf "Search Results Summary\n\n%s" summary