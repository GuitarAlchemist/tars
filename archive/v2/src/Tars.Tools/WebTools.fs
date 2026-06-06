namespace Tars.Tools.Standard

open System
open System.IO
open System.Net.Http
open System.Text.RegularExpressions
open System.Text.Json
open Tars.Tools

/// Tools for web scraping, search, and internet content extraction
module WebTools =

    let private httpClient =
        lazy
            (let client = new HttpClient()
             client.Timeout <- TimeSpan.FromSeconds(30.0)

             client.DefaultRequestHeaders.UserAgent.ParseAdd(
                 "TARS/2.0 (Autonomous Reasoning System; +https://github.com/GuitarAlchemist/tars)"
             )

             client)

    /// Extract readable text from HTML, removing scripts, styles, and tags
    let private extractText (html: string) =
        // Remove script and style blocks
        let noScript =
            Regex.Replace(html, @"<script[^>]*>[\s\S]*?</script>", "", RegexOptions.IgnoreCase)

        let noStyle =
            Regex.Replace(noScript, @"<style[^>]*>[\s\S]*?</style>", "", RegexOptions.IgnoreCase)

        // Extract title
        let titleMatch =
            Regex.Match(noStyle, @"<title[^>]*>(.*?)</title>", RegexOptions.IgnoreCase ||| RegexOptions.Singleline)

        let title =
            if titleMatch.Success then
                titleMatch.Groups.[1].Value.Trim()
            else
                ""

        // Remove all HTML tags
        let text = Regex.Replace(noStyle, @"<[^>]+>", " ")

        // Decode HTML entities
        let decoded = System.Net.WebUtility.HtmlDecode(text)

        // Collapse whitespace
        let cleaned = Regex.Replace(decoded, @"\s+", " ").Trim()

        title, cleaned

    /// Extract links from HTML
    let private extractLinks (html: string) (baseUrl: Uri) =
        let pattern = @"<a\s+[^>]*href=[""']([^""']+)[""'][^>]*>(.*?)</a>"

        Regex.Matches(html, pattern, RegexOptions.IgnoreCase ||| RegexOptions.Singleline)
        |> Seq.cast<Match>
        |> Seq.choose (fun m ->
            try
                let href = m.Groups.[1].Value
                let text = Regex.Replace(m.Groups.[2].Value, @"<[^>]+>", "").Trim()
                let uri = Uri(baseUrl, href)
                Some(uri.AbsoluteUri, text)
            with _ ->
                None)
        |> Seq.distinctBy fst
        |> Seq.truncate 50
        |> Seq.toList

    [<TarsToolAttribute("fetch_webpage",
                        "Fetches a webpage and extracts readable text content. Input JSON: { \"url\": \"https://...\", \"max_length\": 10000 }")>]
    let fetchWebpage (args: string) =
        task {
            try
                let url = ToolHelpers.parseStringArg args "url"

                let maxLength =
                    try
                        let doc = JsonDocument.Parse(args)
                        let mutable prop = Unchecked.defaultof<JsonElement>

                        if doc.RootElement.TryGetProperty("max_length", &prop) then
                            prop.GetInt32()
                        else
                            10000
                    with _ ->
                        10000

                if String.IsNullOrWhiteSpace url then
                    return "fetch_webpage error: missing url"
                else
                    printfn $"🌐 Fetching: {url}"
                    let! response = httpClient.Value.GetAsync(url)
                    response.EnsureSuccessStatusCode() |> ignore
                    let! html = response.Content.ReadAsStringAsync()

                    let title, text = extractText html

                    let truncated =
                        if text.Length > maxLength then
                            text.Substring(0, maxLength) + "... [truncated]"
                        else
                            text

                    return $"Title: {title}\n\nContent ({text.Length} chars):\n{truncated}"
            with ex ->
                return $"fetch_webpage error: {ex.Message}"
        }

    [<TarsToolAttribute("fetch_wikipedia",
                        "Fetches a Wikipedia article and extracts the summary. Input JSON: { \"topic\": \"Artificial intelligence\" }")>]
    let fetchWikipedia (args: string) =
        task {
            try
                let topic = ToolHelpers.parseStringArg args "topic"

                if String.IsNullOrWhiteSpace topic then
                    return "fetch_wikipedia error: missing topic"
                else
                    // Use Wikipedia API for clean extraction
                    let encoded = Uri.EscapeDataString(topic.Replace(" ", "_"))
                    let apiUrl = $"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

                    printfn $"📚 Fetching Wikipedia: {topic}"
                    let! response = httpClient.Value.GetAsync(apiUrl)

                    if not response.IsSuccessStatusCode then
                        return $"Wikipedia article not found: {topic}"
                    else
                        let! json = response.Content.ReadAsStringAsync()
                        let doc = JsonDocument.Parse(json)
                        let root = doc.RootElement

                        let title =
                            let mutable p = Unchecked.defaultof<JsonElement>

                            if root.TryGetProperty("title", &p) then
                                p.GetString()
                            else
                                topic

                        let extract =
                            let mutable p = Unchecked.defaultof<JsonElement>

                            if root.TryGetProperty("extract", &p) then
                                p.GetString()
                            else
                                ""

                        let pageUrl =
                            let mutable cp = Unchecked.defaultof<JsonElement>

                            if root.TryGetProperty("content_urls", &cp) then
                                let mutable dp = Unchecked.defaultof<JsonElement>

                                if cp.TryGetProperty("desktop", &dp) then
                                    let mutable pp = Unchecked.defaultof<JsonElement>

                                    if dp.TryGetProperty("page", &pp) then
                                        pp.GetString()
                                    else
                                        ""
                                else
                                    ""
                            else
                                ""

                        return $"# {title}\n\n{extract}\n\nSource: {pageUrl}"
            with ex ->
                return $"fetch_wikipedia error: {ex.Message}"
        }

    [<TarsToolAttribute("fetch_github_readme",
                        "Fetches the README from a GitHub repository. Input JSON: { \"repo\": \"owner/repo\" }")>]
    let fetchGitHubReadme (args: string) =
        task {
            try
                let repo = ToolHelpers.parseStringArg args "repo"

                if String.IsNullOrWhiteSpace repo then
                    return "fetch_github_readme error: missing repo (format: owner/repo)"
                else
                    // Use GitHub API for raw README
                    let apiUrl = $"https://api.github.com/repos/{repo}/readme"

                    printfn $"🐙 Fetching GitHub README: {repo}"

                    use request = new HttpRequestMessage(HttpMethod.Get, apiUrl)
                    request.Headers.Accept.ParseAdd("application/vnd.github.v3.raw")

                    let! response = httpClient.Value.SendAsync(request)

                    if not response.IsSuccessStatusCode then
                        return $"GitHub repo not found or no README: {repo}"
                    else
                        let! content = response.Content.ReadAsStringAsync()

                        let truncated =
                            if content.Length > 15000 then
                                content.Substring(0, 15000) + "\n\n... [truncated]"
                            else
                                content

                        return $"# README: {repo}\n\n{truncated}"
            with ex ->
                return $"fetch_github_readme error: {ex.Message}"
        }

    [<TarsToolAttribute("extract_links", "Extracts all links from a webpage. Input JSON: { \"url\": \"https://...\" }")>]
    let extractLinksFromPage (args: string) =
        task {
            try
                let url = ToolHelpers.parseStringArg args "url"

                if String.IsNullOrWhiteSpace url then
                    return "extract_links error: missing url"
                else
                    printfn $"🔗 Extracting links from: {url}"
                    let! response = httpClient.Value.GetAsync(url)
                    response.EnsureSuccessStatusCode() |> ignore
                    let! html = response.Content.ReadAsStringAsync()

                    let baseUri = Uri(url)
                    let links = extractLinks html baseUri

                    let formatted =
                        links
                        |> List.mapi (fun i (href, text) ->
                            let displayText =
                                if String.IsNullOrWhiteSpace text then
                                    "(no text)"
                                else
                                    text.Substring(0, min 50 text.Length)

                            $"{i + 1}. [{displayText}]({href})")
                        |> String.concat "\n"

                    return $"Found {links.Length} links:\n\n{formatted}"
            with ex ->
                return $"extract_links error: {ex.Message}"
        }

    [<TarsToolAttribute("search_web",
                        "Searches the web using DuckDuckGo (no API key needed). Input JSON: { \"query\": \"search terms\", \"max_results\": 5 }")>]
    let searchWeb (args: string) =
        task {
            try
                let query = ToolHelpers.parseStringArg args "query"

                let maxResults =
                    try
                        let doc = JsonDocument.Parse(args)
                        let mutable prop = Unchecked.defaultof<JsonElement>

                        if doc.RootElement.TryGetProperty("max_results", &prop) then
                            prop.GetInt32()
                        else
                            5
                    with _ ->
                        5

                if String.IsNullOrWhiteSpace query then
                    return "search_web error: missing query"
                else
                    // Use DuckDuckGo HTML search (no API key needed)
                    let encoded = Uri.EscapeDataString(query)
                    let searchUrl = $"https://html.duckduckgo.com/html/?q={encoded}"

                    printfn $"🔍 Searching: {query}"

                    use request = new HttpRequestMessage(HttpMethod.Get, searchUrl)
                    request.Headers.Add("Accept-Language", "en-US,en;q=0.9")

                    let! response = httpClient.Value.SendAsync(request)
                    let! html = response.Content.ReadAsStringAsync()

                    // Parse DuckDuckGo results
                    let resultPattern =
                        @"<a[^>]*class=""result__a""[^>]*href=""([^""]+)""[^>]*>([^<]+)</a>"

                    let snippetPattern = @"<a[^>]*class=""result__snippet""[^>]*>([^<]+)</a>"

                    let results =
                        Regex.Matches(html, resultPattern, RegexOptions.IgnoreCase)
                        |> Seq.cast<Match>
                        |> Seq.map (fun m -> m.Groups.[1].Value, m.Groups.[2].Value)
                        |> Seq.filter (fun (url, _) -> not (url.Contains("duckduckgo.com")))
                        |> Seq.truncate maxResults
                        |> Seq.toList

                    if results.IsEmpty then
                        return $"No results found for: {query}"
                    else
                        let formatted =
                            results
                            |> List.mapi (fun i (url, title) ->
                                // DuckDuckGo uses redirect URLs, extract actual URL
                                let actualUrl =
                                    let uddgMatch = Regex.Match(url, @"uddg=([^&]+)")

                                    if uddgMatch.Success then
                                        Uri.UnescapeDataString(uddgMatch.Groups.[1].Value)
                                    else
                                        url

                                $"{i + 1}. **{title}**\n   {actualUrl}")
                            |> String.concat "\n\n"

                        return $"Search results for \"{query}\":\n\n{formatted}"
            with ex ->
                return $"search_web error: {ex.Message}"
        }

    [<TarsToolAttribute("fetch_json_api",
                        "Fetches data from a JSON API. Input JSON: { \"url\": \"https://api...\", \"method\": \"GET\", \"headers\": {} }")>]
    let fetchJsonApi (args: string) =
        task {
            try
                let doc = JsonDocument.Parse(args)
                let root = doc.RootElement

                let url = root.GetProperty("url").GetString()

                let method =
                    let mutable p = Unchecked.defaultof<JsonElement>

                    if root.TryGetProperty("method", &p) then
                        p.GetString().ToUpper()
                    else
                        "GET"

                if String.IsNullOrWhiteSpace url then
                    return "fetch_json_api error: missing url"
                else
                    printfn $"📡 API Request: {method} {url}"

                    use request =
                        new HttpRequestMessage(
                            (match method with
                             | "POST" -> HttpMethod.Post
                             | "PUT" -> HttpMethod.Put
                             | "DELETE" -> HttpMethod.Delete
                             | _ -> HttpMethod.Get),
                            url
                        )

                    request.Headers.Accept.ParseAdd("application/json")

                    // Add custom headers
                    let mutable headersProp = Unchecked.defaultof<JsonElement>

                    if root.TryGetProperty("headers", &headersProp) then
                        for prop in headersProp.EnumerateObject() do
                            request.Headers.TryAddWithoutValidation(prop.Name, prop.Value.GetString())
                            |> ignore

                    let! response = httpClient.Value.SendAsync(request)
                    let! content = response.Content.ReadAsStringAsync()

                    let statusCode = int response.StatusCode

                    // Pretty print if valid JSON
                    let formatted =
                        try
                            let parsed = JsonDocument.Parse(content)
                            JsonSerializer.Serialize(parsed, JsonSerializerOptions(WriteIndented = true))
                        with _ ->
                            content

                    let truncated =
                        if formatted.Length > 10000 then
                            formatted.Substring(0, 10000) + "\n... [truncated]"
                        else
                            formatted

                    return $"HTTP {statusCode}\n\n{truncated}"
            with ex ->
                return $"fetch_json_api error: {ex.Message}"
        }

    [<TarsToolAttribute("download_file",
                        "Downloads a file to a local path. Input JSON: { \"url\": \"https://...\", \"path\": \"local/file.ext\" }")>]
    let downloadFile (args: string) =
        task {
            try
                let doc = JsonDocument.Parse(args)
                let root = doc.RootElement

                let url = root.GetProperty("url").GetString()
                let path = root.GetProperty("path").GetString()

                if String.IsNullOrWhiteSpace url || String.IsNullOrWhiteSpace path then
                    return "download_file error: missing url or path"
                else
                    printfn $"⬇️ Downloading: {url} -> {path}"

                    let fullPath = Path.GetFullPath(path)
                    let dir = Path.GetDirectoryName(fullPath)

                    if not (Directory.Exists dir) then
                        Directory.CreateDirectory(dir) |> ignore

                    let! response = httpClient.Value.GetAsync(url)
                    response.EnsureSuccessStatusCode() |> ignore

                    use! stream = response.Content.ReadAsStreamAsync()
                    use fileStream = File.Create(fullPath)
                    do! stream.CopyToAsync(fileStream)

                    let fileInfo = FileInfo(fullPath)
                    return $"Downloaded {fileInfo.Length} bytes to {fullPath}"
            with ex ->
                return $"download_file error: {ex.Message}"
        }
