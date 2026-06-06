namespace Tars.Tools.Research

open System
open System.Net.Http
open System.Text.RegularExpressions
open System.Text.Json
open Tars.Tools

/// Tools for arXiv paper retrieval and PDF handling
module ResearchTools =

    let private httpClient =
        lazy
            (let client = new HttpClient()
             client.Timeout <- TimeSpan.FromSeconds(60.0)

             client.DefaultRequestHeaders.UserAgent.ParseAdd(
                 "TARS/2.0 (Research Assistant; +https://github.com/GuitarAlchemist/tars)"
             )

             client)

    [<TarsToolAttribute("fetch_arxiv",
                        "Fetches an arXiv paper's metadata and abstract. Input JSON: { \"id\": \"2301.00001\" } or { \"query\": \"transformer attention\" }")>]
    let fetchArxiv (args: string) =
        task {
            try
                let doc = JsonDocument.Parse(args)
                let root = doc.RootElement

                let mutable idProp = Unchecked.defaultof<JsonElement>
                let mutable queryProp = Unchecked.defaultof<JsonElement>

                if root.TryGetProperty("id", &idProp) then
                    // Fetch specific paper by ID
                    let arxivId = idProp.GetString()
                    let apiUrl = $"http://export.arxiv.org/api/query?id_list={arxivId}"

                    printfn $"📄 Fetching arXiv paper: {arxivId}"
                    let! response = httpClient.Value.GetAsync(apiUrl)
                    let! xml = response.Content.ReadAsStringAsync()

                    // Parse Atom XML (simplified)
                    let titleMatch =
                        Regex.Match(xml, @"<title[^>]*>([^<]+)</title>", RegexOptions.Singleline)

                    let summaryMatch =
                        Regex.Match(xml, @"<summary[^>]*>([\s\S]*?)</summary>", RegexOptions.Singleline)

                    let authorsMatches =
                        Regex.Matches(xml, @"<author>\s*<name>([^<]+)</name>", RegexOptions.Singleline)

                    let publishedMatch = Regex.Match(xml, @"<published>([^<]+)</published>")

                    let title =
                        if titleMatch.Success then
                            titleMatch.Groups.[1].Value.Trim()
                        else
                            "Unknown"

                    let summary =
                        if summaryMatch.Success then
                            Regex.Replace(summaryMatch.Groups.[1].Value, @"\s+", " ").Trim()
                        else
                            "No abstract available"

                    let authors =
                        authorsMatches
                        |> Seq.cast<Match>
                        |> Seq.map (fun m -> m.Groups.[1].Value)
                        |> String.concat ", "

                    let published =
                        if publishedMatch.Success then
                            publishedMatch.Groups.[1].Value.Substring(0, 10)
                        else
                            "Unknown"

                    return
                        $"# arXiv:{arxivId}\n\n**Title:** {title}\n**Authors:** {authors}\n**Published:** {published}\n**Link:** https://arxiv.org/abs/{arxivId}\n\n## Abstract\n{summary}"

                elif root.TryGetProperty("query", &queryProp) then
                    // Search for papers
                    let query = queryProp.GetString()
                    let encoded = Uri.EscapeDataString(query)

                    let apiUrl =
                        $"http://export.arxiv.org/api/query?search_query=all:{encoded}&start=0&max_results=5"

                    printfn $"🔍 Searching arXiv: {query}"
                    let! response = httpClient.Value.GetAsync(apiUrl)
                    let! xml = response.Content.ReadAsStringAsync()

                    // Parse multiple entries
                    let entries = Regex.Matches(xml, @"<entry>([\s\S]*?)</entry>")

                    let results =
                        entries
                        |> Seq.cast<Match>
                        |> Seq.mapi (fun i entry ->
                            let content = entry.Groups.[1].Value
                            let idMatch = Regex.Match(content, @"<id>http://arxiv.org/abs/([^<]+)</id>")
                            let titleMatch = Regex.Match(content, @"<title>([^<]+)</title>")
                            let summaryMatch = Regex.Match(content, @"<summary>([\s\S]*?)</summary>")

                            let id = if idMatch.Success then idMatch.Groups.[1].Value else "?"

                            let title =
                                if titleMatch.Success then
                                    titleMatch.Groups.[1].Value.Trim()
                                else
                                    "?"

                            let summary =
                                if summaryMatch.Success then
                                    let s = summaryMatch.Groups.[1].Value
                                    if s.Length > 200 then s.Substring(0, 200) + "..." else s
                                else
                                    ""

                            let cleanSummary = Regex.Replace(summary, @"\s+", " ").Trim()
                            let num = i + 1
                            $"{num}. **[{id}]** {title}\n   {cleanSummary}")
                        |> String.concat "\n\n"

                    return $"# arXiv Search: \"{query}\"\n\n{results}"
                else
                    return "fetch_arxiv error: provide either 'id' or 'query' parameter"
            with ex ->
                return $"fetch_arxiv error: {ex.Message}"
        }

    [<TarsToolAttribute("fetch_doi", "Fetches paper metadata by DOI. Input JSON: { \"doi\": \"10.1000/xyz123\" }")>]
    let fetchDoi (args: string) =
        task {
            try
                let doi = ToolHelpers.parseStringArg args "doi"

                if String.IsNullOrWhiteSpace doi then
                    return "fetch_doi error: missing doi"
                else
                    printfn $"📚 Fetching DOI: {doi}"

                    use request = new HttpRequestMessage(HttpMethod.Get, $"https://doi.org/{doi}")
                    request.Headers.Accept.ParseAdd("application/json")

                    let! response = httpClient.Value.SendAsync(request)
                    let! json = response.Content.ReadAsStringAsync()

                    try
                        let doc = JsonDocument.Parse(json)
                        let root = doc.RootElement

                        let title =
                            let mutable p = Unchecked.defaultof<JsonElement>

                            if root.TryGetProperty("title", &p) then
                                p.GetString()
                            else
                                "Unknown"

                        let authors =
                            let mutable p = Unchecked.defaultof<JsonElement>

                            if root.TryGetProperty("author", &p) then
                                p.EnumerateArray()
                                |> Seq.map (fun a ->
                                    let mutable given = Unchecked.defaultof<JsonElement>
                                    let mutable family = Unchecked.defaultof<JsonElement>

                                    let g =
                                        if a.TryGetProperty("given", &given) then
                                            given.GetString()
                                        else
                                            ""

                                    let f =
                                        if a.TryGetProperty("family", &family) then
                                            family.GetString()
                                        else
                                            ""

                                    $"{g} {f}")
                                |> String.concat ", "
                            else
                                "Unknown"

                        return
                            $"# DOI: {doi}\n\n**Title:** {title}\n**Authors:** {authors}\n**Link:** https://doi.org/{doi}"
                    with _ ->
                        return $"Paper found but metadata format unexpected. View at: https://doi.org/{doi}"
            with ex ->
                return $"fetch_doi error: {ex.Message}"
        }

    [<TarsToolAttribute("search_semantic_scholar",
                        "Searches Semantic Scholar for papers. Input JSON: { \"query\": \"neural network\", \"limit\": 5 }")>]
    let searchSemanticScholar (args: string) =
        task {
            try
                let query = ToolHelpers.parseStringArg args "query"

                let limit =
                    try
                        let doc = JsonDocument.Parse(args)
                        let mutable prop = Unchecked.defaultof<JsonElement>

                        if doc.RootElement.TryGetProperty("limit", &prop) then
                            prop.GetInt32()
                        else
                            5
                    with _ ->
                        5

                if String.IsNullOrWhiteSpace query then
                    return "search_semantic_scholar error: missing query"
                else
                    let encoded = Uri.EscapeDataString(query)

                    let apiUrl =
                        $"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit={limit}&fields=title,authors,year,citationCount,abstract"

                    printfn $"🔬 Searching Semantic Scholar: {query}"
                    let! response = httpClient.Value.GetAsync(apiUrl)
                    let! json = response.Content.ReadAsStringAsync()

                    let doc = JsonDocument.Parse(json)
                    let mutable dataProp = Unchecked.defaultof<JsonElement>

                    if doc.RootElement.TryGetProperty("data", &dataProp) then
                        let results =
                            dataProp.EnumerateArray()
                            |> Seq.mapi (fun i paper ->
                                let mutable titleProp = Unchecked.defaultof<JsonElement>
                                let mutable yearProp = Unchecked.defaultof<JsonElement>
                                let mutable citeProp = Unchecked.defaultof<JsonElement>
                                let mutable abstractProp = Unchecked.defaultof<JsonElement>

                                let title =
                                    if paper.TryGetProperty("title", &titleProp) then
                                        titleProp.GetString()
                                    else
                                        "?"

                                let year =
                                    if
                                        paper.TryGetProperty("year", &yearProp)
                                        && yearProp.ValueKind <> JsonValueKind.Null
                                    then
                                        yearProp.GetInt32().ToString()
                                    else
                                        "?"

                                let citations =
                                    if
                                        paper.TryGetProperty("citationCount", &citeProp)
                                        && citeProp.ValueKind <> JsonValueKind.Null
                                    then
                                        citeProp.GetInt32()
                                    else
                                        0

                                let abstr =
                                    if
                                        paper.TryGetProperty("abstract", &abstractProp)
                                        && abstractProp.ValueKind <> JsonValueKind.Null
                                    then
                                        let s = abstractProp.GetString()
                                        if s.Length > 150 then s.Substring(0, 150) + "..." else s
                                    else
                                        ""

                                $"{i + 1}. **{title}** ({year})\n   📊 {citations} citations\n   {abstr}")
                            |> String.concat "\n\n"

                        return $"# Semantic Scholar: \"{query}\"\n\n{results}"
                    else
                        return "No results found"
            with ex ->
                return $"search_semantic_scholar error: {ex.Message}"
        }
