namespace Tars.Tools.Standard

open System
open System.Net.Http
open System.Text.Json
open Tars.Tools
open Tars.Core
open Tars.Connectors.EpisodeIngestion

module KnowledgeTools =

    let private httpClient = new HttpClient()

    [<TarsToolAttribute("search_web", "Searches the web using DuckDuckGo. Input: search query")>]
    let webSearch (query: string) =
        task {
            let q = query.Trim()
            printfn $"🔍 WEB SEARCH: %s{q}"

            try
                let url = $"https://html.duckduckgo.com/html/?q={Uri.EscapeDataString(q)}"
                // Use common browser User-Agent
                httpClient.DefaultRequestHeaders.UserAgent.Clear()

                httpClient.DefaultRequestHeaders.UserAgent.ParseAdd(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                )

                let! response = httpClient.GetAsync(url)

                if response.IsSuccessStatusCode then
                    let! content = response.Content.ReadAsStringAsync()
                    // Simple regex to extract Result Title and URL. Pattern: <a class="result__a" href="URL">TITLE</a>
                    let pattern = """<a class="result__a" href="([^"]+)">([^<]+)</a>"""
                    let matches = System.Text.RegularExpressions.Regex.Matches(content, pattern)

                    let results =
                        matches
                        |> Seq.cast<System.Text.RegularExpressions.Match>
                        |> Seq.map (fun m ->
                            let href = m.Groups.[1].Value
                            let title = System.Net.WebUtility.HtmlDecode(m.Groups.[2].Value)
                            (href, title))
                        |> Seq.filter (fun (h, _) -> not (h.Contains("duckduckgo.com")))
                        |> Seq.truncate 5
                        |> Seq.map (fun (h, t) -> $"- {t}: {h}")
                        |> String.concat "\n"

                    if String.IsNullOrWhiteSpace(results) then
                        return $"No results found for '{q}'."
                    else
                        return $"Web Search Results for '{q}':\n{results}\n\nUse http_get to read specific pages."
                else
                    return $"Web Search error: {int response.StatusCode}"
            with ex ->
                return "search_web error: " + ex.Message
        }

    [<TarsToolAttribute("wikidata_search", "Searches Wikidata for entities. Input: search query")>]
    let wikidataSearch (query: string) =
        task {
            let q = query.Trim()
            printfn $"🔍 WIKIDATA SEARCH: %s{q}"

            try
                let url =
                    $"https://www.wikidata.org/w/api.php?action=wbsearchentities&search=%s{Uri.EscapeDataString(q)}&language=en&format=json"

                let! response = httpClient.GetAsync(url)

                if response.IsSuccessStatusCode then
                    let! content = response.Content.ReadAsStringAsync()
                    let doc = JsonDocument.Parse(content)
                    let search = doc.RootElement.GetProperty("search")

                    let mutable resultList = []

                    for item in search.EnumerateArray() do
                        let id = item.GetProperty("id").GetString()
                        let label = item.GetProperty("label").GetString()
                        let mutable descProp = Unchecked.defaultof<JsonElement>

                        let desc =
                            if item.TryGetProperty("description", &descProp) then
                                descProp.GetString()
                            else
                                ""

                        resultList <- resultList @ [ $"  %s{id}: %s{label} - %s{desc}" ]

                    let results = String.concat "\n" resultList

                    if results.Length = 0 then
                        return $"No Wikidata results for '%s{q}'"
                    else
                        return $"Wikidata results for '%s{q}':\n%s{results}"
                else
                    return $"Wikidata API error: %d{int response.StatusCode}"
            with ex ->
                return "wikidata_search error: " + ex.Message
        }

    [<TarsToolAttribute("nuget_search", "Searches NuGet for packages. Input: package name or keyword")>]
    let nugetSearch (query: string) =
        task {
            let q = query.Trim()
            printfn $"📦 NUGET SEARCH: %s{q}"

            try
                let searchUrl =
                    $"https://azuresearch-usnc.nuget.org/query?q=%s{Uri.EscapeDataString(q)}&take=10"

                let! response = httpClient.GetAsync(searchUrl)

                if response.IsSuccessStatusCode then
                    let! content = response.Content.ReadAsStringAsync()
                    let doc = JsonDocument.Parse(content)
                    let data = doc.RootElement.GetProperty("data")

                    let mutable resultList = []

                    for pkg in data.EnumerateArray() do
                        let id = pkg.GetProperty("id").GetString()
                        let version = pkg.GetProperty("version").GetString()
                        let mutable descProp = Unchecked.defaultof<JsonElement>

                        let desc =
                            if pkg.TryGetProperty("description", &descProp) then
                                let d = descProp.GetString()
                                if d.Length > 60 then d.Substring(0, 60) + "..." else d
                            else
                                ""

                        resultList <- resultList @ [ $"  %s{id} (%s{version}): %s{desc}" ]

                    let results = String.concat "\n" resultList

                    if results.Length = 0 then
                        return $"No NuGet packages found for '%s{q}'"
                    else
                        return $"NuGet packages for '%s{q}':\n%s{results}"
                else
                    return $"NuGet search error: %d{int response.StatusCode}"
            with ex ->
                return "nuget_search error: " + ex.Message
        }

    [<TarsToolAttribute("github_repo",
                        "Gets information about a GitHub repository. Input: owner/repo (e.g., 'dotnet/fsharp')")>]
    let githubRepo (repoPath: string) =
        task {
            let path = repoPath.Trim()
            printfn $"🐙 GITHUB REPO: %s{path}"

            try
                let url = $"https://api.github.com/repos/%s{path}"
                httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("TARS/1.0")
                let! response = httpClient.GetAsync(url)

                if response.IsSuccessStatusCode then
                    let! content = response.Content.ReadAsStringAsync()
                    let doc = JsonDocument.Parse(content)
                    let root = doc.RootElement

                    let name = root.GetProperty("full_name").GetString()
                    let stars = root.GetProperty("stargazers_count").GetInt32()
                    let forks = root.GetProperty("forks_count").GetInt32()

                    return $"GitHub: %s{name}\nStars: %d{stars} | Forks: %d{forks}\nURL: https://github.com/%s{path}"
                else
                    return $"GitHub API error: %d{int response.StatusCode}"
            with ex ->
                return "github_repo error: " + ex.Message
        }

    [<TarsToolAttribute("schema_org",
                        "Gets Schema.org type information. Input: type name (e.g., 'Person', 'Organization')")>]
    let schemaOrg (typeName: string) =
        task {
            let name = typeName.Trim()
            printfn $"📋 SCHEMA.ORG: %s{name}"

            let schemas =
                dict
                    [ ("Person", "A person (alive, dead, undead, or fictional)")
                      ("Organization", "An organization such as a school, NGO, corporation, club, etc.")
                      ("Place", "Entities with a physical location")
                      ("Event", "An event happening at a certain time and location")
                      ("Product", "Any offered product or service")
                      ("CreativeWork", "The most generic kind of creative work")
                      ("SoftwareApplication", "A software application")
                      ("Thing", "The most generic type of item")
                      ("Action", "An action performed by a direct agent upon an indirect object")
                      ("Article", "An article, such as a news article or piece of investigative report")
                      ("BreadcrumbList", "A BreadcrumbList is an ItemList consisting of a chain of linked web pages")
                      ("CollegeOrUniversity", "A college, university, or other third-level educational institution")
                      ("Corporation", "Organization: A business corporation")
                      ("EducationalOrganization", "An educational organization")
                      ("GovernmentOrganization", "A governmental organization or agency")
                      ("LocalBusiness", "A particular physical business or branch of an organization")
                      ("MedicalEntity", "The most generic type of entity relating to health or the practice of medicine")
                      ("MessageType", "A type of message that can be sent or received")
                      ("Offer", "An offer to transfer some rights to an item or to provide a service")
                      ("PostalAddress", "The mailing address")
                      ("Recipe", "A strategy off of which a food item is made")
                      ("Review", "A review of an item - for example, of a restaurant, movie, or store")
                      ("SearchAction", "The act of searching for an object")
                      ("WebPage",
                       "A web page. Every web page is implicitly assumed to be declared to be of type WebPage")
                      ("WebSite",
                       "A WebSite is a set of related web pages and other items typically identified with a common domain name") ]

            if schemas.ContainsKey(name) then
                return
                    $"Schema.org Type: %s{name}\nDescription: %s{schemas.[name]}\nURI: https://schema.org/%s{name}\n\nUse this type for structured data in Knowledge Browser or Web content."
            else
                // Dynamic fallback - suggest the URI even if not in local cache
                return
                    $"Schema type '%s{name}' not in local cache.\nPotential URI: https://schema.org/%s{name}\n\nCommon types include: Person, Organization, Place, Event, Product, SoftwareApplication, LocalBusiness."
        }

    /// Create the search tool for Graphiti memory
    let createSearchMemoryTool (ingestionService: IEpisodeIngestionService) =
        Tars.Core.Tool.Create(
            "search_memory",
            "Searches TARS's long-term episodic memory (Knowledge Graph). Input: natural language query.",
            fun args ->
                task {
                    try
                        let query = Tars.Tools.ToolHelpers.parseStringArg args "query"
                        let! resultsResult = ingestionService.SearchAsync(query, Some 10)

                        match resultsResult with
                        | Result.Ok results ->
                            if results.IsEmpty then
                                return Result.Ok "No relevant memories found."
                            else
                                let hits =
                                    results
                                    |> List.map (fun r ->
                                        let fact = r.Fact |> Option.defaultValue ""
                                        $"- %s{r.Name} (Score: %.2f{r.Score}) %s{fact}")
                                    |> String.concat "\n"

                                return Result.Ok hits
                        | Result.Error err -> return Result.Error $"Error searching memory: {err}"
                    with ex ->
                        return Result.Error $"Error searching memory: {ex.Message}"
                }
        )

    /// Create the save_memory tool
    let createSaveMemoryTool (ingestionService: IEpisodeIngestionService) =
        Tars.Core.Tool.Create(
            "save_memory",
            "Saves a fact or belief to TARS's long-term memory. Input: The fact to string to save.",
            fun args ->
                task {
                    try
                        let fact = Tars.Tools.ToolHelpers.parseStringArg args "fact"
                        let ep = Tars.Core.BeliefUpdate("User", fact, 1.0, DateTime.UtcNow)
                        ingestionService.Queue(ep)
                        // Flush to ensure it persists
                        let! _ = ingestionService.FlushAsync()
                        return Result.Ok $"Memory saved: {fact}"
                    with ex ->
                        return Result.Error $"Error saving memory: {ex.Message}"
                }
        )
