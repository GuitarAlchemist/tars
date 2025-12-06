namespace Tars.Tools.Standard

open System
open System.Net.Http
open System.Text.Json
open Tars.Tools

module KnowledgeTools =

    let private httpClient = new HttpClient()

    [<TarsToolAttribute("wikidata_search", "Searches Wikidata for entities. Input: search query")>]
    let wikidataSearch (query: string) =
        task {
            let q = query.Trim()
            printfn "🔍 WIKIDATA SEARCH: %s" q

            try
                let url =
                    sprintf
                        "https://www.wikidata.org/w/api.php?action=wbsearchentities&search=%s&language=en&format=json"
                        (Uri.EscapeDataString(q))

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

                        resultList <- resultList @ [ sprintf "  %s: %s - %s" id label desc ]

                    let results = String.concat "\n" resultList

                    if results.Length = 0 then
                        return sprintf "No Wikidata results for '%s'" q
                    else
                        return sprintf "Wikidata results for '%s':\n%s" q results
                else
                    return sprintf "Wikidata API error: %d" (int response.StatusCode)
            with ex ->
                return "wikidata_search error: " + ex.Message
        }

    [<TarsToolAttribute("nuget_search", "Searches NuGet for packages. Input: package name or keyword")>]
    let nugetSearch (query: string) =
        task {
            let q = query.Trim()
            printfn "📦 NUGET SEARCH: %s" q

            try
                let searchUrl =
                    sprintf "https://azuresearch-usnc.nuget.org/query?q=%s&take=10" (Uri.EscapeDataString(q))

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

                        resultList <- resultList @ [ sprintf "  %s (%s): %s" id version desc ]

                    let results = String.concat "\n" resultList

                    if results.Length = 0 then
                        return sprintf "No NuGet packages found for '%s'" q
                    else
                        return sprintf "NuGet packages for '%s':\n%s" q results
                else
                    return sprintf "NuGet search error: %d" (int response.StatusCode)
            with ex ->
                return "nuget_search error: " + ex.Message
        }

    [<TarsToolAttribute("github_repo",
                        "Gets information about a GitHub repository. Input: owner/repo (e.g., 'dotnet/fsharp')")>]
    let githubRepo (repoPath: string) =
        task {
            let path = repoPath.Trim()
            printfn "🐙 GITHUB REPO: %s" path

            try
                let url = sprintf "https://api.github.com/repos/%s" path
                httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("TARS/1.0")
                let! response = httpClient.GetAsync(url)

                if response.IsSuccessStatusCode then
                    let! content = response.Content.ReadAsStringAsync()
                    let doc = JsonDocument.Parse(content)
                    let root = doc.RootElement

                    let name = root.GetProperty("full_name").GetString()
                    let stars = root.GetProperty("stargazers_count").GetInt32()
                    let forks = root.GetProperty("forks_count").GetInt32()

                    return sprintf "GitHub: %s\nStars: %d | Forks: %d\nURL: https://github.com/%s" name stars forks path
                else
                    return sprintf "GitHub API error: %d" (int response.StatusCode)
            with ex ->
                return "github_repo error: " + ex.Message
        }

    [<TarsToolAttribute("schema_org",
                        "Gets Schema.org type information. Input: type name (e.g., 'Person', 'Organization')")>]
    let schemaOrg (typeName: string) =
        task {
            let name = typeName.Trim()
            printfn "📋 SCHEMA.ORG: %s" name

            let schemas =
                dict
                    [ ("Person", "A person (alive, dead, undead, or fictional)")
                      ("Organization", "An organization such as a school, NGO, corporation, club, etc.")
                      ("Place", "Entities with a physical location")
                      ("Event", "An event happening at a certain time and location")
                      ("Product", "Any offered product or service")
                      ("CreativeWork", "The most generic kind of creative work")
                      ("SoftwareApplication", "A software application")
                      ("Thing", "The most generic type of item") ]

            if schemas.ContainsKey(name) then
                return sprintf "Schema.org: %s\nDescription: %s\nURI: https://schema.org/%s" name schemas.[name] name
            else
                return
                    sprintf "Schema type '%s' not found. Common types: Person, Organization, Place, Event, Product" name
        }
