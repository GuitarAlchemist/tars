// Test script for new TARS tools
// Run with: dotnet fsi tests/Scripts/test-new-tools.fsx

#r "nuget: System.Text.Json"

open System
open System.Net.Http
open System.Threading.Tasks

let httpClient = new HttpClient()
httpClient.Timeout <- TimeSpan.FromSeconds(30.0)
httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("TARS/2.0 (Test)")

printfn "🧪 Testing new TARS tools..."
printfn ""

// Test 1: Wikipedia API
printfn "📚 Test 1: Wikipedia API"

let wikiTest =
    async {
        try
            let! response =
                httpClient.GetAsync("https://en.wikipedia.org/api/rest_v1/page/summary/F_Sharp_(programming_language)")
                |> Async.AwaitTask

            let! content = response.Content.ReadAsStringAsync() |> Async.AwaitTask

            if response.IsSuccessStatusCode then
                printfn "   ✅ Wikipedia API works! Got %d chars" content.Length
            else
                printfn "   ❌ Wikipedia API failed: %A" response.StatusCode
        with ex ->
            printfn "   ❌ Wikipedia API error: %s" ex.Message
    }

Async.RunSynchronously wikiTest

// Test 2: arXiv API
printfn "📄 Test 2: arXiv API"

let arxivTest =
    async {
        try
            let! response =
                httpClient.GetAsync("http://export.arxiv.org/api/query?search_query=all:attention&max_results=1")
                |> Async.AwaitTask

            let! content = response.Content.ReadAsStringAsync() |> Async.AwaitTask

            if response.IsSuccessStatusCode && content.Contains("<entry>") then
                printfn "   ✅ arXiv API works! Got search results"
            else
                printfn "   ⚠️ arXiv API returned no entries"
        with ex ->
            printfn "   ❌ arXiv API error: %s" ex.Message
    }

Async.RunSynchronously arxivTest

// Test 3: GitHub API
printfn "🐙 Test 3: GitHub API"

let githubTest =
    async {
        try
            use request =
                new HttpRequestMessage(HttpMethod.Get, "https://api.github.com/repos/dotnet/fsharp/readme")

            request.Headers.Accept.ParseAdd("application/vnd.github.v3.raw")
            let! response = httpClient.SendAsync(request) |> Async.AwaitTask

            if response.IsSuccessStatusCode then
                let! content = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                printfn "   ✅ GitHub API works! Got %d chars from F# repo README" content.Length
            else
                printfn "   ⚠️ GitHub API returned: %A" response.StatusCode
        with ex ->
            printfn "   ❌ GitHub API error: %s" ex.Message
    }

Async.RunSynchronously githubTest

// Test 4: DuckDuckGo HTML search
printfn "🔍 Test 4: DuckDuckGo Search"

let ddgTest =
    async {
        try
            let query = Uri.EscapeDataString("F# programming language")

            let! response =
                httpClient.GetAsync($"https://html.duckduckgo.com/html/?q={query}")
                |> Async.AwaitTask

            let! content = response.Content.ReadAsStringAsync() |> Async.AwaitTask

            if response.IsSuccessStatusCode && content.Contains("result__a") then
                printfn "   ✅ DuckDuckGo search works! Found search results"
            else
                printfn "   ⚠️ DuckDuckGo search returned no results"
        with ex ->
            printfn "   ❌ DuckDuckGo error: %s" ex.Message
    }

Async.RunSynchronously ddgTest

// Test 5: Semantic Scholar API
printfn "🔬 Test 5: Semantic Scholar API"

let s2Test =
    async {
        try
            let! response =
                httpClient.GetAsync(
                    "https://api.semanticscholar.org/graph/v1/paper/search?query=neural+network&limit=1&fields=title"
                )
                |> Async.AwaitTask

            let! content = response.Content.ReadAsStringAsync() |> Async.AwaitTask

            if response.IsSuccessStatusCode && content.Contains("\"data\"") then
                printfn "   ✅ Semantic Scholar API works!"
            else
                printfn "   ⚠️ Semantic Scholar returned: %A" response.StatusCode
        with ex ->
            printfn "   ❌ Semantic Scholar error: %s" ex.Message
    }

Async.RunSynchronously s2Test

printfn ""
printfn "🎉 Tool API tests complete!"
printfn ""
printfn "Summary of new TARS tools:"

printfn
    "  📦 WebTools.fs: 7 tools (fetch_webpage, fetch_wikipedia, fetch_github_readme, extract_links, search_web, fetch_json_api, download_file)"

printfn
    "  📦 CodeAnalysisTools.fs: 5 tools (analyze_file_complexity, find_code_smells, extract_symbols, compare_files, find_duplicates)"

printfn "  📦 ResearchTools.fs: 3 tools (fetch_arxiv, fetch_doi, search_semantic_scholar)"

printfn
    "  📦 GraphTools.fs: 8 tools (graph_add_node, graph_add_edge, graph_get_neighborhood, graph_query, graph_stats, graph_export_json, graph_find_contradictions, graph_clear)"

printfn ""
printfn "Total: 124 tools 🔧"
