namespace TarsEngine.FSharp.TARSX.GrammarFetcher

open System
open System.Net.Http
open System.Threading.Tasks
open System.Collections.Generic
open System.IO
open Newtonsoft.Json

/// Internet Grammar Fetcher
/// Downloads EBNF/ANTLR grammars from the internet for dynamic language support
module InternetGrammarFetcher =
    
    /// Grammar Source Information
    type GrammarSource = {
        Language: string
        Url: string
        Format: GrammarFormat
        Description: string
        LastUpdated: DateTime option
        CacheKey: string
    }
    and GrammarFormat = 
        | EBNF | ANTLR | Yacc | PEG | Custom of string
    
    /// Grammar Fetch Result
    type GrammarFetchResult = {
        Success: bool
        Language: string
        Content: string
        Format: GrammarFormat
        Source: string
        FetchedAt: DateTime
        ErrorMessage: string option
        CacheHit: bool
    }
    
    /// Grammar Cache Entry
    type GrammarCacheEntry = {
        Content: string
        Format: GrammarFormat
        FetchedAt: DateTime
        ExpiresAt: DateTime
        Source: string
    }
    
    /// Built-in Grammar Sources
    let builtInGrammarSources = [
        {
            Language = "PYTHON"
            Url = "https://raw.githubusercontent.com/python/cpython/main/Grammar/python.gram"
            Format = PEG
            Description = "Official Python grammar from CPython repository"
            LastUpdated = None
            CacheKey = "python_official"
        }
        {
            Language = "CSHARP"
            Url = "https://raw.githubusercontent.com/antlr/grammars-v4/master/csharp/CSharpLexer.g4"
            Format = ANTLR
            Description = "C# grammar from ANTLR grammars repository"
            LastUpdated = None
            CacheKey = "csharp_antlr"
        }
        {
            Language = "JAVASCRIPT"
            Url = "https://raw.githubusercontent.com/antlr/grammars-v4/master/javascript/javascript/JavaScriptLexer.g4"
            Format = ANTLR
            Description = "JavaScript grammar from ANTLR grammars repository"
            LastUpdated = None
            CacheKey = "javascript_antlr"
        }
        {
            Language = "RUST"
            Url = "https://raw.githubusercontent.com/antlr/grammars-v4/master/rust/RustLexer.g4"
            Format = ANTLR
            Description = "Rust grammar from ANTLR grammars repository"
            LastUpdated = None
            CacheKey = "rust_antlr"
        }
        {
            Language = "SQL"
            Url = "https://raw.githubusercontent.com/antlr/grammars-v4/master/sql/mysql/Positive-Technologies/MySqlLexer.g4"
            Format = ANTLR
            Description = "MySQL grammar from ANTLR grammars repository"
            LastUpdated = None
            CacheKey = "mysql_antlr"
        }
        {
            Language = "MERMAID"
            Url = "https://raw.githubusercontent.com/mermaid-js/mermaid/develop/packages/mermaid/src/diagrams/flowchart/parser/flow.jison"
            Format = Custom "Jison"
            Description = "Mermaid flowchart grammar from official repository"
            LastUpdated = None
            CacheKey = "mermaid_flowchart"
        }
        {
            Language = "GRAPHQL"
            Url = "https://raw.githubusercontent.com/antlr/grammars-v4/master/graphql/GraphQL.g4"
            Format = ANTLR
            Description = "GraphQL grammar from ANTLR grammars repository"
            LastUpdated = None
            CacheKey = "graphql_antlr"
        }
        {
            Language = "YAML"
            Url = "https://raw.githubusercontent.com/antlr/grammars-v4/master/yaml/YAML.g4"
            Format = ANTLR
            Description = "YAML grammar from ANTLR grammars repository"
            LastUpdated = None
            CacheKey = "yaml_antlr"
        }
    ]
    
    /// Grammar Cache (in-memory with persistence)
    let private grammarCache = Dictionary<string, GrammarCacheEntry>()
    let private cacheFilePath = Path.Combine(".tars", "grammar-cache.json")
    
    /// HTTP Client for fetching grammars
    let private httpClient = new HttpClient()
    
    /// Initialize grammar cache
    let initializeGrammarCache () =
        try
            if File.Exists(cacheFilePath) then
                let json = File.ReadAllText(cacheFilePath)
                let cacheData = JsonConvert.DeserializeObject<Dictionary<string, GrammarCacheEntry>>(json)
                for kvp in cacheData do
                    if kvp.Value.ExpiresAt > DateTime.UtcNow then
                        grammarCache.[kvp.Key] <- kvp.Value
                printfn "📦 Grammar cache loaded: %d entries" grammarCache.Count
            else
                printfn "📦 Grammar cache initialized (empty)"
        with
        | ex -> printfn "⚠️ Failed to load grammar cache: %s" ex.Message
    
    /// Save grammar cache to disk
    let saveGrammarCache () =
        try
            let cacheDir = Path.GetDirectoryName(cacheFilePath)
            if not (Directory.Exists(cacheDir)) then
                Directory.CreateDirectory(cacheDir) |> ignore
            
            let json = JsonConvert.SerializeObject(grammarCache, Formatting.Indented)
            File.WriteAllText(cacheFilePath, json)
            printfn "💾 Grammar cache saved: %d entries" grammarCache.Count
        with
        | ex -> printfn "⚠️ Failed to save grammar cache: %s" ex.Message
    
    /// Fetch grammar from URL
    let fetchGrammarFromUrl (url: string) : Task<Result<string, string>> =
        task {
            try
                printfn "🌐 Fetching grammar from: %s" url
                let! response = httpClient.GetAsync(url)
                if response.IsSuccessStatusCode then
                    let! content = response.Content.ReadAsStringAsync()
                    printfn "✅ Grammar fetched successfully (%d bytes)" content.Length
                    return Ok content
                else
                    let errorMsg = sprintf "HTTP %d: %s" (int response.StatusCode) (response.ReasonPhrase)
                    printfn "❌ Failed to fetch grammar: %s" errorMsg
                    return Error errorMsg
            with
            | ex -> 
                let errorMsg = sprintf "Network error: %s" ex.Message
                printfn "❌ Failed to fetch grammar: %s" errorMsg
                return Error errorMsg
        }
    
    /// Get grammar from cache or fetch from internet
    let getGrammar (language: string) : Task<GrammarFetchResult> =
        task {
            let cacheKey = language.ToLowerInvariant()
            
            // Check cache first
            match grammarCache.TryGetValue(cacheKey) with
            | true, cached when cached.ExpiresAt > DateTime.UtcNow ->
                printfn "📦 Grammar cache hit for %s" language
                return {
                    Success = true
                    Language = language
                    Content = cached.Content
                    Format = cached.Format
                    Source = cached.Source
                    FetchedAt = cached.FetchedAt
                    ErrorMessage = None
                    CacheHit = true
                }
            | _ ->
                // Find grammar source
                match builtInGrammarSources |> List.tryFind (fun gs -> gs.Language.Equals(language, StringComparison.OrdinalIgnoreCase)) with
                | Some grammarSource ->
                    // Fetch from internet
                    let! fetchResult = fetchGrammarFromUrl grammarSource.Url
                    match fetchResult with
                    | Ok content ->
                        // Cache the result
                        let cacheEntry = {
                            Content = content
                            Format = grammarSource.Format
                            FetchedAt = DateTime.UtcNow
                            ExpiresAt = DateTime.UtcNow.AddHours(24.0) // Cache for 24 hours
                            Source = grammarSource.Url
                        }
                        grammarCache.[cacheKey] <- cacheEntry
                        saveGrammarCache()
                        
                        return {
                            Success = true
                            Language = language
                            Content = content
                            Format = grammarSource.Format
                            Source = grammarSource.Url
                            FetchedAt = DateTime.UtcNow
                            ErrorMessage = None
                            CacheHit = false
                        }
                    | Error errorMsg ->
                        return {
                            Success = false
                            Language = language
                            Content = ""
                            Format = EBNF
                            Source = grammarSource.Url
                            FetchedAt = DateTime.UtcNow
                            ErrorMessage = Some errorMsg
                            CacheHit = false
                        }
                | None ->
                    let errorMsg = sprintf "No grammar source found for language: %s" language
                    printfn "❌ %s" errorMsg
                    return {
                        Success = false
                        Language = language
                        Content = ""
                        Format = EBNF
                        Source = ""
                        FetchedAt = DateTime.UtcNow
                        ErrorMessage = Some errorMsg
                        CacheHit = false
                    }
        }
    
    /// Fetch grammar from custom URL
    let fetchCustomGrammar (language: string) (url: string) (format: GrammarFormat) : Task<GrammarFetchResult> =
        task {
            let! fetchResult = fetchGrammarFromUrl url
            match fetchResult with
            | Ok content ->
                // Cache the custom grammar
                let cacheKey = sprintf "%s_custom_%s" (language.ToLowerInvariant()) (url.GetHashCode().ToString())
                let cacheEntry = {
                    Content = content
                    Format = format
                    FetchedAt = DateTime.UtcNow
                    ExpiresAt = DateTime.UtcNow.AddHours(24.0)
                    Source = url
                }
                grammarCache.[cacheKey] <- cacheEntry
                saveGrammarCache()
                
                return {
                    Success = true
                    Language = language
                    Content = content
                    Format = format
                    Source = url
                    FetchedAt = DateTime.UtcNow
                    ErrorMessage = None
                    CacheHit = false
                }
            | Error errorMsg ->
                return {
                    Success = false
                    Language = language
                    Content = ""
                    Format = format
                    Source = url
                    FetchedAt = DateTime.UtcNow
                    ErrorMessage = Some errorMsg
                    CacheHit = false
                }
        }
    
    /// Get supported languages
    let getSupportedLanguages () : string list =
        builtInGrammarSources |> List.map (fun gs -> gs.Language)
    
    /// Get grammar source info
    let getGrammarSourceInfo (language: string) : GrammarSource option =
        builtInGrammarSources |> List.tryFind (fun gs -> gs.Language.Equals(language, StringComparison.OrdinalIgnoreCase))
    
    /// Clear grammar cache
    let clearGrammarCache () =
        grammarCache.Clear()
        if File.Exists(cacheFilePath) then
            File.Delete(cacheFilePath)
        printfn "🗑️ Grammar cache cleared"
    
    /// Get cache statistics
    let getCacheStatistics () =
        let totalEntries = grammarCache.Count
        let expiredEntries = grammarCache.Values |> Seq.filter (fun entry -> entry.ExpiresAt <= DateTime.UtcNow) |> Seq.length
        let validEntries = totalEntries - expiredEntries
        
        printfn "📊 Grammar Cache Statistics:"
        printfn "   Total entries: %d" totalEntries
        printfn "   Valid entries: %d" validEntries
        printfn "   Expired entries: %d" expiredEntries
        printfn "   Supported languages: %d" builtInGrammarSources.Length
        
        (totalEntries, validEntries, expiredEntries)
    
    // Initialize cache on module load
    do initializeGrammarCache()
    
    printfn "🌐 Internet Grammar Fetcher Loaded"
    printfn "=================================="
    printfn "✅ Built-in grammar sources: %d" builtInGrammarSources.Length
    printfn "✅ Cache system initialized"
    printfn "✅ HTTP client ready"
    printfn "✅ Supported languages: %s" (String.concat ", " (getSupportedLanguages()))
    printfn ""
    printfn "🎯 Ready to fetch grammars from the internet!"
