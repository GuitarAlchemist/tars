namespace TarsEngine.FSharp.Notebooks.Discovery

open System
open System.IO
open System.Net.Http
open System.Text.Json
open System.Text.RegularExpressions
open TarsEngine.FSharp.Notebooks.Types

/// <summary>
/// Adapters for different notebook sources and platforms
/// </summary>

/// Source adapter interface
type ISourceAdapter =
    /// Search for notebooks on this source
    abstract member SearchAsync: SearchCriteria -> Async<DiscoveredNotebook list>
    
    /// Download notebook from this source
    abstract member DownloadAsync: string -> string -> Async<bool>
    
    /// Get metadata for a notebook
    abstract member GetMetadataAsync: string -> Async<NotebookMetadata option>
    
    /// Check if URL is supported by this adapter
    abstract member SupportsUrl: string -> bool

/// GitHub adapter
type GitHubAdapter(httpClient: HttpClient) =
    
    interface ISourceAdapter with
        member _.SearchAsync(criteria: SearchCriteria) = async {
            try
                let query = $"{criteria.Query} extension:ipynb"
                let languageFilter = 
                    match criteria.Language with
                    | Some lang -> $" language:{lang}"
                    | None -> ""
                
                let topicFilter = 
                    match criteria.Topic with
                    | Some topic -> $" topic:{topic}"
                    | None -> ""
                
                let fullQuery = query + languageFilter + topicFilter
                let url = $"https://api.github.com/search/code?q={Uri.EscapeDataString(fullQuery)}&per_page={criteria.MaxResults}"
                
                let! response = httpClient.GetStringAsync(url) |> Async.AwaitTask
                let jsonDoc = JsonDocument.Parse(response)
                
                let results = 
                    jsonDoc.RootElement.GetProperty("items").EnumerateArray()
                    |> Seq.map (fun item ->
                        let repo = item.GetProperty("repository")
                        {
                            Title = item.GetProperty("name").GetString()
                            Description = 
                                if repo.TryGetProperty("description", &_) && not repo.GetProperty("description").ValueKind.Equals(JsonValueKind.Null) then
                                    Some (repo.GetProperty("description").GetString())
                                else None
                            Url = item.GetProperty("html_url").GetString()
                            Author = Some (repo.GetProperty("owner").GetProperty("login").GetString())
                            Language = Some "jupyter"
                            Topics = 
                                if repo.TryGetProperty("topics", &_) then
                                    repo.GetProperty("topics").EnumerateArray()
                                    |> Seq.map (fun t -> t.GetString())
                                    |> List.ofSeq
                                else []
                            Stars = Some (repo.GetProperty("stargazers_count").GetInt32())
                            LastUpdated = Some (DateTime.Parse(repo.GetProperty("updated_at").GetString()))
                            Size = Some (item.GetProperty("size").GetInt64())
                            Source = GitHub
                        }
                    )
                    |> List.ofSeq
                
                return results
                
            with
            | ex ->
                return []
        }
        
        member _.DownloadAsync(url: string, outputPath: string) = async {
            try
                // Convert GitHub URL to raw content URL
                let rawUrl = 
                    if url.Contains("github.com") && url.Contains("/blob/") then
                        url.Replace("github.com", "raw.githubusercontent.com").Replace("/blob/", "/")
                    else url
                
                let! content = httpClient.GetStringAsync(rawUrl) |> Async.AwaitTask
                File.WriteAllText(outputPath, content)
                return true
            with
            | ex ->
                return false
        }
        
        member _.GetMetadataAsync(url: string) = async {
            try
                // Convert to raw URL and download
                let rawUrl = 
                    if url.Contains("github.com") && url.Contains("/blob/") then
                        url.Replace("github.com", "raw.githubusercontent.com").Replace("/blob/", "/")
                    else url
                
                let! content = httpClient.GetStringAsync(rawUrl) |> Async.AwaitTask
                let jsonDoc = JsonDocument.Parse(content)
                
                if jsonDoc.RootElement.TryGetProperty("metadata", &_) then
                    let metadata = jsonDoc.RootElement.GetProperty("metadata")
                    
                    let title = 
                        if metadata.TryGetProperty("title", &_) then
                            Some (metadata.GetProperty("title").GetString())
                        else None
                    
                    let kernelSpec = 
                        if metadata.TryGetProperty("kernelspec", &_) then
                            let ks = metadata.GetProperty("kernelspec")
                            Some {
                                Name = ks.GetProperty("name").GetString()
                                DisplayName = ks.GetProperty("display_name").GetString()
                                Language = if ks.TryGetProperty("language", &_) then Some (ks.GetProperty("language").GetString()) else None
                            }
                        else None
                    
                    return Some {
                        Title = title
                        Authors = []
                        Description = None
                        Tags = []
                        KernelSpec = kernelSpec
                        LanguageInfo = None
                        CreatedDate = None
                        ModifiedDate = None
                        Version = None
                        Custom = Map.empty
                    }
                else
                    return None
                    
            with
            | ex ->
                return None
        }
        
        member _.SupportsUrl(url: string) =
            url.Contains("github.com") || url.Contains("raw.githubusercontent.com")

/// Kaggle adapter
type KaggleAdapter(httpClient: HttpClient) =
    
    interface ISourceAdapter with
        member _.SearchAsync(criteria: SearchCriteria) = async {
            // Kaggle API implementation would go here
            // For now, return empty list
            return []
        }
        
        member _.DownloadAsync(url: string, outputPath: string) = async {
            // Kaggle download implementation
            return false
        }
        
        member _.GetMetadataAsync(url: string) = async {
            return None
        }
        
        member _.SupportsUrl(url: string) =
            url.Contains("kaggle.com")

/// Google Colab adapter
type ColabAdapter(httpClient: HttpClient) =
    
    interface ISourceAdapter with
        member _.SearchAsync(criteria: SearchCriteria) = async {
            // Google Colab search implementation would go here
            return []
        }
        
        member _.DownloadAsync(url: string, outputPath: string) = async {
            try
                // Convert Colab URL to downloadable format
                let downloadUrl = 
                    if url.Contains("colab.research.google.com") then
                        // Extract file ID and create download URL
                        let pattern = @"fileId=([^&]+)"
                        let m = Regex.Match(url, pattern)
                        if m.Success then
                            let fileId = m.Groups.[1].Value
                            $"https://drive.google.com/uc?export=download&id={fileId}"
                        else url
                    else url
                
                let! content = httpClient.GetStringAsync(downloadUrl) |> Async.AwaitTask
                File.WriteAllText(outputPath, content)
                return true
            with
            | ex ->
                return false
        }
        
        member _.GetMetadataAsync(url: string) = async {
            return None
        }
        
        member _.SupportsUrl(url: string) =
            url.Contains("colab.research.google.com") || url.Contains("drive.google.com")

/// NBViewer adapter
type NBViewerAdapter(httpClient: HttpClient) =
    
    interface ISourceAdapter with
        member _.SearchAsync(criteria: SearchCriteria) = async {
            // NBViewer doesn't have a search API, so return empty
            return []
        }
        
        member _.DownloadAsync(url: string, outputPath: string) = async {
            try
                // NBViewer URLs can be converted to raw GitHub URLs
                let rawUrl = 
                    if url.Contains("nbviewer.jupyter.org") then
                        // Extract the original URL from NBViewer
                        let pattern = @"nbviewer\.jupyter\.org/github/([^/]+)/([^/]+)/blob/([^/]+)/(.+)"
                        let m = Regex.Match(url, pattern)
                        if m.Success then
                            let owner = m.Groups.[1].Value
                            let repo = m.Groups.[2].Value
                            let branch = m.Groups.[3].Value
                            let path = m.Groups.[4].Value
                            $"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
                        else url
                    else url
                
                let! content = httpClient.GetStringAsync(rawUrl) |> Async.AwaitTask
                File.WriteAllText(outputPath, content)
                return true
            with
            | ex ->
                return false
        }
        
        member _.GetMetadataAsync(url: string) = async {
            return None
        }
        
        member _.SupportsUrl(url: string) =
            url.Contains("nbviewer.jupyter.org")

/// Local filesystem adapter
type LocalAdapter() =
    
    interface ISourceAdapter with
        member _.SearchAsync(criteria: SearchCriteria) = async {
            let searchPaths = [
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)
                Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)
                @"C:\Users"
                @"D:\Projects"
                @"C:\Projects"
            ]
            
            let results = ResizeArray<DiscoveredNotebook>()
            
            for searchPath in searchPaths do
                if Directory.Exists(searchPath) then
                    try
                        let files = Directory.GetFiles(searchPath, "*.ipynb", SearchOption.AllDirectories)
                        
                        for file in files do
                            let fileName = Path.GetFileNameWithoutExtension(file)
                            if fileName.Contains(criteria.Query, StringComparison.OrdinalIgnoreCase) then
                                let fileInfo = FileInfo(file)
                                results.Add({
                                    Title = fileName
                                    Description = None
                                    Url = $"file://{file}"
                                    Author = None
                                    Language = Some "jupyter"
                                    Topics = []
                                    Stars = None
                                    LastUpdated = Some fileInfo.LastWriteTime
                                    Size = Some fileInfo.Length
                                    Source = Local
                                })
                    with
                    | _ -> () // Ignore access errors
            
            return results |> List.ofSeq
        }
        
        member _.DownloadAsync(url: string, outputPath: string) = async {
            try
                if url.StartsWith("file://") then
                    let sourcePath = url.Substring(7) // Remove "file://"
                    if File.Exists(sourcePath) then
                        File.Copy(sourcePath, outputPath, true)
                        return true
                    else
                        return false
                else
                    return false
            with
            | ex ->
                return false
        }
        
        member _.GetMetadataAsync(url: string) = async {
            try
                if url.StartsWith("file://") then
                    let filePath = url.Substring(7)
                    if File.Exists(filePath) then
                        let content = File.ReadAllText(filePath)
                        let jsonDoc = JsonDocument.Parse(content)
                        
                        if jsonDoc.RootElement.TryGetProperty("metadata", &_) then
                            let metadata = jsonDoc.RootElement.GetProperty("metadata")
                            
                            let title = 
                                if metadata.TryGetProperty("title", &_) then
                                    Some (metadata.GetProperty("title").GetString())
                                else None
                            
                            return Some {
                                Title = title
                                Authors = []
                                Description = None
                                Tags = []
                                KernelSpec = None
                                LanguageInfo = None
                                CreatedDate = None
                                ModifiedDate = None
                                Version = None
                                Custom = Map.empty
                            }
                        else
                            return None
                    else
                        return None
                else
                    return None
            with
            | ex ->
                return None
        }
        
        member _.SupportsUrl(url: string) =
            url.StartsWith("file://") || (not (url.Contains("://")) && File.Exists(url))

/// Source adapter factory
type SourceAdapterFactory(httpClient: HttpClient) =
    
    let adapters = [
        GitHubAdapter(httpClient) :> ISourceAdapter
        KaggleAdapter(httpClient) :> ISourceAdapter
        ColabAdapter(httpClient) :> ISourceAdapter
        NBViewerAdapter(httpClient) :> ISourceAdapter
        LocalAdapter() :> ISourceAdapter
    ]
    
    /// Get adapter for URL
    member _.GetAdapter(url: string) : ISourceAdapter option =
        adapters |> List.tryFind (fun adapter -> adapter.SupportsUrl(url))
    
    /// Get adapter for source type
    member _.GetAdapterForSource(source: SearchSource) : ISourceAdapter =
        match source with
        | GitHub -> GitHubAdapter(httpClient) :> ISourceAdapter
        | Kaggle -> KaggleAdapter(httpClient) :> ISourceAdapter
        | GoogleColab -> ColabAdapter(httpClient) :> ISourceAdapter
        | NBViewer -> NBViewerAdapter(httpClient) :> ISourceAdapter
        | Local -> LocalAdapter() :> ISourceAdapter
    
    /// Get all adapters
    member _.GetAllAdapters() : ISourceAdapter list = adapters
