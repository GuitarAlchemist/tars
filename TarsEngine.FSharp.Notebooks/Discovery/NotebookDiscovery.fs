namespace TarsEngine.FSharp.Notebooks.Discovery

open System
open System.IO
open System.Net.Http
open System.Text.Json
open System.Text.RegularExpressions
open TarsEngine.FSharp.Notebooks.Types

/// <summary>
/// Discovery and search functionality for Jupyter notebooks
/// </summary>

/// Search source types
type SearchSource = 
    | GitHub
    | Kaggle
    | GoogleColab
    | NBViewer
    | Local

/// Search criteria
type SearchCriteria = {
    Query: string
    Source: SearchSource
    Language: string option
    Topic: string option
    MaxResults: int
    SortBy: SortOption
}

/// Sort options
and SortOption = 
    | Relevance
    | Stars
    | Updated
    | Created
    | Name

/// Discovered notebook
type DiscoveredNotebook = {
    Title: string
    Description: string option
    Url: string
    Author: string option
    Language: string option
    Topics: string list
    Stars: int option
    LastUpdated: DateTime option
    Size: int64 option
    Source: SearchSource
}

/// Search result
type SearchResult = {
    Query: string
    Source: SearchSource
    TotalCount: int
    Results: DiscoveredNotebook list
    SearchTime: TimeSpan
    NextPageToken: string option
}

/// Notebook discovery service
type NotebookDiscoveryService(httpClient: HttpClient) =
    
    /// Search for notebooks
    member _.SearchAsync(criteria: SearchCriteria) : Async<SearchResult> = async {
        let startTime = DateTime.UtcNow
        
        try
            let! results = 
                match criteria.Source with
                | GitHub -> this.SearchGitHubAsync(criteria)
                | Kaggle -> this.SearchKaggleAsync(criteria)
                | GoogleColab -> this.SearchColabAsync(criteria)
                | NBViewer -> this.SearchNBViewerAsync(criteria)
                | Local -> this.SearchLocalAsync(criteria)
            
            let searchTime = DateTime.UtcNow - startTime
            
            return {
                Query = criteria.Query
                Source = criteria.Source
                TotalCount = results.Length
                Results = results |> List.take (min criteria.MaxResults results.Length)
                SearchTime = searchTime
                NextPageToken = None
            }
            
        with
        | ex ->
            return {
                Query = criteria.Query
                Source = criteria.Source
                TotalCount = 0
                Results = []
                SearchTime = DateTime.UtcNow - startTime
                NextPageToken = None
            }
    }
    
    /// Search GitHub for notebooks
    member private _.SearchGitHubAsync(criteria: SearchCriteria) : Async<DiscoveredNotebook list> = async {
        // GitHub API search implementation
        let query = $"{criteria.Query} extension:ipynb"
        let url = $"https://api.github.com/search/code?q={Uri.EscapeDataString(query)}&per_page={criteria.MaxResults}"
        
        try
            let! response = httpClient.GetStringAsync(url) |> Async.AwaitTask
            let jsonDoc = JsonDocument.Parse(response)
            
            let results = 
                jsonDoc.RootElement.GetProperty("items").EnumerateArray()
                |> Seq.map (fun item ->
                    {
                        Title = item.GetProperty("name").GetString()
                        Description = None
                        Url = item.GetProperty("html_url").GetString()
                        Author = Some (item.GetProperty("repository").GetProperty("owner").GetProperty("login").GetString())
                        Language = Some "jupyter"
                        Topics = []
                        Stars = Some (item.GetProperty("repository").GetProperty("stargazers_count").GetInt32())
                        LastUpdated = Some (DateTime.Parse(item.GetProperty("repository").GetProperty("updated_at").GetString()))
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
    
    /// Search Kaggle for notebooks
    member private _.SearchKaggleAsync(criteria: SearchCriteria) : Async<DiscoveredNotebook list> = async {
        // Kaggle search implementation (placeholder)
        return []
    }
    
    /// Search Google Colab for notebooks
    member private _.SearchColabAsync(criteria: SearchCriteria) : Async<DiscoveredNotebook list> = async {
        // Google Colab search implementation (placeholder)
        return []
    }
    
    /// Search NBViewer for notebooks
    member private _.SearchNBViewerAsync(criteria: SearchCriteria) : Async<DiscoveredNotebook list> = async {
        // NBViewer search implementation (placeholder)
        return []
    }
    
    /// Search local filesystem for notebooks
    member private _.SearchLocalAsync(criteria: SearchCriteria) : Async<DiscoveredNotebook list> = async {
        let searchPaths = [
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)
            @"C:\Users"
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
    
    /// Download notebook from URL
    member _.DownloadNotebookAsync(url: string, outputPath: string) : Async<bool> = async {
        try
            let! content = httpClient.GetStringAsync(url) |> Async.AwaitTask
            File.WriteAllText(outputPath, content)
            return true
        with
        | ex ->
            return false
    }
    
    /// Get notebook metadata from URL
    member _.GetNotebookMetadataAsync(url: string) : Async<NotebookMetadata option> = async {
        try
            let! content = httpClient.GetStringAsync(url) |> Async.AwaitTask
            let jsonDoc = JsonDocument.Parse(content)
            
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
            
        with
        | ex ->
            return None
    }

/// Discovery utilities
module DiscoveryUtils =
    
    /// Create search criteria
    let createSearchCriteria query source maxResults =
        {
            Query = query
            Source = source
            Language = None
            Topic = None
            MaxResults = maxResults
            SortBy = Relevance
        }
    
    /// Create search criteria with options
    let createSearchCriteriaWithOptions query source language topic maxResults sortBy =
        {
            Query = query
            Source = source
            Language = language
            Topic = topic
            MaxResults = maxResults
            SortBy = sortBy
        }
    
    /// Extract topics from notebook content
    let extractTopics (notebook: JupyterNotebook) : string list =
        let topics = ResizeArray<string>()
        
        // Extract from metadata tags
        topics.AddRange(notebook.Metadata.Tags)
        
        // Extract from markdown cells
        for cell in notebook.Cells do
            match cell with
            | MarkdownCell markdownData ->
                let content = String.Join(" ", markdownData.Source)
                
                // Look for common data science topics
                let patterns = [
                    @"\b(machine learning|ML)\b"
                    @"\b(deep learning|DL)\b"
                    @"\b(artificial intelligence|AI)\b"
                    @"\b(data science|data analysis)\b"
                    @"\b(natural language processing|NLP)\b"
                    @"\b(computer vision|CV)\b"
                    @"\b(neural network|NN)\b"
                    @"\b(pandas|numpy|matplotlib|seaborn|scikit-learn|tensorflow|pytorch)\b"
                ]
                
                for pattern in patterns do
                    let matches = Regex.Matches(content, pattern, RegexOptions.IgnoreCase)
                    for m in matches do
                        let topic = m.Value.ToLower()
                        if not (topics.Contains(topic)) then
                            topics.Add(topic)
            | _ -> ()
        
        topics |> List.ofSeq
    
    /// Estimate notebook complexity
    let estimateComplexity (notebook: JupyterNotebook) : int =
        let mutable complexity = 0
        
        for cell in notebook.Cells do
            match cell with
            | CodeCell codeData ->
                complexity <- complexity + codeData.Source.Length * 2
            | MarkdownCell markdownData ->
                complexity <- complexity + markdownData.Source.Length
            | RawCell rawData ->
                complexity <- complexity + rawData.Source.Length
        
        complexity
    
    /// Get programming language from notebook
    let getProgrammingLanguage (notebook: JupyterNotebook) : string option =
        match notebook.Metadata.KernelSpec with
        | Some kernelSpec -> kernelSpec.Language
        | None ->
            match notebook.Metadata.LanguageInfo with
            | Some langInfo -> Some langInfo.Name
            | None -> None
    
    /// Format search results as text
    let formatSearchResults (result: SearchResult) : string =
        let sb = System.Text.StringBuilder()
        
        sb.AppendLine($"🔍 Search Results for '{result.Query}' on {result.Source}") |> ignore
        sb.AppendLine($"📊 Found {result.TotalCount} results in {result.SearchTime.TotalMilliseconds:F0}ms") |> ignore
        sb.AppendLine() |> ignore
        
        for (i, notebook) in result.Results |> List.indexed do
            sb.AppendLine($"{i + 1}. 📓 {notebook.Title}") |> ignore
            
            match notebook.Author with
            | Some author -> sb.AppendLine($"   👤 Author: {author}") |> ignore
            | None -> ()
            
            match notebook.Description with
            | Some desc -> sb.AppendLine($"   📝 {desc}") |> ignore
            | None -> ()
            
            sb.AppendLine($"   🔗 {notebook.Url}") |> ignore
            
            match notebook.Stars with
            | Some stars -> sb.AppendLine($"   ⭐ {stars} stars") |> ignore
            | None -> ()
            
            sb.AppendLine() |> ignore
        
        sb.ToString()
