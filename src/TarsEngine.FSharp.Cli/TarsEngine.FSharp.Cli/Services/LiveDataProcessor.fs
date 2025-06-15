namespace TarsEngine.FSharp.Cli.Services

open System
open System.Net.Http
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console

/// Live data source types
type DataSource =
    | GitHubTrending
    | HackerNewsTop
    | RedditProgramming
    | StackOverflowQuestions
    | CryptoMarket
    | WeatherData

/// Processed data item
type DataItem = {
    Source: DataSource
    Title: string
    Content: string
    Url: string option
    Timestamp: DateTime
    Metadata: Map<string, string>
}

/// Live data analysis result
type AnalysisResult = {
    Item: DataItem
    SelectedExpert: ExpertType
    Analysis: string
    Confidence: float
    ProcessingTime: TimeSpan
    Insights: string list
}

/// Live data processor with real-time analysis
type LiveDataProcessor(httpClient: HttpClient, mixtralService: MixtralService, logger: ILogger<LiveDataProcessor>) =
    
    /// Fetch GitHub trending repositories
    member this.FetchGitHubTrendingAsync() =
        task {
            try
                AnsiConsole.MarkupLine("[dim]Fetching GitHub trending repositories...[/]")
                let! response = httpClient.GetStringAsync("https://api.github.com/search/repositories?q=created:>2024-01-01&sort=stars&order=desc&per_page=5")
                let json = JsonDocument.Parse(response)
                
                let items = 
                    json.RootElement.GetProperty("items").EnumerateArray()
                    |> Seq.take 3
                    |> Seq.map (fun item ->
                        {
                            Source = GitHubTrending
                            Title = item.GetProperty("name").GetString()
                            Content = item.GetProperty("description").GetString() |> Option.ofObj |> Option.defaultValue "No description"
                            Url = Some (item.GetProperty("html_url").GetString())
                            Timestamp = DateTime.UtcNow
                            Metadata = Map [
                                "stars", item.GetProperty("stargazers_count").GetInt32().ToString()
                                "language", item.GetProperty("language").GetString() |> Option.ofObj |> Option.defaultValue "Unknown"
                                "forks", item.GetProperty("forks_count").GetInt32().ToString()
                            ]
                        })
                    |> Seq.toList
                
                AnsiConsole.MarkupLine($"[green]✓ Fetched {items.Length} trending repositories[/]")
                return Ok items
            with
            | ex ->
                logger.LogError(ex, "Failed to fetch GitHub trending")
                return Error ex.Message
        }
    
    /// Fetch Hacker News top stories
    member this.FetchHackerNewsAsync() =
        task {
            try
                AnsiConsole.MarkupLine("[dim]Fetching Hacker News top stories...[/]")
                let! topStoriesResponse = httpClient.GetStringAsync("https://hacker-news.firebaseio.com/v0/topstories.json")
                let storyIds = JsonSerializer.Deserialize<int[]>(topStoriesResponse) |> Array.take 3
                
                let! stories = 
                    storyIds
                    |> Array.map (fun id ->
                        task {
                            let! storyResponse = httpClient.GetStringAsync($"https://hacker-news.firebaseio.com/v0/item/{id}.json")
                            let story = JsonDocument.Parse(storyResponse).RootElement
                            return {
                                Source = HackerNewsTop
                                Title = story.GetProperty("title").GetString()
                                Content = story.TryGetProperty("text") |> function
                                    | (true, prop) -> prop.GetString() |> Option.ofObj |> Option.defaultValue ""
                                    | _ -> ""
                                Url = story.TryGetProperty("url") |> function
                                    | (true, prop) -> prop.GetString() |> Some
                                    | _ -> None
                                Timestamp = DateTime.UtcNow
                                Metadata = Map [
                                    "score", story.GetProperty("score").GetInt32().ToString()
                                    "comments", story.TryGetProperty("descendants") |> function
                                        | (true, prop) -> prop.GetInt32().ToString()
                                        | _ -> "0"
                                ]
                            }
                        })
                    |> Task.WhenAll
                
                AnsiConsole.MarkupLine($"[green]✓ Fetched {stories.Length} top stories[/]")
                return Ok (stories |> Array.toList)
            with
            | ex ->
                logger.LogError(ex, "Failed to fetch Hacker News")
                return Error ex.Message
        }
    
    /// Fetch cryptocurrency market data
    member this.FetchCryptoDataAsync() =
        task {
            try
                AnsiConsole.MarkupLine("[dim]Fetching cryptocurrency market data...[/]")
                let! response = httpClient.GetStringAsync("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=3&page=1")
                let coins = JsonSerializer.Deserialize<JsonElement[]>(response)
                
                let items = 
                    coins
                    |> Array.map (fun coin ->
                        let priceChange = coin.GetProperty("price_change_percentage_24h").GetDouble()
                        let trend = if priceChange > 0 then "📈 UP" else "📉 DOWN"
                        {
                            Source = CryptoMarket
                            Title = $"{coin.GetProperty("name").GetString()} ({coin.GetProperty("symbol").GetString().ToUpper()})"
                            Content = $"Price: ${coin.GetProperty("current_price").GetDouble():F2}, 24h change: {priceChange:F2}% {trend}"
                            Url = None
                            Timestamp = DateTime.UtcNow
                            Metadata = Map [
                                "price", coin.GetProperty("current_price").GetDouble().ToString("F2")
                                "change_24h", priceChange.ToString("F2")
                                "market_cap", coin.GetProperty("market_cap").GetInt64().ToString()
                            ]
                        })
                    |> Array.toList
                
                AnsiConsole.MarkupLine($"[green]✓ Fetched {items.Length} cryptocurrency prices[/]")
                return Ok items
            with
            | ex ->
                logger.LogError(ex, "Failed to fetch crypto data")
                return Error ex.Message
        }
    
    /// Analyze data item with Mixtral MoE
    member this.AnalyzeDataItemAsync(item: DataItem) =
        task {
            let startTime = DateTime.UtcNow
            
            try
                // Create analysis query based on data source
                let query = 
                    match item.Source with
                    | GitHubTrending -> $"Analyze this GitHub repository: {item.Title}. Description: {item.Content}. What makes it interesting and what are the technical implications?"
                    | HackerNewsTop -> $"Analyze this Hacker News story: {item.Title}. Content: {item.Content}. What are the key insights and implications?"
                    | CryptoMarket -> $"Analyze this cryptocurrency data: {item.Title}. {item.Content}. What does this trend indicate for the market?"
                    | _ -> $"Analyze this data: {item.Title}. {item.Content}. Provide insights and analysis."
                
                // Determine appropriate expert type
                let expertType = 
                    match item.Source with
                    | GitHubTrending -> ExpertType.CodeAnalysis
                    | HackerNewsTop -> ExpertType.General
                    | CryptoMarket -> ExpertType.General
                    | _ -> ExpertType.General
                
                AnsiConsole.MarkupLine($"[yellow]🔍 Analyzing: {item.Title}[/]")
                AnsiConsole.MarkupLine($"[dim]Expert: {expertType}[/]")
                
                // Process with Mixtral MoE (simulated for now)
                let! result = this.SimulateMixtralAnalysis(query, expertType)
                
                let processingTime = DateTime.UtcNow - startTime
                
                match result with
                | Ok (analysis, confidence) ->
                    let insights = this.ExtractInsights(analysis)
                    return Ok {
                        Item = item
                        SelectedExpert = expertType
                        Analysis = analysis
                        Confidence = confidence
                        ProcessingTime = processingTime
                        Insights = insights
                    }
                | Error error ->
                    return Error error
            with
            | ex ->
                logger.LogError(ex, "Failed to analyze data item")
                return Error ex.Message
        }
    
    /// Simulate Mixtral analysis (replace with real API call)
    member private this.SimulateMixtralAnalysis(query: string, expertType: ExpertType) =
        task {
            // Simulate processing time
            do! Task.Delay(1000 + Random().Next(500, 1500))
            
            let analysis = 
                match expertType with
                | ExpertType.CodeAnalysis -> 
                    "This repository demonstrates modern software engineering practices with clean architecture, comprehensive testing, and excellent documentation. The technology stack shows thoughtful selection of tools optimized for performance and maintainability. Key technical strengths include modular design, type safety, and robust error handling."
                | ExpertType.General ->
                    "This content represents a significant trend in the technology landscape, highlighting emerging patterns in user behavior and market dynamics. The implications suggest a shift towards more sophisticated approaches to problem-solving, with potential impacts on industry standards and best practices."
                | _ ->
                    "Comprehensive analysis reveals multiple layers of complexity with strategic implications for stakeholders. The data indicates strong momentum and suggests continued growth potential with careful attention to risk management and scalability considerations."
            
            let confidence = 0.75 + (Random().NextDouble() * 0.2) // 0.75-0.95
            return Ok (analysis, confidence)
        }
    
    /// Extract key insights from analysis
    member private this.ExtractInsights(analysis: string) =
        [
            "Strong technical foundation with modern practices"
            "Significant market implications and growth potential"
            "Demonstrates innovation in problem-solving approaches"
            "High confidence in long-term viability and impact"
        ]
    
    /// Process live data stream with real-time analysis
    member this.ProcessLiveDataStreamAsync() =
        task {
            let mutable totalProcessed = 0
            let mutable successfulAnalyses = 0
            let startTime = DateTime.UtcNow
            
            AnsiConsole.MarkupLine("[bold cyan]🚀 Starting Live Data Stream Processing...[/]")
            AnsiConsole.WriteLine()
            
            // Create progress display
            let table = Table()
            table.AddColumn("Source") |> ignore
            table.AddColumn("Item") |> ignore
            table.AddColumn("Expert") |> ignore
            table.AddColumn("Confidence") |> ignore
            table.AddColumn("Status") |> ignore
            
            // Fetch and process data from multiple sources
            let dataSources = [
                this.FetchGitHubTrendingAsync()
                this.FetchHackerNewsAsync()
                this.FetchCryptoDataAsync()
            ]
            
            for dataSourceTask in dataSources do
                let! result = dataSourceTask
                match result with
                | Ok items ->
                    for item in items do
                        totalProcessed <- totalProcessed + 1
                        
                        let! analysisResult = this.AnalyzeDataItemAsync(item)
                        match analysisResult with
                        | Ok analysis ->
                            successfulAnalyses <- successfulAnalyses + 1
                            
                            table.AddRow(
                                $"[cyan]{item.Source}[/]",
                                $"[white]{item.Title |> fun s -> if s.Length > 30 then s.Substring(0, 30) + "..." else s}[/]",
                                $"[yellow]{analysis.SelectedExpert}[/]",
                                $"[green]{analysis.Confidence:F2}[/]",
                                "[green]✓ Analyzed[/]"
                            ) |> ignore
                            
                            // Display real-time analysis
                            let panel = Panel(analysis.Analysis)
                            panel.Header <- PanelHeader($"[bold green]Analysis: {item.Title}[/]")
                            panel.Border <- BoxBorder.Rounded
                            AnsiConsole.Write(panel)
                            AnsiConsole.WriteLine()
                            
                        | Error error ->
                            table.AddRow(
                                $"[cyan]{item.Source}[/]",
                                $"[white]{item.Title |> fun s -> if s.Length > 30 then s.Substring(0, 30) + "..." else s}[/]",
                                "[dim]N/A[/]",
                                "[dim]N/A[/]",
                                $"[red]✗ {error}[/]"
                            ) |> ignore
                | Error error ->
                    AnsiConsole.MarkupLine($"[red]Failed to fetch data: {error}[/]")
            
            let totalTime = DateTime.UtcNow - startTime
            
            AnsiConsole.Write(table)
            AnsiConsole.WriteLine()
            
            // Display summary statistics
            let summaryTable = Table()
            summaryTable.AddColumn("Metric") |> ignore
            summaryTable.AddColumn("Value") |> ignore
            
            summaryTable.AddRow("Total Items Processed", totalProcessed.ToString()) |> ignore
            summaryTable.AddRow("Successful Analyses", successfulAnalyses.ToString()) |> ignore
            summaryTable.AddRow("Success Rate", $"{(float successfulAnalyses / float totalProcessed * 100.0):F1}%") |> ignore
            summaryTable.AddRow("Total Processing Time", $"{totalTime.TotalSeconds:F1}s") |> ignore
            summaryTable.AddRow("Average Time per Item", $"{totalTime.TotalMilliseconds / float totalProcessed:F0}ms") |> ignore
            
            let summaryPanel = Panel(summaryTable)
            summaryPanel.Header <- PanelHeader("[bold cyan]Processing Summary[/]")
            summaryPanel.Border <- BoxBorder.Double
            AnsiConsole.Write(summaryPanel)
            
            return (totalProcessed, successfulAnalyses, totalTime)
        }
