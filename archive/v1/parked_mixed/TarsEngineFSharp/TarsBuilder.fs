namespace Tars.DSL

open System
open System.Threading.Tasks
open Tars.DSL.DataSources
open Tars.DSL.AsyncExecution
open Tars.DSL.PromptEngine

/// The main builder for TARS computational expressions
type TarsBuilder() =
    member _.Bind(asyncOperation: Async<'T>, continuation: 'T -> Async<'U>) : Async<'U> =
        async {
            let! result = asyncOperation
            return! continuation result
        }
        
    member _.Return(value: 'T) : Async<'T> = 
        async.Return value
        
    member _.ReturnFrom(asyncOperation: Async<'T>) : Async<'T> = 
        asyncOperation
        
    member _.Zero() : Async<unit> = 
        async.Return ()
        
    member _.Delay(generator: unit -> Async<'T>) : Async<'T> =
        async.Delay generator
        
    member _.Combine(operation1: Async<unit>, operation2: Async<'T>) : Async<'T> =
        async {
            do! operation1
            return! operation2
        }
        
    member _.Using(resource: 'T when 'T :> IDisposable, body: 'T -> Async<'U>) : Async<'U> =
        async.Using(resource, body)
        
    member _.While(condition: unit -> bool, body: unit -> Async<unit>) : Async<unit> =
        async.While(condition, body)
        
    member _.For(sequence: seq<'T>, body: 'T -> Async<unit>) : Async<unit> =
        async.For(sequence, body)
        
    member _.TryWith(operation: Async<'T>, handler: exn -> Async<'T>) : Async<'T> =
        async.TryWith(operation, handler)
        
    member _.TryFinally(operation: Async<'T>, compensation: unit -> unit) : Async<'T> =
        async.TryFinally(operation, compensation)
        
    // TARS-specific operations
    
    /// Load data from a file
    member _.FileData(path: string, parser: string -> 'T) : Async<'T> =
        loadFile path parser
        
    /// Load data from a CSV file
    member _.CsvData<'T>(path: string, rowParser: string[] -> 'T, delimiter: char) : Async<'T[]> =
        loadCsv<'T> path rowParser delimiter
        
    /// Fetch data from an API
    member _.ApiData(url: string, parser: string -> 'T, ?headers: seq<string * string>) : Async<'T> =
        fetchApi url parser headers
        
    /// Fetch JSON data from an API
    member _.JsonApiData<'T>(url: string, ?headers: seq<string * string>) : Async<'T> =
        fetchJsonApi<'T> url headers
        
    /// Perform a web search
    member _.WebSearch(query: string) : Async<string list> =
        webSearch query
        
    /// Execute an async task with tracking
    member _.ExecuteTask<'T>(name: string, operation: Async<'T>) : Async<TaskInfo * 'T> = async {
        let! taskInfo = taskManager.ExecuteAsync(name, operation)
        let! result = operation
        return (taskInfo, result)
    }
    
    /// Wait for a task to complete
    member _.WaitForTask(taskId: Guid, ?timeout: TimeSpan) : Async<TaskInfo> =
        taskManager.WaitForTask(taskId, timeout)
        
    /// Summarize content using AI
    member _.Summarize(content: string) : Async<PromptResult> =
        summarize content
        
    /// Analyze content using AI
    member _.Analyze(content: string) : Async<PromptResult> =
        analyze content
        
    /// Generate content using AI
    member _.Generate(prompt: string) : Async<PromptResult> =
        generate prompt
        
    /// Improve content using AI
    member _.Improve(content: string, improvementPrompt: string) : Async<PromptResult> =
        improve content improvementPrompt

/// Module containing TARS DSL operations
module TarsDsl =
    /// Create a new TARS builder instance
    let tars = TarsBuilder()
    
    /// DATA namespace for data operations
    module DATA =
        /// Load data from a file
        let FILE path parser = tars.FileData(path, parser)
        
        /// Load data from a CSV file
        let CSV<'T> path rowParser delimiter = tars.CsvData<'T>(path, rowParser, delimiter)
        
        /// Fetch data from an API
        let API url parser headers = tars.ApiData(url, parser, headers)
        
        /// Fetch JSON data from an API
        let JSON_API<'T> url headers = tars.JsonApiData<'T>(url, headers)
        
        /// Perform a web search
        let WEB_SEARCH query = tars.WebSearch(query)
    
    /// ASYNC namespace for async operations
    module ASYNC =
        /// Execute an async task with tracking
        let EXECUTE name operation = tars.ExecuteTask(name, operation)
        
        /// Wait for a task to complete
        let WAIT_FOR taskId timeout = tars.WaitForTask(taskId, timeout)
        
        /// Execute a web search asynchronously
        let WEB_SEARCH query = tars.WebSearch(query)
    
    /// AI namespace for AI operations
    module AI =
        /// Summarize content using AI
        let SUMMARIZE content = tars.Summarize(content)
        
        /// Analyze content using AI
        let ANALYZE content = tars.Analyze(content)
        
        /// Generate content using AI
        let GENERATE prompt = tars.Generate(prompt)
        
        /// Improve content using AI
        let IMPROVE content improvementPrompt = tars.Improve(content, improvementPrompt)