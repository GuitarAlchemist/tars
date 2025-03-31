namespace Tars.DSL

open System
open System.Threading.Tasks
open Tars.DSL.DataSources
open Tars.DSL.AsyncExecution
open Tars.DSL.PromptEngine

/// Module containing the core TARS DSL implementation
module TarsDsl =
    /// Type representing a TARS computation
    type TarsBuilder() =
        member _.Bind(m: Async<'T>, f: 'T -> Async<'U>) : Async<'U> = 
            async.Bind(m, f)
            
        member _.Return(x: 'T) : Async<'T> = 
            async.Return(x)
            
        member _.ReturnFrom(m: Async<'T>) : Async<'T> = 
            m
            
        member _.Delay(f: unit -> Async<'T>) : Async<'T> = 
            async.Delay(f)
            
        member _.Combine(m1: Async<unit>, m2: Async<'T>) : Async<'T> = 
            async.Combine(m1, m2)
            
        member _.Zero() : Async<unit> = 
            async.Return()
            
        member _.TryWith(m: Async<'T>, handler: exn -> Async<'T>) : Async<'T> = 
            async.TryWith(m, handler)
            
        member _.TryFinally(m: Async<'T>, compensation: unit -> unit) : Async<'T> = 
            async.TryFinally(m, compensation)
            
        member _.Using(resource: 'T when 'T :> IDisposable, binder: 'T -> Async<'U>) : Async<'U> = 
            async.Using(resource, binder)
            
        member _.While(guard: unit -> bool, body: unit -> Async<unit>) : Async<unit> = 
            let rec loop() = async {
                if guard() then
                    do! body()
                    return! loop()
            }
            loop()
            
        member _.For(sequence: seq<'T>, body: 'T -> Async<unit>) : Async<unit> = 
            async.For(sequence, body)
            
        // Custom operations for TARS DSL
        
        /// Load data from a file
        member _.FileData(path: string, transform: string -> 'T) : Async<'T> = async {
            let! content = loadFile path
            return transform content
        }
            
        /// Load data from an API
        member _.ApiData(url: string, transform: string -> 'T) : Async<'T> = async {
            let! content = loadApi url
            return transform content
        }
            
        /// Perform a web search
        member _.WebSearch(query: string) : Async<string[]> = async {
            let! results = webSearch query
            return Array.ofList results
        }
            
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
        member _.Improve(content: string, instructions: string) : Async<PromptResult> =
            let prompt = $"Improve the following content according to these instructions: {instructions}\n\nContent: {content}"
            generate prompt
            
        /// Execute an async task
        member _.ExecuteTask(name: string, task: Async<'T>) : Async<TaskInfo * 'T> = 
            executeTask name task
        
        /// Wait for a task to complete
        member _.WaitForTask(taskId: Guid, timeout: TimeSpan) : Async<TaskResult<obj>> =
            waitForTask taskId timeout
    
    /// The TARS computation expression instance
    let tars = TarsBuilder()
    
    /// Module for data operations in the alternative syntax
    module DATA =
        /// Load data from a CSV file
        let CSV (path: string) (rowParser: string -> 'T) (delimiter: char) : Async<'T[]> = async {
            let! content = loadFile path
            let rows = content.Split('\n')
                      |> Array.filter (fun s -> not (String.IsNullOrWhiteSpace(s)))
                      |> Array.map rowParser
            return rows
        }
        
        /// Perform a web search
        let WEB_SEARCH (query: string) : Async<string[]> = async {
            let! results = webSearch query
            return Array.ofList results
        }
    
    /// Module for AI operations in the alternative syntax
    module AI =
        /// Analyze content using AI
        let ANALYZE (content: string) : Async<PromptResult> =
            analyze content
        
        /// Generate content using AI
        let GENERATE (prompt: string) : Async<PromptResult> =
            generate prompt
        
        /// Summarize content using AI
        let SUMMARIZE (content: string) : Async<PromptResult> =
            summarize content
