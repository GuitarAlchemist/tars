namespace Tars.DSL

open System
open System.Threading.Tasks

/// Module for asynchronous execution utilities
module AsyncExecution =
    /// Information about a running task
    type TaskInfo = {
        Id: Guid
        Name: string
        StartTime: DateTime
        Status: string
    }
    
    /// Task result with metadata
    type TaskResult<'T> = {
        TaskInfo: TaskInfo
        Result: 'T
        CompletionTime: DateTime
    }
    
    /// Task execution context
    type TaskExecutionContext = {
        Tasks: Map<Guid, TaskInfo * Async<obj>>
    }
    
    /// Global task execution context
    let mutable private context = {
        Tasks = Map.empty
    }
    
    /// Execute a task asynchronously
    let executeTask (name: string) (task: Async<'T>) : Async<TaskInfo * 'T> = async {
        let taskId = Guid.NewGuid()
        let taskInfo = {
            Id = taskId
            Name = name
            StartTime = DateTime.Now
            Status = "Running"
        }
        
        // Store task in context
        let boxedTask = async {
            let! result = task
            return box result
        }
        
        context <- { context with Tasks = context.Tasks.Add(taskId, (taskInfo, boxedTask)) }
        
        // Execute task
        let! result = task
        
        return (taskInfo, result)
    }
    
    /// Wait for a task to complete
    let waitForTask (taskId: Guid) (timeout: TimeSpan) : Async<TaskResult<obj>> = async {
        match context.Tasks.TryFind taskId with
        | Some (taskInfo, task) ->
            let! result = task
            return {
                TaskInfo = { taskInfo with Status = "Completed" }
                Result = result
                CompletionTime = DateTime.Now
            }
        | None ->
            return failwith $"Task with ID {taskId} not found"
    }
    
    /// Extension methods for Async
    module Async =
        /// Map a function over an async value
        let map (f: 'T -> 'U) (a: Async<'T>) : Async<'U> = async {
            let! result = a
            return f result
        }