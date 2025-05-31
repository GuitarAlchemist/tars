namespace TarsEngine.FSharp.Core.FileWatcher

open System
open System.IO
open Microsoft.Extensions.Logging

/// TARS Autonomous File Watcher Service
type TarsFileWatcherService(logger: ILogger<TarsFileWatcherService>) =
    
    let mutable fileWatcher: FileSystemWatcher option = None
    let watchPath = ".tars"
    
    member this.StartWatching() =
        async {
            logger.LogInformation("🔍 TARS File Watcher: Starting")
            
            let watcher = new FileSystemWatcher(watchPath)
            watcher.IncludeSubdirectories <- true
            watcher.EnableRaisingEvents <- true
            
            watcher.Created.Add(fun e ->
                logger.LogInformation("📄 TARS detected: {File}", e.Name)
                
                match Path.GetExtension(e.Name).ToLower() with
                | ".trsx" -> logger.LogInformation("🤖 Autonomous metascript detected")
                | ".md" -> logger.LogInformation("📚 Documentation detected")
                | _ -> logger.LogInformation("📋 File added to knowledge base")
            )
            
            fileWatcher <- Some watcher
            logger.LogInformation("✅ TARS File Watcher: ACTIVE")
        }
    
    member this.StopWatching() =
        match fileWatcher with
        | Some watcher -> watcher.Dispose()
        | None -> ()
    
    interface IDisposable with
        member this.Dispose() = this.StopWatching()
