namespace TarsEngine.FSharp.Metascripts.Discovery

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascripts.Core

/// <summary>
/// Service for discovering metascripts in the file system.
/// </summary>
type MetascriptDiscovery(registry: MetascriptRegistry, manager: MetascriptManager, logger: ILogger<MetascriptDiscovery>) =
    
    /// <summary>
    /// Discovers metascripts in the specified directory.
    /// </summary>
    member _.DiscoverMetascriptsAsync(directory: string, recursive: bool) =
        task {
            try
                logger.LogInformation(sprintf "Discovering metascripts in: %s (recursive: %b)" directory recursive)
                
                if Directory.Exists(directory) then
                    let searchOption = if recursive then SearchOption.AllDirectories else SearchOption.TopDirectoryOnly
                    let files = Directory.GetFiles(directory, "*.tars", searchOption)
                    
                    let discoveredMetascripts = ResizeArray<RegisteredMetascript>()
                    
                    for filePath in files do
                        try
                            let loadResult = manager.LoadMetascriptAsync(filePath).Result
                            match loadResult with
                            | Ok registered ->
                                discoveredMetascripts.Add(registered)
                                logger.LogDebug(sprintf "Discovered metascript: %s" registered.Source.Name)
                            | Error error ->
                                logger.LogWarning(sprintf "Failed to load metascript from %s: %s" filePath error)
                        with
                        | ex ->
                            logger.LogWarning(ex, sprintf "Error processing file: %s" filePath)
                    
                    let results = discoveredMetascripts |> Seq.toList
                    logger.LogInformation(sprintf "Discovery completed. Found %d metascripts" results.Length)
                    
                    return Ok results
                else
                    let error = sprintf "Directory not found: %s" directory
                    logger.LogWarning(error)
                    return Error error
            with
            | ex ->
                let error = sprintf "Error during metascript discovery: %s" ex.Message
                logger.LogError(ex, error)
                return Error error
        }
    
    /// <summary>
    /// Discovers metascripts in common locations.
    /// </summary>
    member this.DiscoverInCommonLocationsAsync() =
        task {
            try
                logger.LogInformation("Discovering metascripts in common locations")
                
                let commonPaths = [
                    "TarsCli/Metascripts"
                    "Metascripts"
                    "Scripts"
                    "."
                ]
                
                let allDiscovered = ResizeArray<RegisteredMetascript>()
                
                for path in commonPaths do
                    if Directory.Exists(path) then
                        let! result = this.DiscoverMetascriptsAsync(path, true)
                        match result with
                        | Ok metascripts ->
                            allDiscovered.AddRange(metascripts)
                        | Error error ->
                            logger.LogDebug(sprintf "No metascripts found in %s: %s" path error)
                
                let results = allDiscovered |> Seq.toList
                logger.LogInformation(sprintf "Common location discovery completed. Found %d metascripts" results.Length)
                
                return Ok results
            with
            | ex ->
                let error = sprintf "Error during common location discovery: %s" ex.Message
                logger.LogError(ex, error)
                return Error error
        }
    
    /// <summary>
    /// Watches a directory for new metascripts.
    /// </summary>
    member _.StartWatching(directory: string, callback: RegisteredMetascript -> unit) =
        try
            if Directory.Exists(directory) then
                let watcher = new FileSystemWatcher(directory, "*.tars")
                watcher.IncludeSubdirectories <- true
                watcher.EnableRaisingEvents <- true
                
                watcher.Created.Add(fun args ->
                    try
                        logger.LogInformation(sprintf "New metascript detected: %s" args.FullPath)
                        let loadResult = manager.LoadMetascriptAsync(args.FullPath).Result
                        match loadResult with
                        | Ok registered ->
                            callback registered
                        | Error error ->
                            logger.LogWarning(sprintf "Failed to load new metascript: %s" error)
                    with
                    | ex ->
                        logger.LogError(ex, sprintf "Error processing new metascript: %s" args.FullPath)
                )
                
                logger.LogInformation(sprintf "Started watching directory: %s" directory)
                Some watcher
            else
                logger.LogWarning(sprintf "Cannot watch non-existent directory: %s" directory)
                None
        with
        | ex ->
            logger.LogError(ex, sprintf "Error starting directory watch: %s" directory)
            None
