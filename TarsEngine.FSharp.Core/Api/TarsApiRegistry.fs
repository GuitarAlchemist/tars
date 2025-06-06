namespace TarsEngine.FSharp.Core.Api

open System
open System.Threading
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Api

/// <summary>
/// Global registry for TARS Engine API instances with thread-safe access
/// </summary>
type TarsApiRegistry private () =
    
    static let mutable instance: ITarsEngineApi option = None
    static let lockObj = obj()
    static let mutable logger: ILogger option = None
    
    /// <summary>
    /// Register the TARS Engine API instance
    /// </summary>
    /// <param name="api">The TARS Engine API instance to register</param>
    /// <param name="loggerInstance">Optional logger for registry operations</param>
    static member Register(api: ITarsEngineApi, ?loggerInstance: ILogger) =
        lock lockObj (fun () ->
            match loggerInstance with
            | Some log -> 
                logger <- Some log
                log.LogInformation("TarsApiRegistry.Register: Registering TARS Engine API instance")
            | None -> ()
            
            instance <- Some api
            
            match logger with
            | Some log -> log.LogInformation("TarsApiRegistry.Register: TARS Engine API successfully registered")
            | None -> printfn "TarsApiRegistry: TARS Engine API successfully registered"
        )
    
    /// <summary>
    /// Get the registered TARS Engine API instance
    /// </summary>
    /// <returns>The registered TARS Engine API instance</returns>
    /// <exception cref="InvalidOperationException">Thrown when no API instance is registered</exception>
    static member GetApi() =
        lock lockObj (fun () ->
            match instance with
            | Some api -> 
                match logger with
                | Some log -> log.LogDebug("TarsApiRegistry.GetApi: Returning registered API instance")
                | None -> ()
                api
            | None -> 
                let errorMsg = "TARS Engine API not registered. Call TarsApiRegistry.Register() first."
                match logger with
                | Some log -> log.LogError("TarsApiRegistry.GetApi: {ErrorMessage}", errorMsg)
                | None -> printfn "ERROR: %s" errorMsg
                failwith errorMsg
        )
    
    /// <summary>
    /// Check if a TARS Engine API instance is registered
    /// </summary>
    /// <returns>True if an API instance is registered, false otherwise</returns>
    static member IsAvailable =
        lock lockObj (fun () ->
            instance.IsSome
        )
    
    /// <summary>
    /// Unregister the current TARS Engine API instance
    /// </summary>
    static member Unregister() =
        lock lockObj (fun () ->
            match logger with
            | Some log -> log.LogInformation("TarsApiRegistry.Unregister: Unregistering TARS Engine API instance")
            | None -> printfn "TarsApiRegistry: Unregistering TARS Engine API instance"
            
            instance <- None
            
            match logger with
            | Some log -> log.LogInformation("TarsApiRegistry.Unregister: TARS Engine API successfully unregistered")
            | None -> printfn "TarsApiRegistry: TARS Engine API successfully unregistered"
        )
    
    /// <summary>
    /// Get information about the current registration status
    /// </summary>
    /// <returns>Registration status information</returns>
    static member GetRegistrationInfo() =
        lock lockObj (fun () ->
            let isRegistered = instance.IsSome
            let hasLogger = logger.IsSome
            
            {|
                IsRegistered = isRegistered
                HasLogger = hasLogger
                RegistrationTime = if isRegistered then Some DateTime.UtcNow else None
                ApiTypeName = if isRegistered then Some (instance.Value.GetType().FullName) else None
            |}
        )

/// <summary>
/// Factory for creating and configuring TARS Engine API instances
/// </summary>
type TarsApiFactory() =
    
    /// <summary>
    /// Create a new TARS Engine API instance with default configuration
    /// </summary>
    /// <param name="logger">Logger for the API instance</param>
    /// <returns>A new TARS Engine API instance</returns>
    static member CreateDefault(logger: ILogger<TarsEngineApiImpl>) =
        new TarsEngineApiImpl(logger) :> ITarsEngineApi
    
    /// <summary>
    /// Create and register a new TARS Engine API instance
    /// </summary>
    /// <param name="logger">Logger for the API instance</param>
    /// <param name="registryLogger">Optional logger for registry operations</param>
    /// <returns>The created and registered TARS Engine API instance</returns>
    static member CreateAndRegister(logger: ILogger<TarsEngineApiImpl>, ?registryLogger: ILogger) =
        let api = TarsApiFactory.CreateDefault(logger)
        TarsApiRegistry.Register(api, ?loggerInstance = registryLogger)
        api
    
    /// <summary>
    /// Initialize the TARS API registry with a default implementation
    /// </summary>
    /// <param name="loggerFactory">Logger factory for creating loggers</param>
    static member InitializeRegistry(loggerFactory: ILoggerFactory) =
        let apiLogger = loggerFactory.CreateLogger<TarsEngineApiImpl>()
        let registryLogger = loggerFactory.CreateLogger("TarsApiRegistry")
        
        if not TarsApiRegistry.IsAvailable then
            let api = TarsApiFactory.CreateAndRegister(apiLogger, registryLogger)
            registryLogger.LogInformation("TarsApiFactory.InitializeRegistry: TARS API registry initialized successfully")
            api
        else
            registryLogger.LogInformation("TarsApiFactory.InitializeRegistry: TARS API registry already initialized")
            TarsApiRegistry.GetApi()

/// <summary>
/// Extension methods for easier TARS API access
/// </summary>
[<AutoOpen>]
module TarsApiExtensions =
    
    /// <summary>
    /// Extension method to get TARS API with better error handling
    /// </summary>
    type TarsApiRegistry with
        
        /// <summary>
        /// Try to get the TARS API instance without throwing exceptions
        /// </summary>
        /// <returns>Some API instance if registered, None otherwise</returns>
        static member TryGetApi() =
            try
                Some (TarsApiRegistry.GetApi())
            with
            | _ -> None
        
        /// <summary>
        /// Get the TARS API instance or create a default one if not registered
        /// </summary>
        /// <param name="loggerFactory">Logger factory for creating a default instance</param>
        /// <returns>The TARS API instance</returns>
        static member GetOrCreateApi(loggerFactory: ILoggerFactory) =
            match TarsApiRegistry.TryGetApi() with
            | Some api -> api
            | None -> TarsApiFactory.InitializeRegistry(loggerFactory)

/// <summary>
/// Helper functions for common TARS API operations
/// </summary>
module TarsApiHelpers =
    
    /// <summary>
    /// Execute a function with the TARS API, handling registration errors gracefully
    /// </summary>
    /// <param name="operation">The operation to execute with the API</param>
    /// <param name="onError">Function to call if API is not available</param>
    /// <returns>Result of the operation or error handling</returns>
    let withTarsApi (operation: ITarsEngineApi -> 'T) (onError: unit -> 'T) =
        match TarsApiRegistry.TryGetApi() with
        | Some api -> operation api
        | None -> onError ()
    
    /// <summary>
    /// Execute an async function with the TARS API
    /// </summary>
    /// <param name="operation">The async operation to execute with the API</param>
    /// <param name="onError">Function to call if API is not available</param>
    /// <returns>Task result of the operation or error handling</returns>
    let withTarsApiAsync (operation: ITarsEngineApi -> System.Threading.Tasks.Task<'T>) (onError: unit -> System.Threading.Tasks.Task<'T>) =
        match TarsApiRegistry.TryGetApi() with
        | Some api -> operation api
        | None -> onError ()
    
    /// <summary>
    /// Log an operation using the TARS API execution context
    /// </summary>
    /// <param name="level">Log level</param>
    /// <param name="message">Log message</param>
    let logWithTars (level: LogLevel) (message: string) =
        withTarsApi
            (fun api -> api.ExecutionContext.LogEvent(level, message))
            (fun () -> printfn "[TARS API NOT AVAILABLE] %A: %s" level message)
    
    /// <summary>
    /// Search the vector store with error handling
    /// </summary>
    /// <param name="query">Search query</param>
    /// <param name="limit">Maximum number of results</param>
    /// <returns>Search results or empty array if API not available</returns>
    let searchVectorStore (query: string) (limit: int) =
        withTarsApiAsync
            (fun api -> api.VectorStore.SearchAsync(query, limit))
            (fun () -> System.Threading.Tasks.Task.FromResult([||]))
    
    /// <summary>
    /// Complete text using LLM with error handling
    /// </summary>
    /// <param name="prompt">Text prompt</param>
    /// <param name="model">Model name</param>
    /// <returns>Completion result or error message if API not available</returns>
    let completeWithLlm (prompt: string) (model: string) =
        withTarsApiAsync
            (fun api -> api.LlmService.CompleteAsync(prompt, model))
            (fun () -> System.Threading.Tasks.Task.FromResult("TARS API not available"))
    
    /// <summary>
    /// Spawn an agent with error handling
    /// </summary>
    /// <param name="agentType">Type of agent to spawn</param>
    /// <param name="config">Agent configuration</param>
    /// <returns>Agent ID or error message if API not available</returns>
    let spawnAgent (agentType: string) (config: AgentConfig) =
        withTarsApiAsync
            (fun api -> api.AgentCoordinator.SpawnAsync(agentType, config))
            (fun () -> System.Threading.Tasks.Task.FromResult("TARS API not available"))
