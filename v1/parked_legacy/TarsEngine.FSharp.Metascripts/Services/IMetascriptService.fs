namespace TarsEngine.FSharp.Metascripts.Services

open System.Threading.Tasks
open TarsEngine.FSharp.Metascripts.Core

/// <summary>
/// Interface for metascript services.
/// </summary>
type IMetascriptService =
    /// <summary>
    /// Discovers metascripts in the specified directory.
    /// </summary>
    abstract member DiscoverMetascriptsAsync: directory: string -> Task<Result<RegisteredMetascript list, string>>
    
    /// <summary>
    /// Lists all registered metascripts.
    /// </summary>
    abstract member ListMetascriptsAsync: unit -> Task<Result<RegisteredMetascript list, string>>
    
    /// <summary>
    /// Gets a specific metascript by name.
    /// </summary>
    abstract member GetMetascriptAsync: name: string -> Task<Result<RegisteredMetascript option, string>>
    
    /// <summary>
    /// Executes a metascript by name.
    /// </summary>
    abstract member ExecuteMetascriptAsync: name: string -> Task<Result<MetascriptExecutionResult, string>>
    
    /// <summary>
    /// Gets metascript statistics.
    /// </summary>
    abstract member GetStatisticsAsync: unit -> Task<Result<MetascriptStats, string>>
    
    /// <summary>
    /// Validates a metascript.
    /// </summary>
    abstract member ValidateMetascriptAsync: source: MetascriptSource -> Task<Result<MetascriptSource, string>>

    /// <summary>
    /// Registers a metascript for execution.
    /// </summary>
    abstract member RegisterMetascriptAsync: source: MetascriptSource -> Task<Result<unit, string>>
