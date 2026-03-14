namespace TarsEngine.FSharp.ML.Core

/// <summary>
/// Options for configuring the ML framework.
/// </summary>
type MLFrameworkOptions = {
    /// <summary>
    /// The seed value for random number generation.
    /// </summary>
    Seed: int option
    
    /// <summary>
    /// The base path for storing models.
    /// </summary>
    ModelBasePath: string option
    
    /// <summary>
    /// Whether to automatically reload models when they change on disk.
    /// </summary>
    AutoReloadModels: bool
    
    /// <summary>
    /// The interval in seconds to check for model changes when AutoReloadModels is true.
    /// </summary>
    ModelReloadIntervalSeconds: int
}

/// <summary>
/// Default values for MLFrameworkOptions.
/// </summary>
module MLFrameworkOptionsDefaults =
    /// <summary>
    /// Default options for the ML framework.
    /// </summary>
    let defaultOptions = {
        Seed = Some 42
        ModelBasePath = None
        AutoReloadModels = false
        ModelReloadIntervalSeconds = 60
    }
