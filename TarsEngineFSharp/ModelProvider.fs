namespace TarsEngineFSharp

/// <summary>
/// Discriminated union for model providers
/// </summary>
type ModelProvider =
    /// <summary>
    /// Ollama model provider
    /// </summary>
    | Ollama
    /// <summary>
    /// Docker Model Runner provider
    /// </summary>
    | DockerModelRunner

/// <summary>
/// Module containing helper functions for ModelProvider
/// </summary>
module ModelProvider =
    /// <summary>
    /// Convert a string to a ModelProvider
    /// </summary>
    /// <param name="providerName">The provider name</param>
    /// <returns>The corresponding ModelProvider or None if not found</returns>
    let fromString (providerName: string) : ModelProvider option =
        match providerName.ToLowerInvariant() with
        | "ollama" -> Some Ollama
        | "dockermodelrunner" | "docker-model-runner" | "dmr" -> Some DockerModelRunner
        | _ -> None

    /// <summary>
    /// Convert a ModelProvider to a string
    /// </summary>
    /// <param name="provider">The provider</param>
    /// <returns>The string representation of the provider</returns>
    let toString (provider: ModelProvider) : string =
        match provider with
        | Ollama -> "Ollama"
        | DockerModelRunner -> "DockerModelRunner"

    /// <summary>
    /// Get the default model for a provider
    /// </summary>
    /// <param name="provider">The provider</param>
    /// <returns>The default model for the provider</returns>
    let getDefaultModel (provider: ModelProvider) : string =
        match provider with
        | Ollama -> "llama3"
        | DockerModelRunner -> "llama3:8b"
