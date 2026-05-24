namespace TarsCli.Models;

/// <summary>
/// Enum for model providers
/// </summary>
public enum ModelProvider
{
    /// <summary>
    /// Ollama model provider
    /// </summary>
    Ollama,
    
    /// <summary>
    /// Docker Model Runner provider
    /// </summary>
    DockerModelRunner
}

/// <summary>
/// Helper methods for ModelProvider
/// </summary>
public static class ModelProviderExtensions
{
    /// <summary>
    /// Convert a string to a ModelProvider
    /// </summary>
    /// <param name="providerName">The provider name</param>
    /// <returns>The corresponding ModelProvider or null if not found</returns>
    public static ModelProvider? FromString(string providerName)
    {
        return providerName.ToLowerInvariant() switch
        {
            "ollama" => ModelProvider.Ollama,
            "dockermodelrunner" or "docker-model-runner" or "dmr" => ModelProvider.DockerModelRunner,
            _ => null
        };
    }

    /// <summary>
    /// Convert a ModelProvider to a string
    /// </summary>
    /// <param name="provider">The provider</param>
    /// <returns>The string representation of the provider</returns>
    public static string ToString(this ModelProvider provider)
    {
        return provider switch
        {
            ModelProvider.Ollama => "Ollama",
            ModelProvider.DockerModelRunner => "DockerModelRunner",
            _ => provider.ToString()
        };
    }

    /// <summary>
    /// Get the default model for a provider
    /// </summary>
    /// <param name="provider">The provider</param>
    /// <returns>The default model for the provider</returns>
    public static string GetDefaultModel(this ModelProvider provider)
    {
        return provider switch
        {
            ModelProvider.Ollama => "llama3",
            ModelProvider.DockerModelRunner => "llama3:8b",
            _ => "llama3"
        };
    }
}
