namespace TarsCli.Constants;

/// <summary>
/// Configuration keys for the application
/// </summary>
public static class ConfigurationKeys
{
    /// <summary>
    /// Docker Model Runner configuration section
    /// </summary>
    public static class DockerModelRunner
    {
        /// <summary>
        /// Base section
        /// </summary>
        public const string Section = "DockerModelRunner";
        
        /// <summary>
        /// Base URL for the Docker Model Runner API
        /// </summary>
        public const string BaseUrl = Section + ":BaseUrl";
        
        /// <summary>
        /// Default model for Docker Model Runner
        /// </summary>
        public const string DefaultModel = Section + ":DefaultModel";
    }
    
    /// <summary>
    /// Ollama configuration section
    /// </summary>
    public static class Ollama
    {
        /// <summary>
        /// Base section
        /// </summary>
        public const string Section = "Ollama";
        
        /// <summary>
        /// Base URL for the Ollama API
        /// </summary>
        public const string BaseUrl = Section + ":BaseUrl";
        
        /// <summary>
        /// Default model for Ollama
        /// </summary>
        public const string DefaultModel = Section + ":DefaultModel";
    }
    
    /// <summary>
    /// Model provider configuration section
    /// </summary>
    public static class ModelProvider
    {
        /// <summary>
        /// Base section
        /// </summary>
        public const string Section = "ModelProvider";
        
        /// <summary>
        /// Default model provider
        /// </summary>
        public const string Default = Section + ":Default";
    }
}
