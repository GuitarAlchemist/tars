using System.Text.Json;
using System.Text.Json.Serialization;

namespace TarsCli.Services
{
    /// <summary>
    /// Provides standardized JSON serialization configuration for the application
    /// </summary>
    public static class JsonSerializerConfig
    {
        /// <summary>
        /// Gets the default JSON serializer options for the application
        /// </summary>
        public static JsonSerializerOptions DefaultOptions => new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNameCaseInsensitive = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Converters = 
            {
                new JsonStringEnumConverter()
            }
        };

        /// <summary>
        /// Gets JSON serializer options optimized for deserializing AI responses
        /// </summary>
        public static JsonSerializerOptions AiResponseOptions => new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true,
            Converters = 
            {
                new JsonStringEnumConverter()
            }
        };
    }
}
