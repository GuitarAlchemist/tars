using Microsoft.Extensions.Configuration;
using System.Text;
using System.Text.Json;

namespace TarsCli.Services;

/// <summary>
/// Service for managing configuration settings
/// </summary>
public class ConfigurationService
{
    private readonly ILogger<ConfigurationService> _logger;
    private readonly IConfiguration _configuration;
    private readonly string _appSettingsPath;

    /// <summary>
    /// Create a new configuration service
    /// </summary>
    /// <param name="logger">Logger</param>
    /// <param name="configuration">Configuration</param>
    public ConfigurationService(
        ILogger<ConfigurationService> logger,
        IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
        _appSettingsPath = Path.Combine(AppContext.BaseDirectory, "appsettings.json");
    }

    /// <summary>
    /// Get a configuration value
    /// </summary>
    /// <param name="key">Configuration key</param>
    /// <returns>Configuration value</returns>
    public string? GetConfigurationValue(string key)
    {
        return _configuration[key];
    }

    /// <summary>
    /// Set a configuration value
    /// </summary>
    /// <param name="key">Configuration key</param>
    /// <param name="value">Configuration value</param>
    /// <returns>True if the value was set successfully, false otherwise</returns>
    public async Task<bool> SetConfigurationValueAsync(string key, string value)
    {
        try
        {
            // Read the current appsettings.json file
            string json = await File.ReadAllTextAsync(_appSettingsPath);
            var jsonDocument = JsonDocument.Parse(json);
            var root = jsonDocument.RootElement;

            // Create a new JSON object with the updated value
            using var memoryStream = new MemoryStream();
            using var jsonWriter = new Utf8JsonWriter(memoryStream, new JsonWriterOptions { Indented = true });

            jsonWriter.WriteStartObject();

            // Copy all existing properties
            foreach (var property in root.EnumerateObject())
            {
                if (property.Name == key.Split(':')[0])
                {
                    // This is the section we want to update
                    jsonWriter.WritePropertyName(property.Name);
                    WriteUpdatedSection(jsonWriter, property.Value, key.Split(':').Skip(1).ToArray(), value);
                }
                else
                {
                    // Copy the property as is
                    property.WriteTo(jsonWriter);
                }
            }

            // If the key doesn't exist in the root, add it
            if (!root.EnumerateObject().Any(p => p.Name == key.Split(':')[0]))
            {
                var sections = key.Split(':');
                jsonWriter.WritePropertyName(sections[0]);
                WriteNewSection(jsonWriter, sections.Skip(1).ToArray(), value);
            }

            jsonWriter.WriteEndObject();
            jsonWriter.Flush();

            // Write the updated JSON back to the file
            var updatedJson = Encoding.UTF8.GetString(memoryStream.ToArray());
            await File.WriteAllTextAsync(_appSettingsPath, updatedJson);

            _logger.LogInformation($"Configuration value {key} updated to {value}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error setting configuration value {key}");
            return false;
        }
    }

    private void WriteUpdatedSection(Utf8JsonWriter jsonWriter, JsonElement element, string[] remainingKeyParts, string value)
    {
        if (remainingKeyParts.Length == 0)
        {
            // We've reached the end of the key, write the value
            jsonWriter.WriteStringValue(value);
            return;
        }

        jsonWriter.WriteStartObject();

        // Copy all existing properties in this section
        foreach (var property in element.EnumerateObject())
        {
            if (property.Name == remainingKeyParts[0])
            {
                // This is the section we want to update
                jsonWriter.WritePropertyName(property.Name);
                WriteUpdatedSection(jsonWriter, property.Value, remainingKeyParts.Skip(1).ToArray(), value);
            }
            else
            {
                // Copy the property as is
                property.WriteTo(jsonWriter);
            }
        }

        // If the key doesn't exist in this section, add it
        if (!element.EnumerateObject().Any(p => p.Name == remainingKeyParts[0]))
        {
            jsonWriter.WritePropertyName(remainingKeyParts[0]);
            WriteNewSection(jsonWriter, remainingKeyParts.Skip(1).ToArray(), value);
        }

        jsonWriter.WriteEndObject();
    }

    private void WriteNewSection(Utf8JsonWriter jsonWriter, string[] remainingKeyParts, string value)
    {
        if (remainingKeyParts.Length == 0)
        {
            // We've reached the end of the key, write the value
            jsonWriter.WriteStringValue(value);
            return;
        }

        jsonWriter.WriteStartObject();
        jsonWriter.WritePropertyName(remainingKeyParts[0]);
        WriteNewSection(jsonWriter, remainingKeyParts.Skip(1).ToArray(), value);
        jsonWriter.WriteEndObject();
    }
}
