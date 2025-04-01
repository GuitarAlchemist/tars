using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services;

/// <summary>
/// Service for managing user secrets
/// </summary>
public class SecretsService
{
    private readonly ILogger<SecretsService> _logger;
    private readonly IConfiguration _configuration;
    private readonly string _userSecretsId;
    private readonly string _secretsFilePath;

    public SecretsService(
        ILogger<SecretsService> logger,
        IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
            
        // Get or generate a user secrets ID
        _userSecretsId = _configuration["UserSecretsId"] ?? "tars-cli-secrets";
            
        // Determine the secrets file path based on the platform
        _secretsFilePath = GetSecretsFilePath();
            
        // Ensure the secrets directory exists
        EnsureSecretsDirectoryExists();
    }

    /// <summary>
    /// Get a secret value
    /// </summary>
    /// <param name="key">The secret key</param>
    /// <returns>The secret value, or null if not found</returns>
    public async Task<string?> GetSecretAsync(string key)
    {
        try
        {
            var secrets = await LoadSecretsAsync();
                
            if (secrets.TryGetValue(key, out var value))
            {
                return value;
            }
                
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting secret: {key}");
            return null;
        }
    }

    /// <summary>
    /// Set a secret value
    /// </summary>
    /// <param name="key">The secret key</param>
    /// <param name="value">The secret value</param>
    /// <returns>True if successful, false otherwise</returns>
    public async Task<bool> SetSecretAsync(string key, string value)
    {
        try
        {
            var secrets = await LoadSecretsAsync();
            secrets[key] = value;
                
            await SaveSecretsAsync(secrets);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error setting secret: {key}");
            return false;
        }
    }

    /// <summary>
    /// Remove a secret
    /// </summary>
    /// <param name="key">The secret key</param>
    /// <returns>True if successful, false otherwise</returns>
    public async Task<bool> RemoveSecretAsync(string key)
    {
        try
        {
            var secrets = await LoadSecretsAsync();
                
            if (secrets.ContainsKey(key))
            {
                secrets.Remove(key);
                await SaveSecretsAsync(secrets);
                return true;
            }
                
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error removing secret: {key}");
            return false;
        }
    }

    /// <summary>
    /// List all secret keys
    /// </summary>
    /// <returns>A list of secret keys</returns>
    public async Task<List<string>> ListSecretKeysAsync()
    {
        try
        {
            var secrets = await LoadSecretsAsync();
            return new List<string>(secrets.Keys);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error listing secret keys");
            return new List<string>();
        }
    }

    /// <summary>
    /// Clear all secrets
    /// </summary>
    /// <returns>True if successful, false otherwise</returns>
    public async Task<bool> ClearSecretsAsync()
    {
        try
        {
            await SaveSecretsAsync(new Dictionary<string, string>());
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error clearing secrets");
            return false;
        }
    }

    /// <summary>
    /// Load secrets from the secrets file
    /// </summary>
    /// <returns>A dictionary of secrets</returns>
    private async Task<Dictionary<string, string>> LoadSecretsAsync()
    {
        if (!File.Exists(_secretsFilePath))
        {
            return new Dictionary<string, string>();
        }
            
        try
        {
            var json = await File.ReadAllTextAsync(_secretsFilePath);
            var secrets = JsonSerializer.Deserialize<Dictionary<string, string>>(json);
            return secrets ?? new Dictionary<string, string>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading secrets");
            return new Dictionary<string, string>();
        }
    }

    /// <summary>
    /// Save secrets to the secrets file
    /// </summary>
    /// <param name="secrets">The secrets to save</param>
    private async Task SaveSecretsAsync(Dictionary<string, string> secrets)
    {
        try
        {
            var json = JsonSerializer.Serialize(secrets, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(_secretsFilePath, json);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving secrets");
            throw;
        }
    }

    /// <summary>
    /// Get the path to the secrets file
    /// </summary>
    /// <returns>The path to the secrets file</returns>
    private string GetSecretsFilePath()
    {
        string userSecretsPath;
            
        if (OperatingSystem.IsWindows())
        {
            userSecretsPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "Microsoft", "UserSecrets", _userSecretsId);
        }
        else if (OperatingSystem.IsLinux() || OperatingSystem.IsMacOS())
        {
            userSecretsPath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".microsoft", "usersecrets", _userSecretsId);
        }
        else
        {
            throw new PlatformNotSupportedException("Unsupported operating system");
        }
            
        return Path.Combine(userSecretsPath, "secrets.json");
    }

    /// <summary>
    /// Ensure the secrets directory exists
    /// </summary>
    private void EnsureSecretsDirectoryExists()
    {
        var directory = Path.GetDirectoryName(_secretsFilePath);
            
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
    }
}