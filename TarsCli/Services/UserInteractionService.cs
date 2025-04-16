using Microsoft.Extensions.Configuration;
using System.Security;

namespace TarsCli.Services;

/// <summary>
/// Service for handling user interaction and input
/// </summary>
public class UserInteractionService
{
    private readonly ILogger<UserInteractionService> _logger;
    private readonly IConfiguration _configuration;
    private readonly SecretsService _secretsService;

    public UserInteractionService(
        ILogger<UserInteractionService> logger,
        IConfiguration configuration,
        SecretsService secretsService)
    {
        _logger = logger;
        _configuration = configuration;
        _secretsService = secretsService;
    }

    /// <summary>
    /// Ask the user for input
    /// </summary>
    /// <param name="prompt">The prompt to display to the user</param>
    /// <param name="defaultValue">Optional default value</param>
    /// <returns>The user's input</returns>
    public string AskForInput(string prompt, string defaultValue = "")
    {
        Console.Write($"{prompt}");
            
        if (!string.IsNullOrEmpty(defaultValue))
        {
            Console.Write($" [default: {defaultValue}]");
        }
            
        Console.Write(": ");
            
        var input = Console.ReadLine();
            
        if (string.IsNullOrWhiteSpace(input) && !string.IsNullOrEmpty(defaultValue))
        {
            return defaultValue;
        }
            
        return input ?? string.Empty;
    }

    /// <summary>
    /// Ask the user for a secret (like an API key)
    /// </summary>
    /// <param name="prompt">The prompt to display to the user</param>
    /// <returns>The user's secret input</returns>
    public string AskForSecret(string prompt)
    {
        Console.Write($"{prompt}: ");
            
        var secret = new SecureString();
        ConsoleKeyInfo key;
            
        do
        {
            key = Console.ReadKey(true);
                
            // Ignore control keys like Backspace
            if (key.Key != ConsoleKey.Backspace && key.Key != ConsoleKey.Enter)
            {
                secret.AppendChar(key.KeyChar);
                Console.Write("*");
            }
            else if (key.Key == ConsoleKey.Backspace && secret.Length > 0)
            {
                secret.RemoveAt(secret.Length - 1);
                Console.Write("\b \b");
            }
        } while (key.Key != ConsoleKey.Enter);
            
        Console.WriteLine();
            
        return new System.Net.NetworkCredential(string.Empty, secret).Password;
    }

    /// <summary>
    /// Ask the user for confirmation
    /// </summary>
    /// <param name="prompt">The prompt to display to the user</param>
    /// <param name="defaultValue">Optional default value</param>
    /// <returns>True if the user confirmed, false otherwise</returns>
    public bool AskForConfirmation(string prompt, bool defaultValue = false)
    {
        var defaultText = defaultValue ? "Y/n" : "y/N";
        Console.Write($"{prompt} [{defaultText}]: ");
            
        var input = Console.ReadLine()?.Trim().ToLower();
            
        if (string.IsNullOrWhiteSpace(input))
        {
            return defaultValue;
        }
            
        return input == "y" || input == "yes";
    }

    /// <summary>
    /// Ask the user to select an option from a list
    /// </summary>
    /// <param name="prompt">The prompt to display to the user</param>
    /// <param name="options">The options to choose from</param>
    /// <param name="defaultIndex">Optional default index</param>
    /// <returns>The selected option</returns>
    public string AskForSelection(string prompt, List<string> options, int defaultIndex = 0)
    {
        Console.WriteLine(prompt);
            
        for (var i = 0; i < options.Count; i++)
        {
            Console.WriteLine($"{i + 1}. {options[i]}{(i == defaultIndex ? " (default)" : "")}");
        }
            
        Console.Write($"Enter selection [1-{options.Count}]: ");
            
        var input = Console.ReadLine();
            
        if (string.IsNullOrWhiteSpace(input))
        {
            return options[defaultIndex];
        }
            
        if (int.TryParse(input, out var selection) && selection >= 1 && selection <= options.Count)
        {
            return options[selection - 1];
        }
            
        Console.WriteLine($"Invalid selection. Using default: {options[defaultIndex]}");
        return options[defaultIndex];
    }

    /// <summary>
    /// Ask for an API key and store it in secrets
    /// </summary>
    /// <param name="serviceName">The name of the service (e.g., "HuggingFace", "OpenAI")</param>
    /// <param name="keyName">The name of the key (e.g., "ApiKey")</param>
    /// <param name="required">Whether the key is required</param>
    /// <returns>The API key</returns>
    public async Task<string> AskForApiKeyAsync(string serviceName, string keyName, bool required = true)
    {
        var secretName = $"{serviceName}:{keyName}";
        var apiKey = await _secretsService.GetSecretAsync(secretName);
            
        if (string.IsNullOrEmpty(apiKey))
        {
            CliSupport.WriteColorLine($"No {serviceName} {keyName} found in secrets.", ConsoleColor.Yellow);
                
            if (required || AskForConfirmation($"Would you like to set up a {serviceName} {keyName}?", true))
            {
                apiKey = AskForSecret($"Enter your {serviceName} {keyName}");
                    
                if (!string.IsNullOrEmpty(apiKey))
                {
                    await _secretsService.SetSecretAsync(secretName, apiKey);
                    CliSupport.WriteColorLine($"{serviceName} {keyName} saved to secrets.", ConsoleColor.Green);
                }
                else if (required)
                {
                    CliSupport.WriteColorLine($"{serviceName} {keyName} is required.", ConsoleColor.Red);
                    throw new InvalidOperationException($"{serviceName} {keyName} is required but was not provided.");
                }
            }
        }
            
        return apiKey ?? string.Empty;
    }
}