using System.Text.Json;

namespace TarsEngine.A2A;

/// <summary>
/// Resolves agent cards from URLs
/// </summary>
public class AgentCardResolver
{
    private readonly HttpClient _httpClient;
    private readonly JsonSerializerOptions _jsonOptions;

    /// <summary>
    /// Initializes a new instance of the AgentCardResolver class
    /// </summary>
    public AgentCardResolver()
    {
        _httpClient = new HttpClient();
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true
        };
    }

    /// <summary>
    /// Resolves an agent card from a URL
    /// </summary>
    /// <param name="url">The URL of the agent</param>
    /// <returns>The agent card</returns>
    public async Task<AgentCard> ResolveAgentCardAsync(string url)
    {
        if (string.IsNullOrEmpty(url))
            throw new ArgumentNullException(nameof(url));

        // Try to get the agent card from the well-known location
        var uri = new Uri(url);
        var wellKnownUrl = new Uri(uri, "/.well-known/agent.json");

        try
        {
            var response = await _httpClient.GetAsync(wellKnownUrl);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<AgentCard>(json, _jsonOptions);
        }
        catch (HttpRequestException)
        {
            // If the well-known location fails, try a direct request to the URL
            try
            {
                var response = await _httpClient.GetAsync(url);
                response.EnsureSuccessStatusCode();

                var json = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize<AgentCard>(json, _jsonOptions);
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to resolve agent card from {url}: {ex.Message}", ex);
            }
        }
        catch (Exception ex)
        {
            throw new Exception($"Failed to resolve agent card from {wellKnownUrl}: {ex.Message}", ex);
        }
    }
}