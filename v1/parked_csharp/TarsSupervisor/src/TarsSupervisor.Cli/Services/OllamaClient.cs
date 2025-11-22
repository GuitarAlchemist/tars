using System.Net.Http.Json;
using System.Text.Json;

namespace TarsSupervisor.Cli.Services;

public sealed class OllamaClient
{
    private readonly HttpClient _http = new();
    private readonly string _endpoint;
    private readonly string _model;
    private readonly double _temp;
    private readonly int _numCtx;

    public OllamaClient(string endpoint, string model, double temp, int numCtx)
    {
        _endpoint = endpoint.TrimEnd('/');
        _model = model;
        _temp = temp;
        _numCtx = numCtx;
    }

    private sealed record GenerateReq(string model, string prompt, bool stream, object options);
    private sealed record GenerateResp(string response);

    public async Task<string> GenerateAsync(string prompt)
    {
        var url = $"{_endpoint}/api/generate";
        var req = new GenerateReq(_model, prompt, false, new { temperature = _temp, num_ctx = _numCtx });
        var resp = await _http.PostAsJsonAsync(url, req);
        resp.EnsureSuccessStatusCode();
        var body = await resp.Content.ReadFromJsonAsync<GenerateResp>(new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
        return body?.response ?? string.Empty;
    }
}
