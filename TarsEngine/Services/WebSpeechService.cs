using Microsoft.JSInterop;

namespace TarsEngine.Services;

public class WebSpeechService : ISpeechService
{
    private readonly IJSRuntime _jsRuntime;
    
    public string ServiceName => "Web Speech API";

    public WebSpeechService(IJSRuntime jsRuntime)
    {
        _jsRuntime = jsRuntime;
    }

    public async Task<string> TranscribeAudioAsync(byte[] audioData)
    {
        // Web Speech API doesn't support direct audio buffer transcription
        // We'll need to stream it through the microphone
        throw new NotImplementedException("Direct audio transcription not supported in Web Speech API");
    }

    public async Task<byte[]> SynthesizeSpeechAsync(string text, string? voiceName = null, float? rate = null, float? pitch = null)
    {
        try
        {
            // Call JavaScript with voice parameters
            await _jsRuntime.InvokeVoidAsync("speechService.speak", text, voiceName, rate, pitch);
            return Array.Empty<byte>();
        }
        catch (Exception ex)
        {
            throw new Exception($"Speech synthesis failed: {ex.Message}", ex);
        }
    }
}