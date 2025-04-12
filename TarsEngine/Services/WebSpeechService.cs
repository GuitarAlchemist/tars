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

    public Task<string> TranscribeAudioAsync(byte[] audioData)
    {
        // Web Speech API doesn't support direct audio buffer transcription
        // We'll need to stream it through the microphone
        return Task.FromException<string>(new NotImplementedException("Direct audio transcription not supported in Web Speech API"));
    }

    public async Task<byte[]> SynthesizeSpeechAsync(string text, string? voiceName = null, float? rate = null, float? pitch = null)
    {
        try
        {
            // Use improved speech synthesis with better parameters
            await _jsRuntime.InvokeVoidAsync("speechService.speak",
                text,
                voiceName ?? "Google US English", // Default to a high-quality voice if available
                rate ?? 1.0f,  // Normal rate
                pitch ?? 1.0f  // Normal pitch
            );

            return [];
        }
        catch (Exception ex)
        {
            throw new Exception($"Speech synthesis failed: {ex.Message}", ex);
        }
    }

    public async Task<List<VoiceInfo>> GetAvailableVoicesAsync()
    {
        try
        {
            var voices = await _jsRuntime.InvokeAsync<List<VoiceInfo>>("speechService.getVoices");
            return voices;
        }
        catch (Exception ex)
        {
            throw new Exception($"Failed to get available voices: {ex.Message}", ex);
        }
    }
}

public class VoiceInfo
{
    public string Name { get; set; } = string.Empty;
    public string Lang { get; set; } = string.Empty;
    public bool Default { get; set; }
}
