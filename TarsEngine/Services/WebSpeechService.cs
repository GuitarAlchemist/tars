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

    public async Task<byte[]> SynthesizeSpeechAsync(string text)
    {
        try
        {
            // Call JavaScript to handle speech synthesis
            await _jsRuntime.InvokeVoidAsync("speechService.speak", text);
            return Array.Empty<byte>(); // Web Speech API handles playback directly
        }
        catch (Exception ex)
        {
            throw new Exception($"Speech synthesis failed: {ex.Message}", ex);
        }
    }
}