namespace TarsEngine.Services;

public interface ISpeechService
{
    Task<string> TranscribeAudioAsync(byte[] audioData);
    Task<byte[]> SynthesizeSpeechAsync(string text, string? voiceName = null, float? rate = null, float? pitch = null);
    string ServiceName { get; }
}