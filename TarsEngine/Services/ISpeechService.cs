namespace TarsEngine.Services;

public interface ISpeechService
{
    Task<string> TranscribeAudioAsync(byte[] audioData);
    Task<byte[]> SynthesizeSpeechAsync(string text);
    string ServiceName { get; }
}