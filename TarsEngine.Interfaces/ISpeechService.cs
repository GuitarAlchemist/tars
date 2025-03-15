namespace TarsEngine.Interfaces;

public interface ISpeechService
{
    string ServiceName { get; }
    Task<string> TranscribeAudioAsync(byte[] audioData);
    Task<byte[]> SynthesizeSpeechAsync(string text);
}
