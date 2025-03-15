using TarsEngineFSharp;
using Microsoft.FSharp.Control;
using Microsoft.FSharp.Core;
using TarsEngine.Interfaces;

namespace TarsEngine.Services;

public class RivaWrapperService : ISpeechService, IDisposable
{
    private readonly RivaService.RivaClient _client = RivaService.createDefaultClient();
    
    public string ServiceName => "NVIDIA Riva";

    public async Task<string> TranscribeAudioAsync(byte[] audioData)
    {
        if (audioData == null || audioData.Length == 0)
            throw new ArgumentException("Audio data cannot be null or empty", nameof(audioData));

        var result = await FSharpAsync.StartAsTask(
            _client.ProcessQuery(audioData),
            FSharpOption<TaskCreationOptions>.None,
            FSharpOption<CancellationToken>.None
        );
        return result.OriginalText;
    }

    public async Task<byte[]> SynthesizeSpeechAsync(string text)
    {
        if (string.IsNullOrEmpty(text))
            throw new ArgumentException("Text cannot be null or empty", nameof(text));

        var result = await FSharpAsync.StartAsTask(
            _client.SynthesizeSpeech(text),
            FSharpOption<TaskCreationOptions>.None,
            FSharpOption<CancellationToken>.None
        );
        return result.AudioData;
    }

    public void Dispose()
    {
        _client?.Dispose();
    }
}
