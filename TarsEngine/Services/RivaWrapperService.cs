using TarsEngineFSharp;
using Microsoft.FSharp.Control;
using Microsoft.FSharp.Core;

namespace TarsEngine.Services;

public class RivaWrapperService : IDisposable, ISpeechService
{
    private readonly RivaService.RivaClient _client;
    
    public string ServiceName => "Riva";
    
    public RivaWrapperService()
    {
        _client = RivaService.createDefaultClient();
    }

    public async Task<string> TranscribeAudioAsync(byte[] audioData)
    {
        var result = await FSharpAsync.StartAsTask(
            _client.ProcessQuery(audioData),
            FSharpOption<TaskCreationOptions>.None,
            FSharpOption<CancellationToken>.None);
        return result.OriginalText;
    }

    public async Task<byte[]> SynthesizeSpeechAsync(string text)
    {
        var result = await FSharpAsync.StartAsTask(
            _client.GenerateResponse(text),
            FSharpOption<TaskCreationOptions>.None,
            FSharpOption<CancellationToken>.None);
        return result.AudioData;
    }

    public void Dispose()
    {
        ((IDisposable)_client).Dispose();
    }
}