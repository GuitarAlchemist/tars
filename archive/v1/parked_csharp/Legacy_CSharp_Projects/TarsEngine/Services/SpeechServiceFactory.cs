using Microsoft.Extensions.DependencyInjection;

namespace TarsEngine.Services;

public class SpeechServiceFactory(IServiceProvider serviceProvider)
{
    public SpeechServiceType DefaultService => SpeechServiceType.Riva;

    public ISpeechService GetSpeechService(SpeechServiceType serviceType)
    {
        return serviceType switch
        {
            SpeechServiceType.Riva => serviceProvider.GetRequiredService<RivaWrapperService>(),
            SpeechServiceType.WebSpeech => serviceProvider.GetRequiredService<WebSpeechService>(),
            _ => throw new ArgumentException($"Unknown speech service: {serviceType}")
        };
    }

    public IEnumerable<SpeechServiceType> AvailableServices => Enum.GetValues<SpeechServiceType>();
}