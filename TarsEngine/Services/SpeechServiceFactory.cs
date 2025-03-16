using Microsoft.Extensions.DependencyInjection;

namespace TarsEngine.Services;

public class SpeechServiceFactory
{
    private readonly IServiceProvider _serviceProvider;
    
    public SpeechServiceType DefaultService => SpeechServiceType.Riva;

    public SpeechServiceFactory(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public ISpeechService GetSpeechService(SpeechServiceType serviceType)
    {
        return serviceType switch
        {
            SpeechServiceType.Riva => _serviceProvider.GetRequiredService<RivaWrapperService>(),
            SpeechServiceType.WebSpeech => _serviceProvider.GetRequiredService<WebSpeechService>(),
            _ => throw new ArgumentException($"Unknown speech service: {serviceType}")
        };
    }

    public IEnumerable<SpeechServiceType> AvailableServices => Enum.GetValues<SpeechServiceType>();
}