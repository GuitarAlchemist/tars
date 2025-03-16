using Microsoft.Extensions.DependencyInjection;

namespace TarsEngine.Services;

public class SpeechServiceFactory
{
    private readonly IServiceProvider _serviceProvider;

    public SpeechServiceFactory(IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public ISpeechService GetSpeechService(string serviceName)
    {
        return serviceName.ToLower() switch
        {
            "riva" => _serviceProvider.GetRequiredService<RivaWrapperService>(),
            "webspeech" => _serviceProvider.GetRequiredService<WebSpeechService>(),
            _ => throw new ArgumentException($"Unknown speech service: {serviceName}")
        };
    }

    public IEnumerable<string> AvailableServices => new[] { "riva", "webspeech" };
}