using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using TarsEngine.Services;
using TarsEngine.Services.Interfaces;

namespace TarsCli.Services;

public static class ServiceProviderFactory
{
    private static IServiceProvider? _serviceProvider;

    public static IServiceProvider CreateServiceProvider()
    {
        if (_serviceProvider != null)
            return _serviceProvider;

        // Create a new service collection
        var services = new ServiceCollection();

        // Add configuration
        var configuration = new ConfigurationBuilder()
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
            .Build();
        services.AddSingleton<IConfiguration>(configuration);

        // Register services
        services.AddSingleton<ICodeComplexityAnalyzer, CSharpComplexityAnalyzer>();
        services.AddSingleton<ICodeAnalysisService, CodeAnalysisService>();
        // Temporarily comment out the duplication analyzer
        // services.AddSingleton<IDuplicationAnalyzer>(provider => {
        //     var logger = provider.GetRequiredService<ILogger<DuplicationAnalyzer>>();
        //     return new DuplicationAnalyzer(logger);
        // });
        services.AddLogging();

        // Build the service provider
        _serviceProvider = services.BuildServiceProvider();
        return _serviceProvider;
    }
}
