using Microsoft.Extensions.DependencyInjection;
using TarsCli.Services;
using TarsCli.Services.Agents;
using TarsCli.Services.CodeGeneration;
using TarsCli.Services.Testing;
using TarsEngine.Services;
using SelfImprovementService = TarsCli.Services.SelfImprovementService;

namespace TarsCli.Extensions;

/// <summary>
/// Extension methods for IServiceCollection
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Adds TARS services to the service collection
    /// </summary>
    /// <param name="services">The service collection</param>
    /// <returns>The service collection</returns>
    public static IServiceCollection AddTarsServices(this IServiceCollection services)
    {
        // Add core services
        services.AddSingleton<ConsoleService>();
        services.AddSingleton<DockerService>();
        services.AddSingleton<McpService>();
        services.AddSingleton<TarsMcpSwarmService>();
        services.AddSingleton<SelfImprovementService>();
        services.AddSingleton<SwarmSelfImprovementService>();
        
        // Add agent services
        services.AddAgentServices();
        
        return services;
    }
    
    /// <summary>
    /// Adds agent services to the service collection
    /// </summary>
    /// <param name="services">The service collection</param>
    /// <returns>The service collection</returns>
    public static IServiceCollection AddAgentServices(this IServiceCollection services)
    {
        // Add agent factory
        services.AddSingleton<AgentFactory>();
        
        // Add agent implementations
        services.AddSingleton<CodeAnalyzerAgent>();
        services.AddSingleton<CodeGeneratorAgent>();
        services.AddSingleton<TestGeneratorAgent>();
        services.AddSingleton<DocumentationGeneratorAgent>();
        services.AddSingleton<ProjectManagerAgent>();
        
        // Add supporting services
        services.AddSingleton<CSharpCodeGenerator>();
        services.AddSingleton<FSharpCodeGenerator>();
        services.AddSingleton<CSharpTestGenerator>();
        services.AddSingleton<FSharpTestGenerator>();
        services.AddSingleton<TestResultAnalyzer>();
        services.AddSingleton<ImprovementPrioritizer>();
        services.AddSingleton<ImprovementSuggestionGenerator>();
        services.AddSingleton<SecurityVulnerabilityAnalyzer>();
        
        return services;
    }
}
