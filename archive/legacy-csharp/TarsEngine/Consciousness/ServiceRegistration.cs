using Microsoft.Extensions.DependencyInjection;
using TarsEngine.Consciousness.Core;
using TarsEngine.Consciousness.Intelligence;
using TarsEngine.Consciousness.Intelligence.Divergent;
using TarsEngine.Consciousness.Intelligence.Conceptual;
using TarsEngine.Consciousness.Intelligence.Pattern;
using TarsEngine.Consciousness.Intelligence.Solution;

namespace TarsEngine.Consciousness;

/// <summary>
/// Extension methods for registering Consciousness services
/// </summary>
public static class ServiceRegistration
{
    /// <summary>
    /// Adds Consciousness services to the service collection
    /// </summary>
    /// <param name="services">The service collection</param>
    /// <returns>The service collection</returns>
    public static IServiceCollection AddConsciousnessServices(this IServiceCollection services)
    {
        // Core services
        services.AddSingleton<ConsciousnessCore>();
        services.AddSingleton<SelfModel>();
        services.AddSingleton<EmotionalState>();
        services.AddSingleton<ValueSystem>();
        services.AddSingleton<MentalState>();
        services.AddSingleton<ConsciousnessLevel>();
        
        // Intelligence services
        services.AddSingleton<IntelligenceSpark>();
        
        // Creative thinking services
        services.AddSingleton<DivergentThinking>();
        services.AddSingleton<ConceptualBlending>();
        services.AddSingleton<PatternDisruption>();
        services.AddSingleton<CreativeSolutionGeneration>();
        services.AddSingleton<CreativeThinking>();
        
        // Intuitive reasoning services
        services.AddSingleton<IntuitiveReasoning>();
        
        // Spontaneous thought services
        services.AddSingleton<SpontaneousThought>();
        
        // Curiosity drive services
        services.AddSingleton<CuriosityDrive>();
        
        // Insight generation services
        services.AddSingleton<InsightGeneration>();
        services.AddSingleton<ConnectionDiscovery>();
        
        return services;
    }
}
