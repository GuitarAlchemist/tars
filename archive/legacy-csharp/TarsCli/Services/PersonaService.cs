using TarsCli.Models;
using TarsCli.Models.Personas;

namespace TarsCli.Services;

/// <summary>
/// Service for managing TARS personas
/// </summary>
public class PersonaService
{
    private readonly ILogger<PersonaService> _logger;
    private readonly Dictionary<string, IPersona> _personas = new(StringComparer.OrdinalIgnoreCase);
    private IPersona _currentPersona;

    /// <summary>
    /// Initializes a new instance of the <see cref="PersonaService"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    public PersonaService(ILogger<PersonaService> logger)
    {
        _logger = logger;
        
        // Register default personas
        RegisterPersona(new TarsPersona());
        RegisterPersona(new DaneelPersona());
        RegisterPersona(new JarvisPersona());
        
        // Set default persona
        _currentPersona = _personas["TARS"];
    }

    /// <summary>
    /// Gets the current persona
    /// </summary>
    public IPersona CurrentPersona => _currentPersona;

    /// <summary>
    /// Gets all available personas
    /// </summary>
    /// <returns>A collection of personas</returns>
    public IEnumerable<IPersona> GetAllPersonas()
    {
        return _personas.Values;
    }

    /// <summary>
    /// Gets a persona by name
    /// </summary>
    /// <param name="name">The name of the persona</param>
    /// <returns>The persona, or null if not found</returns>
    public IPersona GetPersona(string name)
    {
        if (_personas.TryGetValue(name, out var persona))
        {
            return persona;
        }
        
        return null;
    }

    /// <summary>
    /// Sets the current persona
    /// </summary>
    /// <param name="name">The name of the persona</param>
    /// <returns>True if the persona was set, false otherwise</returns>
    public bool SetPersona(string name)
    {
        if (_personas.TryGetValue(name, out var persona))
        {
            _currentPersona = persona;
            _logger.LogInformation("Persona set to {PersonaName}", persona.Name);
            return true;
        }
        
        _logger.LogWarning("Persona {PersonaName} not found", name);
        return false;
    }

    /// <summary>
    /// Registers a new persona
    /// </summary>
    /// <param name="persona">The persona to register</param>
    public void RegisterPersona(IPersona persona)
    {
        _personas[persona.Name] = persona;
        _logger.LogInformation("Registered persona {PersonaName}", persona.Name);
    }

    /// <summary>
    /// Transforms a response according to the current persona's style
    /// </summary>
    /// <param name="response">The original response</param>
    /// <returns>The transformed response</returns>
    public string TransformResponse(string response)
    {
        return _currentPersona.TransformResponse(response);
    }

    /// <summary>
    /// Gets a greeting from the current persona
    /// </summary>
    /// <returns>A greeting message</returns>
    public string GetGreeting()
    {
        return _currentPersona.GetGreeting();
    }
}
