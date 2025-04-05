using System;
using System.Collections.Generic;

namespace TarsCli.Models;

/// <summary>
/// Base class for TARS personas
/// </summary>
public abstract class PersonaBase : IPersona
{
    /// <summary>
    /// Gets the name of the persona
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Gets the description of the persona
    /// </summary>
    public abstract string Description { get; }

    /// <summary>
    /// Gets the humor level of the persona (0.0 to 1.0)
    /// </summary>
    public abstract float HumorLevel { get; }

    /// <summary>
    /// Gets the formality level of the persona (0.0 to 1.0)
    /// </summary>
    public abstract float Formality { get; }

    /// <summary>
    /// Gets a greeting message from the persona
    /// </summary>
    /// <returns>A greeting message</returns>
    public abstract string GetGreeting();

    /// <summary>
    /// Transforms a response according to the persona's style
    /// </summary>
    /// <param name="response">The original response</param>
    /// <returns>The transformed response</returns>
    public virtual string TransformResponse(string response)
    {
        // Default implementation returns the original response
        return response;
    }

    /// <summary>
    /// Gets the traits of the persona
    /// </summary>
    /// <returns>A dictionary of traits</returns>
    public virtual Dictionary<string, object> GetTraits()
    {
        return new Dictionary<string, object>
        {
            { "Name", Name },
            { "Description", Description },
            { "HumorLevel", HumorLevel },
            { "Formality", Formality }
        };
    }
}
