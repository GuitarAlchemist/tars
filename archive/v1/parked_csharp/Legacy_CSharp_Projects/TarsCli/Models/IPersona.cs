namespace TarsCli.Models;

/// <summary>
/// Interface for TARS personas
/// </summary>
public interface IPersona
{
    /// <summary>
    /// Gets the name of the persona
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the description of the persona
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Gets the humor level of the persona (0.0 to 1.0)
    /// </summary>
    float HumorLevel { get; }

    /// <summary>
    /// Gets the formality level of the persona (0.0 to 1.0)
    /// </summary>
    float Formality { get; }

    /// <summary>
    /// Gets a greeting message from the persona
    /// </summary>
    /// <returns>A greeting message</returns>
    string GetGreeting();

    /// <summary>
    /// Transforms a response according to the persona's style
    /// </summary>
    /// <param name="response">The original response</param>
    /// <returns>The transformed response</returns>
    string TransformResponse(string response);

    /// <summary>
    /// Gets the traits of the persona
    /// </summary>
    /// <returns>A dictionary of traits</returns>
    Dictionary<string, object> GetTraits();
}
