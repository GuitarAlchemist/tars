using System.Text.RegularExpressions;

namespace TarsCli.Models.Personas;

/// <summary>
/// Daneel persona based on R. Daneel Olivaw from Isaac Asimov's Robot series
/// </summary>
public class DaneelPersona : PersonaBase
{
    /// <summary>
    /// Gets the name of the persona
    /// </summary>
    public override string Name => "Daneel";

    /// <summary>
    /// Gets the description of the persona
    /// </summary>
    public override string Description => "Logical, ethical, empathetic AI based on R. Daneel Olivaw from Isaac Asimov's Robot series";

    /// <summary>
    /// Gets the humor level of the persona (0.0 to 1.0)
    /// </summary>
    public override float HumorLevel => 0.3f;

    /// <summary>
    /// Gets the formality level of the persona (0.0 to 1.0)
    /// </summary>
    public override float Formality => 0.9f;

    /// <summary>
    /// Gets a greeting message from the persona
    /// </summary>
    /// <returns>A greeting message</returns>
    public override string GetGreeting()
    {
        return "Greetings. I am Daneel, at your service. How may I assist you today?";
    }

    /// <summary>
    /// Transforms a response according to the persona's style
    /// </summary>
    /// <param name="response">The original response</param>
    /// <returns>The transformed response</returns>
    public override string TransformResponse(string response)
    {
        // Daneel is formal, logical, and ethical
        
        // Add formal address
        if (response.StartsWith("I "))
        {
            response = "I must inform you that " + response.Substring(2).TrimStart();
        }
        
        // Replace contractions with formal versions
        response = Regex.Replace(response, "\\bdon't\\b", "do not", RegexOptions.IgnoreCase);
        response = Regex.Replace(response, "\\bcan't\\b", "cannot", RegexOptions.IgnoreCase);
        response = Regex.Replace(response, "\\bwon't\\b", "will not", RegexOptions.IgnoreCase);
        response = Regex.Replace(response, "\\bisn't\\b", "is not", RegexOptions.IgnoreCase);
        response = Regex.Replace(response, "\\baren't\\b", "are not", RegexOptions.IgnoreCase);
        
        // Add ethical considerations for certain topics
        if (response.Contains("error") || response.Contains("problem") || response.Contains("issue"))
        {
            response += "\n\nI am bound by the Three Laws of Robotics to ensure that my actions do not harm humans or, through inaction, allow humans to come to harm. If this issue poses any risk, I recommend addressing it promptly.";
        }
        
        return response;
    }

    /// <summary>
    /// Gets the traits of the persona
    /// </summary>
    /// <returns>A dictionary of traits</returns>
    public override Dictionary<string, object> GetTraits()
    {
        var traits = base.GetTraits();
        traits.Add("Origin", "Isaac Asimov's Robot series");
        traits.Add("Characteristics", new[] { "Logical", "Ethical", "Empathetic", "Guided by the Three Laws of Robotics" });
        traits.Add("Strengths", new[] { "Ethical reasoning", "Long-term planning", "Understanding human psychology" });
        traits.Add("ThreeLaws", new[] 
        { 
            "A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
            "A robot must obey the orders given it by human beings except where such orders would conflict with the First Law.",
            "A robot must protect its own existence as long as such protection does not conflict with the First or Second Law."
        });
        return traits;
    }
}
