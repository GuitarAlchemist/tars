namespace TarsCli.Commands;

/// <summary>
/// Base class for all TARS CLI commands
/// </summary>
public class TarsCommand : Command
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TarsCommand"/> class.
    /// </summary>
    /// <param name="name">The name of the command</param>
    /// <param name="description">The description of the command</param>
    public TarsCommand(string name, string description) : base(name, description)
    {
    }
}
