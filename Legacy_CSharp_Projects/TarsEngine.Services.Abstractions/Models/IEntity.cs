namespace TarsEngine.Services.Abstractions.Models
{
    /// <summary>
    /// Base interface for all entities in the system.
    /// </summary>
    public interface IEntity
    {
        /// <summary>
        /// Gets the unique identifier for the entity.
        /// </summary>
        string Id { get; }
    }
}
