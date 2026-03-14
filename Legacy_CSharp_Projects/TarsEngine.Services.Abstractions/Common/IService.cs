namespace TarsEngine.Services.Abstractions.Common
{
    /// <summary>
    /// Base interface for all services in the TarsEngine.
    /// </summary>
    public interface IService
    {
        /// <summary>
        /// Gets the name of the service.
        /// </summary>
        string Name { get; }
    }
}
