namespace TarsEngine.Services.Abstractions.Models.Docker
{
    /// <summary>
    /// Represents the health status of a Docker container.
    /// </summary>
    public enum ContainerHealth
    {
        /// <summary>
        /// The container health is unknown.
        /// </summary>
        Unknown = 0,

        /// <summary>
        /// The container is starting.
        /// </summary>
        Starting = 1,

        /// <summary>
        /// The container is healthy.
        /// </summary>
        Healthy = 2,

        /// <summary>
        /// The container is unhealthy.
        /// </summary>
        Unhealthy = 3
    }
}
