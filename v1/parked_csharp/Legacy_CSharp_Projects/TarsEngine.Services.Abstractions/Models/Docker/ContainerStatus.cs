namespace TarsEngine.Services.Abstractions.Models.Docker
{
    /// <summary>
    /// Represents the status of a Docker container.
    /// </summary>
    public enum ContainerStatus
    {
        /// <summary>
        /// The container status is unknown.
        /// </summary>
        Unknown = 0,

        /// <summary>
        /// The container is created but not started.
        /// </summary>
        Created = 1,

        /// <summary>
        /// The container is running.
        /// </summary>
        Running = 2,

        /// <summary>
        /// The container is paused.
        /// </summary>
        Paused = 3,

        /// <summary>
        /// The container is stopped.
        /// </summary>
        Stopped = 4,

        /// <summary>
        /// The container is exited.
        /// </summary>
        Exited = 5,

        /// <summary>
        /// The container is dead.
        /// </summary>
        Dead = 6,

        /// <summary>
        /// The container is restarting.
        /// </summary>
        Restarting = 7
    }
}
