using TarsEngine.Services.Abstractions.Common;
using TarsEngine.Services.Abstractions.Models.Docker;

namespace TarsEngine.Services.Abstractions.Docker
{
    /// <summary>
    /// Interface for services that interact with Docker.
    /// </summary>
    public interface IDockerService : IService
    {
        /// <summary>
        /// Gets a list of all Docker containers.
        /// </summary>
        /// <param name="includeInactive">Whether to include inactive containers.</param>
        /// <returns>A list of Docker containers.</returns>
        Task<IEnumerable<DockerContainer>> GetContainersAsync(bool includeInactive = false);

        /// <summary>
        /// Starts a Docker container.
        /// </summary>
        /// <param name="containerId">The ID of the container to start.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task StartContainerAsync(string containerId);

        /// <summary>
        /// Stops a Docker container.
        /// </summary>
        /// <param name="containerId">The ID of the container to stop.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        Task StopContainerAsync(string containerId);

        /// <summary>
        /// Creates a new Docker container.
        /// </summary>
        /// <param name="image">The Docker image to use.</param>
        /// <param name="name">The name for the container.</param>
        /// <param name="environmentVariables">The environment variables to set.</param>
        /// <param name="ports">The port mappings to set.</param>
        /// <returns>The created Docker container.</returns>
        Task<DockerContainer> CreateContainerAsync(string image, string name, Dictionary<string, string>? environmentVariables = null, Dictionary<int, int>? ports = null);
    }
}
