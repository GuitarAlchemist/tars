namespace TarsEngine.Services.Abstractions.Models.Docker
{
    /// <summary>
    /// Represents a Docker container.
    /// </summary>
    public class DockerContainer
    {
        /// <summary>
        /// Gets or sets the ID of the container.
        /// </summary>
        public string Id { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the name of the container.
        /// </summary>
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the image used by the container.
        /// </summary>
        public string Image { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the status of the container.
        /// </summary>
        public ContainerStatus Status { get; set; } = ContainerStatus.Unknown;

        /// <summary>
        /// Gets or sets the creation time of the container.
        /// </summary>
        public DateTime CreatedAt { get; set; }

        /// <summary>
        /// Gets or sets the port mappings for the container.
        /// </summary>
        public Dictionary<int, int> Ports { get; set; } = new Dictionary<int, int>();

        /// <summary>
        /// Gets or sets the environment variables for the container.
        /// </summary>
        public Dictionary<string, string> EnvironmentVariables { get; set; } = new Dictionary<string, string>();

        /// <summary>
        /// Gets or sets the health status of the container.
        /// </summary>
        public ContainerHealth Health { get; set; } = ContainerHealth.Unknown;

        /// <summary>
        /// Gets or sets the container statistics.
        /// </summary>
        public ContainerStats? Stats { get; set; }
    }
}
