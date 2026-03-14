using Microsoft.Extensions.Logging;
using TarsEngine.Services.Abstractions.Docker;
using TarsEngine.Services.Abstractions.Models.Docker;
using TarsEngine.Services.Core.Base;

namespace TarsEngine.Services.Docker
{
    /// <summary>
    /// Implementation of the IDockerService interface.
    /// </summary>
    public class DockerService : ServiceBase, IDockerService
    {
        private readonly Dictionary<string, DockerContainer> _containers = new();

        /// <summary>
        /// Initializes a new instance of the <see cref="DockerService"/> class.
        /// </summary>
        /// <param name="logger">The logger instance.</param>
        public DockerService(ILogger<DockerService> logger)
            : base(logger)
        {
        }

        /// <inheritdoc/>
        public override string Name => "Docker Service";

        /// <inheritdoc/>
        public async Task<DockerContainer> CreateContainerAsync(string image, string name, Dictionary<string, string>? environmentVariables = null, Dictionary<int, int>? ports = null)
        {
            Logger.LogInformation("Creating Docker container: {Name} from image: {Image}", name, image);
            
            // Simulate Docker API call
            await Task.Delay(100);
            
            var container = new DockerContainer
            {
                Id = Guid.NewGuid().ToString(),
                Name = name,
                Image = image,
                Status = ContainerStatus.Created,
                CreatedAt = DateTime.UtcNow,
                EnvironmentVariables = environmentVariables ?? new Dictionary<string, string>(),
                Ports = ports ?? new Dictionary<int, int>(),
                Health = ContainerHealth.Unknown
            };
            
            _containers[container.Id] = container;
            
            return container;
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<DockerContainer>> GetContainersAsync(bool includeInactive = false)
        {
            Logger.LogInformation("Getting Docker containers, includeInactive: {IncludeInactive}", includeInactive);
            
            // Simulate Docker API call
            await Task.Delay(100);
            
            if (includeInactive)
            {
                return _containers.Values;
            }
            
            return _containers.Values.Where(c => 
                c.Status == ContainerStatus.Running || 
                c.Status == ContainerStatus.Paused || 
                c.Status == ContainerStatus.Restarting);
        }

        /// <inheritdoc/>
        public async Task StartContainerAsync(string containerId)
        {
            Logger.LogInformation("Starting Docker container: {ContainerId}", containerId);
            
            // Simulate Docker API call
            await Task.Delay(100);
            
            if (_containers.TryGetValue(containerId, out var container))
            {
                container.Status = ContainerStatus.Running;
                container.Health = ContainerHealth.Starting;
            }
            else
            {
                Logger.LogWarning("Attempted to start non-existent container: {ContainerId}", containerId);
            }
        }

        /// <inheritdoc/>
        public async Task StopContainerAsync(string containerId)
        {
            Logger.LogInformation("Stopping Docker container: {ContainerId}", containerId);
            
            // Simulate Docker API call
            await Task.Delay(100);
            
            if (_containers.TryGetValue(containerId, out var container))
            {
                container.Status = ContainerStatus.Stopped;
            }
            else
            {
                Logger.LogWarning("Attempted to stop non-existent container: {ContainerId}", containerId);
            }
        }
    }
}
