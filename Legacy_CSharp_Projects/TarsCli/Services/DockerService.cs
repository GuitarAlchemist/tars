using System.Diagnostics;

namespace TarsCli.Services;

/// <summary>
/// Service for interacting with Docker
/// </summary>
public class DockerService
{
    private readonly ILogger<DockerService> _logger;

    /// <summary>
    /// Create a new Docker service
    /// </summary>
    /// <param name="logger">Logger</param>
    public DockerService(ILogger<DockerService> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Check if Docker is running
    /// </summary>
    /// <returns>True if Docker is running, false otherwise</returns>
    public async Task<bool> IsDockerRunning()
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "docker",
                    Arguments = "info",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            await process.WaitForExitAsync();

            return process.ExitCode == 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking if Docker is running");
            return false;
        }
    }

    /// <summary>
    /// Start a Docker container using docker-compose
    /// </summary>
    /// <param name="composeFile">Path to the docker-compose file</param>
    /// <param name="serviceName">Name of the service to start</param>
    /// <returns>True if the container was started successfully, false otherwise</returns>
    public async Task<bool> StartContainer(string composeFile, string serviceName)
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "docker-compose",
                    Arguments = $"-f {composeFile} up -d {serviceName}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            await process.WaitForExitAsync();

            return process.ExitCode == 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error starting container {serviceName}");
            return false;
        }
    }

    /// <summary>
    /// Stop a Docker container using docker-compose
    /// </summary>
    /// <param name="composeFile">Path to the docker-compose file</param>
    /// <param name="serviceName">Name of the service to stop</param>
    /// <returns>True if the container was stopped successfully, false otherwise</returns>
    public async Task<bool> StopContainer(string composeFile, string serviceName)
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "docker-compose",
                    Arguments = $"-f {composeFile} stop {serviceName}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            await process.WaitForExitAsync();

            return process.ExitCode == 0;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error stopping container {serviceName}");
            return false;
        }
    }

    /// <summary>
    /// Get the status of a Docker container
    /// </summary>
    /// <param name="containerName">Name of the container</param>
    /// <returns>Status of the container (running, stopped, etc.)</returns>
    public async Task<string> GetContainerStatus(string containerName)
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "docker",
                    Arguments = $"inspect --format=\"{{{{.State.Status}}}}\" {containerName}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode == 0)
            {
                return output.Trim();
            }
            else
            {
                return "not found";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error getting container status for {containerName}");
            return "error";
        }
    }

    /// <summary>
    /// Execute a command in a Docker container
    /// </summary>
    /// <param name="containerName">Name of the container</param>
    /// <param name="command">Command to execute</param>
    /// <returns>Output of the command</returns>
    public async Task<string> ExecuteCommand(string containerName, string command)
    {
        try
        {
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "docker",
                    Arguments = $"exec {containerName} {command}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode == 0)
            {
                return output.Trim();
            }
            else
            {
                var error = await process.StandardError.ReadToEndAsync();
                return $"Error: {error}";
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing command in container {containerName}");
            return $"Error: {ex.Message}";
        }
    }
}
