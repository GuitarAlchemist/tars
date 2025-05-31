namespace TarsEngine.Services.Abstractions.Models.Docker
{
    /// <summary>
    /// Represents statistics for a Docker container.
    /// </summary>
    public class ContainerStats
    {
        /// <summary>
        /// Gets or sets the CPU usage percentage.
        /// </summary>
        public double CpuUsagePercentage { get; set; }

        /// <summary>
        /// Gets or sets the memory usage in bytes.
        /// </summary>
        public long MemoryUsageBytes { get; set; }

        /// <summary>
        /// Gets or sets the memory limit in bytes.
        /// </summary>
        public long MemoryLimitBytes { get; set; }

        /// <summary>
        /// Gets or sets the memory usage percentage.
        /// </summary>
        public double MemoryUsagePercentage { get; set; }

        /// <summary>
        /// Gets or sets the network input in bytes.
        /// </summary>
        public long NetworkInputBytes { get; set; }

        /// <summary>
        /// Gets or sets the network output in bytes.
        /// </summary>
        public long NetworkOutputBytes { get; set; }

        /// <summary>
        /// Gets or sets the block input in bytes.
        /// </summary>
        public long BlockInputBytes { get; set; }

        /// <summary>
        /// Gets or sets the block output in bytes.
        /// </summary>
        public long BlockOutputBytes { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when the stats were collected.
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }
}
