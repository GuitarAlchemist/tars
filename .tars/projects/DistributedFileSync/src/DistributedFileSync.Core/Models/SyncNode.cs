namespace DistributedFileSync.Core.Models;

/// <summary>
/// Represents a node in the distributed file synchronization network
/// Designed by: Architect Agent (Alice)
/// Security reviewed by: Security Specialist Agent (Eve)
/// </summary>
public class SyncNode
{
    /// <summary>
    /// Unique identifier for the node
    /// </summary>
    public Guid Id { get; set; } = Guid.NewGuid();

    /// <summary>
    /// Human-readable name for the node
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Network address of the node
    /// </summary>
    public string Address { get; set; } = string.Empty;

    /// <summary>
    /// Port number for gRPC communication
    /// </summary>
    public int Port { get; set; } = 5000;

    /// <summary>
    /// Node status
    /// </summary>
    public NodeStatus Status { get; set; } = NodeStatus.Offline;

    /// <summary>
    /// Last time this node was seen online
    /// </summary>
    public DateTime LastSeen { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Node capabilities and features
    /// </summary>
    public NodeCapabilities Capabilities { get; set; } = new();

    /// <summary>
    /// Authentication token for secure communication
    /// </summary>
    public string AuthToken { get; set; } = string.Empty;

    /// <summary>
    /// Public key for encryption
    /// </summary>
    public string PublicKey { get; set; } = string.Empty;

    /// <summary>
    /// Node version information
    /// </summary>
    public string Version { get; set; } = "1.0.0";

    /// <summary>
    /// Synchronization statistics
    /// </summary>
    public NodeStatistics Statistics { get; set; } = new();

    /// <summary>
    /// List of synchronized directories on this node
    /// </summary>
    public List<string> SyncDirectories { get; set; } = new();

    /// <summary>
    /// Node configuration settings
    /// </summary>
    public Dictionary<string, string> Configuration { get; set; } = new();
}

/// <summary>
/// Node status enumeration
/// </summary>
public enum NodeStatus
{
    Offline,
    Online,
    Syncing,
    Error,
    Maintenance
}

/// <summary>
/// Node capabilities
/// Performance optimized by: Performance Engineer Agent (Dave)
/// </summary>
public class NodeCapabilities
{
    /// <summary>
    /// Maximum file size this node can handle (in bytes)
    /// </summary>
    public long MaxFileSize { get; set; } = 1024 * 1024 * 1024; // 1GB

    /// <summary>
    /// Maximum number of concurrent synchronizations
    /// </summary>
    public int MaxConcurrentSyncs { get; set; } = 10;

    /// <summary>
    /// Supported compression algorithms
    /// </summary>
    public List<string> SupportedCompression { get; set; } = new() { "gzip", "lz4" };

    /// <summary>
    /// Supported encryption algorithms
    /// </summary>
    public List<string> SupportedEncryption { get; set; } = new() { "AES-256", "ChaCha20" };

    /// <summary>
    /// Whether this node supports conflict resolution
    /// </summary>
    public bool SupportsConflictResolution { get; set; } = true;

    /// <summary>
    /// Whether this node can act as a relay
    /// </summary>
    public bool CanRelay { get; set; } = true;

    /// <summary>
    /// Available storage space in bytes
    /// </summary>
    public long AvailableStorage { get; set; }

    /// <summary>
    /// Network bandwidth in bytes per second
    /// </summary>
    public long NetworkBandwidth { get; set; }
}

/// <summary>
/// Node performance and synchronization statistics
/// Monitored by: Performance Engineer Agent (Dave)
/// </summary>
public class NodeStatistics
{
    /// <summary>
    /// Total number of files synchronized
    /// </summary>
    public long TotalFilesSynced { get; set; }

    /// <summary>
    /// Total bytes transferred
    /// </summary>
    public long TotalBytesTransferred { get; set; }

    /// <summary>
    /// Number of successful synchronizations
    /// </summary>
    public long SuccessfulSyncs { get; set; }

    /// <summary>
    /// Number of failed synchronizations
    /// </summary>
    public long FailedSyncs { get; set; }

    /// <summary>
    /// Average synchronization time in milliseconds
    /// </summary>
    public double AverageSyncTime { get; set; }

    /// <summary>
    /// Current CPU usage percentage
    /// </summary>
    public double CpuUsage { get; set; }

    /// <summary>
    /// Current memory usage in bytes
    /// </summary>
    public long MemoryUsage { get; set; }

    /// <summary>
    /// Current network utilization percentage
    /// </summary>
    public double NetworkUtilization { get; set; }

    /// <summary>
    /// Number of active connections
    /// </summary>
    public int ActiveConnections { get; set; }

    /// <summary>
    /// Last statistics update time
    /// </summary>
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
}
