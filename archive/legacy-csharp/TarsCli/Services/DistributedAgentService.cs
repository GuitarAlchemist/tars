using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TarsCli.Services;

/// <summary>
/// Service for distributed agent execution
/// </summary>
public class DistributedAgentService
{
    private readonly ILogger<DistributedAgentService> _logger;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly List<AgentNode> _registeredNodes = new();
    private readonly JsonSerializerOptions _jsonOptions;

    public DistributedAgentService(
        ILogger<DistributedAgentService> logger,
        IHttpClientFactory httpClientFactory)
    {
        _logger = logger;
        _httpClientFactory = httpClientFactory;
            
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };
    }

    /// <summary>
    /// Registers a node for distributed agent execution
    /// </summary>
    /// <param name="nodeUrl">URL of the node</param>
    /// <param name="capabilities">Capabilities of the node</param>
    public async Task<bool> RegisterNodeAsync(string nodeUrl, List<string> capabilities)
    {
        try
        {
            _logger.LogInformation($"Registering node: {nodeUrl}");
                
            // Check if the node is already registered
            if (_registeredNodes.Any(n => n.Url == nodeUrl))
            {
                _logger.LogWarning($"Node already registered: {nodeUrl}");
                return false;
            }
                
            // Check if the node is available
            var client = _httpClientFactory.CreateClient();
            var response = await client.GetAsync($"{nodeUrl}/health");
                
            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning($"Node is not available: {nodeUrl}");
                return false;
            }
                
            // Register the node
            _registeredNodes.Add(new AgentNode
            {
                Url = nodeUrl,
                Capabilities = capabilities,
                Status = NodeStatus.Available
            });
                
            _logger.LogInformation($"Node registered: {nodeUrl}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error registering node: {nodeUrl}");
            return false;
        }
    }

    /// <summary>
    /// Unregisters a node
    /// </summary>
    /// <param name="nodeUrl">URL of the node</param>
    public bool UnregisterNode(string nodeUrl)
    {
        try
        {
            _logger.LogInformation($"Unregistering node: {nodeUrl}");
                
            // Find the node
            var node = _registeredNodes.FirstOrDefault(n => n.Url == nodeUrl);
                
            if (node == null)
            {
                _logger.LogWarning($"Node not found: {nodeUrl}");
                return false;
            }
                
            // Remove the node
            _registeredNodes.Remove(node);
                
            _logger.LogInformation($"Node unregistered: {nodeUrl}");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error unregistering node: {nodeUrl}");
            return false;
        }
    }

    /// <summary>
    /// Gets all registered nodes
    /// </summary>
    public List<AgentNode> GetRegisteredNodes()
    {
        return _registeredNodes.ToList();
    }

    /// <summary>
    /// Executes an agent task on a distributed node
    /// </summary>
    /// <param name="task">The task to execute</param>
    public async Task<AgentTaskResult> ExecuteTaskAsync(AgentTask task)
    {
        try
        {
            _logger.LogInformation($"Executing task: {task.TaskType}");
                
            // Find a node with the required capability
            var node = FindNodeForTask(task);
                
            if (node == null)
            {
                _logger.LogWarning($"No node found for task: {task.TaskType}");
                return new AgentTaskResult
                {
                    Success = false,
                    ErrorMessage = $"No node found for task: {task.TaskType}"
                };
            }
                
            // Mark the node as busy
            node.Status = NodeStatus.Busy;
                
            try
            {
                // Send the task to the node
                var client = _httpClientFactory.CreateClient();
                var response = await client.PostAsJsonAsync($"{node.Url}/tasks", task, _jsonOptions);
                    
                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogWarning($"Error executing task on node {node.Url}: {response.StatusCode}");
                    return new AgentTaskResult
                    {
                        Success = false,
                        ErrorMessage = $"Error executing task on node {node.Url}: {response.StatusCode}"
                    };
                }
                    
                // Get the result
                var result = await response.Content.ReadFromJsonAsync<AgentTaskResult>(_jsonOptions);
                    
                _logger.LogInformation($"Task executed successfully: {task.TaskType}");
                return result;
            }
            finally
            {
                // Mark the node as available
                node.Status = NodeStatus.Available;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, $"Error executing task: {task.TaskType}");
            return new AgentTaskResult
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    /// <summary>
    /// Executes multiple agent tasks in parallel on distributed nodes
    /// </summary>
    /// <param name="tasks">The tasks to execute</param>
    public async Task<List<AgentTaskResult>> ExecuteTasksAsync(List<AgentTask> tasks)
    {
        try
        {
            _logger.LogInformation($"Executing {tasks.Count} tasks");
                
            // Execute tasks in parallel
            var results = await Task.WhenAll(tasks.Select(ExecuteTaskAsync));
                
            return results.ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing tasks");
            return new List<AgentTaskResult>();
        }
    }

    /// <summary>
    /// Finds a node that can execute the task
    /// </summary>
    private AgentNode FindNodeForTask(AgentTask task)
    {
        // Find a node with the required capability
        return _registeredNodes
            .Where(n => n.Status == NodeStatus.Available && n.Capabilities.Contains(task.TaskType))
            .OrderBy(_ => Guid.NewGuid()) // Randomize to distribute load
            .FirstOrDefault();
    }
}

/// <summary>
/// Represents a node in the distributed agent network
/// </summary>
public class AgentNode
{
    public string Url { get; set; }
    public List<string> Capabilities { get; set; } = new();
    public NodeStatus Status { get; set; }
}

/// <summary>
/// Status of a node
/// </summary>
public enum NodeStatus
{
    Available,
    Busy,
    Offline
}

/// <summary>
/// Represents a task to be executed by an agent
/// </summary>
public class AgentTask
{
    public string TaskType { get; set; }
    public string TaskId { get; set; } = Guid.NewGuid().ToString();
    public Dictionary<string, object> Parameters { get; set; } = new();
}

/// <summary>
/// Result of an agent task
/// </summary>
public class AgentTaskResult
{
    public bool Success { get; set; }
    public string TaskId { get; set; }
    public Dictionary<string, object> Results { get; set; } = new();
    public string ErrorMessage { get; set; }
}