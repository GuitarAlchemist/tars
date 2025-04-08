using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TarsEngine.Models.Metrics;

namespace TarsEngine.Data;

/// <summary>
/// Repository for storing and retrieving metrics
/// </summary>
public class MetricsRepository
{
    private readonly ILogger<MetricsRepository> _logger;
    private readonly string _storageBasePath;
    private readonly Dictionary<string, List<BaseMetric>> _metricsCache = new();
    private readonly SemaphoreSlim _semaphore = new(1, 1);
    
    /// <summary>
    /// Initializes a new instance of the <see cref="MetricsRepository"/> class
    /// </summary>
    /// <param name="logger">The logger</param>
    /// <param name="storageBasePath">The storage base path</param>
    public MetricsRepository(ILogger<MetricsRepository> logger, string storageBasePath)
    {
        _logger = logger;
        _storageBasePath = storageBasePath;
        
        // Create the storage directory if it doesn't exist
        Directory.CreateDirectory(Path.Combine(_storageBasePath, "Metrics"));
    }
    
    /// <summary>
    /// Adds a metric
    /// </summary>
    /// <param name="metric">The metric to add</param>
    /// <returns>True if the metric was added successfully</returns>
    public async Task<bool> AddMetricAsync(BaseMetric metric)
    {
        try
        {
            await _semaphore.WaitAsync();
            
            // Add to cache
            if (!_metricsCache.TryGetValue(metric.Name, out var metrics))
            {
                metrics = [];
                _metricsCache[metric.Name] = metrics;
            }
            
            metrics.Add(metric);
            
            // Save to disk
            await SaveMetricsToDiskAsync(metric.Name);
            
            _logger.LogInformation("Added metric: {MetricName} with value: {MetricValue}", metric.Name, metric.Value);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding metric: {MetricName}", metric.Name);
            return false;
        }
        finally
        {
            _semaphore.Release();
        }
    }
    
    /// <summary>
    /// Adds multiple metrics
    /// </summary>
    /// <param name="metrics">The metrics to add</param>
    /// <returns>True if the metrics were added successfully</returns>
    public async Task<bool> AddMetricsAsync(IEnumerable<BaseMetric> metrics)
    {
        try
        {
            await _semaphore.WaitAsync();
            
            var metricsByName = metrics.GroupBy(m => m.Name).ToDictionary(g => g.Key, g => g.ToList());
            
            foreach (var (name, nameMetrics) in metricsByName)
            {
                // Add to cache
                if (!_metricsCache.TryGetValue(name, out var existingMetrics))
                {
                    existingMetrics = [];
                    _metricsCache[name] = existingMetrics;
                }
                
                existingMetrics.AddRange(nameMetrics);
                
                // Save to disk
                await SaveMetricsToDiskAsync(name);
            }
            
            _logger.LogInformation("Added {MetricCount} metrics", metrics.Count());
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error adding metrics");
            return false;
        }
        finally
        {
            _semaphore.Release();
        }
    }
    
    /// <summary>
    /// Gets metrics by name
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The metrics</returns>
    public async Task<IEnumerable<BaseMetric>> GetMetricsByNameAsync(string name, DateTime? startTime = null, DateTime? endTime = null)
    {
        try
        {
            await _semaphore.WaitAsync();
            
            // Load from disk if not in cache
            if (!_metricsCache.TryGetValue(name, out var metrics))
            {
                metrics = await LoadMetricsFromDiskAsync(name);
                _metricsCache[name] = metrics;
            }
            
            // Filter by time range
            var filteredMetrics = metrics.AsEnumerable();
            
            if (startTime.HasValue)
            {
                filteredMetrics = filteredMetrics.Where(m => m.Timestamp >= startTime.Value);
            }
            
            if (endTime.HasValue)
            {
                filteredMetrics = filteredMetrics.Where(m => m.Timestamp <= endTime.Value);
            }
            
            return filteredMetrics.OrderBy(m => m.Timestamp).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting metrics by name: {MetricName}", name);
            return [];
        }
        finally
        {
            _semaphore.Release();
        }
    }
    
    /// <summary>
    /// Gets metrics by category
    /// </summary>
    /// <param name="category">The metric category</param>
    /// <param name="startTime">The start time</param>
    /// <param name="endTime">The end time</param>
    /// <returns>The metrics</returns>
    public async Task<IEnumerable<BaseMetric>> GetMetricsByCategoryAsync(MetricCategory category, DateTime? startTime = null, DateTime? endTime = null)
    {
        try
        {
            await _semaphore.WaitAsync();
            
            // Load all metrics if cache is empty
            if (_metricsCache.Count == 0)
            {
                await LoadAllMetricsFromDiskAsync();
            }
            
            // Filter by category and time range
            var filteredMetrics = _metricsCache.Values
                .SelectMany(m => m)
                .Where(m => m.Category == category);
            
            if (startTime.HasValue)
            {
                filteredMetrics = filteredMetrics.Where(m => m.Timestamp >= startTime.Value);
            }
            
            if (endTime.HasValue)
            {
                filteredMetrics = filteredMetrics.Where(m => m.Timestamp <= endTime.Value);
            }
            
            return filteredMetrics.OrderBy(m => m.Timestamp).ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting metrics by category: {Category}", category);
            return [];
        }
        finally
        {
            _semaphore.Release();
        }
    }
    
    /// <summary>
    /// Gets the latest metric value by name
    /// </summary>
    /// <param name="name">The metric name</param>
    /// <returns>The latest metric value</returns>
    public async Task<double?> GetLatestMetricValueAsync(string name)
    {
        try
        {
            await _semaphore.WaitAsync();
            
            // Load from disk if not in cache
            if (!_metricsCache.TryGetValue(name, out var metrics))
            {
                metrics = await LoadMetricsFromDiskAsync(name);
                _metricsCache[name] = metrics;
            }
            
            // Get the latest metric
            var latestMetric = metrics.OrderByDescending(m => m.Timestamp).FirstOrDefault();
            
            return latestMetric?.Value;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting latest metric value by name: {MetricName}", name);
            return null;
        }
        finally
        {
            _semaphore.Release();
        }
    }
    
    /// <summary>
    /// Gets all metric names
    /// </summary>
    /// <returns>All metric names</returns>
    public async Task<IEnumerable<string>> GetAllMetricNamesAsync()
    {
        try
        {
            await _semaphore.WaitAsync();
            
            // Load all metrics if cache is empty
            if (_metricsCache.Count == 0)
            {
                await LoadAllMetricsFromDiskAsync();
            }
            
            return _metricsCache.Keys.ToList();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting all metric names");
            return [];
        }
        finally
        {
            _semaphore.Release();
        }
    }
    
    /// <summary>
    /// Clears all metrics
    /// </summary>
    /// <returns>True if the metrics were cleared successfully</returns>
    public async Task<bool> ClearAllMetricsAsync()
    {
        try
        {
            await _semaphore.WaitAsync();
            
            // Clear cache
            _metricsCache.Clear();
            
            // Clear disk
            var metricsDir = Path.Combine(_storageBasePath, "Metrics");
            if (Directory.Exists(metricsDir))
            {
                var files = Directory.GetFiles(metricsDir, "*.json");
                foreach (var file in files)
                {
                    File.Delete(file);
                }
            }
            
            _logger.LogInformation("Cleared all metrics");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error clearing all metrics");
            return false;
        }
        finally
        {
            _semaphore.Release();
        }
    }
    
    /// <summary>
    /// Saves metrics to disk
    /// </summary>
    /// <param name="metricName">The metric name</param>
    private async Task SaveMetricsToDiskAsync(string metricName)
    {
        if (!_metricsCache.TryGetValue(metricName, out var metrics))
        {
            return;
        }
        
        var filePath = GetMetricFilePath(metricName);
        var json = JsonSerializer.Serialize(metrics, new JsonSerializerOptions
        {
            WriteIndented = true
        });
        
        await File.WriteAllTextAsync(filePath, json);
    }
    
    /// <summary>
    /// Loads metrics from disk
    /// </summary>
    /// <param name="metricName">The metric name</param>
    /// <returns>The metrics</returns>
    private async Task<List<BaseMetric>> LoadMetricsFromDiskAsync(string metricName)
    {
        var filePath = GetMetricFilePath(metricName);
        
        if (!File.Exists(filePath))
        {
            return [];
        }
        
        var json = await File.ReadAllTextAsync(filePath);
        
        try
        {
            return JsonSerializer.Deserialize<List<BaseMetric>>(json) ?? [];
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error deserializing metrics from disk: {MetricName}", metricName);
            return [];
        }
    }
    
    /// <summary>
    /// Loads all metrics from disk
    /// </summary>
    private async Task LoadAllMetricsFromDiskAsync()
    {
        var metricsDir = Path.Combine(_storageBasePath, "Metrics");
        
        if (!Directory.Exists(metricsDir))
        {
            return;
        }
        
        var files = Directory.GetFiles(metricsDir, "*.json");
        
        foreach (var file in files)
        {
            var metricName = Path.GetFileNameWithoutExtension(file);
            var metrics = await LoadMetricsFromDiskAsync(metricName);
            _metricsCache[metricName] = metrics;
        }
    }
    
    /// <summary>
    /// Gets the metric file path
    /// </summary>
    /// <param name="metricName">The metric name</param>
    /// <returns>The metric file path</returns>
    private string GetMetricFilePath(string metricName)
    {
        // Sanitize the metric name for use as a file name
        var sanitizedName = string.Join("_", metricName.Split(Path.GetInvalidFileNameChars()));
        return Path.Combine(_storageBasePath, "Metrics", $"{sanitizedName}.json");
    }
}
