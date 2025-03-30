using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace TarsCli.Services
{
    /// <summary>
    /// Service for detecting and checking GPU capabilities
    /// </summary>
    public class GpuService
    {
        private readonly ILogger<GpuService> _logger;
        private readonly IConfiguration _configuration;
        private bool? _isGpuAvailable;
        private List<GpuInfo>? _gpuInfo;

        public GpuService(
            ILogger<GpuService> logger,
            IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;
        }

        /// <summary>
        /// Check if a GPU is available for acceleration
        /// </summary>
        /// <returns>True if a compatible GPU is available, false otherwise</returns>
        public bool IsGpuAvailable()
        {
            // Return cached result if available
            if (_isGpuAvailable.HasValue)
            {
                return _isGpuAvailable.Value;
            }

            try
            {
                // Check configuration override
                var configOverride = _configuration["Ollama:EnableGpu"];
                if (!string.IsNullOrEmpty(configOverride))
                {
                    if (bool.TryParse(configOverride, out bool enableGpu))
                    {
                        _logger.LogInformation($"Using GPU acceleration setting from configuration: {enableGpu}");
                        _isGpuAvailable = enableGpu;
                        return enableGpu;
                    }
                }

                // Get GPU info
                var gpuInfo = GetGpuInfo();
                
                // Check if any compatible GPUs are found
                _isGpuAvailable = gpuInfo.Any(gpu => IsGpuCompatible(gpu));
                
                _logger.LogInformation($"GPU acceleration available: {_isGpuAvailable}");
                
                if (_isGpuAvailable.Value && gpuInfo.Any())
                {
                    var compatibleGpus = gpuInfo.Where(gpu => IsGpuCompatible(gpu)).ToList();
                    foreach (var gpu in compatibleGpus)
                    {
                        _logger.LogInformation($"Compatible GPU found: {gpu.Name} with {gpu.MemoryMB}MB memory");
                    }
                }
                
                return _isGpuAvailable.Value;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking GPU availability");
                _isGpuAvailable = false;
                return false;
            }
        }

        /// <summary>
        /// Get information about available GPUs
        /// </summary>
        /// <returns>List of GPU information</returns>
        public List<GpuInfo> GetGpuInfo()
        {
            if (_gpuInfo != null)
            {
                return _gpuInfo;
            }

            _gpuInfo = new List<GpuInfo>();
            
            try
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    _gpuInfo = GetWindowsGpuInfo();
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    _gpuInfo = GetLinuxGpuInfo();
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                {
                    _gpuInfo = GetMacGpuInfo();
                }
                
                _logger.LogInformation($"Found {_gpuInfo.Count} GPUs");
                foreach (var gpu in _gpuInfo)
                {
                    _logger.LogInformation($"GPU: {gpu.Name}, Memory: {gpu.MemoryMB}MB, Type: {gpu.Type}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting GPU information");
            }
            
            return _gpuInfo;
        }

        /// <summary>
        /// Get Ollama GPU parameters based on available GPUs
        /// </summary>
        /// <returns>Dictionary of GPU parameters for Ollama</returns>
        public Dictionary<string, object> GetOllamaGpuParameters()
        {
            var parameters = new Dictionary<string, object>();
            
            if (!IsGpuAvailable())
            {
                return parameters;
            }
            
            try
            {
                // Get compatible GPUs
                var compatibleGpus = _gpuInfo?.Where(gpu => IsGpuCompatible(gpu)).ToList() ?? new List<GpuInfo>();
                
                if (compatibleGpus.Any())
                {
                    // Enable GPU acceleration
                    parameters["use_gpu"] = true;
                    
                    // Set GPU layers based on available memory
                    // This is a simple heuristic and can be improved
                    var bestGpu = compatibleGpus.OrderByDescending(g => g.MemoryMB).First();
                    
                    if (bestGpu.MemoryMB >= 24000) // 24GB or more
                    {
                        parameters["gpu_layers"] = 100; // Use all layers on GPU
                    }
                    else if (bestGpu.MemoryMB >= 16000) // 16GB
                    {
                        parameters["gpu_layers"] = 80;
                    }
                    else if (bestGpu.MemoryMB >= 12000) // 12GB
                    {
                        parameters["gpu_layers"] = 60;
                    }
                    else if (bestGpu.MemoryMB >= 8000) // 8GB
                    {
                        parameters["gpu_layers"] = 40;
                    }
                    else if (bestGpu.MemoryMB >= 6000) // 6GB
                    {
                        parameters["gpu_layers"] = 20;
                    }
                    else
                    {
                        parameters["gpu_layers"] = 10; // Minimal GPU acceleration
                    }
                    
                    _logger.LogInformation($"Using GPU acceleration with {parameters["gpu_layers"]} layers");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting Ollama GPU parameters");
            }
            
            return parameters;
        }

        /// <summary>
        /// Check if a GPU is compatible with Ollama
        /// </summary>
        /// <param name="gpu">GPU information</param>
        /// <returns>True if the GPU is compatible, false otherwise</returns>
        private bool IsGpuCompatible(GpuInfo gpu)
        {
            // NVIDIA GPUs with CUDA support
            if (gpu.Type == GpuType.Nvidia && gpu.MemoryMB >= 4000) // At least 4GB VRAM
            {
                return true;
            }
            
            // AMD GPUs with ROCm support (Linux only)
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && 
                gpu.Type == GpuType.Amd && 
                gpu.MemoryMB >= 8000) // At least 8GB VRAM for AMD
            {
                return true;
            }
            
            // Apple Silicon (Metal)
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX) && 
                gpu.Type == GpuType.Apple && 
                gpu.MemoryMB >= 4000) // At least 4GB unified memory
            {
                return true;
            }
            
            return false;
        }

        /// <summary>
        /// Get GPU information on Windows
        /// </summary>
        private List<GpuInfo> GetWindowsGpuInfo()
        {
            var result = new List<GpuInfo>();
            
            try
            {
                // Use PowerShell to get GPU information
                var startInfo = new ProcessStartInfo
                {
                    FileName = "powershell",
                    Arguments = "-Command \"Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM | Format-List\"",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                
                using var process = Process.Start(startInfo);
                if (process == null)
                {
                    _logger.LogError("Failed to start PowerShell process");
                    return result;
                }
                
                var output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();
                
                // Parse the output
                var gpuBlocks = output.Split(new[] { "\r\n\r\n" }, StringSplitOptions.RemoveEmptyEntries);
                
                foreach (var block in gpuBlocks)
                {
                    var nameMatch = Regex.Match(block, @"Name\s+:\s+(.+)");
                    var ramMatch = Regex.Match(block, @"AdapterRAM\s+:\s+(\d+)");
                    
                    if (nameMatch.Success)
                    {
                        var name = nameMatch.Groups[1].Value.Trim();
                        var memoryBytes = ramMatch.Success ? long.Parse(ramMatch.Groups[1].Value) : 0;
                        var memoryMB = (int)(memoryBytes / (1024 * 1024));
                        
                        var type = GpuType.Unknown;
                        if (name.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase))
                        {
                            type = GpuType.Nvidia;
                        }
                        else if (name.Contains("AMD", StringComparison.OrdinalIgnoreCase) || 
                                 name.Contains("Radeon", StringComparison.OrdinalIgnoreCase))
                        {
                            type = GpuType.Amd;
                        }
                        else if (name.Contains("Intel", StringComparison.OrdinalIgnoreCase))
                        {
                            type = GpuType.Intel;
                        }
                        
                        result.Add(new GpuInfo
                        {
                            Name = name,
                            MemoryMB = memoryMB,
                            Type = type
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting Windows GPU information");
            }
            
            return result;
        }

        /// <summary>
        /// Get GPU information on Linux
        /// </summary>
        private List<GpuInfo> GetLinuxGpuInfo()
        {
            var result = new List<GpuInfo>();
            
            try
            {
                // Try to get NVIDIA GPU info using nvidia-smi
                try
                {
                    var startInfo = new ProcessStartInfo
                    {
                        FileName = "nvidia-smi",
                        Arguments = "--query-gpu=name,memory.total --format=csv,noheader",
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };
                    
                    using var process = Process.Start(startInfo);
                    if (process != null)
                    {
                        var output = process.StandardOutput.ReadToEnd();
                        process.WaitForExit();
                        
                        if (process.ExitCode == 0)
                        {
                            var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                            
                            foreach (var line in lines)
                            {
                                var parts = line.Split(',', StringSplitOptions.RemoveEmptyEntries);
                                if (parts.Length >= 2)
                                {
                                    var name = parts[0].Trim();
                                    var memoryStr = parts[1].Trim();
                                    
                                    // Parse memory (format is typically "16384 MiB")
                                    var memoryMatch = Regex.Match(memoryStr, @"(\d+)\s*MiB");
                                    var memoryMB = memoryMatch.Success ? int.Parse(memoryMatch.Groups[1].Value) : 0;
                                    
                                    result.Add(new GpuInfo
                                    {
                                        Name = name,
                                        MemoryMB = memoryMB,
                                        Type = GpuType.Nvidia
                                    });
                                }
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogDebug(ex, "Error getting NVIDIA GPU information on Linux");
                }
                
                // Try to get AMD GPU info using rocm-smi
                try
                {
                    var startInfo = new ProcessStartInfo
                    {
                        FileName = "rocm-smi",
                        Arguments = "--showproductname --showmeminfo vram",
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };
                    
                    using var process = Process.Start(startInfo);
                    if (process != null)
                    {
                        var output = process.StandardOutput.ReadToEnd();
                        process.WaitForExit();
                        
                        if (process.ExitCode == 0)
                        {
                            // Parse the output (format is more complex)
                            var nameMatch = Regex.Match(output, @"GPU\[[\d]+\]:\s+([^\n]+)");
                            var memoryMatch = Regex.Match(output, @"VRAM Total Memory \(B\):\s+(\d+)");
                            
                            if (nameMatch.Success && memoryMatch.Success)
                            {
                                var name = nameMatch.Groups[1].Value.Trim();
                                var memoryBytes = long.Parse(memoryMatch.Groups[1].Value);
                                var memoryMB = (int)(memoryBytes / (1024 * 1024));
                                
                                result.Add(new GpuInfo
                                {
                                    Name = name,
                                    MemoryMB = memoryMB,
                                    Type = GpuType.Amd
                                });
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogDebug(ex, "Error getting AMD GPU information on Linux");
                }
                
                // If no GPUs found, try lspci as a fallback
                if (result.Count == 0)
                {
                    var startInfo = new ProcessStartInfo
                    {
                        FileName = "lspci",
                        Arguments = "-v | grep -i vga",
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };
                    
                    using var process = Process.Start(startInfo);
                    if (process != null)
                    {
                        var output = process.StandardOutput.ReadToEnd();
                        process.WaitForExit();
                        
                        var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                        
                        foreach (var line in lines)
                        {
                            var type = GpuType.Unknown;
                            
                            if (line.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase))
                            {
                                type = GpuType.Nvidia;
                            }
                            else if (line.Contains("AMD", StringComparison.OrdinalIgnoreCase) || 
                                     line.Contains("Radeon", StringComparison.OrdinalIgnoreCase))
                            {
                                type = GpuType.Amd;
                            }
                            else if (line.Contains("Intel", StringComparison.OrdinalIgnoreCase))
                            {
                                type = GpuType.Intel;
                            }
                            
                            result.Add(new GpuInfo
                            {
                                Name = line.Trim(),
                                MemoryMB = 0, // Memory unknown
                                Type = type
                            });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting Linux GPU information");
            }
            
            return result;
        }

        /// <summary>
        /// Get GPU information on macOS
        /// </summary>
        private List<GpuInfo> GetMacGpuInfo()
        {
            var result = new List<GpuInfo>();
            
            try
            {
                // Use system_profiler to get GPU information
                var startInfo = new ProcessStartInfo
                {
                    FileName = "system_profiler",
                    Arguments = "SPDisplaysDataType",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };
                
                using var process = Process.Start(startInfo);
                if (process == null)
                {
                    _logger.LogError("Failed to start system_profiler process");
                    return result;
                }
                
                var output = process.StandardOutput.ReadToEnd();
                process.WaitForExit();
                
                // Check if it's Apple Silicon
                var isAppleSilicon = output.Contains("Apple M", StringComparison.OrdinalIgnoreCase);
                
                if (isAppleSilicon)
                {
                    // For Apple Silicon, we need to get the memory from sysctl
                    var memoryStartInfo = new ProcessStartInfo
                    {
                        FileName = "sysctl",
                        Arguments = "hw.memsize",
                        RedirectStandardOutput = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };
                    
                    using var memoryProcess = Process.Start(memoryStartInfo);
                    if (memoryProcess != null)
                    {
                        var memoryOutput = memoryProcess.StandardOutput.ReadToEnd();
                        memoryProcess.WaitForExit();
                        
                        var memoryMatch = Regex.Match(memoryOutput, @"hw.memsize:\s*(\d+)");
                        var memoryBytes = memoryMatch.Success ? long.Parse(memoryMatch.Groups[1].Value) : 0;
                        var memoryMB = (int)(memoryBytes / (1024 * 1024));
                        
                        // For Apple Silicon, we'll assume half of system memory is available for GPU
                        var gpuMemoryMB = memoryMB / 2;
                        
                        // Get the chip model
                        var modelMatch = Regex.Match(output, @"Chipset Model:\s*(.+)");
                        var model = modelMatch.Success ? modelMatch.Groups[1].Value.Trim() : "Apple Silicon";
                        
                        result.Add(new GpuInfo
                        {
                            Name = model,
                            MemoryMB = gpuMemoryMB,
                            Type = GpuType.Apple
                        });
                    }
                }
                else
                {
                    // For discrete GPUs, parse the output
                    var chipsetMatch = Regex.Match(output, @"Chipset Model:\s*(.+)");
                    var vramMatch = Regex.Match(output, @"VRAM \(Total\):\s*(\d+)\s*MB");
                    
                    if (chipsetMatch.Success)
                    {
                        var name = chipsetMatch.Groups[1].Value.Trim();
                        var memoryMB = vramMatch.Success ? int.Parse(vramMatch.Groups[1].Value) : 0;
                        
                        var type = GpuType.Unknown;
                        if (name.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase))
                        {
                            type = GpuType.Nvidia;
                        }
                        else if (name.Contains("AMD", StringComparison.OrdinalIgnoreCase) || 
                                 name.Contains("Radeon", StringComparison.OrdinalIgnoreCase))
                        {
                            type = GpuType.Amd;
                        }
                        else if (name.Contains("Intel", StringComparison.OrdinalIgnoreCase))
                        {
                            type = GpuType.Intel;
                        }
                        
                        result.Add(new GpuInfo
                        {
                            Name = name,
                            MemoryMB = memoryMB,
                            Type = type
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting macOS GPU information");
            }
            
            return result;
        }
    }

    /// <summary>
    /// GPU information
    /// </summary>
    public class GpuInfo
    {
        /// <summary>
        /// GPU name
        /// </summary>
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// GPU memory in MB
        /// </summary>
        public int MemoryMB { get; set; }
        
        /// <summary>
        /// GPU type
        /// </summary>
        public GpuType Type { get; set; }
    }

    /// <summary>
    /// GPU type
    /// </summary>
    public enum GpuType
    {
        /// <summary>
        /// Unknown GPU type
        /// </summary>
        Unknown,
        
        /// <summary>
        /// NVIDIA GPU
        /// </summary>
        Nvidia,
        
        /// <summary>
        /// AMD GPU
        /// </summary>
        Amd,
        
        /// <summary>
        /// Intel GPU
        /// </summary>
        Intel,
        
        /// <summary>
        /// Apple Silicon GPU
        /// </summary>
        Apple
    }
}
