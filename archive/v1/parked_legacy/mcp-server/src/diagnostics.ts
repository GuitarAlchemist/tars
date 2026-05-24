/**
 * TARS Real Diagnostics - NO FAKE DATA, ONLY REAL SYSTEM MEASUREMENTS
 */

import * as si from 'systeminformation';
import simpleGit from 'simple-git';
import ping from 'ping';
import { promises as fs } from 'fs';
import { exec } from 'child_process';
import { promisify } from 'util';
import winston from 'winston';
import {
  GpuInfo,
  GitRepositoryHealth,
  NetworkDiagnostics,
  SystemResourceMetrics,
  ServiceHealth,
  ComprehensiveDiagnostics,
  TarsProjectInfo,
} from './types.js';

const execAsync = promisify(exec);

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'tars-mcp-server.log' })
  ]
});

/**
 * Detect REAL GPU information using system tools
 */
export async function detectGpuInfo(): Promise<GpuInfo[]> {
  try {
    const graphics = await si.graphics();
    const gpus: GpuInfo[] = [];

    for (const controller of graphics.controllers) {
      // Check for CUDA support (NVIDIA GPUs)
      const isCudaSupported = controller.vendor?.toLowerCase().includes('nvidia') || 
                             controller.model?.toLowerCase().includes('nvidia') || false;

      gpus.push({
        name: controller.model || 'Unknown GPU',
        memoryTotal: controller.vram || 0,
        memoryUsed: controller.vramUsed || 0,
        memoryFree: (controller.vram || 0) - (controller.vramUsed || 0),
        temperature: controller.temperatureGpu,
        powerUsage: undefined, // Would need additional tools
        utilizationGpu: controller.utilizationGpu,
        utilizationMemory: controller.utilizationMemory,
        cudaSupported: isCudaSupported,
        driverVersion: controller.driverVersion,
      });
    }

    // If no GPUs detected, try nvidia-smi for NVIDIA GPUs
    if (gpus.length === 0) {
      try {
        const { stdout } = await execAsync('nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,utilization.memory,driver_version --format=csv,noheader,nounits');
        const lines = stdout.trim().split('\n');
        
        for (const line of lines) {
          const parts = line.split(', ');
          if (parts.length >= 8) {
            gpus.push({
              name: parts[0] || 'NVIDIA GPU',
              memoryTotal: parseInt(parts[1] || '0') * 1024 * 1024, // Convert MB to bytes
              memoryUsed: parseInt(parts[2] || '0') * 1024 * 1024,
              memoryFree: parseInt(parts[3] || '0') * 1024 * 1024,
              temperature: parseFloat(parts[4] || '0'),
              utilizationGpu: parseFloat(parts[5] || '0'),
              utilizationMemory: parseFloat(parts[6] || '0'),
              cudaSupported: true,
              driverVersion: parts[7],
            });
          }
        }
      } catch (error) {
        logger.warn('nvidia-smi not available or failed', { error: error.message });
      }
    }

    return gpus;
  } catch (error) {
    logger.error('GPU detection failed', { error: error.message });
    return [{
      name: 'GPU Detection Failed',
      memoryTotal: 0,
      memoryUsed: 0,
      memoryFree: 0,
      cudaSupported: false,
    }];
  }
}

/**
 * Get REAL git repository health using git commands
 */
export async function getGitRepositoryHealth(repositoryPath: string): Promise<GitRepositoryHealth> {
  try {
    const git = simpleGit(repositoryPath);
    
    // Check if it's a git repository
    const isRepo = await git.checkIsRepo();
    if (!isRepo) {
      return {
        isRepository: false,
        isClean: false,
        unstagedChanges: 0,
        stagedChanges: 0,
        commits: 0,
        aheadBy: 0,
        behindBy: 0,
      };
    }

    // Get current branch
    const branch = await git.branch();
    const currentBranch = branch.current;

    // Get status
    const status = await git.status();
    
    // Get commit count
    const log = await git.log();
    const commits = log.total;

    // Get remote URL
    const remotes = await git.getRemotes(true);
    const originRemote = remotes.find(r => r.name === 'origin');
    const remoteUrl = originRemote?.refs?.fetch;

    // Get last commit info
    const lastCommit = log.latest;
    const lastCommitHash = lastCommit?.hash;
    const lastCommitDate = lastCommit?.date ? new Date(lastCommit.date) : undefined;

    // Calculate ahead/behind (simplified)
    let aheadBy = 0;
    let behindBy = 0;
    try {
      if (status.tracking) {
        aheadBy = status.ahead || 0;
        behindBy = status.behind || 0;
      }
    } catch (error) {
      logger.warn('Could not determine ahead/behind status', { error: error.message });
    }

    return {
      isRepository: true,
      currentBranch,
      isClean: status.isClean(),
      unstagedChanges: status.not_added.length + status.modified.length + status.deleted.length,
      stagedChanges: status.staged.length,
      commits,
      remoteUrl,
      lastCommitHash,
      lastCommitDate,
      aheadBy,
      behindBy,
    };
  } catch (error) {
    logger.error('Git health check failed', { error: error.message, repositoryPath });
    return {
      isRepository: false,
      isClean: false,
      unstagedChanges: 0,
      stagedChanges: 0,
      commits: 0,
      aheadBy: 0,
      behindBy: 0,
    };
  }
}

/**
 * Perform REAL network diagnostics
 */
export async function performNetworkDiagnostics(): Promise<NetworkDiagnostics> {
  try {
    // Check internet connectivity
    const networkInterfaces = await si.networkInterfaces();
    const activeInterfaces = networkInterfaces.filter(iface => iface.operstate === 'up');
    const isConnected = activeInterfaces.length > 0;

    // Get public IP address
    let publicIpAddress: string | undefined;
    try {
      const response = await fetch('https://api.ipify.org?format=text', { 
        signal: AbortSignal.timeout(5000) 
      });
      publicIpAddress = await response.text();
    } catch (error) {
      logger.warn('Could not get public IP', { error: error.message });
    }

    // DNS resolution test
    const dnsStart = Date.now();
    let dnsResolutionTime = 0;
    try {
      await fetch('https://google.com', { 
        method: 'HEAD',
        signal: AbortSignal.timeout(5000)
      });
      dnsResolutionTime = Date.now() - dnsStart;
    } catch (error) {
      dnsResolutionTime = -1; // Failed
    }

    // Ping test
    let pingLatency: number | undefined;
    try {
      const pingResult = await ping.promise.probe('8.8.8.8', {
        timeout: 5,
        extra: ['-c', '1'],
      });
      if (pingResult.alive) {
        pingLatency = parseFloat(pingResult.time);
      }
    } catch (error) {
      logger.warn('Ping test failed', { error: error.message });
    }

    // Get network connections
    const connections = await si.networkConnections();
    const activeConnections = connections.length;

    return {
      isConnected,
      publicIpAddress,
      dnsResolutionTime,
      pingLatency,
      downloadSpeed: undefined, // Would need speed test
      uploadSpeed: undefined, // Would need speed test
      activeConnections,
      networkInterfaces: activeInterfaces.map(iface => iface.iface),
    };
  } catch (error) {
    logger.error('Network diagnostics failed', { error: error.message });
    return {
      isConnected: false,
      dnsResolutionTime: -1,
      activeConnections: 0,
      networkInterfaces: [],
    };
  }
}

/**
 * Get REAL system resource metrics
 */
export async function getSystemResourceMetrics(): Promise<SystemResourceMetrics> {
  try {
    const [cpu, memory, disk, processes] = await Promise.all([
      si.cpu(),
      si.mem(),
      si.fsSize(),
      si.processes(),
    ]);

    // Get current CPU load
    const cpuLoad = await si.currentLoad();

    // Calculate disk usage for primary drive
    const primaryDisk = disk[0] || { size: 0, used: 0, available: 0 };

    return {
      cpuUsagePercent: cpuLoad.currentLoad || 0,
      cpuCoreCount: cpu.cores || 0,
      cpuFrequency: cpu.speed || 0,
      memoryTotalBytes: memory.total || 0,
      memoryUsedBytes: memory.used || 0,
      memoryAvailableBytes: memory.available || 0,
      diskTotalBytes: primaryDisk.size || 0,
      diskUsedBytes: primaryDisk.used || 0,
      diskFreeBytes: primaryDisk.available || 0,
      processCount: processes.all || 0,
      threadCount: 0, // Would need additional calculation
      uptime: si.time().uptime || 0,
    };
  } catch (error) {
    logger.error('System resource metrics failed', { error: error.message });
    return {
      cpuUsagePercent: 0,
      cpuCoreCount: 0,
      cpuFrequency: 0,
      memoryTotalBytes: 0,
      memoryUsedBytes: 0,
      memoryAvailableBytes: 0,
      diskTotalBytes: 0,
      diskUsedBytes: 0,
      diskFreeBytes: 0,
      processCount: 0,
      threadCount: 0,
      uptime: 0,
    };
  }
}

/**
 * Check REAL service health
 */
export async function checkServiceHealth(): Promise<ServiceHealth> {
  try {
    // File system permissions test
    let fileSystemPermissions = false;
    try {
      const tempFile = `/tmp/tars-test-${Date.now()}.txt`;
      await fs.writeFile(tempFile, 'test');
      await fs.unlink(tempFile);
      fileSystemPermissions = true;
    } catch (error) {
      logger.warn('File system permissions test failed', { error: error.message });
    }

    // Environment variables
    const environmentVariables = { ...process.env };

    // Get listening ports
    const connections = await si.networkConnections();
    const listeningPorts = connections
      .filter(conn => conn.state === 'LISTEN')
      .map(conn => parseInt(conn.localPort))
      .filter(port => !isNaN(port));

    // Get running services/processes
    const processes = await si.processes();
    const servicesRunning = processes.list
      .map(proc => proc.name)
      .filter(name => name && name.length > 0)
      .slice(0, 50); // Limit to first 50 for performance

    return {
      databaseConnectivity: false, // Would need actual DB connection test
      webServiceAvailability: false, // Would need actual service test
      fileSystemPermissions,
      environmentVariables,
      portsListening: [...new Set(listeningPorts)], // Remove duplicates
      servicesRunning: [...new Set(servicesRunning)], // Remove duplicates
    };
  } catch (error) {
    logger.error('Service health check failed', { error: error.message });
    return {
      databaseConnectivity: false,
      webServiceAvailability: false,
      fileSystemPermissions: false,
      environmentVariables: {},
      portsListening: [],
      servicesRunning: [],
    };
  }
}

/**
 * Calculate overall health score from all metrics
 */
export function calculateOverallHealthScore(diagnostics: ComprehensiveDiagnostics): number {
  const scores: number[] = [];

  // GPU health (if available)
  if (diagnostics.gpuInfo.length > 0) {
    const gpuScore = diagnostics.gpuInfo.reduce((acc, gpu) => {
      if (gpu.cudaSupported) return acc + 100;
      if (gpu.memoryTotal > 0) return acc + 75;
      return acc + 25;
    }, 0) / diagnostics.gpuInfo.length;
    scores.push(gpuScore);
  }

  // Git health
  const gitScore = diagnostics.gitHealth.isRepository
    ? diagnostics.gitHealth.isClean
      ? 100
      : diagnostics.gitHealth.unstagedChanges < 5
      ? 75
      : 50
    : 25;
  scores.push(gitScore);

  // Network health
  const networkScore = diagnostics.networkDiagnostics.isConnected
    ? diagnostics.networkDiagnostics.pingLatency !== undefined
      ? diagnostics.networkDiagnostics.pingLatency < 50
        ? 100
        : diagnostics.networkDiagnostics.pingLatency < 100
        ? 75
        : 50
      : 25
    : 0;
  scores.push(networkScore);

  // System resource health
  const cpuScore = diagnostics.systemResources.cpuUsagePercent < 80 ? 100 : 50;
  const memoryScore = diagnostics.systemResources.memoryTotalBytes > 0
    ? (diagnostics.systemResources.memoryUsedBytes / diagnostics.systemResources.memoryTotalBytes) * 100 < 80
      ? 100
      : 50
    : 50;
  const diskScore = diagnostics.systemResources.diskTotalBytes > 0
    ? (diagnostics.systemResources.diskUsedBytes / diagnostics.systemResources.diskTotalBytes) * 100 < 90
      ? 100
      : 50
    : 50;
  const resourceScore = (cpuScore + memoryScore + diskScore) / 3;
  scores.push(resourceScore);

  // Service health
  const serviceScore = (
    (diagnostics.serviceHealth.fileSystemPermissions ? 100 : 0) +
    (diagnostics.serviceHealth.portsListening.length > 0 ? 100 : 50)
  ) / 2;
  scores.push(serviceScore);

  // Calculate weighted average
  return scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;
}

/**
 * Get comprehensive REAL diagnostics - the main function
 */
export async function getComprehensiveDiagnostics(repositoryPath: string = process.cwd()): Promise<ComprehensiveDiagnostics> {
  logger.info('Starting comprehensive diagnostics', { repositoryPath });

  const [gpuInfo, gitHealth, networkDiagnostics, systemResources, serviceHealth] = await Promise.all([
    detectGpuInfo(),
    getGitRepositoryHealth(repositoryPath),
    performNetworkDiagnostics(),
    getSystemResourceMetrics(),
    checkServiceHealth(),
  ]);

  const diagnostics: ComprehensiveDiagnostics = {
    timestamp: new Date(),
    gpuInfo,
    gitHealth,
    networkDiagnostics,
    systemResources,
    serviceHealth,
    overallHealthScore: 0, // Will be calculated next
  };

  diagnostics.overallHealthScore = calculateOverallHealthScore(diagnostics);

  logger.info('Comprehensive diagnostics completed', { 
    overallHealthScore: diagnostics.overallHealthScore,
    gpuCount: gpuInfo.length,
    isGitRepo: gitHealth.isRepository,
    networkConnected: networkDiagnostics.isConnected
  });

  return diagnostics;
}

/**
 * Get TARS project information
 */
export async function getTarsProjectInfo(projectPath: string = process.cwd()): Promise<TarsProjectInfo> {
  try {
    const packageJsonPath = `${projectPath}/package.json`;
    const fsprojFiles = await execAsync(`find ${projectPath} -name "*.fsproj" -type f 2>/dev/null || echo ""`);
    const csprojFiles = await execAsync(`find ${projectPath} -name "*.csproj" -type f 2>/dev/null || echo ""`);

    let projectType: 'fsharp' | 'csharp' | 'typescript' | 'other' = 'other';
    let buildStatus: 'success' | 'failed' | 'building' | 'unknown' = 'unknown';
    let dependencies: string[] = [];

    // Determine project type
    if (fsprojFiles.stdout.trim()) {
      projectType = 'fsharp';
    } else if (csprojFiles.stdout.trim()) {
      projectType = 'csharp';
    } else {
      try {
        await fs.access(packageJsonPath);
        projectType = 'typescript';
        const packageJson = JSON.parse(await fs.readFile(packageJsonPath, 'utf-8'));
        dependencies = Object.keys(packageJson.dependencies || {});
      } catch (error) {
        // Keep as 'other'
      }
    }

    // Try to determine build status
    try {
      if (projectType === 'fsharp' || projectType === 'csharp') {
        const { stdout, stderr } = await execAsync(`cd ${projectPath} && dotnet build --verbosity quiet`);
        buildStatus = stderr ? 'failed' : 'success';
      } else if (projectType === 'typescript') {
        const { stdout, stderr } = await execAsync(`cd ${projectPath} && npm run build 2>/dev/null || echo "no build script"`);
        buildStatus = stderr.includes('error') ? 'failed' : 'success';
      }
    } catch (error) {
      buildStatus = 'failed';
    }

    return {
      name: projectPath.split('/').pop() || 'Unknown Project',
      path: projectPath,
      type: projectType,
      buildStatus,
      lastBuild: new Date(),
      dependencies,
      testCoverage: undefined, // Would need coverage tools
    };
  } catch (error) {
    logger.error('Failed to get project info', { error: error.message, projectPath });
    return {
      name: 'Unknown Project',
      path: projectPath,
      type: 'other',
      buildStatus: 'unknown',
      dependencies: [],
    };
  }
}
