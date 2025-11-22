/**
 * TARS MCP Server Types - Real system diagnostics for Augment Code
 */

export interface GpuInfo {
  name: string;
  memoryTotal: number;
  memoryUsed: number;
  memoryFree: number;
  temperature?: number;
  powerUsage?: number;
  utilizationGpu?: number;
  utilizationMemory?: number;
  cudaSupported: boolean;
  driverVersion?: string;
}

export interface GitRepositoryHealth {
  isRepository: boolean;
  currentBranch?: string;
  isClean: boolean;
  unstagedChanges: number;
  stagedChanges: number;
  commits: number;
  remoteUrl?: string;
  lastCommitHash?: string;
  lastCommitDate?: Date;
  aheadBy: number;
  behindBy: number;
}

export interface NetworkDiagnostics {
  isConnected: boolean;
  publicIpAddress?: string;
  dnsResolutionTime: number;
  pingLatency?: number;
  downloadSpeed?: number;
  uploadSpeed?: number;
  activeConnections: number;
  networkInterfaces: string[];
}

export interface SystemResourceMetrics {
  cpuUsagePercent: number;
  cpuCoreCount: number;
  cpuFrequency: number;
  memoryTotalBytes: number;
  memoryUsedBytes: number;
  memoryAvailableBytes: number;
  diskTotalBytes: number;
  diskUsedBytes: number;
  diskFreeBytes: number;
  processCount: number;
  threadCount: number;
  uptime: number; // seconds
}

export interface ServiceHealth {
  databaseConnectivity: boolean;
  webServiceAvailability: boolean;
  fileSystemPermissions: boolean;
  environmentVariables: Record<string, string>;
  portsListening: number[];
  servicesRunning: string[];
}

export interface ComprehensiveDiagnostics {
  timestamp: Date;
  gpuInfo: GpuInfo[];
  gitHealth: GitRepositoryHealth;
  networkDiagnostics: NetworkDiagnostics;
  systemResources: SystemResourceMetrics;
  serviceHealth: ServiceHealth;
  overallHealthScore: number;
}

export interface TarsProjectInfo {
  name: string;
  path: string;
  type: 'fsharp' | 'csharp' | 'typescript' | 'other';
  buildStatus: 'success' | 'failed' | 'building' | 'unknown';
  lastBuild?: Date;
  dependencies: string[];
  testCoverage?: number;
}

export interface FluxScriptExecution {
  scriptId: string;
  script: string;
  status: 'running' | 'completed' | 'failed';
  startTime: Date;
  endTime?: Date;
  result?: any;
  error?: string;
  logs: string[];
}

export interface AgentCoordination {
  agentId: string;
  agentType: string;
  status: 'active' | 'idle' | 'busy' | 'error';
  currentTask?: string;
  capabilities: string[];
  performance: {
    tasksCompleted: number;
    averageExecutionTime: number;
    successRate: number;
  };
}

export interface TarsSystemStatus {
  version: string;
  uptime: number;
  activeAgents: number;
  runningScripts: number;
  systemLoad: number;
  memoryUsage: number;
  diagnosticsEnabled: boolean;
  mcpConnections: number;
}
