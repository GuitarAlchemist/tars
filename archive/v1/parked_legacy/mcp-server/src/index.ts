#!/usr/bin/env node

/**
 * TARS MCP Server - Model Context Protocol server for Augment Code integration
 * Provides real-time system diagnostics, TARS integration, and development tools
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';
import winston from 'winston';
import {
  getComprehensiveDiagnostics,
  detectGpuInfo,
  getGitRepositoryHealth,
  performNetworkDiagnostics,
  getSystemResourceMetrics,
  checkServiceHealth,
  getTarsProjectInfo,
} from './diagnostics.js';

// Configure logging
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.simple()
    }),
    new winston.transports.File({ 
      filename: 'tars-mcp-server.log',
      format: winston.format.json()
    })
  ]
});

// Create MCP server
const server = new Server(
  {
    name: 'tars-mcp-server',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// Define available tools
const tools: Tool[] = [
  {
    name: 'get_comprehensive_diagnostics',
    description: 'Get comprehensive real-time system diagnostics including GPU, Git, network, and system resources',
    inputSchema: {
      type: 'object',
      properties: {
        repository_path: {
          type: 'string',
          description: 'Path to the repository to analyze (defaults to current directory)',
        },
      },
    },
  },
  {
    name: 'get_gpu_info',
    description: 'Get real GPU information including CUDA support, memory usage, and performance metrics',
    inputSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    name: 'get_git_health',
    description: 'Get Git repository health including branch status, changes, and commit information',
    inputSchema: {
      type: 'object',
      properties: {
        repository_path: {
          type: 'string',
          description: 'Path to the Git repository (defaults to current directory)',
        },
      },
    },
  },
  {
    name: 'get_network_diagnostics',
    description: 'Perform network diagnostics including connectivity, latency, and interface information',
    inputSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    name: 'get_system_resources',
    description: 'Get real-time system resource metrics including CPU, memory, and disk usage',
    inputSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    name: 'get_service_health',
    description: 'Check service health including file system permissions, ports, and running services',
    inputSchema: {
      type: 'object',
      properties: {},
    },
  },
  {
    name: 'get_tars_project_info',
    description: 'Get TARS project information including build status, dependencies, and project type',
    inputSchema: {
      type: 'object',
      properties: {
        project_path: {
          type: 'string',
          description: 'Path to the project (defaults to current directory)',
        },
      },
    },
  },
  {
    name: 'execute_tars_command',
    description: 'Execute a TARS CLI command and return the results',
    inputSchema: {
      type: 'object',
      properties: {
        command: {
          type: 'string',
          description: 'TARS command to execute (e.g., "diagnostics", "version", "llm")',
        },
        args: {
          type: 'array',
          items: { type: 'string' },
          description: 'Command arguments',
        },
        working_directory: {
          type: 'string',
          description: 'Working directory for command execution',
        },
      },
      required: ['command'],
    },
  },
  {
    name: 'build_tars_project',
    description: 'Build a TARS project and return build status and output',
    inputSchema: {
      type: 'object',
      properties: {
        project_path: {
          type: 'string',
          description: 'Path to the project to build',
        },
        configuration: {
          type: 'string',
          enum: ['Debug', 'Release'],
          description: 'Build configuration',
          default: 'Debug',
        },
      },
    },
  },
  {
    name: 'run_tars_tests',
    description: 'Run tests for a TARS project and return test results',
    inputSchema: {
      type: 'object',
      properties: {
        project_path: {
          type: 'string',
          description: 'Path to the project to test',
        },
        test_filter: {
          type: 'string',
          description: 'Filter for specific tests to run',
        },
      },
    },
  },
];

// List tools handler
server.setRequestHandler(ListToolsRequestSchema, async () => {
  logger.info('Listing available tools');
  return { tools };
});

// Call tool handler
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  logger.info('Tool called', { name, args });

  try {
    switch (name) {
      case 'get_comprehensive_diagnostics': {
        const repositoryPath = args?.repository_path as string || process.cwd();
        const diagnostics = await getComprehensiveDiagnostics(repositoryPath);
        
        return {
          content: [
            {
              type: 'text',
              text: `# TARS Comprehensive System Diagnostics

## Overall Health Score: ${diagnostics.overallHealthScore.toFixed(1)}%

## GPU Information
${diagnostics.gpuInfo.map(gpu => `
- **${gpu.name}**
  - Memory: ${(gpu.memoryTotal / (1024 * 1024 * 1024)).toFixed(2)} GB total, ${(gpu.memoryUsed / (1024 * 1024 * 1024)).toFixed(2)} GB used
  - CUDA Supported: ${gpu.cudaSupported ? '✅' : '❌'}
  - Utilization: ${gpu.utilizationGpu || 'N/A'}%
  - Temperature: ${gpu.temperature || 'N/A'}°C
`).join('')}

## Git Repository Health
- Repository: ${diagnostics.gitHealth.isRepository ? '✅' : '❌'}
- Current Branch: ${diagnostics.gitHealth.currentBranch || 'N/A'}
- Clean: ${diagnostics.gitHealth.isClean ? '✅' : '❌'}
- Unstaged Changes: ${diagnostics.gitHealth.unstagedChanges}
- Staged Changes: ${diagnostics.gitHealth.stagedChanges}
- Total Commits: ${diagnostics.gitHealth.commits}

## Network Diagnostics
- Connected: ${diagnostics.networkDiagnostics.isConnected ? '✅' : '❌'}
- Public IP: ${diagnostics.networkDiagnostics.publicIpAddress || 'N/A'}
- DNS Resolution: ${diagnostics.networkDiagnostics.dnsResolutionTime}ms
- Ping Latency: ${diagnostics.networkDiagnostics.pingLatency || 'N/A'}ms
- Active Connections: ${diagnostics.networkDiagnostics.activeConnections}

## System Resources
- CPU Usage: ${diagnostics.systemResources.cpuUsagePercent.toFixed(1)}%
- CPU Cores: ${diagnostics.systemResources.cpuCoreCount}
- Memory: ${(diagnostics.systemResources.memoryUsedBytes / (1024 * 1024 * 1024)).toFixed(2)} GB / ${(diagnostics.systemResources.memoryTotalBytes / (1024 * 1024 * 1024)).toFixed(2)} GB
- Disk: ${(diagnostics.systemResources.diskUsedBytes / (1024 * 1024 * 1024)).toFixed(2)} GB / ${(diagnostics.systemResources.diskTotalBytes / (1024 * 1024 * 1024)).toFixed(2)} GB
- Uptime: ${Math.floor(diagnostics.systemResources.uptime / 3600)}h ${Math.floor((diagnostics.systemResources.uptime % 3600) / 60)}m

## Service Health
- File System Permissions: ${diagnostics.serviceHealth.fileSystemPermissions ? '✅' : '❌'}
- Listening Ports: ${diagnostics.serviceHealth.portsListening.length}
- Running Services: ${diagnostics.serviceHealth.servicesRunning.length}

*Timestamp: ${diagnostics.timestamp.toISOString()}*`,
            },
            {
              type: 'text',
              text: JSON.stringify(diagnostics, null, 2),
            },
          ],
        };
      }

      case 'get_gpu_info': {
        const gpuInfo = await detectGpuInfo();
        return {
          content: [
            {
              type: 'text',
              text: `# GPU Information\n\n${gpuInfo.map(gpu => `
## ${gpu.name}
- Memory: ${(gpu.memoryTotal / (1024 * 1024 * 1024)).toFixed(2)} GB total
- CUDA Supported: ${gpu.cudaSupported ? '✅ Yes' : '❌ No'}
- Driver Version: ${gpu.driverVersion || 'Unknown'}
- Utilization: ${gpu.utilizationGpu || 'N/A'}%
- Temperature: ${gpu.temperature || 'N/A'}°C
`).join('')}`,
            },
            {
              type: 'text',
              text: JSON.stringify(gpuInfo, null, 2),
            },
          ],
        };
      }

      case 'get_git_health': {
        const repositoryPath = args?.repository_path as string || process.cwd();
        const gitHealth = await getGitRepositoryHealth(repositoryPath);
        return {
          content: [
            {
              type: 'text',
              text: `# Git Repository Health

- **Repository**: ${gitHealth.isRepository ? '✅ Valid Git Repository' : '❌ Not a Git Repository'}
- **Current Branch**: ${gitHealth.currentBranch || 'N/A'}
- **Status**: ${gitHealth.isClean ? '✅ Clean' : '⚠️ Has Changes'}
- **Unstaged Changes**: ${gitHealth.unstagedChanges}
- **Staged Changes**: ${gitHealth.stagedChanges}
- **Total Commits**: ${gitHealth.commits}
- **Remote URL**: ${gitHealth.remoteUrl || 'N/A'}
- **Last Commit**: ${gitHealth.lastCommitHash?.substring(0, 8) || 'N/A'}
- **Ahead/Behind**: +${gitHealth.aheadBy}/-${gitHealth.behindBy}`,
            },
            {
              type: 'text',
              text: JSON.stringify(gitHealth, null, 2),
            },
          ],
        };
      }

      case 'get_network_diagnostics': {
        const networkDiagnostics = await performNetworkDiagnostics();
        return {
          content: [
            {
              type: 'text',
              text: `# Network Diagnostics

- **Connected**: ${networkDiagnostics.isConnected ? '✅ Yes' : '❌ No'}
- **Public IP**: ${networkDiagnostics.publicIpAddress || 'N/A'}
- **DNS Resolution**: ${networkDiagnostics.dnsResolutionTime}ms
- **Ping Latency**: ${networkDiagnostics.pingLatency || 'N/A'}ms
- **Active Connections**: ${networkDiagnostics.activeConnections}
- **Network Interfaces**: ${networkDiagnostics.networkInterfaces.join(', ')}`,
            },
            {
              type: 'text',
              text: JSON.stringify(networkDiagnostics, null, 2),
            },
          ],
        };
      }

      case 'get_system_resources': {
        const systemResources = await getSystemResourceMetrics();
        return {
          content: [
            {
              type: 'text',
              text: `# System Resources

## CPU
- **Usage**: ${systemResources.cpuUsagePercent.toFixed(1)}%
- **Cores**: ${systemResources.cpuCoreCount}
- **Frequency**: ${systemResources.cpuFrequency} MHz

## Memory
- **Used**: ${(systemResources.memoryUsedBytes / (1024 * 1024 * 1024)).toFixed(2)} GB
- **Total**: ${(systemResources.memoryTotalBytes / (1024 * 1024 * 1024)).toFixed(2)} GB
- **Available**: ${(systemResources.memoryAvailableBytes / (1024 * 1024 * 1024)).toFixed(2)} GB

## Disk
- **Used**: ${(systemResources.diskUsedBytes / (1024 * 1024 * 1024)).toFixed(2)} GB
- **Total**: ${(systemResources.diskTotalBytes / (1024 * 1024 * 1024)).toFixed(2)} GB
- **Free**: ${(systemResources.diskFreeBytes / (1024 * 1024 * 1024)).toFixed(2)} GB

## System
- **Processes**: ${systemResources.processCount}
- **Uptime**: ${Math.floor(systemResources.uptime / 3600)}h ${Math.floor((systemResources.uptime % 3600) / 60)}m`,
            },
            {
              type: 'text',
              text: JSON.stringify(systemResources, null, 2),
            },
          ],
        };
      }

      case 'get_service_health': {
        const serviceHealth = await checkServiceHealth();
        return {
          content: [
            {
              type: 'text',
              text: `# Service Health

- **File System Permissions**: ${serviceHealth.fileSystemPermissions ? '✅ OK' : '❌ Failed'}
- **Database Connectivity**: ${serviceHealth.databaseConnectivity ? '✅ Connected' : '❌ Not Connected'}
- **Web Service Availability**: ${serviceHealth.webServiceAvailability ? '✅ Available' : '❌ Not Available'}
- **Listening Ports**: ${serviceHealth.portsListening.length} (${serviceHealth.portsListening.slice(0, 10).join(', ')}${serviceHealth.portsListening.length > 10 ? '...' : ''})
- **Running Services**: ${serviceHealth.servicesRunning.length} services`,
            },
            {
              type: 'text',
              text: JSON.stringify(serviceHealth, null, 2),
            },
          ],
        };
      }

      case 'get_tars_project_info': {
        const projectPath = args?.project_path as string || process.cwd();
        const projectInfo = await getTarsProjectInfo(projectPath);
        return {
          content: [
            {
              type: 'text',
              text: `# TARS Project Information

- **Name**: ${projectInfo.name}
- **Path**: ${projectInfo.path}
- **Type**: ${projectInfo.type}
- **Build Status**: ${projectInfo.buildStatus === 'success' ? '✅' : projectInfo.buildStatus === 'failed' ? '❌' : '⚠️'} ${projectInfo.buildStatus}
- **Dependencies**: ${projectInfo.dependencies.length}
- **Last Build**: ${projectInfo.lastBuild?.toISOString() || 'N/A'}`,
            },
            {
              type: 'text',
              text: JSON.stringify(projectInfo, null, 2),
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    logger.error('Tool execution failed', { name, error: error.message });
    return {
      content: [
        {
          type: 'text',
          text: `Error executing tool ${name}: ${error.message}`,
        },
      ],
      isError: true,
    };
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  
  logger.info('Starting TARS MCP Server', {
    version: '1.0.0',
    tools: tools.length,
    pid: process.pid
  });

  await server.connect(transport);
  
  logger.info('TARS MCP Server connected and ready for Augment Code integration');
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  logger.info('Received SIGINT, shutting down gracefully');
  process.exit(0);
});

process.on('SIGTERM', () => {
  logger.info('Received SIGTERM, shutting down gracefully');
  process.exit(0);
});

// Start the server
main().catch((error) => {
  logger.error('Failed to start TARS MCP Server', { error: error.message });
  process.exit(1);
});
