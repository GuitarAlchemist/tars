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
  // NOTE: diagnose_and_remediate currently emits free-text markdown.
  // The eventual Path C upgrade (governed cross-repo remediation) will
  // require strict JSON output so the ix harness can consume it via a
  // future `ix_dispatch_action` MCP endpoint. See the design doc in
  // the ix repo: docs/brainstorms/2026-04-11-triage-session-scenario.md
  // — that scenario (ix_triage_session) is the in-process prototype;
  // once it lands, this tool's system prompt should be tightened to
  // emit { issues: [{ severity, remediation_tool, params }] } matching
  // the ix AgentAction::InvokeTool shape.
  {
    name: 'diagnose_and_remediate',
    description:
      'Collect comprehensive system diagnostics and ask the client\'s LLM (via MCP sampling) to analyze them. Returns the top 3 issues, severity classifications, and actionable remediations. Gracefully falls back to raw diagnostics if the client does not support sampling.',
    inputSchema: {
      type: 'object',
      properties: {
        repository_path: {
          type: 'string',
          description: 'Path to the repository to analyze (defaults to current directory)',
        },
        max_tokens: {
          type: 'number',
          description: 'Maximum tokens for the LLM analysis response (default: 800)',
          default: 800,
        },
      },
    },
  },
];

/**
 * Format a ComprehensiveDiagnostics object into a compact text payload
 * suitable for LLM analysis via MCP sampling.
 */
function formatDiagnosticsForLlm(
  diagnostics: Awaited<ReturnType<typeof getComprehensiveDiagnostics>>
): string {
  const gb = (bytes: number): string => (bytes / (1024 * 1024 * 1024)).toFixed(2);
  const gpuSection = diagnostics.gpuInfo
    .map(
      (gpu) =>
        `- ${gpu.name}: ${gb(gpu.memoryUsed)}/${gb(gpu.memoryTotal)} GB VRAM, ` +
        `util=${gpu.utilizationGpu ?? 'N/A'}%, temp=${gpu.temperature ?? 'N/A'}C, ` +
        `cuda=${gpu.cudaSupported}`
    )
    .join('\n');

  return [
    `TARS System Diagnostics Snapshot (timestamp: ${diagnostics.timestamp.toISOString()})`,
    `Overall health score: ${diagnostics.overallHealthScore.toFixed(1)}%`,
    '',
    '## GPU',
    gpuSection || '- none detected',
    '',
    '## Git repository',
    `- isRepository=${diagnostics.gitHealth.isRepository}, branch=${diagnostics.gitHealth.currentBranch ?? 'N/A'}`,
    `- clean=${diagnostics.gitHealth.isClean}, unstaged=${diagnostics.gitHealth.unstagedChanges}, staged=${diagnostics.gitHealth.stagedChanges}`,
    `- ahead=${diagnostics.gitHealth.aheadBy}, behind=${diagnostics.gitHealth.behindBy}, commits=${diagnostics.gitHealth.commits}`,
    '',
    '## Network',
    `- connected=${diagnostics.networkDiagnostics.isConnected}, publicIp=${diagnostics.networkDiagnostics.publicIpAddress ?? 'N/A'}`,
    `- dnsResolutionMs=${diagnostics.networkDiagnostics.dnsResolutionTime}, pingMs=${diagnostics.networkDiagnostics.pingLatency ?? 'N/A'}`,
    `- activeConnections=${diagnostics.networkDiagnostics.activeConnections}`,
    '',
    '## System resources',
    `- cpu=${diagnostics.systemResources.cpuUsagePercent.toFixed(1)}% across ${diagnostics.systemResources.cpuCoreCount} cores @ ${diagnostics.systemResources.cpuFrequency}MHz`,
    `- memory=${gb(diagnostics.systemResources.memoryUsedBytes)}/${gb(diagnostics.systemResources.memoryTotalBytes)} GB (avail ${gb(diagnostics.systemResources.memoryAvailableBytes)} GB)`,
    `- disk=${gb(diagnostics.systemResources.diskUsedBytes)}/${gb(diagnostics.systemResources.diskTotalBytes)} GB (free ${gb(diagnostics.systemResources.diskFreeBytes)} GB)`,
    `- processes=${diagnostics.systemResources.processCount}, uptimeHours=${Math.floor(diagnostics.systemResources.uptime / 3600)}`,
    '',
    '## Services',
    `- fsPermissions=${diagnostics.serviceHealth.fileSystemPermissions}`,
    `- dbConnectivity=${diagnostics.serviceHealth.databaseConnectivity}`,
    `- webServiceAvailable=${diagnostics.serviceHealth.webServiceAvailability}`,
    `- portsListening=${diagnostics.serviceHealth.portsListening.length}, servicesRunning=${diagnostics.serviceHealth.servicesRunning.length}`,
  ].join('\n');
}

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

      case 'diagnose_and_remediate': {
        const repositoryPath = (args?.repository_path as string | undefined) ?? process.cwd();
        const maxTokens = (args?.max_tokens as number | undefined) ?? 800;

        // Step 1: collect raw diagnostics using the same logic as get_comprehensive_diagnostics
        const diagnostics = await getComprehensiveDiagnostics(repositoryPath);
        const formatted = formatDiagnosticsForLlm(diagnostics);

        const systemPrompt =
          'You are an SRE analyzing TARS system diagnostics. ' +
          'Identify the top 3 issues (ordered by urgency), classify each as CRITICAL, WARNING, or INFO, ' +
          'and suggest specific, actionable remediations. Be concise and prescriptive. ' +
          'If the system appears healthy, say so and list the top 3 optimization opportunities instead.';

        const userMessage =
          'Analyze this system state snapshot and produce the ranked issue list with severity ' +
          'and remediation steps:\n\n' +
          formatted;

        // Step 2: ask the client's LLM via MCP sampling. If the client did not declare
        // the `sampling` capability, this will reject — fall back to returning the raw
        // diagnostics with an explanatory note so the tool still provides value.
        try {
          const samplingResult = await server.createMessage({
            messages: [
              {
                role: 'user',
                content: { type: 'text', text: userMessage },
              },
            ],
            systemPrompt,
            maxTokens,
            modelPreferences: {
              hints: [{ name: 'claude-3-sonnet' }],
              intelligencePriority: 0.8,
              speedPriority: 0.3,
            },
          });

          const analysisText =
            samplingResult.content.type === 'text'
              ? samplingResult.content.text
              : '[non-text response from sampling client]';

          logger.info('diagnose_and_remediate sampling succeeded', {
            model: samplingResult.model,
            stopReason: samplingResult.stopReason,
          });

          return {
            content: [
              {
                type: 'text',
                text:
                  `# TARS Diagnose & Remediate\n\n` +
                  `**Health score**: ${diagnostics.overallHealthScore.toFixed(1)}%  \n` +
                  `**Analysis model**: ${samplingResult.model}  \n` +
                  `**Stop reason**: ${samplingResult.stopReason ?? 'n/a'}\n\n` +
                  `## LLM Analysis\n\n${analysisText}\n\n` +
                  `---\n\n## Raw diagnostics input\n\n\`\`\`\n${formatted}\n\`\`\``,
              },
            ],
          };
        } catch (samplingError) {
          const samplingErrMsg =
            samplingError instanceof Error ? samplingError.message : String(samplingError);
          logger.warn('diagnose_and_remediate sampling failed, falling back to raw diagnostics', {
            error: samplingErrMsg,
          });

          return {
            content: [
              {
                type: 'text',
                text:
                  `# TARS Diagnose & Remediate (fallback mode)\n\n` +
                  `**Note**: MCP sampling request failed — the connected client likely does ` +
                  `not declare the \`sampling\` capability, or the user declined the request. ` +
                  `Returning raw diagnostics instead so you can analyze them directly.\n\n` +
                  `**Sampling error**: ${samplingErrMsg}\n\n` +
                  `**Health score**: ${diagnostics.overallHealthScore.toFixed(1)}%\n\n` +
                  `## Raw diagnostics\n\n\`\`\`\n${formatted}\n\`\`\``,
              },
              {
                type: 'text',
                text: JSON.stringify(diagnostics, null, 2),
              },
            ],
          };
        }
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
