import axios from 'axios';
import { TarsSystemStatus, TarsProject, TarsMetascript, TarsAgent, PerformanceMetrics, CommandResult } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
});

// TARS API Service - Created by TARS for TARS
export const tarsApi = {
  // System Status
  getSystemStatus: (): Promise<TarsSystemStatus> =>
    api.get('/system/status').then(res => res.data),

  // Projects
  getProjects: (): Promise<TarsProject[]> =>
    api.get('/projects').then(res => res.data),
  
  createProject: (prompt: string): Promise<TarsProject> =>
    api.post('/projects', { prompt }).then(res => res.data),
  
  getProject: (id: string): Promise<TarsProject> =>
    api.get(`/projects/${id}`).then(res => res.data),

  // Metascripts
  getMetascripts: (): Promise<TarsMetascript[]> =>
    api.get('/metascripts').then(res => res.data),
  
  executeMetascript: (name: string, params?: any): Promise<TarsMetascript> =>
    api.post('/metascripts/execute', { name, params }).then(res => res.data),

  // Agents
  getAgents: (): Promise<TarsAgent[]> =>
    api.get('/agents').then(res => res.data),
  
  startAgentSystem: (): Promise<void> =>
    api.post('/agents/start').then(res => res.data),
  
  stopAgentSystem: (): Promise<void> =>
    api.post('/agents/stop').then(res => res.data),
  
  assignTask: (task: string, capabilities?: string[]): Promise<TarsAgent> =>
    api.post('/agents/assign-task', { task, capabilities }).then(res => res.data),

  // Performance
  getPerformanceMetrics: (): Promise<PerformanceMetrics> =>
    api.get('/performance').then(res => res.data),

  // Commands
  executeCommand: (command: string): Promise<CommandResult> =>
    api.post('/commands/execute', { command }).then(res => res.data),
};

// Mock data for development (TARS creates its own mock data)
export const mockTarsApi = {
  getSystemStatus: (): Promise<TarsSystemStatus> =>
    Promise.resolve({
      isOnline: true,
      version: '2.0.0',
      uptime: 86400,
      cpuUsage: 45.2,
      memoryUsage: 67.8,
      cudaAvailable: true,
      agentCount: 5,
    }),

  getProjects: (): Promise<TarsProject[]> =>
    Promise.resolve([
      {
        id: '1',
        name: 'File Sync System',
        description: 'Distributed file synchronization with encryption',
        status: 'completed',
        createdAt: '2024-01-15T10:30:00Z',
        lastModified: '2024-01-15T14:45:00Z',
        files: ['src/main.rs', 'src/sync.rs', 'tests/integration.rs'],
        tests: [
          { id: '1', name: 'sync_test', status: 'passed', duration: 1200, message: 'All sync operations successful' },
          { id: '2', name: 'encryption_test', status: 'passed', duration: 800 }
        ]
      },
      {
        id: '2',
        name: 'TARS UI',
        description: 'React TypeScript UI for TARS system',
        status: 'active',
        createdAt: '2024-01-16T09:00:00Z',
        lastModified: '2024-01-16T11:30:00Z',
        files: ['src/App.tsx', 'src/components/Layout.tsx', 'src/pages/Dashboard.tsx'],
        tests: []
      }
    ]),

  getAgents: (): Promise<TarsAgent[]> =>
    Promise.resolve([
      {
        id: '1',
        name: 'CodeAnalysisAgent',
        persona: 'Senior Software Architect',
        status: 'active',
        capabilities: ['code_analysis', 'architecture_review', 'performance_optimization'],
        currentTask: 'Analyzing React component structure',
        performance: {
          tasksCompleted: 42,
          averageExecutionTime: 2.3,
          successRate: 0.95,
          lastActivity: '2024-01-16T11:25:00Z'
        }
      },
      {
        id: '2',
        name: 'CudaOptimizationAgent',
        persona: 'GPU Performance Specialist',
        status: 'busy',
        capabilities: ['cuda_optimization', 'memory_management', 'kernel_tuning'],
        currentTask: 'Optimizing vector search performance',
        performance: {
          tasksCompleted: 18,
          averageExecutionTime: 5.7,
          successRate: 0.98,
          lastActivity: '2024-01-16T11:28:00Z'
        }
      }
    ]),

  getPerformanceMetrics: (): Promise<PerformanceMetrics> =>
    Promise.resolve({
      cuda: {
        available: true,
        deviceName: 'NVIDIA GeForce RTX 3070',
        memoryUsed: 2048,
        memoryTotal: 8192,
        searchesPerSecond: 184000000
      },
      system: {
        cpuUsage: 45.2,
        memoryUsage: 67.8,
        diskUsage: 23.4,
        networkActivity: 12.1
      },
      agents: {
        totalAgents: 5,
        activeAgents: 3,
        tasksInQueue: 7,
        averageResponseTime: 1.2
      }
    })
};
