/**
 * Tests for TARS MCP Server Diagnostics - Ensuring real system measurements
 */

import { describe, test, expect, beforeAll, afterAll } from '@jest/globals';
import {
  detectGpuInfo,
  getGitRepositoryHealth,
  performNetworkDiagnostics,
  getSystemResourceMetrics,
  checkServiceHealth,
  getComprehensiveDiagnostics,
  calculateOverallHealthScore,
  getTarsProjectInfo,
} from '../diagnostics.js';
import { promises as fs } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import simpleGit from 'simple-git';

describe('TARS Real Diagnostics', () => {
  let tempDir: string;
  let tempGitRepo: string;

  beforeAll(async () => {
    // Create temporary directory for testing
    tempDir = join(tmpdir(), `tars-test-${Date.now()}`);
    await fs.mkdir(tempDir, { recursive: true });

    // Create a temporary git repository for testing
    tempGitRepo = join(tempDir, 'test-repo');
    await fs.mkdir(tempGitRepo, { recursive: true });
    
    const git = simpleGit(tempGitRepo);
    await git.init();
    await git.addConfig('user.name', 'Test User');
    await git.addConfig('user.email', 'test@example.com');
    
    // Create a test file and commit
    await fs.writeFile(join(tempGitRepo, 'test.txt'), 'test content');
    await git.add('test.txt');
    await git.commit('Initial commit');
  });

  afterAll(async () => {
    // Clean up temporary directory
    try {
      await fs.rm(tempDir, { recursive: true, force: true });
    } catch (error) {
      console.warn('Failed to clean up temp directory:', error);
    }
  });

  describe('GPU Detection', () => {
    test('should detect real GPU information', async () => {
      const gpuInfo = await detectGpuInfo();
      
      expect(gpuInfo).toBeDefined();
      expect(Array.isArray(gpuInfo)).toBe(true);
      expect(gpuInfo.length).toBeGreaterThan(0);
      
      for (const gpu of gpuInfo) {
        // GPU name should not be fake
        expect(gpu.name).toBeDefined();
        expect(gpu.name).not.toBe('');
        expect(gpu.name.toLowerCase()).not.toContain('fake');
        expect(gpu.name.toLowerCase()).not.toContain('mock');
        
        // Memory values should be realistic
        expect(gpu.memoryTotal).toBeGreaterThanOrEqual(0);
        expect(gpu.memoryUsed).toBeGreaterThanOrEqual(0);
        expect(gpu.memoryFree).toBeGreaterThanOrEqual(0);
        
        if (gpu.memoryTotal > 0) {
          expect(gpu.memoryUsed + gpu.memoryFree).toBeLessThanOrEqual(gpu.memoryTotal);
        }
        
        // Temperature should be realistic if present
        if (gpu.temperature !== undefined) {
          expect(gpu.temperature).toBeGreaterThan(0);
          expect(gpu.temperature).toBeLessThan(150); // Reasonable GPU temp range
        }
        
        // Utilization should be percentage if present
        if (gpu.utilizationGpu !== undefined) {
          expect(gpu.utilizationGpu).toBeGreaterThanOrEqual(0);
          expect(gpu.utilizationGpu).toBeLessThanOrEqual(100);
        }
        
        // CUDA support should be boolean
        expect(typeof gpu.cudaSupported).toBe('boolean');
      }
    }, 10000); // Allow 10 seconds for GPU detection
  });

  describe('Git Repository Health', () => {
    test('should detect real git repository', async () => {
      const gitHealth = await getGitRepositoryHealth(tempGitRepo);
      
      expect(gitHealth.isRepository).toBe(true);
      expect(gitHealth.currentBranch).toBeDefined();
      expect(gitHealth.commits).toBeGreaterThan(0);
      expect(gitHealth.unstagedChanges).toBeGreaterThanOrEqual(0);
      expect(gitHealth.stagedChanges).toBeGreaterThanOrEqual(0);
      
      // Should have a real commit hash
      expect(gitHealth.lastCommitHash).toBeDefined();
      expect(gitHealth.lastCommitHash!.length).toBeGreaterThan(7);
      expect(gitHealth.lastCommitHash).not.toBe('fake_hash');
      
      // Should have a real commit date
      expect(gitHealth.lastCommitDate).toBeDefined();
      expect(gitHealth.lastCommitDate).toBeInstanceOf(Date);
    });

    test('should handle non-git directory', async () => {
      const nonGitDir = join(tempDir, 'non-git');
      await fs.mkdir(nonGitDir, { recursive: true });
      
      const gitHealth = await getGitRepositoryHealth(nonGitDir);
      
      expect(gitHealth.isRepository).toBe(false);
      expect(gitHealth.commits).toBe(0);
      expect(gitHealth.unstagedChanges).toBe(0);
      expect(gitHealth.stagedChanges).toBe(0);
    });
  });

  describe('Network Diagnostics', () => {
    test('should perform real network diagnostics', async () => {
      const networkDiagnostics = await performNetworkDiagnostics();
      
      expect(networkDiagnostics).toBeDefined();
      expect(typeof networkDiagnostics.isConnected).toBe('boolean');
      expect(networkDiagnostics.dnsResolutionTime).toBeGreaterThanOrEqual(0);
      expect(networkDiagnostics.activeConnections).toBeGreaterThanOrEqual(0);
      expect(Array.isArray(networkDiagnostics.networkInterfaces)).toBe(true);
      
      // Should have at least one network interface (loopback)
      expect(networkDiagnostics.networkInterfaces.length).toBeGreaterThan(0);
      
      // Interface names should not be fake
      for (const interfaceName of networkDiagnostics.networkInterfaces) {
        expect(interfaceName).toBeDefined();
        expect(interfaceName).not.toBe('');
        expect(interfaceName.toLowerCase()).not.toContain('fake');
        expect(interfaceName.toLowerCase()).not.toContain('mock');
      }
      
      // If connected, should have reasonable ping latency
      if (networkDiagnostics.isConnected && networkDiagnostics.pingLatency !== undefined) {
        expect(networkDiagnostics.pingLatency).toBeGreaterThan(0);
        expect(networkDiagnostics.pingLatency).toBeLessThan(5000); // 5 seconds max
      }
      
      // Public IP should be valid format if present
      if (networkDiagnostics.publicIpAddress) {
        expect(networkDiagnostics.publicIpAddress).toMatch(/^\d+\.\d+\.\d+\.\d+$/);
      }
    }, 15000); // Allow 15 seconds for network tests
  });

  describe('System Resource Metrics', () => {
    test('should get real system resource metrics', async () => {
      const systemResources = await getSystemResourceMetrics();
      
      expect(systemResources).toBeDefined();
      
      // CPU metrics should be realistic
      expect(systemResources.cpuUsagePercent).toBeGreaterThanOrEqual(0);
      expect(systemResources.cpuUsagePercent).toBeLessThanOrEqual(100);
      expect(systemResources.cpuCoreCount).toBeGreaterThan(0);
      expect(systemResources.cpuCoreCount).toBeLessThan(1000); // Reasonable upper bound
      
      // Memory metrics should be realistic
      expect(systemResources.memoryTotalBytes).toBeGreaterThan(0);
      expect(systemResources.memoryUsedBytes).toBeGreaterThanOrEqual(0);
      expect(systemResources.memoryAvailableBytes).toBeGreaterThanOrEqual(0);
      expect(systemResources.memoryUsedBytes).toBeLessThanOrEqual(systemResources.memoryTotalBytes);
      
      // Disk metrics should be realistic
      expect(systemResources.diskTotalBytes).toBeGreaterThan(0);
      expect(systemResources.diskUsedBytes).toBeGreaterThanOrEqual(0);
      expect(systemResources.diskFreeBytes).toBeGreaterThanOrEqual(0);
      expect(systemResources.diskUsedBytes).toBeLessThanOrEqual(systemResources.diskTotalBytes);
      
      // Process metrics should be realistic
      expect(systemResources.processCount).toBeGreaterThan(0);
      expect(systemResources.processCount).toBeLessThan(100000); // Reasonable upper bound
      
      // Uptime should be positive
      expect(systemResources.uptime).toBeGreaterThan(0);
    });
  });

  describe('Service Health', () => {
    test('should check real service health', async () => {
      const serviceHealth = await checkServiceHealth();
      
      expect(serviceHealth).toBeDefined();
      expect(typeof serviceHealth.fileSystemPermissions).toBe('boolean');
      expect(typeof serviceHealth.databaseConnectivity).toBe('boolean');
      expect(typeof serviceHealth.webServiceAvailability).toBe('boolean');
      expect(typeof serviceHealth.environmentVariables).toBe('object');
      expect(Array.isArray(serviceHealth.portsListening)).toBe(true);
      expect(Array.isArray(serviceHealth.servicesRunning)).toBe(true);
      
      // Environment variables should contain real system variables
      expect(Object.keys(serviceHealth.environmentVariables).length).toBeGreaterThan(0);
      expect(serviceHealth.environmentVariables).toHaveProperty('PATH');
      
      // Should have some running services
      expect(serviceHealth.servicesRunning.length).toBeGreaterThan(0);
      
      // Service names should not be fake
      for (const serviceName of serviceHealth.servicesRunning.slice(0, 10)) {
        expect(serviceName).toBeDefined();
        expect(serviceName).not.toBe('');
        expect(serviceName.toLowerCase()).not.toContain('fake');
        expect(serviceName.toLowerCase()).not.toContain('mock');
      }
    });
  });

  describe('Comprehensive Diagnostics', () => {
    test('should get comprehensive real diagnostics', async () => {
      const diagnostics = await getComprehensiveDiagnostics(tempGitRepo);
      
      expect(diagnostics).toBeDefined();
      expect(diagnostics.timestamp).toBeInstanceOf(Date);
      
      // Timestamp should be recent
      const timeDiff = Date.now() - diagnostics.timestamp.getTime();
      expect(timeDiff).toBeLessThan(60000); // Within last minute
      
      // Should have all diagnostic components
      expect(Array.isArray(diagnostics.gpuInfo)).toBe(true);
      expect(diagnostics.gitHealth).toBeDefined();
      expect(diagnostics.networkDiagnostics).toBeDefined();
      expect(diagnostics.systemResources).toBeDefined();
      expect(diagnostics.serviceHealth).toBeDefined();
      
      // Overall health should be calculated
      expect(diagnostics.overallHealthScore).toBeGreaterThanOrEqual(0);
      expect(diagnostics.overallHealthScore).toBeLessThanOrEqual(100);
      
      // Should not be obvious fake values
      expect(diagnostics.overallHealthScore).not.toBe(95.5);
      expect(diagnostics.overallHealthScore).not.toBe(100.0);
    });

    test('should calculate consistent health scores', async () => {
      const diagnostics1 = await getComprehensiveDiagnostics(tempGitRepo);
      
      // Wait a small amount
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const diagnostics2 = await getComprehensiveDiagnostics(tempGitRepo);
      
      // Health scores should be similar (within reasonable variance)
      const healthDiff = Math.abs(diagnostics1.overallHealthScore - diagnostics2.overallHealthScore);
      expect(healthDiff).toBeLessThan(50); // Should not vary wildly
    });
  });

  describe('Health Score Calculation', () => {
    test('should calculate deterministic health scores', () => {
      const mockDiagnostics = {
        timestamp: new Date(),
        gpuInfo: [{
          name: 'Test GPU',
          memoryTotal: 8000000000,
          memoryUsed: 2000000000,
          memoryFree: 6000000000,
          cudaSupported: true,
        }],
        gitHealth: {
          isRepository: true,
          isClean: true,
          unstagedChanges: 0,
          stagedChanges: 0,
          commits: 10,
          aheadBy: 0,
          behindBy: 0,
        },
        networkDiagnostics: {
          isConnected: true,
          dnsResolutionTime: 50,
          pingLatency: 25,
          activeConnections: 5,
          networkInterfaces: ['eth0'],
        },
        systemResources: {
          cpuUsagePercent: 50,
          cpuCoreCount: 8,
          cpuFrequency: 3000,
          memoryTotalBytes: 16000000000,
          memoryUsedBytes: 8000000000,
          memoryAvailableBytes: 8000000000,
          diskTotalBytes: 1000000000000,
          diskUsedBytes: 500000000000,
          diskFreeBytes: 500000000000,
          processCount: 200,
          threadCount: 1000,
          uptime: 86400,
        },
        serviceHealth: {
          databaseConnectivity: false,
          webServiceAvailability: false,
          fileSystemPermissions: true,
          environmentVariables: {},
          portsListening: [80, 443],
          servicesRunning: ['nginx', 'node'],
        },
        overallHealthScore: 0,
      };
      
      const healthScore1 = calculateOverallHealthScore(mockDiagnostics);
      const healthScore2 = calculateOverallHealthScore(mockDiagnostics);
      
      // Same input should give same output
      expect(healthScore1).toBe(healthScore2);
      expect(healthScore1).toBeGreaterThan(0);
      expect(healthScore1).toBeLessThanOrEqual(100);
    });
  });

  describe('TARS Project Info', () => {
    test('should get real project information', async () => {
      // Create a test F# project
      const projectDir = join(tempDir, 'test-fsharp-project');
      await fs.mkdir(projectDir, { recursive: true });
      
      const fsprojContent = `<?xml version="1.0" encoding="utf-8"?>
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
  </PropertyGroup>
</Project>`;
      
      await fs.writeFile(join(projectDir, 'test.fsproj'), fsprojContent);
      
      const projectInfo = await getTarsProjectInfo(projectDir);
      
      expect(projectInfo).toBeDefined();
      expect(projectInfo.name).toBe('test-fsharp-project');
      expect(projectInfo.path).toBe(projectDir);
      expect(projectInfo.type).toBe('fsharp');
      expect(['success', 'failed', 'building', 'unknown']).toContain(projectInfo.buildStatus);
      expect(Array.isArray(projectInfo.dependencies)).toBe(true);
    });

    test('should handle TypeScript project', async () => {
      // Create a test TypeScript project
      const projectDir = join(tempDir, 'test-ts-project');
      await fs.mkdir(projectDir, { recursive: true });
      
      const packageJsonContent = {
        name: 'test-ts-project',
        version: '1.0.0',
        dependencies: {
          'typescript': '^5.0.0',
          'express': '^4.18.0'
        }
      };
      
      await fs.writeFile(join(projectDir, 'package.json'), JSON.stringify(packageJsonContent, null, 2));
      
      const projectInfo = await getTarsProjectInfo(projectDir);
      
      expect(projectInfo).toBeDefined();
      expect(projectInfo.name).toBe('test-ts-project');
      expect(projectInfo.type).toBe('typescript');
      expect(projectInfo.dependencies).toContain('typescript');
      expect(projectInfo.dependencies).toContain('express');
    });
  });

  describe('Performance Tests', () => {
    test('should complete diagnostics within reasonable time', async () => {
      const startTime = Date.now();
      
      await getComprehensiveDiagnostics(tempGitRepo);
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should complete within 30 seconds
      expect(duration).toBeLessThan(30000);
    }, 35000);

    test('should handle concurrent diagnostic requests', async () => {
      const promises = Array.from({ length: 3 }, () => 
        getComprehensiveDiagnostics(tempGitRepo)
      );
      
      const results = await Promise.all(promises);
      
      expect(results).toHaveLength(3);
      
      for (const result of results) {
        expect(result).toBeDefined();
        expect(result.overallHealthScore).toBeGreaterThanOrEqual(0);
        expect(result.overallHealthScore).toBeLessThanOrEqual(100);
      }
    });
  });
});
