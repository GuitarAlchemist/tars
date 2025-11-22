/**
 * Jest test setup for TARS MCP Server
 */

// Increase timeout for system diagnostic tests
jest.setTimeout(30000);

// Mock console methods to reduce noise during tests
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;
const originalConsoleError = console.error;

beforeAll(() => {
  // Only show errors during tests
  console.log = jest.fn();
  console.warn = jest.fn();
  console.error = originalConsoleError;
});

afterAll(() => {
  // Restore console methods
  console.log = originalConsoleLog;
  console.warn = originalConsoleWarn;
  console.error = originalConsoleError;
});

// Global test utilities
global.testUtils = {
  createTempDir: async () => {
    const { tmpdir } = await import('os');
    const { join } = await import('path');
    const { promises: fs } = await import('fs');
    
    const tempDir = join(tmpdir(), `tars-test-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);
    await fs.mkdir(tempDir, { recursive: true });
    return tempDir;
  },
  
  cleanupTempDir: async (dir: string) => {
    const { promises: fs } = await import('fs');
    try {
      await fs.rm(dir, { recursive: true, force: true });
    } catch (error) {
      console.warn(`Failed to cleanup temp directory ${dir}:`, error);
    }
  },
  
  waitFor: (ms: number) => new Promise(resolve => setTimeout(resolve, ms)),
  
  isCI: () => process.env.CI === 'true' || process.env.GITHUB_ACTIONS === 'true',
  
  skipIfCI: (reason: string) => {
    if (global.testUtils.isCI()) {
      console.log(`Skipping test in CI: ${reason}`);
      return true;
    }
    return false;
  }
};

// Declare global types
declare global {
  var testUtils: {
    createTempDir: () => Promise<string>;
    cleanupTempDir: (dir: string) => Promise<void>;
    waitFor: (ms: number) => Promise<void>;
    isCI: () => boolean;
    skipIfCI: (reason: string) => boolean;
  };
}
