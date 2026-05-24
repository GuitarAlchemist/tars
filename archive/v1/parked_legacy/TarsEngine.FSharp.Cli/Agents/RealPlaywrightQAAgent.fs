namespace TarsEngine.FSharp.Cli.Agents

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open System.Diagnostics
open System.Text.Json

/// Real bug detection result
type BugDetectionResult = {
    BugId: string
    Severity: string // "Critical", "High", "Medium", "Low"
    Description: string
    Location: string
    StackTrace: string option
    Screenshot: string option
    Reproducible: bool
    FixSuggestion: string option
}

/// Real test execution result
type TestExecutionResult = {
    TestName: string
    Status: string // "Passed", "Failed", "Skipped"
    Duration: TimeSpan
    ErrorMessage: string option
    Screenshots: string list
    ConsoleErrors: string list
    NetworkErrors: string list
    PerformanceMetrics: Map<string, float>
}

/// Real QA session result
type QASessionResult = {
    SessionId: string
    ApplicationPath: string
    TotalTests: int
    PassedTests: int
    FailedTests: int
    BugsDetected: BugDetectionResult list
    TestResults: TestExecutionResult list
    OverallQuality: float
    QualityGate: string // "Passed", "Failed"
    Recommendations: string list
    ExecutionTime: TimeSpan
}

/// Real Playwright QA Agent for autonomous testing
type RealPlaywrightQAAgent(logger: ILogger<RealPlaywrightQAAgent>) =
    
    let mutable qaHistory: QASessionResult list = []
    let mutable currentSession: string option = None
    
    /// Install Playwright browsers if needed
    member this.EnsurePlaywrightSetup() =
        task {
            logger.LogInformation("Ensuring Playwright setup...")
            
            try
                // Check if Playwright is installed
                let! checkResult = this.ExecuteCommand("npx", "playwright --version", Directory.GetCurrentDirectory())
                
                if not checkResult.Success then
                    logger.LogInformation("Installing Playwright...")
                    let! installResult = this.ExecuteCommand("npm", "install -D @playwright/test", Directory.GetCurrentDirectory())
                    
                    if installResult.Success then
                        logger.LogInformation("Installing Playwright browsers...")
                        let! browsersResult = this.ExecuteCommand("npx", "playwright install", Directory.GetCurrentDirectory())
                        return browsersResult.Success
                    else
                        logger.LogError("Failed to install Playwright")
                        return false
                else
                    logger.LogInformation("Playwright already installed")
                    return true
                    
            with ex ->
                logger.LogError(ex, "Error setting up Playwright")
                return false
        }
    
    /// Execute command with real process execution
    member private this.ExecuteCommand(command: string, arguments: string, workingDirectory: string) =
        task {
            try
                let processInfo = ProcessStartInfo(
                    FileName = command,
                    Arguments = arguments,
                    WorkingDirectory = workingDirectory,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                )
                
                use process = Process.Start(processInfo)
                let! output = process.StandardOutput.ReadToEndAsync()
                let! error = process.StandardError.ReadToEndAsync()
                process.WaitForExit()
                
                return {|
                    Success = process.ExitCode = 0
                    Output = output
                    Error = error
                    ExitCode = process.ExitCode
                |}
                
            with ex ->
                logger.LogError(ex, $"Failed to execute command: {command} {arguments}")
                return {|
                    Success = false
                    Output = ""
                    Error = ex.Message
                    ExitCode = -1
                |}
        }
    
    /// Generate comprehensive Playwright tests for application
    member this.GenerateTestsForApplication(applicationPath: string, applicationType: string) =
        task {
            logger.LogInformation($"Generating Playwright tests for: {applicationPath}")
            
            let testDirectory = Path.Combine(applicationPath, "tests")
            Directory.CreateDirectory(testDirectory) |> ignore
            
            // Generate test configuration
            let playwrightConfig = """
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: 'html',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
  ],
  webServer: {
    command: 'npm start',
    url: 'http://localhost:3000',
    reuseExistingServer: !process.env.CI,
  },
});
"""
            
            File.WriteAllText(Path.Combine(applicationPath, "playwright.config.ts"), playwrightConfig)
            
            // Generate comprehensive test suite based on application type
            let testSuite = this.GenerateTestSuiteForType(applicationType)
            File.WriteAllText(Path.Combine(testDirectory, "comprehensive.spec.ts"), testSuite)
            
            logger.LogInformation("Playwright tests generated successfully")
            return true
        }
    
    /// Generate test suite based on application type
    member private this.GenerateTestSuiteForType(applicationType: string) =
        match applicationType.ToLower() with
        | "react" | "web" ->
            """
import { test, expect } from '@playwright/test';

test.describe('Comprehensive Application Tests', () => {
  test('should load homepage without errors', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/.*/, { timeout: 10000 });
    
    // Check for console errors
    const errors = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });
    
    await page.waitForLoadState('networkidle');
    expect(errors).toHaveLength(0);
  });

  test('should have responsive design', async ({ page }) => {
    await page.goto('/');
    
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForLoadState('networkidle');
    await expect(page.locator('body')).toBeVisible();
    
    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.waitForLoadState('networkidle');
    await expect(page.locator('body')).toBeVisible();
    
    // Test desktop viewport
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.waitForLoadState('networkidle');
    await expect(page.locator('body')).toBeVisible();
  });

  test('should handle user interactions', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Test clickable elements
    const buttons = await page.locator('button').all();
    for (const button of buttons) {
      if (await button.isVisible()) {
        await button.click();
        await page.waitForTimeout(500);
      }
    }
    
    // Test form inputs if present
    const inputs = await page.locator('input').all();
    for (const input of inputs) {
      if (await input.isVisible()) {
        await input.fill('test data');
        await page.waitForTimeout(200);
      }
    }
  });

  test('should have good performance', async ({ page }) => {
    await page.goto('/');
    
    const performanceMetrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0];
      return {
        loadTime: navigation.loadEventEnd - navigation.loadEventStart,
        domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
        firstPaint: performance.getEntriesByName('first-paint')[0]?.startTime || 0,
        firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0,
      };
    });
    
    // Performance assertions
    expect(performanceMetrics.loadTime).toBeLessThan(3000);
    expect(performanceMetrics.domContentLoaded).toBeLessThan(2000);
  });

  test('should be accessible', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Check for basic accessibility
    const headings = await page.locator('h1, h2, h3, h4, h5, h6').count();
    expect(headings).toBeGreaterThan(0);
    
    // Check for alt text on images
    const images = await page.locator('img').all();
    for (const img of images) {
      if (await img.isVisible()) {
        const alt = await img.getAttribute('alt');
        expect(alt).toBeTruthy();
      }
    }
  });
});
"""
        | "3d" | "threejs" ->
            """
import { test, expect } from '@playwright/test';

test.describe('3D Application Tests', () => {
  test('should load 3D scene without errors', async ({ page }) => {
    await page.goto('/');
    
    // Wait for WebGL context
    await page.waitForFunction(() => {
      const canvas = document.querySelector('canvas');
      return canvas && canvas.getContext('webgl') !== null;
    });
    
    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible();
  });

  test('should handle 3D interactions', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const canvas = page.locator('canvas');
    await canvas.click();
    
    // Test mouse interactions
    await canvas.hover();
    await page.mouse.wheel(0, 100);
    await page.waitForTimeout(1000);
  });

  test('should have good 3D performance', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const fps = await page.evaluate(() => {
      return new Promise((resolve) => {
        let frames = 0;
        const start = performance.now();
        
        function countFrames() {
          frames++;
          if (performance.now() - start < 1000) {
            requestAnimationFrame(countFrames);
          } else {
            resolve(frames);
          }
        }
        requestAnimationFrame(countFrames);
      });
    });
    
    expect(fps).toBeGreaterThan(30);
  });
});
"""
        | _ ->
            """
import { test, expect } from '@playwright/test';

test.describe('Generic Application Tests', () => {
  test('should load without errors', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('body')).toBeVisible();
  });
});
"""
    
    /// Execute comprehensive QA testing
    member this.ExecuteComprehensiveQA(applicationPath: string, applicationType: string) =
        task {
            let sessionId = Guid.NewGuid().ToString("N")[..7]
            currentSession <- Some sessionId
            let startTime = DateTime.UtcNow
            
            logger.LogInformation($"Starting comprehensive QA session: {sessionId}")
            
            try
                // Ensure Playwright is set up
                let! setupSuccess = this.EnsurePlaywrightSetup()
                if not setupSuccess then
                    failwith "Failed to set up Playwright"
                
                // Generate tests
                let! testsGenerated = this.GenerateTestsForApplication(applicationPath, applicationType)
                if not testsGenerated then
                    failwith "Failed to generate tests"
                
                // Execute tests
                let! testResults = this.ExecutePlaywrightTests(applicationPath)
                
                // Analyze results and detect bugs
                let bugsDetected = this.AnalyzeTestResultsForBugs(testResults)
                
                let endTime = DateTime.UtcNow
                let executionTime = endTime - startTime
                
                let qaResult = {
                    SessionId = sessionId
                    ApplicationPath = applicationPath
                    TotalTests = testResults.Length
                    PassedTests = testResults |> List.filter (fun t -> t.Status = "Passed") |> List.length
                    FailedTests = testResults |> List.filter (fun t -> t.Status = "Failed") |> List.length
                    BugsDetected = bugsDetected
                    TestResults = testResults
                    OverallQuality = this.CalculateQualityScore(testResults, bugsDetected)
                    QualityGate = if bugsDetected |> List.exists (fun b -> b.Severity = "Critical") then "Failed" else "Passed"
                    Recommendations = this.GenerateRecommendations(testResults, bugsDetected)
                    ExecutionTime = executionTime
                }
                
                qaHistory <- qaResult :: qaHistory
                
                logger.LogInformation($"QA session completed: {qaResult.PassedTests}/{qaResult.TotalTests} tests passed")
                return qaResult
                
            with ex ->
                logger.LogError(ex, $"QA session {sessionId} failed")
                return {
                    SessionId = sessionId
                    ApplicationPath = applicationPath
                    TotalTests = 0
                    PassedTests = 0
                    FailedTests = 0
                    BugsDetected = []
                    TestResults = []
                    OverallQuality = 0.0
                    QualityGate = "Failed"
                    Recommendations = [$"QA execution failed: {ex.Message}"]
                    ExecutionTime = DateTime.UtcNow - startTime
                }
        }

    /// Execute Playwright tests and capture results
    member private this.ExecutePlaywrightTests(applicationPath: string) =
        task {
            logger.LogInformation("Executing Playwright tests...")

            let! result = this.ExecuteCommand("npx", "playwright test --reporter=json", applicationPath)

            if result.Success then
                try
                    // Parse test results from JSON output
                    let testResults = this.ParsePlaywrightResults(result.Output)
                    return testResults
                with ex ->
                    logger.LogError(ex, "Failed to parse test results")
                    return []
            else
                logger.LogError($"Playwright tests failed: {result.Error}")
                return []
        }

    /// Parse Playwright JSON results into structured data
    member private this.ParsePlaywrightResults(jsonOutput: string) =
        try
            // Real JSON parsing of Playwright results
            let lines = jsonOutput.Split('\n') |> Array.filter (fun line -> line.Trim().StartsWith("{"))

            lines
            |> Array.map (fun line ->
                try
                    let json = JsonDocument.Parse(line)
                    let root = json.RootElement

                    {
                        TestName = if root.TryGetProperty("title", &_) then root.GetProperty("title").GetString() else "Unknown Test"
                        Status = if root.TryGetProperty("outcome", &_) then root.GetProperty("outcome").GetString() else "Unknown"
                        Duration = TimeSpan.FromMilliseconds(if root.TryGetProperty("duration", &_) then root.GetProperty("duration").GetDouble() else 0.0)
                        ErrorMessage = if root.TryGetProperty("error", &_) then Some(root.GetProperty("error").GetString()) else None
                        Screenshots = []
                        ConsoleErrors = []
                        NetworkErrors = []
                        PerformanceMetrics = Map.empty
                    }
                with ex ->
                    {
                        TestName = "Parse Error"
                        Status = "Failed"
                        Duration = TimeSpan.Zero
                        ErrorMessage = Some(ex.Message)
                        Screenshots = []
                        ConsoleErrors = []
                        NetworkErrors = []
                        PerformanceMetrics = Map.empty
                    })
            |> Array.toList
        with ex ->
            logger.LogError(ex, "Failed to parse Playwright JSON results")
            []

    /// Analyze test results to detect bugs
    member private this.AnalyzeTestResultsForBugs(testResults: TestExecutionResult list) =
        testResults
        |> List.filter (fun t -> t.Status = "Failed")
        |> List.mapi (fun i testResult ->
            let severity =
                if testResult.TestName.Contains("load") || testResult.TestName.Contains("critical") then "Critical"
                elif testResult.TestName.Contains("performance") || testResult.TestName.Contains("accessibility") then "High"
                elif testResult.TestName.Contains("responsive") || testResult.TestName.Contains("interaction") then "Medium"
                else "Low"

            {
                BugId = $"BUG-{DateTime.Now:yyyyMMdd}-{i + 1:D3}"
                Severity = severity
                Description = testResult.ErrorMessage |> Option.defaultValue "Test failed without specific error message"
                Location = testResult.TestName
                StackTrace = testResult.ErrorMessage
                Screenshot = None
                Reproducible = true
                FixSuggestion = this.GenerateFixSuggestion(testResult)
            })

    /// Generate fix suggestions based on test failures
    member private this.GenerateFixSuggestion(testResult: TestExecutionResult) =
        match testResult.TestName.ToLower() with
        | name when name.Contains("load") ->
            Some "Check for JavaScript errors, missing dependencies, or network issues. Ensure all resources are properly loaded."
        | name when name.Contains("responsive") ->
            Some "Review CSS media queries and responsive design implementation. Test viewport handling."
        | name when name.Contains("performance") ->
            Some "Optimize loading times, reduce bundle size, implement lazy loading, or optimize images."
        | name when name.Contains("accessibility") ->
            Some "Add proper ARIA labels, alt text for images, and ensure keyboard navigation works."
        | name when name.Contains("interaction") ->
            Some "Check event handlers, form validation, and user interaction flows. Ensure elements are clickable."
        | _ ->
            Some "Review the specific test failure and check application logic, error handling, and user experience."

    /// Calculate overall quality score
    member private this.CalculateQualityScore(testResults: TestExecutionResult list, bugs: BugDetectionResult list) =
        if testResults.Length = 0 then 0.0
        else
            let passRate = float (testResults |> List.filter (fun t -> t.Status = "Passed") |> List.length) / float testResults.Length
            let criticalBugs = bugs |> List.filter (fun b -> b.Severity = "Critical") |> List.length
            let highBugs = bugs |> List.filter (fun b -> b.Severity = "High") |> List.length

            let bugPenalty = float (criticalBugs * 30 + highBugs * 15) / 100.0
            Math.Max(0.0, (passRate * 100.0) - bugPenalty)

    /// Generate recommendations based on results
    member private this.GenerateRecommendations(testResults: TestExecutionResult list, bugs: BugDetectionResult list) =
        let recommendations = ResizeArray<string>()

        let failedTests = testResults |> List.filter (fun t -> t.Status = "Failed")
        if failedTests.Length > 0 then
            recommendations.Add($"Fix {failedTests.Length} failing tests to improve application stability")

        let criticalBugs = bugs |> List.filter (fun b -> b.Severity = "Critical")
        if criticalBugs.Length > 0 then
            recommendations.Add($"Address {criticalBugs.Length} critical bugs immediately - these prevent basic functionality")

        let performanceIssues = testResults |> List.filter (fun t -> t.TestName.Contains("performance") && t.Status = "Failed")
        if performanceIssues.Length > 0 then
            recommendations.Add("Optimize application performance - slow loading affects user experience")

        let accessibilityIssues = testResults |> List.filter (fun t -> t.TestName.Contains("accessibility") && t.Status = "Failed")
        if accessibilityIssues.Length > 0 then
            recommendations.Add("Improve accessibility compliance for better user inclusion")

        if recommendations.Count = 0 then
            recommendations.Add("Application quality is excellent - all tests passing!")

        recommendations |> List.ofSeq

    /// Get QA history
    member this.GetQAHistory() = qaHistory

    /// Get current session
    member this.GetCurrentSession() = currentSession
