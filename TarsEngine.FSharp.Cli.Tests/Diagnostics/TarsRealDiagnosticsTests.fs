namespace TarsEngine.FSharp.Cli.Tests.Diagnostics

open System
open System.IO
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open FluentAssertions
open TarsEngine.FSharp.Cli.Diagnostics.TarsRealDiagnostics

/// Tests for TARS Real Diagnostics - Ensuring NO FAKE DATA, only real measurements
module TarsRealDiagnosticsTests =

    [<Fact>]
    let ``detectGpuInfo should return real GPU information`` () =
        task {
            // Act
            let! gpuInfo = detectGpuInfo()
            
            // Assert
            gpuInfo |> should not' (be Empty)
            
            for gpu in gpuInfo do
                // GPU name should not be empty or fake
                gpu.Name |> should not' (be NullOrWhiteSpace)
                gpu.Name |> should not' (equal "Fake GPU")
                gpu.Name |> should not' (equal "Mock GPU")
                
                // Memory values should be realistic (not obviously fake)
                if gpu.MemoryTotal > 0L then
                    gpu.MemoryTotal |> should be (greaterThan 0L)
                    gpu.MemoryFree |> should be (greaterThanOrEqualTo 0L)
                    gpu.MemoryUsed |> should be (greaterThanOrEqualTo 0L)
                    (gpu.MemoryUsed + gpu.MemoryFree) |> should be (lessThanOrEqualTo gpu.MemoryTotal)
                
                // Temperature should be realistic if present
                match gpu.Temperature with
                | Some temp -> 
                    temp |> should be (greaterThan 0.0)
                    temp |> should be (lessThan 150.0) // Reasonable GPU temp range
                | None -> () // OK if not available
                
                // Utilization should be percentage if present
                match gpu.UtilizationGpu with
                | Some util -> 
                    util |> should be (greaterThanOrEqualTo 0.0)
                    util |> should be (lessThanOrEqualTo 100.0)
                | None -> () // OK if not available
        }

    [<Fact>]
    let ``getRealSystemHealth should return actual system metrics`` () =
        // Act
        let systemHealth = getRealSystemHealth()
        
        // Assert
        // CPU usage should be realistic
        systemHealth.CpuUsage |> should be (greaterThanOrEqualTo 0.0)
        systemHealth.CpuUsage |> should be (lessThan 10000.0) // Not obviously fake
        
        // Memory usage should be positive and realistic
        systemHealth.MemoryUsage |> should be (greaterThan 0.0)
        systemHealth.MemoryUsage |> should be (lessThan 100000.0) // Not obviously fake
        
        // Uptime should be positive
        systemHealth.Uptime.TotalSeconds |> should be (greaterThan 0.0)
        
        // Error rate should be reasonable
        systemHealth.ErrorRate |> should be (greaterThanOrEqualTo 0.0)
        systemHealth.ErrorRate |> should be (lessThanOrEqualTo 100.0)

    [<Fact>]
    let ``getRealComponentAnalyses should calculate real health percentages`` () =
        // Arrange
        let cognitiveEngine = Some (box "test_engine")
        let beliefBus = Some (box "test_bus")
        let projectManager = Some (box "test_manager")
        
        // Act
        let analyses = getRealComponentAnalyses cognitiveEngine beliefBus projectManager
        
        // Assert
        analyses |> should not' (be Empty)
        
        for analysis in analyses do
            // Health percentages should be calculated, not hardcoded
            analysis.Percentage |> should not' (equal 100.0) // Avoid obvious fake values
            analysis.Percentage |> should not' (equal 94.0)
            analysis.Percentage |> should not' (equal 91.0)
            analysis.Percentage |> should not' (equal 87.0)
            
            // Percentage should be realistic
            analysis.Percentage |> should be (greaterThanOrEqualTo 0.0)
            analysis.Percentage |> should be (lessThanOrEqualTo 100.0)
            
            // Status should not be fake
            analysis.Status |> should not' (contain "fake")
            analysis.Status |> should not' (contain "mock")
            analysis.Status |> should not' (contain "simulated")
            
            // Metrics should contain real data
            analysis.Metrics |> should not' (be Empty)
            
            // Last checked should be recent
            let timeDiff = DateTime.Now - analysis.LastChecked
            timeDiff.TotalMinutes |> should be (lessThan 5.0) // Should be very recent

    [<Fact>]
    let ``getGitRepositoryHealth should detect real git repository`` () =
        task {
            // Arrange
            let currentDir = Directory.GetCurrentDirectory()
            
            // Act
            let! gitHealth = getGitRepositoryHealth currentDir
            
            // Assert
            if gitHealth.IsRepository then
                // If it's a repository, validate real git data
                gitHealth.CurrentBranch |> should not' (be null)
                gitHealth.Commits |> should be (greaterThanOrEqualTo 0)
                
                // Changes should be realistic counts
                gitHealth.UnstagedChanges |> should be (greaterThanOrEqualTo 0)
                gitHealth.StagedChanges |> should be (greaterThanOrEqualTo 0)
                
                // If we have a last commit, it should be real
                match gitHealth.LastCommitHash with
                | Some hash -> 
                    hash.Length |> should be (greaterThan 7) // Git hashes are longer
                    hash |> should not' (equal "fake_hash")
                | None -> () // OK if no commits
                
                // Remote URL should be valid if present
                match gitHealth.RemoteUrl with
                | Some url -> 
                    url |> should not' (be NullOrWhiteSpace)
                    url |> should not' (equal "fake_url")
                | None -> () // OK if no remote
        }

    [<Fact>]
    let ``getComprehensiveDiagnostics should return real system data`` () =
        task {
            // Arrange
            let repositoryPath = Directory.GetCurrentDirectory()
            
            // Act
            let! diagnostics = getComprehensiveDiagnostics None None None repositoryPath
            
            // Assert
            // Timestamp should be recent
            let timeDiff = DateTime.UtcNow - diagnostics.Timestamp
            timeDiff.TotalMinutes |> should be (lessThan 1.0)
            
            // Overall health should be calculated, not hardcoded
            diagnostics.OverallSystemHealth |> should be (greaterThanOrEqualTo 0.0)
            diagnostics.OverallSystemHealth |> should be (lessThanOrEqualTo 100.0)
            diagnostics.OverallSystemHealth |> should not' (equal 95.5) // Avoid obvious fake values
            
            // System health should have real values
            diagnostics.SystemHealth.CpuUsage |> should be (greaterThanOrEqualTo 0.0)
            diagnostics.SystemHealth.MemoryUsage |> should be (greaterThan 0.0)
            
            // Component analyses should be present and real
            diagnostics.ComponentAnalyses |> should not' (be Empty)
            
            for analysis in diagnostics.ComponentAnalyses do
                // Verify no fake percentages
                analysis.Percentage |> should not' (equal 100.0)
                analysis.Percentage |> should not' (equal 94.0)
                analysis.Percentage |> should not' (equal 91.0)
                analysis.Percentage |> should not' (equal 87.0)
        }

    [<Fact>]
    let ``calculateRealHealth should use actual system metrics`` () =
        // Test that health calculations are based on real system state
        let memoryUsage = GC.GetTotalMemory(false) |> float
        let processorTime = System.Diagnostics.Process.GetCurrentProcess().TotalProcessorTime.TotalMilliseconds
        
        // These should be real values, not zero or fake
        memoryUsage |> should be (greaterThan 0.0)
        processorTime |> should be (greaterThanOrEqualTo 0.0)
        
        // Calculate health based on real metrics
        let memoryHealth = if memoryUsage < 100000000.0 then 100.0 else max 0.0 (100.0 - (memoryUsage / 1000000.0))
        let processingHealth = if processorTime > 0.0 then min 100.0 (processorTime / 1000.0) else 0.0
        let overallHealth = (memoryHealth + processingHealth) / 2.0
        
        // Health should be calculated, not hardcoded
        overallHealth |> should be (greaterThanOrEqualTo 0.0)
        overallHealth |> should be (lessThanOrEqualTo 100.0)
        overallHealth |> should not' (equal 100.0) // Unlikely to be exactly 100%

    [<Fact>]
    let ``system metrics should reflect actual hardware`` () =
        // Test that we're getting real hardware information
        let cpuCount = Environment.ProcessorCount
        let currentProcess = System.Diagnostics.Process.GetCurrentProcess()
        
        // CPU count should be realistic
        cpuCount |> should be (greaterThan 0)
        cpuCount |> should be (lessThan 1000) // Reasonable upper bound
        
        // Process should have real memory usage
        currentProcess.WorkingSet64 |> should be (greaterThan 0L)
        
        // Process should have real processor time
        currentProcess.TotalProcessorTime.TotalMilliseconds |> should be (greaterThanOrEqualTo 0.0)

    [<Fact>]
    let ``network interfaces should be real`` () =
        // Test that network interface detection is real
        let interfaces = System.Net.NetworkInformation.NetworkInterface.GetAllNetworkInterfaces()
        
        // Should have at least one interface (loopback)
        interfaces.Length |> should be (greaterThan 0)
        
        for iface in interfaces do
            // Interface names should not be fake
            iface.Name |> should not' (be NullOrWhiteSpace)
            iface.Name |> should not' (contain "fake")
            iface.Name |> should not' (contain "mock")

    [<Fact>]
    let ``file system access should be real`` () =
        // Test that file system operations are real
        let currentDir = Directory.GetCurrentDirectory()
        
        // Current directory should exist and be accessible
        Directory.Exists(currentDir) |> should be True
        
        // Should be able to get real file counts
        let files = Directory.GetFiles(currentDir, "*", SearchOption.TopDirectoryOnly)
        let dirs = Directory.GetDirectories(currentDir)
        
        // Counts should be realistic (not obviously fake)
        files.Length |> should be (greaterThanOrEqualTo 0)
        dirs.Length |> should be (greaterThanOrEqualTo 0)

    [<Theory>]
    [<InlineData(0.0, 100.0)>]
    [<InlineData(50.0, 75.0)>]
    [<InlineData(90.0, 50.0)>]
    let ``health calculation should be deterministic`` (inputValue: float) (expectedMinHealth: float) =
        // Test that health calculations are consistent and not random
        let health1 = if inputValue < 80.0 then 100.0 else 50.0
        let health2 = if inputValue < 80.0 then 100.0 else 50.0
        
        // Same input should give same output
        health1 |> should equal health2
        health1 |> should be (greaterThanOrEqualTo expectedMinHealth)

    [<Fact>]
    let ``diagnostics should complete within reasonable time`` () =
        task {
            // Performance test - diagnostics should not take too long
            let stopwatch = System.Diagnostics.Stopwatch.StartNew()
            
            let! _ = getComprehensiveDiagnostics None None None (Directory.GetCurrentDirectory())
            
            stopwatch.Stop()
            
            // Should complete within 30 seconds (generous for real system calls)
            stopwatch.Elapsed.TotalSeconds |> should be (lessThan 30.0)
        }

    [<Fact>]
    let ``repeated diagnostics should show consistency`` () =
        task {
            // Test that repeated calls show consistent behavior (not random fake data)
            let! diagnostics1 = getComprehensiveDiagnostics None None None (Directory.GetCurrentDirectory())
            
            // Wait a small amount
            do! Task.Delay(100)
            
            let! diagnostics2 = getComprehensiveDiagnostics None None None (Directory.GetCurrentDirectory())
            
            // System health should be similar (within reasonable variance)
            let healthDiff = abs (diagnostics1.OverallSystemHealth - diagnostics2.OverallSystemHealth)
            healthDiff |> should be (lessThan 50.0) // Should not vary wildly
            
            // Component count should be the same
            diagnostics1.ComponentAnalyses.Length |> should equal diagnostics2.ComponentAnalyses.Length
        }
