namespace TarsEngine.FSharp.WindowsService.Monitoring

open System
open System.Collections.Concurrent
open System.Diagnostics
open System.IO
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.Core.ServiceConfiguration

/// <summary>
/// Diagnostic data types
/// </summary>
type DiagnosticDataType =
    | SystemInfo
    | ProcessInfo
    | MemoryDump
    | ThreadDump
    | EventLogs
    | PerformanceCounters
    | NetworkInfo
    | DiskInfo
    | EnvironmentInfo

/// <summary>
/// Diagnostic data entry
/// </summary>
type DiagnosticData = {
    Id: string
    Type: DiagnosticDataType
    Timestamp: DateTime
    Source: string
    Data: Map<string, obj>
    FilePath: string option
    Size: int64
    Metadata: Map<string, string>
}

/// <summary>
/// System diagnostic snapshot
/// </summary>
type SystemDiagnosticSnapshot = {
    Timestamp: DateTime
    SystemInfo: SystemInfo
    ProcessInfo: ProcessInfo
    MemoryInfo: MemoryInfo
    DiskInfo: DiskInfo
    NetworkInfo: NetworkInfo
    EnvironmentInfo: EnvironmentInfo
    EventLogSummary: EventLogSummary
}

/// <summary>
/// System information
/// </summary>
and SystemInfo = {
    MachineName: string
    OSVersion: string
    ProcessorCount: int
    TotalPhysicalMemory: int64
    AvailablePhysicalMemory: int64
    SystemUptime: TimeSpan
    DotNetVersion: string
}

/// <summary>
/// Process information
/// </summary>
and ProcessInfo = {
    ProcessId: int
    ProcessName: string
    StartTime: DateTime
    TotalProcessorTime: TimeSpan
    WorkingSet: int64
    PrivateMemorySize: int64
    VirtualMemorySize: int64
    ThreadCount: int
    HandleCount: int
    ModuleCount: int
}

/// <summary>
/// Memory information
/// </summary>
and MemoryInfo = {
    TotalMemory: int64
    AvailableMemory: int64
    UsedMemory: int64
    MemoryUsagePercent: float
    GCTotalMemory: int64
    GCCollections: Map<int, int>
    LargeObjectHeapSize: int64
}

/// <summary>
/// Disk information
/// </summary>
and DiskInfo = {
    Drives: DriveInfo list
    TotalSpace: int64
    FreeSpace: int64
    UsedSpace: int64
    UsagePercent: float
}

/// <summary>
/// Drive information
/// </summary>
and DriveInfo = {
    Name: string
    DriveType: string
    TotalSize: int64
    FreeSpace: int64
    UsedSpace: int64
    UsagePercent: float
    FileSystem: string
}

/// <summary>
/// Network information
/// </summary>
and NetworkInfo = {
    NetworkInterfaces: NetworkInterfaceInfo list
    ActiveConnections: int
    TotalBytesReceived: int64
    TotalBytesSent: int64
}

/// <summary>
/// Network interface information
/// </summary>
and NetworkInterfaceInfo = {
    Name: string
    Description: string
    Status: string
    Speed: int64
    BytesReceived: int64
    BytesSent: int64
}

/// <summary>
/// Environment information
/// </summary>
and EnvironmentInfo = {
    EnvironmentVariables: Map<string, string>
    CommandLineArgs: string list
    CurrentDirectory: string
    UserName: string
    UserDomainName: string
    Is64BitOperatingSystem: bool
    Is64BitProcess: bool
}

/// <summary>
/// Event log summary
/// </summary>
and EventLogSummary = {
    ErrorCount: int
    WarningCount: int
    InfoCount: int
    RecentErrors: EventLogEntry list
    RecentWarnings: EventLogEntry list
}

/// <summary>
/// Event log entry
/// </summary>
and EventLogEntry = {
    Timestamp: DateTime
    Level: string
    Source: string
    Message: string
    EventId: int
}

/// <summary>
/// Comprehensive diagnostics collection and analysis system
/// </summary>
type DiagnosticsCollector(logger: ILogger<DiagnosticsCollector>) =
    
    let diagnosticData = ConcurrentDictionary<string, DiagnosticData>()
    let systemSnapshots = ConcurrentQueue<SystemDiagnosticSnapshot>()
    let diagnosticFiles = ConcurrentDictionary<string, string>()
    
    let mutable isRunning = false
    let mutable cancellationTokenSource: CancellationTokenSource option = None
    let mutable collectionTask: Task option = None
    let mutable config: MonitoringConfig option = None
    
    let maxSnapshotHistory = 1440 // 24 hours of minute-by-minute data
    let maxDiagnosticData = 10000
    let diagnosticsDirectory = "diagnostics"
    
    /// Start the diagnostics collector
    member this.StartAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Starting diagnostics collector...")
            
            cancellationTokenSource <- Some (CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
            isRunning <- true
            
            // Ensure diagnostics directory exists
            this.EnsureDiagnosticsDirectory()
            
            // Start collection loop
            let collectionLoop = this.DiagnosticsCollectionLoopAsync(cancellationTokenSource.Value.Token)
            collectionTask <- Some collectionLoop
            
            logger.LogInformation("Diagnostics collector started successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to start diagnostics collector")
            isRunning <- false
            raise
    }
    
    /// Stop the diagnostics collector
    member this.StopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogInformation("Stopping diagnostics collector...")
            
            isRunning <- false
            
            // Cancel all operations
            match cancellationTokenSource with
            | Some cts -> cts.Cancel()
            | None -> ()
            
            // Wait for collection task to complete
            match collectionTask with
            | Some task ->
                try
                    do! task.WaitAsync(TimeSpan.FromSeconds(10.0), cancellationToken)
                with
                | :? TimeoutException ->
                    logger.LogWarning("Diagnostics collection task did not complete within timeout")
                | ex ->
                    logger.LogWarning(ex, "Error waiting for diagnostics collection task to complete")
            | None -> ()
            
            // Cleanup
            match cancellationTokenSource with
            | Some cts -> 
                cts.Dispose()
                cancellationTokenSource <- None
            | None -> ()
            
            collectionTask <- None
            
            logger.LogInformation("Diagnostics collector stopped successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Error stopping diagnostics collector")
    }
    
    /// Ensure diagnostics directory exists
    member private this.EnsureDiagnosticsDirectory() =
        try
            if not (Directory.Exists(diagnosticsDirectory)) then
                Directory.CreateDirectory(diagnosticsDirectory) |> ignore
                logger.LogDebug($"Created diagnostics directory: {diagnosticsDirectory}")
        with
        | ex ->
            logger.LogWarning(ex, $"Failed to create diagnostics directory: {diagnosticsDirectory}")
    
    /// Main diagnostics collection loop
    member private this.DiagnosticsCollectionLoopAsync(cancellationToken: CancellationToken) = task {
        try
            logger.LogDebug("Starting diagnostics collection loop")
            
            while not cancellationToken.IsCancellationRequested && isRunning do
                try
                    // Collect system diagnostic snapshot
                    let! snapshot = this.CollectSystemDiagnosticSnapshotAsync()
                    systemSnapshots.Enqueue(snapshot)
                    
                    // Keep snapshot history manageable
                    while systemSnapshots.Count > maxSnapshotHistory do
                        systemSnapshots.TryDequeue() |> ignore
                    
                    // Collect detailed diagnostics periodically
                    if DateTime.UtcNow.Minute % 15 = 0 then // Every 15 minutes
                        do! this.CollectDetailedDiagnosticsAsync()
                    
                    // Wait for next collection cycle
                    do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                    
                with
                | :? OperationCanceledException ->
                    break
                | ex ->
                    logger.LogWarning(ex, "Error in diagnostics collection loop")
                    do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)
                    
        with
        | :? OperationCanceledException ->
            logger.LogDebug("Diagnostics collection loop cancelled")
        | ex ->
            logger.LogError(ex, "Diagnostics collection loop failed")
    }
    
    /// Collect system diagnostic snapshot
    member private this.CollectSystemDiagnosticSnapshotAsync() = task {
        try
            let timestamp = DateTime.UtcNow
            
            // Collect system information
            let systemInfo = this.CollectSystemInfo()
            
            // Collect process information
            let processInfo = this.CollectProcessInfo()
            
            // Collect memory information
            let memoryInfo = this.CollectMemoryInfo()
            
            // Collect disk information
            let diskInfo = this.CollectDiskInfo()
            
            // Collect network information
            let networkInfo = this.CollectNetworkInfo()
            
            // Collect environment information
            let environmentInfo = this.CollectEnvironmentInfo()
            
            // Collect event log summary
            let eventLogSummary = this.CollectEventLogSummary()
            
            return {
                Timestamp = timestamp
                SystemInfo = systemInfo
                ProcessInfo = processInfo
                MemoryInfo = memoryInfo
                DiskInfo = diskInfo
                NetworkInfo = networkInfo
                EnvironmentInfo = environmentInfo
                EventLogSummary = eventLogSummary
            }
            
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting system diagnostic snapshot")
            return {
                Timestamp = DateTime.UtcNow
                SystemInfo = { MachineName = ""; OSVersion = ""; ProcessorCount = 0; TotalPhysicalMemory = 0L; AvailablePhysicalMemory = 0L; SystemUptime = TimeSpan.Zero; DotNetVersion = "" }
                ProcessInfo = { ProcessId = 0; ProcessName = ""; StartTime = DateTime.MinValue; TotalProcessorTime = TimeSpan.Zero; WorkingSet = 0L; PrivateMemorySize = 0L; VirtualMemorySize = 0L; ThreadCount = 0; HandleCount = 0; ModuleCount = 0 }
                MemoryInfo = { TotalMemory = 0L; AvailableMemory = 0L; UsedMemory = 0L; MemoryUsagePercent = 0.0; GCTotalMemory = 0L; GCCollections = Map.empty; LargeObjectHeapSize = 0L }
                DiskInfo = { Drives = []; TotalSpace = 0L; FreeSpace = 0L; UsedSpace = 0L; UsagePercent = 0.0 }
                NetworkInfo = { NetworkInterfaces = []; ActiveConnections = 0; TotalBytesReceived = 0L; TotalBytesSent = 0L }
                EnvironmentInfo = { EnvironmentVariables = Map.empty; CommandLineArgs = []; CurrentDirectory = ""; UserName = ""; UserDomainName = ""; Is64BitOperatingSystem = false; Is64BitProcess = false }
                EventLogSummary = { ErrorCount = 0; WarningCount = 0; InfoCount = 0; RecentErrors = []; RecentWarnings = [] }
            }
    }
    
    /// Collect system information
    member private this.CollectSystemInfo() =
        try
            {
                MachineName = Environment.MachineName
                OSVersion = Environment.OSVersion.ToString()
                ProcessorCount = Environment.ProcessorCount
                TotalPhysicalMemory = 0L // Would use WMI or P/Invoke in real implementation
                AvailablePhysicalMemory = 0L // Would use WMI or P/Invoke in real implementation
                SystemUptime = TimeSpan.FromMilliseconds(float Environment.TickCount)
                DotNetVersion = Environment.Version.ToString()
            }
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting system information")
            { MachineName = ""; OSVersion = ""; ProcessorCount = 0; TotalPhysicalMemory = 0L; AvailablePhysicalMemory = 0L; SystemUptime = TimeSpan.Zero; DotNetVersion = "" }
    
    /// Collect process information
    member private this.CollectProcessInfo() =
        try
            let currentProcess = Process.GetCurrentProcess()
            {
                ProcessId = currentProcess.Id
                ProcessName = currentProcess.ProcessName
                StartTime = currentProcess.StartTime
                TotalProcessorTime = currentProcess.TotalProcessorTime
                WorkingSet = currentProcess.WorkingSet64
                PrivateMemorySize = currentProcess.PrivateMemorySize64
                VirtualMemorySize = currentProcess.VirtualMemorySize64
                ThreadCount = currentProcess.Threads.Count
                HandleCount = currentProcess.HandleCount
                ModuleCount = currentProcess.Modules.Count
            }
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting process information")
            { ProcessId = 0; ProcessName = ""; StartTime = DateTime.MinValue; TotalProcessorTime = TimeSpan.Zero; WorkingSet = 0L; PrivateMemorySize = 0L; VirtualMemorySize = 0L; ThreadCount = 0; HandleCount = 0; ModuleCount = 0 }
    
    /// Collect memory information
    member private this.CollectMemoryInfo() =
        try
            let gcMemory = GC.GetTotalMemory(false)
            let gcCollections = 
                [0..2] 
                |> List.map (fun gen -> (gen, GC.CollectionCount(gen)))
                |> Map.ofList
            
            {
                TotalMemory = 0L // Would use WMI in real implementation
                AvailableMemory = 0L // Would use WMI in real implementation
                UsedMemory = gcMemory
                MemoryUsagePercent = 0.0 // Would calculate from total/available
                GCTotalMemory = gcMemory
                GCCollections = gcCollections
                LargeObjectHeapSize = 0L // Would use GC.GetTotalMemory with LOH info
            }
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting memory information")
            { TotalMemory = 0L; AvailableMemory = 0L; UsedMemory = 0L; MemoryUsagePercent = 0.0; GCTotalMemory = 0L; GCCollections = Map.empty; LargeObjectHeapSize = 0L }
    
    /// Collect disk information
    member private this.CollectDiskInfo() =
        try
            let drives = 
                DriveInfo.GetDrives()
                |> Array.filter (fun d -> d.IsReady)
                |> Array.map (fun d -> 
                    let usedSpace = d.TotalSize - d.AvailableFreeSpace
                    let usagePercent = if d.TotalSize > 0L then (float usedSpace / float d.TotalSize) * 100.0 else 0.0
                    {
                        Name = d.Name
                        DriveType = d.DriveType.ToString()
                        TotalSize = d.TotalSize
                        FreeSpace = d.AvailableFreeSpace
                        UsedSpace = usedSpace
                        UsagePercent = usagePercent
                        FileSystem = d.DriveFormat
                    })
                |> List.ofArray
            
            let totalSpace = drives |> List.sumBy (fun d -> d.TotalSize)
            let freeSpace = drives |> List.sumBy (fun d -> d.FreeSpace)
            let usedSpace = totalSpace - freeSpace
            let usagePercent = if totalSpace > 0L then (float usedSpace / float totalSpace) * 100.0 else 0.0
            
            {
                Drives = drives
                TotalSpace = totalSpace
                FreeSpace = freeSpace
                UsedSpace = usedSpace
                UsagePercent = usagePercent
            }
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting disk information")
            { Drives = []; TotalSpace = 0L; FreeSpace = 0L; UsedSpace = 0L; UsagePercent = 0.0 }
    
    /// Collect network information
    member private this.CollectNetworkInfo() =
        try
            // In a real implementation, we'd use NetworkInterface.GetAllNetworkInterfaces()
            {
                NetworkInterfaces = []
                ActiveConnections = 0
                TotalBytesReceived = 0L
                TotalBytesSent = 0L
            }
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting network information")
            { NetworkInterfaces = []; ActiveConnections = 0; TotalBytesReceived = 0L; TotalBytesSent = 0L }
    
    /// Collect environment information
    member private this.CollectEnvironmentInfo() =
        try
            let envVars = 
                Environment.GetEnvironmentVariables()
                |> Seq.cast<System.Collections.DictionaryEntry>
                |> Seq.map (fun entry -> (entry.Key.ToString(), entry.Value.ToString()))
                |> Map.ofSeq
            
            {
                EnvironmentVariables = envVars
                CommandLineArgs = Environment.GetCommandLineArgs() |> List.ofArray
                CurrentDirectory = Environment.CurrentDirectory
                UserName = Environment.UserName
                UserDomainName = Environment.UserDomainName
                Is64BitOperatingSystem = Environment.Is64BitOperatingSystem
                Is64BitProcess = Environment.Is64BitProcess
            }
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting environment information")
            { EnvironmentVariables = Map.empty; CommandLineArgs = []; CurrentDirectory = ""; UserName = ""; UserDomainName = ""; Is64BitOperatingSystem = false; Is64BitProcess = false }
    
    /// Collect event log summary
    member private this.CollectEventLogSummary() =
        try
            // In a real implementation, we'd read from Windows Event Log
            {
                ErrorCount = 0
                WarningCount = 0
                InfoCount = 0
                RecentErrors = []
                RecentWarnings = []
            }
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting event log summary")
            { ErrorCount = 0; WarningCount = 0; InfoCount = 0; RecentErrors = []; RecentWarnings = [] }
    
    /// Collect detailed diagnostics
    member private this.CollectDetailedDiagnosticsAsync() = task {
        try
            logger.LogDebug("Collecting detailed diagnostics...")
            
            // Collect thread dump
            do! this.CollectThreadDumpAsync()
            
            // Collect performance counters
            do! this.CollectPerformanceCountersAsync()
            
            // Clean up old diagnostic files
            this.CleanupOldDiagnosticFiles()
            
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting detailed diagnostics")
    }
    
    /// Collect thread dump
    member private this.CollectThreadDumpAsync() = task {
        try
            let timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss")
            let fileName = $"threaddump_{timestamp}.txt"
            let filePath = Path.Combine(diagnosticsDirectory, fileName)
            
            let currentProcess = Process.GetCurrentProcess()
            let threadInfo = 
                currentProcess.Threads
                |> Seq.cast<ProcessThread>
                |> Seq.map (fun thread -> 
                    $"Thread ID: {thread.Id}, State: {thread.ThreadState}, Priority: {thread.PriorityLevel}, Start Time: {thread.StartTime}")
                |> String.concat Environment.NewLine
            
            do! File.WriteAllTextAsync(filePath, threadInfo)
            
            let diagnosticData = {
                Id = Guid.NewGuid().ToString()
                Type = ThreadDump
                Timestamp = DateTime.UtcNow
                Source = "DiagnosticsCollector"
                Data = Map.ofList [("ThreadCount", currentProcess.Threads.Count :> obj)]
                FilePath = Some filePath
                Size = FileInfo(filePath).Length
                Metadata = Map.ofList [("ProcessId", currentProcess.Id.ToString())]
            }
            
            diagnosticData |> ignore // Would store in diagnosticData dictionary
            
            logger.LogDebug($"Thread dump collected: {fileName}")
            
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting thread dump")
    }
    
    /// Collect performance counters
    member private this.CollectPerformanceCountersAsync() = task {
        try
            let timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss")
            let fileName = $"perfcounters_{timestamp}.txt"
            let filePath = Path.Combine(diagnosticsDirectory, fileName)
            
            // In a real implementation, we'd collect all available performance counters
            let perfData = $"Performance Counters Snapshot - {DateTime.UtcNow}{Environment.NewLine}CPU Usage: N/A{Environment.NewLine}Memory Usage: N/A{Environment.NewLine}"
            
            do! File.WriteAllTextAsync(filePath, perfData)
            
            logger.LogDebug($"Performance counters collected: {fileName}")
            
        with
        | ex ->
            logger.LogWarning(ex, "Error collecting performance counters")
    }
    
    /// Clean up old diagnostic files
    member private this.CleanupOldDiagnosticFiles() =
        try
            let cutoffTime = DateTime.UtcNow.AddDays(-7.0) // Keep files for 7 days
            let files = Directory.GetFiles(diagnosticsDirectory)
            
            for file in files do
                let fileInfo = FileInfo(file)
                if fileInfo.CreationTime < cutoffTime then
                    File.Delete(file)
                    logger.LogDebug($"Deleted old diagnostic file: {fileInfo.Name}")
                    
        with
        | ex ->
            logger.LogWarning(ex, "Error cleaning up old diagnostic files")
    
    /// Get current system diagnostic snapshot
    member this.GetCurrentSystemDiagnosticSnapshot() = task {
        return! this.CollectSystemDiagnosticSnapshotAsync()
    }
    
    /// Get system diagnostic snapshots
    member this.GetSystemDiagnosticSnapshots(hours: int) =
        let cutoffTime = DateTime.UtcNow.AddHours(-float hours)
        systemSnapshots
        |> Seq.filter (fun s -> s.Timestamp >= cutoffTime)
        |> Seq.sortBy (fun s -> s.Timestamp)
        |> List.ofSeq
    
    /// Generate diagnostic report
    member this.GenerateDiagnosticReportAsync() = task {
        try
            let timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss")
            let fileName = $"diagnostic_report_{timestamp}.txt"
            let filePath = Path.Combine(diagnosticsDirectory, fileName)
            
            let! currentSnapshot = this.GetCurrentSystemDiagnosticSnapshot()
            
            let report = 
                $"TARS System Diagnostic Report{Environment.NewLine}" +
                $"Generated: {DateTime.UtcNow}{Environment.NewLine}" +
                $"========================================{Environment.NewLine}" +
                $"System Information:{Environment.NewLine}" +
                $"  Machine Name: {currentSnapshot.SystemInfo.MachineName}{Environment.NewLine}" +
                $"  OS Version: {currentSnapshot.SystemInfo.OSVersion}{Environment.NewLine}" +
                $"  Processor Count: {currentSnapshot.SystemInfo.ProcessorCount}{Environment.NewLine}" +
                $"  .NET Version: {currentSnapshot.SystemInfo.DotNetVersion}{Environment.NewLine}" +
                $"  System Uptime: {currentSnapshot.SystemInfo.SystemUptime}{Environment.NewLine}" +
                $"{Environment.NewLine}" +
                $"Process Information:{Environment.NewLine}" +
                $"  Process ID: {currentSnapshot.ProcessInfo.ProcessId}{Environment.NewLine}" +
                $"  Process Name: {currentSnapshot.ProcessInfo.ProcessName}{Environment.NewLine}" +
                $"  Start Time: {currentSnapshot.ProcessInfo.StartTime}{Environment.NewLine}" +
                $"  Working Set: {currentSnapshot.ProcessInfo.WorkingSet / (1024L * 1024L)} MB{Environment.NewLine}" +
                $"  Thread Count: {currentSnapshot.ProcessInfo.ThreadCount}{Environment.NewLine}" +
                $"  Handle Count: {currentSnapshot.ProcessInfo.HandleCount}{Environment.NewLine}" +
                $"{Environment.NewLine}" +
                $"Memory Information:{Environment.NewLine}" +
                $"  GC Total Memory: {currentSnapshot.MemoryInfo.GCTotalMemory / (1024L * 1024L)} MB{Environment.NewLine}" +
                $"  Memory Usage: {currentSnapshot.MemoryInfo.MemoryUsagePercent:F1}%{Environment.NewLine}" +
                $"{Environment.NewLine}" +
                $"Disk Information:{Environment.NewLine}" +
                $"  Total Space: {currentSnapshot.DiskInfo.TotalSpace / (1024L * 1024L * 1024L)} GB{Environment.NewLine}" +
                $"  Free Space: {currentSnapshot.DiskInfo.FreeSpace / (1024L * 1024L * 1024L)} GB{Environment.NewLine}" +
                $"  Usage: {currentSnapshot.DiskInfo.UsagePercent:F1}%{Environment.NewLine}"
            
            do! File.WriteAllTextAsync(filePath, report)
            
            logger.LogInformation($"Diagnostic report generated: {fileName}")
            return Ok filePath
            
        with
        | ex ->
            logger.LogError(ex, "Error generating diagnostic report")
            return Error ex.Message
    }
    
    /// Get diagnostic files
    member this.GetDiagnosticFiles() =
        try
            if Directory.Exists(diagnosticsDirectory) then
                Directory.GetFiles(diagnosticsDirectory)
                |> Array.map (fun file -> 
                    let fileInfo = FileInfo(file)
                    {|
                        Name = fileInfo.Name
                        Path = file
                        Size = fileInfo.Length
                        Created = fileInfo.CreationTime
                        Modified = fileInfo.LastWriteTime
                    |})
                |> List.ofArray
            else
                []
        with
        | ex ->
            logger.LogWarning(ex, "Error getting diagnostic files")
            []
