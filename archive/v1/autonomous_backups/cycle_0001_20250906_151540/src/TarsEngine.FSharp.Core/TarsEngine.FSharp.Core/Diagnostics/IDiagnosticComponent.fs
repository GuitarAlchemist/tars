namespace TarsEngine.FSharp.Core.Diagnostics

open System
open System.Threading.Tasks

/// Health status levels
type HealthStatus =
    | Healthy
    | Warning
    | Critical
    | Unknown

/// Diagnostic test result
type DiagnosticResult = {
    ComponentName: string
    TestName: string
    Status: HealthStatus
    Message: string
    Details: string
    ExecutionTimeMs: float
    Timestamp: DateTime
    Metadata: Map<string, obj>
}

/// Performance metrics for a component
type PerformanceMetrics = {
    CpuUsagePercent: float option
    MemoryUsageMB: int64 option
    ThroughputPerSecond: float option
    LatencyMs: float option
    ErrorRate: float option
    CustomMetrics: Map<string, obj>
}

/// Component information
type ComponentInfo = {
    Name: string
    Version: string
    Description: string
    ComponentType: string
    Dependencies: string[]
    IsEnabled: bool
    LastHealthCheck: DateTime option
}

/// Interface that all diagnostic components must implement
type IDiagnosticComponent =
    /// Get basic component information
    abstract member GetComponentInfo: unit -> ComponentInfo
    
    /// Perform health check and return diagnostic result
    abstract member PerformHealthCheck: unit -> Task<DiagnosticResult>
    
    /// Get current performance metrics
    abstract member GetPerformanceMetrics: unit -> Task<PerformanceMetrics>
    
    /// Get detailed diagnostic information
    abstract member GetDetailedDiagnostics: unit -> Task<Map<string, obj>>
    
    /// Test if component can be initialized/started
    abstract member TestInitialization: unit -> Task<DiagnosticResult>
    
    /// Test component's core functionality
    abstract member TestCoreFunctionality: unit -> Task<DiagnosticResult>

/// Diagnostic registry to manage all components
type IDiagnosticRegistry =
    /// Register a diagnostic component
    abstract member RegisterComponent: IDiagnosticComponent -> unit
    
    /// Unregister a diagnostic component
    abstract member UnregisterComponent: string -> unit
    
    /// Get all registered components
    abstract member GetAllComponents: unit -> IDiagnosticComponent[]
    
    /// Get component by name
    abstract member GetComponent: string -> IDiagnosticComponent option
    
    /// Run health checks on all components
    abstract member RunAllHealthChecks: unit -> Task<DiagnosticResult[]>
    
    /// Run health check on specific component
    abstract member RunHealthCheck: string -> Task<DiagnosticResult option>
    
    /// Get system-wide health summary
    abstract member GetSystemHealthSummary: unit -> Task<{| 
        OverallStatus: HealthStatus
        ComponentCount: int
        HealthyCount: int
        WarningCount: int
        CriticalCount: int
        Results: DiagnosticResult[]
    |}>

/// Diagnostic engine interface
type IDiagnosticEngine =
    /// Get the diagnostic registry
    abstract member Registry: IDiagnosticRegistry
    
    /// Generate comprehensive diagnostic report
    abstract member GenerateReport: unit -> Task<string>
    
    /// Generate report in specific format (markdown, json, yaml)
    abstract member GenerateReportInFormat: string -> Task<string>
    
    /// Start continuous health monitoring
    abstract member StartMonitoring: TimeSpan -> Task<unit>
    
    /// Stop continuous health monitoring
    abstract member StopMonitoring: unit -> Task<unit>
    
    /// Get monitoring status
    abstract member IsMonitoring: bool
