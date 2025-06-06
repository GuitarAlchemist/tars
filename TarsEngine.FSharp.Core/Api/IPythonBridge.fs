namespace TarsEngine.FSharp.Core.Api

open System
open System.Threading.Tasks
open System.Collections.Generic

/// Python execution result
type PythonExecutionResult = {
    Success: bool
    Output: string
    Errors: string[]
    Variables: Map<string, obj>
    ExecutionTime: TimeSpan
}

/// Python variable type information
type PythonVariableInfo = {
    Name: string
    Type: string
    Value: obj
    IsCallable: bool
}

/// Python module information
type PythonModuleInfo = {
    Name: string
    Version: string option
    Description: string
    Functions: string[]
    Classes: string[]
}

/// Python environment configuration
type PythonEnvironmentConfig = {
    PythonPath: string option
    VirtualEnvironment: string option
    RequiredPackages: string[]
    EnvironmentVariables: Map<string, string>
    WorkingDirectory: string option
}

/// Python bridge API for executing Python code within TARS metascripts
type IPythonBridge =
    
    /// Execute Python code and return results
    abstract member ExecuteAsync: code: string -> Task<PythonExecutionResult>
    
    /// Execute Python code with specific variables in scope
    abstract member ExecuteWithVariablesAsync: code: string * variables: Map<string, obj> -> Task<PythonExecutionResult>
    
    /// Execute Python script from file
    abstract member ExecuteFileAsync: filePath: string -> Task<PythonExecutionResult>
    
    /// Get all variables in the current Python scope
    abstract member GetVariablesAsync: unit -> Task<PythonVariableInfo[]>
    
    /// Set a variable in the Python scope
    abstract member SetVariableAsync: name: string * value: obj -> Task<bool>
    
    /// Get a variable from the Python scope
    abstract member GetVariableAsync: name: string -> Task<obj option>
    
    /// Import a Python module
    abstract member ImportModuleAsync: moduleName: string -> Task<PythonModuleInfo>
    
    /// Install a Python package using pip
    abstract member InstallPackageAsync: packageName: string -> Task<bool>
    
    /// List installed Python packages
    abstract member ListPackagesAsync: unit -> Task<string[]>
    
    /// Check if a Python package is available
    abstract member IsPackageAvailableAsync: packageName: string -> Task<bool>
    
    /// Configure Python environment
    abstract member ConfigureEnvironmentAsync: config: PythonEnvironmentConfig -> Task<bool>
    
    /// Get Python version information
    abstract member GetVersionInfoAsync: unit -> Task<string>
    
    /// Reset Python environment (clear all variables)
    abstract member ResetEnvironmentAsync: unit -> Task<bool>
    
    /// Evaluate Python expression and return result
    abstract member EvaluateExpressionAsync: expression: string -> Task<obj>
    
    /// Check if Python environment is available
    abstract member IsAvailable: bool

/// Python bridge factory for creating Python execution environments
type IPythonBridgeFactory =
    
    /// Create a new Python bridge instance
    abstract member CreateBridge: config: PythonEnvironmentConfig option -> IPythonBridge
    
    /// Create a sandboxed Python bridge with restricted capabilities
    abstract member CreateSandboxedBridge: allowedModules: string[] -> IPythonBridge
    
    /// Get default Python configuration
    abstract member GetDefaultConfig: unit -> PythonEnvironmentConfig
    
    /// Validate Python installation
    abstract member ValidateInstallation: unit -> Task<bool>

/// Python security policy for sandboxed execution
type PythonSecurityPolicy = {
    AllowedModules: Set<string>
    AllowedBuiltins: Set<string>
    AllowFileAccess: bool
    AllowNetworkAccess: bool
    AllowSubprocesses: bool
    MaxExecutionTime: TimeSpan
    MaxMemoryMB: int
    AllowedPaths: string[]
}

/// Enhanced Python bridge with security features
type ISecurePythonBridge =
    inherit IPythonBridge
    
    /// Execute Python code with security policy
    abstract member ExecuteSecureAsync: code: string * policy: PythonSecurityPolicy -> Task<PythonExecutionResult>
    
    /// Validate Python code against security policy
    abstract member ValidateCodeAsync: code: string * policy: PythonSecurityPolicy -> Task<string[]>
    
    /// Get current security policy
    abstract member GetSecurityPolicy: unit -> PythonSecurityPolicy
    
    /// Set security policy
    abstract member SetSecurityPolicy: policy: PythonSecurityPolicy -> bool

/// Python integration with TARS API
type IPythonTarsIntegration =
    
    /// Inject TARS API into Python environment
    abstract member InjectTarsApiAsync: tarsApi: ITarsEngineApi -> Task<bool>
    
    /// Execute Python code with TARS API access
    abstract member ExecuteWithTarsAsync: code: string * tarsApi: ITarsEngineApi -> Task<PythonExecutionResult>
    
    /// Create Python wrapper for TARS services
    abstract member CreateTarsWrapperAsync: serviceName: string -> Task<string>
    
    /// Get Python code for TARS API usage examples
    abstract member GetTarsExamplesAsync: unit -> Task<Map<string, string>>

/// Python metascript execution context
type PythonMetascriptContext = {
    ExecutionId: string
    Variables: Map<string, obj>
    ImportedModules: string[]
    SecurityPolicy: PythonSecurityPolicy option
    TarsApi: ITarsEngineApi option
    StartTime: DateTime
    WorkingDirectory: string
}

/// Python metascript executor
type IPythonMetascriptExecutor =
    
    /// Execute Python metascript block
    abstract member ExecuteBlockAsync: code: string * context: PythonMetascriptContext -> Task<PythonExecutionResult>
    
    /// Parse Python metascript for dependencies
    abstract member ParseDependenciesAsync: code: string -> Task<string[]>
    
    /// Validate Python metascript syntax
    abstract member ValidateSyntaxAsync: code: string -> Task<string[]>
    
    /// Create execution context
    abstract member CreateContextAsync: executionId: string * tarsApi: ITarsEngineApi option -> Task<PythonMetascriptContext>
    
    /// Cleanup execution context
    abstract member CleanupContextAsync: context: PythonMetascriptContext -> Task<bool>

/// Default security policies for Python execution
type DefaultPythonSecurityPolicies =
    
    /// Restrictive policy for untrusted Python code
    static member Restrictive = {
        AllowedModules = Set.ofList ["math"; "datetime"; "json"; "re"; "collections"]
        AllowedBuiltins = Set.ofList ["len"; "str"; "int"; "float"; "bool"; "list"; "dict"; "tuple"; "set"]
        AllowFileAccess = false
        AllowNetworkAccess = false
        AllowSubprocesses = false
        MaxExecutionTime = TimeSpan.FromSeconds(10.0)
        MaxMemoryMB = 64
        AllowedPaths = []
    }
    
    /// Standard policy for trusted Python code
    static member Standard = {
        AllowedModules = Set.ofList ["math"; "datetime"; "json"; "re"; "collections"; "itertools"; "functools"; "operator"; "numpy"; "pandas"]
        AllowedBuiltins = Set.ofList ["len"; "str"; "int"; "float"; "bool"; "list"; "dict"; "tuple"; "set"; "map"; "filter"; "reduce"; "zip"; "enumerate"]
        AllowFileAccess = true
        AllowNetworkAccess = false
        AllowSubprocesses = false
        MaxExecutionTime = TimeSpan.FromMinutes(5.0)
        MaxMemoryMB = 256
        AllowedPaths = [".tars"; "temp"; "output"]
    }
    
    /// Unrestricted policy for system Python code
    static member Unrestricted = {
        AllowedModules = Set.empty // Allow all modules
        AllowedBuiltins = Set.empty // Allow all builtins
        AllowFileAccess = true
        AllowNetworkAccess = true
        AllowSubprocesses = true
        MaxExecutionTime = TimeSpan.FromMinutes(30.0)
        MaxMemoryMB = 1024
        AllowedPaths = ["/"; "C:\\"] // Allow all paths
    }
