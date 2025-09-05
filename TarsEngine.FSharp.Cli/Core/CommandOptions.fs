namespace TarsEngine.FSharp.Cli.Core

/// Command line options for TARS CLI commands
type CommandOptions = {
    /// Enable verbose output
    Verbose: bool
    /// Enable debug mode
    Debug: bool
    /// Output format (json, yaml, text)
    OutputFormat: string
    /// Working directory
    WorkingDirectory: string option
    /// Configuration file path
    ConfigFile: string option
    /// Enable dry run mode
    DryRun: bool
    /// Force execution without prompts
    Force: bool
    /// Additional parameters
    Parameters: Map<string, string>
}

module CommandOptions =
    
    /// Default command options
    let defaultOptions = {
        Verbose = false
        Debug = false
        OutputFormat = "text"
        WorkingDirectory = None
        ConfigFile = None
        DryRun = false
        Force = false
        Parameters = Map.empty
    }
    
    /// Create options with verbose enabled
    let withVerbose options = { options with Verbose = true }
    
    /// Create options with debug enabled
    let withDebug options = { options with Debug = true }
    
    /// Create options with specific output format
    let withOutputFormat format options = { options with OutputFormat = format }
    
    /// Create options with working directory
    let withWorkingDirectory dir options = { options with WorkingDirectory = Some dir }
    
    /// Create options with config file
    let withConfigFile file options = { options with ConfigFile = Some file }
    
    /// Create options with dry run enabled
    let withDryRun options = { options with DryRun = true }
    
    /// Create options with force enabled
    let withForce options = { options with Force = true }
    
    /// Add parameter to options
    let addParameter key value options = 
        { options with Parameters = options.Parameters.Add(key, value) }
    
    /// Get parameter from options
    let getParameter key options =
        options.Parameters.TryFind(key)
    
    /// Check if parameter exists
    let hasParameter key options =
        options.Parameters.ContainsKey(key)
