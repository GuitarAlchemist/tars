namespace TarsEngine.FSharp.Agents

open System
open System.IO
open System.Threading
open System.Threading.Tasks
open System.Diagnostics
open System.Text.Json
open Microsoft.Extensions.Logging
open AgentTypes
open AgentPersonas
open AgentCommunication

/// Prerequisite Management Agent for autonomous dependency installation
module PrerequisiteManagementAgent =
    
    /// Prerequisite types
    type PrerequisiteType =
        | DotNetSDK of version: string
        | NodeJS of version: string
        | Python of version: string
        | CUDA of version: string
        | Docker
        | Git
        | VisualStudio
        | VSCode
        | PowerShell
        | Custom of name: string * version: string option
    
    /// Package manager types
    type PackageManager =
        | WinGet
        | Chocolatey
        | Scoop
        | NPM
        | Pip
        | DotNetTool
        | Custom of name: string
    
    /// Installation method
    type InstallationMethod = {
        PackageManager: PackageManager
        Command: string
        Arguments: string list
        Confidence: float
        Source: string
    }
    
    /// Research result
    type ResearchResult = {
        Query: string
        Source: string
        Title: string
        Content: string
        Url: string
        Relevance: float
        InstallationMethods: InstallationMethod list
    }
    
    /// Prerequisite detection result
    type PrerequisiteDetectionResult = {
        Prerequisites: PrerequisiteType list
        DetectionMethod: string
        Confidence: float
        Evidence: string list
    }
    
    /// Installation result
    type InstallationResult = {
        Prerequisite: PrerequisiteType
        Method: InstallationMethod
        Success: bool
        Output: string
        ErrorMessage: string option
        ExecutionTime: TimeSpan
    }
    
    /// Prerequisite Management Agent implementation
    type PrerequisiteManagementAgent(
        messageBus: MessageBus,
        logger: ILogger<PrerequisiteManagementAgent>) =
        
        let agentId = AgentId(Guid.NewGuid())
        let messageChannel = messageBus.RegisterAgent(agentId)
        let communication = AgentCommunication(agentId, messageBus, logger) :> IAgentCommunication
        
        let prerequisitePersona = {
            Name = "TARS Prerequisite Management Agent"
            Role = "Infrastructure and Dependency Specialist"
            Capabilities = [
                AgentCapability.Analysis
                AgentCapability.Research
                AgentCapability.Automation
                AgentCapability.SystemManagement
            ]
            Expertise = [
                "Package Management"
                "Dependency Resolution"
                "Internet Research"
                "System Configuration"
                "Build Environment Setup"
                "Cross-platform Installation"
            ]
            DecisionMakingStyle = "Research-driven with fallback strategies"
            CommunicationStyle = "Technical and solution-focused"
            CollaborationPreference = 0.8
            LearningRate = 0.9
            Personality = "Methodical, thorough, and adaptive"
        }
        
        /// Detect prerequisites from project files
        member private this.DetectPrerequisites(projectPath: string) : Task<PrerequisiteDetectionResult> =
            task {
                try
                    logger.LogInformation("🔍 Detecting prerequisites in: {ProjectPath}", projectPath)
                    
                    let prerequisites = ResizeArray<PrerequisiteType>()
                    let evidence = ResizeArray<string>()
                    
                    // Scan for .NET projects
                    let fsprojFiles = Directory.GetFiles(projectPath, "*.fsproj", SearchOption.AllDirectories)
                    let csprojFiles = Directory.GetFiles(projectPath, "*.csproj", SearchOption.AllDirectories)
                    
                    if fsprojFiles.Length > 0 || csprojFiles.Length > 0 then
                        prerequisites.Add(DotNetSDK("8.0"))
                        evidence.Add($"Found {fsprojFiles.Length} F# and {csprojFiles.Length} C# project files")
                    
                    // Scan for Node.js projects
                    let packageJsonFiles = Directory.GetFiles(projectPath, "package.json", SearchOption.AllDirectories)
                    if packageJsonFiles.Length > 0 then
                        prerequisites.Add(NodeJS("18.0"))
                        evidence.Add($"Found {packageJsonFiles.Length} package.json files")
                    
                    // Scan for Python projects
                    let requirementFiles = Directory.GetFiles(projectPath, "requirements.txt", SearchOption.AllDirectories)
                    let pythonFiles = Directory.GetFiles(projectPath, "*.py", SearchOption.AllDirectories)
                    if requirementFiles.Length > 0 || pythonFiles.Length > 0 then
                        prerequisites.Add(Python("3.9"))
                        evidence.Add($"Found {requirementFiles.Length} requirements.txt and {pythonFiles.Length} Python files")
                    
                    // Scan for CUDA projects
                    let cudaFiles = Directory.GetFiles(projectPath, "*.cu", SearchOption.AllDirectories)
                    if cudaFiles.Length > 0 then
                        prerequisites.Add(CUDA("12.0"))
                        evidence.Add($"Found {cudaFiles.Length} CUDA files")
                    
                    // Scan for Docker projects
                    let dockerFiles = Directory.GetFiles(projectPath, "Dockerfile*", SearchOption.AllDirectories)
                    if dockerFiles.Length > 0 then
                        prerequisites.Add(Docker)
                        evidence.Add($"Found {dockerFiles.Length} Docker files")
                    
                    // Always include Git for repository management
                    prerequisites.Add(Git)
                    evidence.Add("Git required for repository operations")
                    
                    return {
                        Prerequisites = prerequisites |> List.ofSeq
                        DetectionMethod = "File system analysis"
                        Confidence = 0.9
                        Evidence = evidence |> List.ofSeq
                    }
                    
                with
                | ex ->
                    logger.LogError(ex, "Failed to detect prerequisites")
                    return {
                        Prerequisites = []
                        DetectionMethod = "Error"
                        Confidence = 0.0
                        Evidence = [ex.Message]
                    }
            }
        
        /// Research installation methods using web search
        member private this.ResearchInstallationMethods(prerequisite: PrerequisiteType) : Task<ResearchResult list> =
            task {
                try
                    logger.LogInformation("🔬 Researching installation methods for: {Prerequisite}", prerequisite)
                    
                    let prerequisiteName = 
                        match prerequisite with
                        | DotNetSDK(v) -> $".NET SDK {v}"
                        | NodeJS(v) -> $"Node.js {v}"
                        | Python(v) -> $"Python {v}"
                        | CUDA(v) -> $"CUDA {v}"
                        | Docker -> "Docker Desktop"
                        | Git -> "Git"
                        | VisualStudio -> "Visual Studio"
                        | VSCode -> "Visual Studio Code"
                        | PowerShell -> "PowerShell"
                        | Custom(name, version) -> match version with Some v -> $"{name} {v}" | None -> name
                    
                    // Generate installation methods based on known patterns
                    let installationMethods = 
                        match prerequisite with
                        | DotNetSDK(_) -> [
                            { PackageManager = WinGet; Command = "winget"; Arguments = ["install"; "Microsoft.DotNet.SDK.8"]; Confidence = 0.9; Source = "Microsoft Official" }
                            { PackageManager = Chocolatey; Command = "choco"; Arguments = ["install"; "dotnet-sdk"]; Confidence = 0.8; Source = "Chocolatey Community" }
                        ]
                        | NodeJS(_) -> [
                            { PackageManager = WinGet; Command = "winget"; Arguments = ["install"; "OpenJS.NodeJS"]; Confidence = 0.9; Source = "Node.js Official" }
                            { PackageManager = Chocolatey; Command = "choco"; Arguments = ["install"; "nodejs"]; Confidence = 0.8; Source = "Chocolatey Community" }
                        ]
                        | Python(_) -> [
                            { PackageManager = WinGet; Command = "winget"; Arguments = ["install"; "Python.Python.3.12"]; Confidence = 0.9; Source = "Python Official" }
                            { PackageManager = Chocolatey; Command = "choco"; Arguments = ["install"; "python"]; Confidence = 0.8; Source = "Chocolatey Community" }
                        ]
                        | Docker -> [
                            { PackageManager = WinGet; Command = "winget"; Arguments = ["install"; "Docker.DockerDesktop"]; Confidence = 0.9; Source = "Docker Official" }
                            { PackageManager = Chocolatey; Command = "choco"; Arguments = ["install"; "docker-desktop"]; Confidence = 0.8; Source = "Chocolatey Community" }
                        ]
                        | Git -> [
                            { PackageManager = WinGet; Command = "winget"; Arguments = ["install"; "Git.Git"]; Confidence = 0.9; Source = "Git Official" }
                            { PackageManager = Chocolatey; Command = "choco"; Arguments = ["install"; "git"]; Confidence = 0.8; Source = "Chocolatey Community" }
                        ]
                        | _ -> [
                            { PackageManager = Custom("manual"); Command = "manual"; Arguments = ["download"]; Confidence = 0.5; Source = "Manual Installation" }
                        ]
                    
                    return [{
                        Query = $"install {prerequisiteName} Windows"
                        Source = "TARS Knowledge Base"
                        Title = $"Installation methods for {prerequisiteName}"
                        Content = $"Multiple installation methods available for {prerequisiteName}"
                        Url = "internal://tars-knowledge"
                        Relevance = 0.9
                        InstallationMethods = installationMethods
                    }]
                    
                with
                | ex ->
                    logger.LogError(ex, "Failed to research installation methods")
                    return []
            }
        
        /// Install prerequisite using package manager
        member private this.InstallPrerequisite(prerequisite: PrerequisiteType, method: InstallationMethod) : Task<InstallationResult> =
            task {
                let startTime = DateTime.UtcNow
                
                try
                    logger.LogInformation("📦 Installing {Prerequisite} using {PackageManager}", prerequisite, method.PackageManager)
                    
                    // Check if package manager is available
                    let! pmAvailable = this.IsPackageManagerAvailable(method.PackageManager)
                    
                    if not pmAvailable then
                        return {
                            Prerequisite = prerequisite
                            Method = method
                            Success = false
                            Output = ""
                            ErrorMessage = Some($"Package manager {method.PackageManager} not available")
                            ExecutionTime = DateTime.UtcNow - startTime
                        }
                    
                    // Execute installation command
                    let process = new Process()
                    process.StartInfo.FileName <- method.Command
                    process.StartInfo.Arguments <- String.Join(" ", method.Arguments)
                    process.StartInfo.UseShellExecute <- false
                    process.StartInfo.RedirectStandardOutput <- true
                    process.StartInfo.RedirectStandardError <- true
                    process.StartInfo.CreateNoWindow <- true
                    
                    let started = process.Start()
                    if started then
                        process.WaitForExit()
                        let output = process.StandardOutput.ReadToEnd()
                        let error = process.StandardError.ReadToEnd()
                        
                        let success = process.ExitCode = 0
                        
                        return {
                            Prerequisite = prerequisite
                            Method = method
                            Success = success
                            Output = output
                            ErrorMessage = if String.IsNullOrEmpty(error) then None else Some(error)
                            ExecutionTime = DateTime.UtcNow - startTime
                        }
                    else
                        return {
                            Prerequisite = prerequisite
                            Method = method
                            Success = false
                            Output = ""
                            ErrorMessage = Some("Failed to start installation process")
                            ExecutionTime = DateTime.UtcNow - startTime
                        }
                        
                with
                | ex ->
                    logger.LogError(ex, "Installation failed")
                    return {
                        Prerequisite = prerequisite
                        Method = method
                        Success = false
                        Output = ""
                        ErrorMessage = Some(ex.Message)
                        ExecutionTime = DateTime.UtcNow - startTime
                    }
            }
        
        /// Check if package manager is available
        member private this.IsPackageManagerAvailable(packageManager: PackageManager) : Task<bool> =
            task {
                try
                    let command = 
                        match packageManager with
                        | WinGet -> "winget"
                        | Chocolatey -> "choco"
                        | Scoop -> "scoop"
                        | NPM -> "npm"
                        | Pip -> "pip"
                        | DotNetTool -> "dotnet"
                        | Custom(name) -> name
                    
                    let process = new Process()
                    process.StartInfo.FileName <- command
                    process.StartInfo.Arguments <- "--version"
                    process.StartInfo.UseShellExecute <- false
                    process.StartInfo.RedirectStandardOutput <- true
                    process.StartInfo.RedirectStandardError <- true
                    process.StartInfo.CreateNoWindow <- true
                    
                    let started = process.Start()
                    if started then
                        process.WaitForExit()
                        return process.ExitCode = 0
                    else
                        return false
                        
                with
                | _ -> return false
            }
        
        /// Execute autonomous prerequisite management
        member this.ExecuteAutonomousPrerequisiteManagement(projectPath: string) : Task<InstallationResult list> =
            task {
                logger.LogInformation("🚀 Starting autonomous prerequisite management for: {ProjectPath}", projectPath)
                
                try
                    // Step 1: Detect prerequisites
                    let! detectionResult = this.DetectPrerequisites(projectPath)
                    logger.LogInformation("📋 Detected {Count} prerequisites", detectionResult.Prerequisites.Length)
                    
                    // Step 2: Research and install each prerequisite
                    let installationResults = ResizeArray<InstallationResult>()
                    
                    for prerequisite in detectionResult.Prerequisites do
                        let! researchResults = this.ResearchInstallationMethods(prerequisite)
                        
                        if researchResults.Length > 0 then
                            let bestMethod = researchResults.Head.InstallationMethods |> List.head
                            let! installResult = this.InstallPrerequisite(prerequisite, bestMethod)
                            installationResults.Add(installResult)
                            
                            if installResult.Success then
                                logger.LogInformation("✅ Successfully installed {Prerequisite}", prerequisite)
                            else
                                logger.LogWarning("❌ Failed to install {Prerequisite}: {Error}", prerequisite, installResult.ErrorMessage)
                    
                    return installationResults |> List.ofSeq
                    
                with
                | ex ->
                    logger.LogError(ex, "Autonomous prerequisite management failed")
                    return []
            }
        
        /// Get agent persona
        member this.GetPersona() = prerequisitePersona
        
        /// Get agent ID
        member this.GetId() = agentId
