// TARS Autonomous Prerequisite Management Demo
// Demonstrates dynamic prerequisite detection, research, and installation

#r "nuget: Microsoft.Extensions.Logging"
#r "nuget: Microsoft.Extensions.Logging.Console"
#r "nuget: System.Text.Json"

open System
open System.IO
open System.Diagnostics
open System.Threading.Tasks
open Microsoft.Extensions.Logging

// Prerequisite Management Types
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

type PackageManager =
    | WinGet
    | Chocolatey
    | Scoop
    | NPM
    | Pip
    | DotNetTool
    | Custom of name: string

type InstallationMethod = {
    PackageManager: PackageManager
    Command: string
    Arguments: string list
    Confidence: float
    Source: string
}

type InstallationResult = {
    Prerequisite: PrerequisiteType
    Method: InstallationMethod
    Success: bool
    Output: string
    ErrorMessage: string option
    ExecutionTime: TimeSpan
}

// Autonomous Prerequisite Management System
module AutonomousPrerequisiteManager =
    
    let logger = 
        let factory = LoggerFactory.Create(fun builder ->
            builder.AddConsole() |> ignore
        )
        factory.CreateLogger("TARS.PrerequisiteManager")
    
    // Detect prerequisites from project structure
    let detectPrerequisites (projectPath: string) =
        async {
            logger.LogInformation("🔍 Detecting prerequisites in: {ProjectPath}", projectPath)
            
            let prerequisites = ResizeArray<PrerequisiteType>()
            let evidence = ResizeArray<string>()
            
            try
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
                
                logger.LogInformation("📋 Detected {Count} prerequisites", prerequisites.Count)
                for evidence in evidence do
                    logger.LogInformation("  📌 {Evidence}", evidence)
                
                return (prerequisites |> List.ofSeq, evidence |> List.ofSeq)
                
            with
            | ex ->
                logger.LogError(ex, "Failed to detect prerequisites")
                return ([], [ex.Message])
        }
    
    // Generate installation methods for prerequisites
    let generateInstallationMethods (prerequisite: PrerequisiteType) =
        async {
            logger.LogInformation("🔬 Generating installation methods for: {Prerequisite}", prerequisite)
            
            let methods = 
                match prerequisite with
                | DotNetSDK(_) -> [
                    { PackageManager = WinGet; Command = "winget"; Arguments = ["install"; "Microsoft.DotNet.SDK.8"]; Confidence = 0.9; Source = "Microsoft Official" }
                    { PackageManager = Chocolatey; Command = "choco"; Arguments = ["install"; "dotnet-sdk"; "-y"]; Confidence = 0.8; Source = "Chocolatey Community" }
                ]
                | NodeJS(_) -> [
                    { PackageManager = WinGet; Command = "winget"; Arguments = ["install"; "OpenJS.NodeJS"]; Confidence = 0.9; Source = "Node.js Official" }
                    { PackageManager = Chocolatey; Command = "choco"; Arguments = ["install"; "nodejs"; "-y"]; Confidence = 0.8; Source = "Chocolatey Community" }
                ]
                | Python(_) -> [
                    { PackageManager = WinGet; Command = "winget"; Arguments = ["install"; "Python.Python.3.12"]; Confidence = 0.9; Source = "Python Official" }
                    { PackageManager = Chocolatey; Command = "choco"; Arguments = ["install"; "python"; "-y"]; Confidence = 0.8; Source = "Chocolatey Community" }
                ]
                | Docker -> [
                    { PackageManager = WinGet; Command = "winget"; Arguments = ["install"; "Docker.DockerDesktop"]; Confidence = 0.9; Source = "Docker Official" }
                    { PackageManager = Chocolatey; Command = "choco"; Arguments = ["install"; "docker-desktop"; "-y"]; Confidence = 0.8; Source = "Chocolatey Community" }
                ]
                | Git -> [
                    { PackageManager = WinGet; Command = "winget"; Arguments = ["install"; "Git.Git"]; Confidence = 0.9; Source = "Git Official" }
                    { PackageManager = Chocolatey; Command = "choco"; Arguments = ["install"; "git"; "-y"]; Confidence = 0.8; Source = "Chocolatey Community" }
                ]
                | VSCode -> [
                    { PackageManager = WinGet; Command = "winget"; Arguments = ["install"; "Microsoft.VisualStudioCode"]; Confidence = 0.9; Source = "Microsoft Official" }
                    { PackageManager = Chocolatey; Command = "choco"; Arguments = ["install"; "vscode"; "-y"]; Confidence = 0.8; Source = "Chocolatey Community" }
                ]
                | _ -> [
                    { PackageManager = Custom("manual"); Command = "manual"; Arguments = ["download"]; Confidence = 0.5; Source = "Manual Installation" }
                ]
            
            logger.LogInformation("📦 Generated {Count} installation methods", methods.Length)
            return methods
        }
    
    // Check if package manager is available
    let isPackageManagerAvailable (packageManager: PackageManager) =
        async {
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

                let proc = new Process()
                proc.StartInfo.FileName <- command
                proc.StartInfo.Arguments <- "--version"
                proc.StartInfo.UseShellExecute <- false
                proc.StartInfo.RedirectStandardOutput <- true
                proc.StartInfo.RedirectStandardError <- true
                proc.StartInfo.CreateNoWindow <- true

                let started = proc.Start()
                if started then
                    proc.WaitForExit()
                    let available = proc.ExitCode = 0
                    logger.LogInformation("🔧 Package manager {PackageManager}: {Status}", packageManager, (if available then "Available" else "Not Available"))
                    return available
                else
                    logger.LogWarning("🔧 Package manager {PackageManager}: Failed to start", packageManager)
                    return false

            with
            | ex ->
                logger.LogWarning("🔧 Package manager {PackageManager}: Error - {Error}", packageManager, ex.Message)
                return false
        }
    
    // Install prerequisite using package manager
    let installPrerequisite (prerequisite: PrerequisiteType) (method: InstallationMethod) =
        async {
            let startTime = DateTime.UtcNow
            
            try
                logger.LogInformation("📦 Installing {Prerequisite} using {PackageManager}", prerequisite, method.PackageManager)
                
                // Check if package manager is available
                let! pmAvailable = isPackageManagerAvailable method.PackageManager
                
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
                let proc = new Process()
                proc.StartInfo.FileName <- method.Command
                proc.StartInfo.Arguments <- String.Join(" ", method.Arguments)
                proc.StartInfo.UseShellExecute <- false
                proc.StartInfo.RedirectStandardOutput <- true
                proc.StartInfo.RedirectStandardError <- true
                proc.StartInfo.CreateNoWindow <- true

                let started = proc.Start()
                if started then
                    proc.WaitForExit()
                    let output = proc.StandardOutput.ReadToEnd()
                    let error = proc.StandardError.ReadToEnd()

                    let success = proc.ExitCode = 0
                    
                    if success then
                        logger.LogInformation("✅ Successfully installed {Prerequisite}", prerequisite)
                    else
                        logger.LogWarning("❌ Failed to install {Prerequisite}: Exit code {ExitCode}", prerequisite, proc.ExitCode)
                    
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
                logger.LogError(ex, "Installation failed for {Prerequisite}", prerequisite)
                return {
                    Prerequisite = prerequisite
                    Method = method
                    Success = false
                    Output = ""
                    ErrorMessage = Some(ex.Message)
                    ExecutionTime = DateTime.UtcNow - startTime
                }
        }
    
    // Execute autonomous prerequisite management
    let executeAutonomousPrerequisiteManagement (projectPath: string) =
        async {
            logger.LogInformation("🚀 Starting autonomous prerequisite management for: {ProjectPath}", projectPath)
            
            try
                // Step 1: Detect prerequisites
                let! (prerequisites, evidence) = detectPrerequisites projectPath
                
                // Step 2: Generate and execute installation methods
                let installationResults = ResizeArray<InstallationResult>()
                
                for prerequisite in prerequisites do
                    let! methods = generateInstallationMethods prerequisite
                    
                    // Try each method until one succeeds
                    let mutable installed = false
                    for method in methods do
                        if not installed then
                            let! result = installPrerequisite prerequisite method
                            installationResults.Add(result)
                            if result.Success then
                                installed <- true
                
                let results = installationResults |> List.ofSeq
                let successCount = results |> List.filter (fun r -> r.Success) |> List.length
                let totalCount = results.Length
                
                logger.LogInformation("🎉 Prerequisite management completed: {Success}/{Total} successful", successCount, totalCount)
                
                return results
                
            with
            | ex ->
                logger.LogError(ex, "Autonomous prerequisite management failed")
                return []
        }

// Demo execution
let runDemo () =
    async {
        printfn "🚀 TARS AUTONOMOUS PREREQUISITE MANAGEMENT DEMO"
        printfn "================================================="
        
        let projectPath = Directory.GetCurrentDirectory()
        printfn $"📁 Project Path: {projectPath}"
        
        let! results = AutonomousPrerequisiteManager.executeAutonomousPrerequisiteManagement projectPath
        
        printfn "\n📊 INSTALLATION RESULTS"
        printfn "========================"
        
        for result in results do
            let status = if result.Success then "✅ SUCCESS" else "❌ FAILED"
            let duration = result.ExecutionTime.TotalSeconds.ToString("F2")
            printfn $"{status} {result.Prerequisite} ({duration}s)"
            
            if result.ErrorMessage.IsSome then
                printfn $"   Error: {result.ErrorMessage.Value}"
        
        let successCount = results |> List.filter (fun r -> r.Success) |> List.length
        let totalCount = results.Length
        
        printfn $"\n🎯 Summary: {successCount}/{totalCount} prerequisites installed successfully"
        
        if successCount = totalCount then
            printfn "🎉 All prerequisites installed! TARS is ready to build."
        else
            printfn "⚠️  Some prerequisites failed to install. Manual intervention may be required."
    }

// Execute the demo
runDemo () |> Async.RunSynchronously

printfn "\n🤖 TARS Autonomous Prerequisite Management Demo completed!"
