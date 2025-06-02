namespace TarsEngine.FSharp.Packaging.Agents

open System
open System.IO
open System.Diagnostics
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Packaging.Core
open TarsEngine.FSharp.Packaging.Generators

/// Packaging agent for creating MSI installers using WiX
type PackagingAgent(logger: ILogger<PackagingAgent>) =
    
    let wixGenerator = WixGenerator()
    
    /// Creates TARS self-packaging installer
    member _.CreateTarsInstaller(version: string, outputDir: string) =
        async {
            try
                logger.LogInformation("Creating TARS MSI installer version {Version}", version)
                
                // Define TARS installation files
                let tarsFiles = this.GetTarsInstallationFiles()
                
                // Create WiX project for TARS
                let project = this.CreateTarsWixProject(version, tarsFiles)
                
                // Generate WiX project
                let generatedProject = wixGenerator.GenerateWixProject(project, outputDir)
                
                logger.LogInformation("TARS installer project generated at {OutputDir}", outputDir)
                
                return Ok {|
                    Type = "TARS_INSTALLER"
                    Version = version
                    OutputDirectory = outputDir
                    ProjectFiles = generatedProject.GeneratedFiles
                    InstallerName = $"TARS-{version}.msi"
                    BuildCommand = "build.cmd"
                |}
                
            with
            | ex ->
                logger.LogError(ex, "Failed to create TARS installer")
                return Error ex.Message
        }
    
    /// Creates custom application installer
    member _.CreateApplicationInstaller(appName: string, version: string, files: string list, outputDir: string) =
        async {
            try
                logger.LogInformation("Creating installer for {AppName} version {Version}", appName, version)
                
                // Create WiX project for application
                let project = this.CreateApplicationWixProject(appName, version, files)
                
                // Generate WiX project
                let generatedProject = wixGenerator.GenerateWixProject(project, outputDir)
                
                logger.LogInformation("Application installer project generated at {OutputDir}", outputDir)
                
                return Ok {|
                    Type = "APPLICATION_INSTALLER"
                    ApplicationName = appName
                    Version = version
                    OutputDirectory = outputDir
                    ProjectFiles = generatedProject.GeneratedFiles
                    InstallerName = $"{appName}-{version}.msi"
                    BuildCommand = "build.cmd"
                |}
                
            with
            | ex ->
                logger.LogError(ex, "Failed to create application installer")
                return Error ex.Message
        }
    
    /// Builds MSI installer using WiX
    member _.BuildInstaller(projectDir: string) =
        async {
            try
                logger.LogInformation("Building MSI installer in {ProjectDir}", projectDir)
                
                // Check if WiX is installed
                let! wixCheck = this.CheckWixInstallation()
                match wixCheck with
                | Error error -> return Error error
                | Ok _ -> ()
                
                // Find build script
                let buildScript = Path.Combine(projectDir, "build.cmd")
                if not (File.Exists(buildScript)) then
                    return Error $"Build script not found: {buildScript}"
                
                // Execute build script
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- buildScript
                startInfo.WorkingDirectory <- projectDir
                startInfo.UseShellExecute <- false
                startInfo.RedirectStandardOutput <- true
                startInfo.RedirectStandardError <- true
                startInfo.CreateNoWindow <- true
                
                let process = Process.Start(startInfo)
                process.WaitForExit()
                
                let output = process.StandardOutput.ReadToEnd()
                let error = process.StandardError.ReadToEnd()
                
                if process.ExitCode = 0 then
                    logger.LogInformation("MSI installer built successfully")
                    
                    // Find generated MSI file
                    let binDir = Path.Combine(projectDir, "bin", "Release")
                    let msiFiles = 
                        if Directory.Exists(binDir) then
                            Directory.GetFiles(binDir, "*.msi")
                        else
                            [||]
                    
                    return Ok {|
                        Success = true
                        Output = output
                        MsiFiles = msiFiles |> Array.toList
                        BuildDirectory = binDir
                    |}
                else
                    logger.LogError("MSI build failed: {Error}", error)
                    return Error $"Build failed: {error}"
                    
            with
            | ex ->
                logger.LogError(ex, "Error building installer")
                return Error ex.Message
        }
    
    /// Checks if WiX Toolset is installed
    member _.CheckWixInstallation() =
        async {
            try
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- "candle"
                startInfo.Arguments <- "-?"
                startInfo.UseShellExecute <- false
                startInfo.RedirectStandardOutput <- true
                startInfo.RedirectStandardError <- true
                startInfo.CreateNoWindow <- true
                
                let process = Process.Start(startInfo)
                process.WaitForExit()
                
                if process.ExitCode = 0 then
                    return Ok "WiX Toolset is installed"
                else
                    return Error "WiX Toolset not found. Please install WiX Toolset from https://wixtoolset.org/releases/"
                    
            with
            | ex ->
                return Error $"Failed to check WiX installation: {ex.Message}"
        }
    
    /// Gets TARS installation files
    member _.GetTarsInstallationFiles() =
        [
            "TarsEngine.FSharp.Cli.exe"
            "TarsEngine.FSharp.Cli.dll"
            "TarsEngine.FSharp.DataSources.dll"
            "TarsEngine.FSharp.Metascripts.dll"
            "TarsEngine.FSharp.Packaging.dll"
            "Microsoft.Extensions.Logging.dll"
            "Microsoft.Extensions.DependencyInjection.dll"
            "Newtonsoft.Json.dll"
            "FSharp.Core.dll"
            "README.md"
            "LICENSE"
        ]
    
    /// Creates WiX project for TARS
    member _.CreateTarsWixProject(version: string, files: string list) =
        let upgradeCode = Guid.Parse("12345678-1234-5678-9ABC-123456789012")  // Fixed GUID for TARS
        
        // Create directories
        let directories = WixHelpers.createAppDirectories "TARS" files
        
        // Create components
        let componentIds = 
            files 
            |> List.mapi (fun i _ -> $"Component{i}")
        
        // Create main feature
        let mainFeature = WixHelpers.createMainFeature "TARS" componentIds
        
        // Create shortcuts
        let startMenuShortcut = {
            Id = "StartMenuShortcut"
            Name = "TARS"
            Description = "TARS Autonomous Reasoning System"
            Target = "[INSTALLFOLDER]TarsEngine.FSharp.Cli.exe"
            Arguments = None
            WorkingDirectory = Some "[INSTALLFOLDER]"
            Icon = None
            IconIndex = 0
            ShowCommand = Normal
        }
        
        // Add shortcut to main component
        let mainComponent = {
            Id = "MainComponent"
            Guid = Guid.NewGuid()
            Directory = "INSTALLFOLDER"
            Files = []
            RegistryKeys = []
            Shortcuts = [startMenuShortcut]
            Services = []
        }
        
        // Create UI
        let ui = WixHelpers.createStandardUI None
        
        WixHelpers.project "TARS"
            .Version(version)
            .Manufacturer("TARS Development Team")
            .Description("TARS Autonomous Reasoning System - AI-powered development assistant")
            .Platform(X64)
            .InstallScope(PerMachine)
            .Directory(directories.[0])
            .Directory(directories.[1])
            .Directory(directories.[2])
            .Feature(mainFeature)
            .UI(ui)
            .Property("ARPPRODUCTICON", "tars.ico")
            .Property("ARPHELPLINK", "https://github.com/GuitarAlchemist/tars")
            .Property("ARPURLINFOABOUT", "https://github.com/GuitarAlchemist/tars")
            .UpgradeRule(upgradeCode, "0.0.0", version, "PREVIOUSVERSIONSINSTALLED")
            .Condition("This application requires Windows 10 or later.", "VersionNT >= 1000")
            .Build()
    
    /// Creates WiX project for generic application
    member _.CreateApplicationWixProject(appName: string, version: string, files: string list) =
        // Create directories
        let directories = WixHelpers.createAppDirectories appName files
        
        // Create components
        let componentIds = 
            files 
            |> List.mapi (fun i _ -> $"Component{i}")
        
        // Create main feature
        let mainFeature = WixHelpers.createMainFeature appName componentIds
        
        // Create UI
        let ui = WixHelpers.createStandardUI None
        
        WixHelpers.project appName
            .Version(version)
            .Manufacturer("TARS Generated")
            .Description($"{appName} application installer")
            .Platform(X64)
            .InstallScope(PerMachine)
            .Directory(directories.[0])
            .Directory(directories.[1])
            .Directory(directories.[2])
            .Feature(mainFeature)
            .UI(ui)
            .Build()
    
    /// Creates installer for infrastructure components
    member _.CreateInfrastructureInstaller(stackName: string, version: string, outputDir: string) =
        async {
            try
                logger.LogInformation("Creating infrastructure installer for {StackName}", stackName)
                
                // Create WiX project for infrastructure stack
                let project = this.CreateInfrastructureWixProject(stackName, version)
                
                // Generate WiX project
                let generatedProject = wixGenerator.GenerateWixProject(project, outputDir)
                
                logger.LogInformation("Infrastructure installer project generated at {OutputDir}", outputDir)
                
                return Ok {|
                    Type = "INFRASTRUCTURE_INSTALLER"
                    StackName = stackName
                    Version = version
                    OutputDirectory = outputDir
                    ProjectFiles = generatedProject.GeneratedFiles
                    InstallerName = $"{stackName}-Infrastructure-{version}.msi"
                    BuildCommand = "build.cmd"
                |}
                
            with
            | ex ->
                logger.LogError(ex, "Failed to create infrastructure installer")
                return Error ex.Message
        }
    
    /// Creates WiX project for infrastructure components
    member _.CreateInfrastructureWixProject(stackName: string, version: string) =
        let files = [
            "docker-compose.yml"
            ".env"
            "start.sh"
            "stop.sh"
            "monitor.sh"
            "README.md"
        ]
        
        // Create directories
        let directories = WixHelpers.createAppDirectories $"{stackName}Infrastructure" files
        
        // Create components
        let componentIds = 
            files 
            |> List.mapi (fun i _ -> $"InfraComponent{i}")
        
        // Create main feature
        let mainFeature = {
            Id = "InfrastructureFeature"
            Title = $"{stackName} Infrastructure"
            Description = $"Docker-based infrastructure stack for {stackName}"
            Level = 1
            Display = Expand
            AllowAdvertise = false
            InstallDefault = Local
            TypicalDefault = Install
            Components = componentIds
            Features = []
        }
        
        // Create UI
        let ui = WixHelpers.createStandardUI None
        
        WixHelpers.project $"{stackName}Infrastructure"
            .Version(version)
            .Manufacturer("TARS Infrastructure Generator")
            .Description($"{stackName} infrastructure stack installer")
            .Platform(X64)
            .InstallScope(PerMachine)
            .Directory(directories.[0])
            .Directory(directories.[1])
            .Directory(directories.[2])
            .Feature(mainFeature)
            .UI(ui)
            .Condition("Docker Desktop is recommended for this infrastructure stack.", "1")
            .Build()
