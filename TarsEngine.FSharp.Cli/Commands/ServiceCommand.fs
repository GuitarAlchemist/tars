namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Diagnostics
open System.ServiceProcess
open System.Security.Principal
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Core

/// <summary>
/// Service management command for TARS Windows Service
/// Handles installation, uninstallation, and management of TARS as a Windows service
/// </summary>
type ServiceCommand() =
    
    let serviceName = "TarsService"
    let serviceDisplayName = "TARS Autonomous Development Platform"
    let serviceDescription = "Autonomous development platform with multi-agent orchestration, semantic coordination, and continuous improvement capabilities"
    
    /// Check if running as administrator
    member private self.IsRunningAsAdmin() =
        let identity = WindowsIdentity.GetCurrent()
        let principal = WindowsPrincipal(identity)
        principal.IsInRole(WindowsBuiltInRole.Administrator)
    
    /// Get the Windows service executable path
    member private self.GetServiceExecutablePath() =
        let currentDir = Directory.GetCurrentDirectory()
        let servicePath = Path.Combine(currentDir, "TarsEngine.FSharp.WindowsService", "bin", "Debug", "net9.0", "TarsEngine.FSharp.WindowsService.exe")
        
        if File.Exists(servicePath) then
            Some servicePath
        else
            // Try alternative paths
            let altPath1 = Path.Combine(currentDir, "..", "TarsEngine.FSharp.WindowsService", "bin", "Debug", "net9.0", "TarsEngine.FSharp.WindowsService.exe")
            let altPath2 = Path.Combine(currentDir, "bin", "TarsEngine.FSharp.WindowsService.exe")
            
            if File.Exists(altPath1) then Some altPath1
            elif File.Exists(altPath2) then Some altPath2
            else None
    
    /// Check if service is installed
    member private self.IsServiceInstalled() =
        try
            use serviceController = new ServiceController(serviceName)
            serviceController.Refresh()
            true
        with
        | :? InvalidOperationException -> false
        | _ -> false
    
    /// Get service status
    member private self.GetServiceStatus() =
        try
            use serviceController = new ServiceController(serviceName)
            serviceController.Refresh()
            Some serviceController.Status
        with
        | :? InvalidOperationException -> None
        | _ -> None
    
    /// Install the Windows service
    member self.InstallService(force: bool) =
        printfn "🤖 TARS Windows Service Installation"
        printfn "═══════════════════════════════════"
        printfn ""
        
        // Check admin privileges
        if not (self.IsRunningAsAdmin()) then
            printfn "❌ Administrator privileges required!"
            printfn "   Please run TARS CLI as Administrator to install the service."
            printfn "   Right-click Command Prompt or PowerShell and select 'Run as Administrator'"
            1
        else
            // Check if service executable exists
            match self.GetServiceExecutablePath() with
            | None ->
                printfn "❌ Service executable not found!"
                printfn "   Please build the Windows service first:"
                printfn "   dotnet build TarsEngine.FSharp.WindowsService"
                1
            | Some executablePath ->
                printfn $"📁 Service executable: {executablePath}"
                
                // Check if service already exists
                if self.IsServiceInstalled() && not force then
                    printfn "⚠️ TARS service is already installed!"
                    printfn "   Use 'tars service install --force' to reinstall"
                    printfn "   Use 'tars service uninstall' to remove first"
                    1
                else
                    try
                        // Uninstall existing service if force is specified
                        if self.IsServiceInstalled() && force then
                            printfn "🔄 Reinstalling existing service..."
                            self.UninstallServiceInternal() |> ignore
                            System.Threading.Thread.Sleep(2000) // Wait for cleanup
                        
                        // Install the service using sc.exe
                        printfn "🔧 Installing TARS Windows Service..."
                        printfn $"   Service Name: {serviceName}"
                        printfn $"   Display Name: {serviceDisplayName}"
                        printfn $"   Executable: {executablePath}"
                        
                        let createArgs = $"create \"{serviceName}\" binPath= \"\"{executablePath}\"\" DisplayName= \"{serviceDisplayName}\" start= auto"
                        let createProcess = Process.Start("sc.exe", createArgs)
                        createProcess.WaitForExit()
                        
                        if createProcess.ExitCode = 0 then
                            // Set service description
                            let descArgs = $"description \"{serviceName}\" \"{serviceDescription}\""
                            let descProcess = Process.Start("sc.exe", descArgs)
                            descProcess.WaitForExit()
                            
                            // Configure service recovery
                            let recoveryArgs = $"failure \"{serviceName}\" reset= 86400 actions= restart/5000/restart/10000/restart/30000"
                            let recoveryProcess = Process.Start("sc.exe", recoveryArgs)
                            recoveryProcess.WaitForExit()
                            
                            printfn "✅ TARS Windows Service installed successfully!"
                            printfn ""
                            printfn "🎯 Service Management Commands:"
                            printfn "   Start:     tars service start"
                            printfn "   Stop:      tars service stop"
                            printfn "   Status:    tars service status"
                            printfn "   Restart:   tars service restart"
                            printfn "   Uninstall: tars service uninstall"
                            printfn ""
                            
                            // Ask if user wants to start the service
                            printf "🚀 Start TARS service now? (y/N): "
                            let response = Console.ReadLine()
                            if response.ToLower() = "y" || response.ToLower() = "yes" then
                                self.StartService()
                            else
                                printfn "ℹ️ Service installed but not started. Use 'tars service start' when ready."
                                0
                        else
                            printfn $"❌ Failed to install service. Exit code: {createProcess.ExitCode}"
                            1
                            
                    with
                    | ex ->
                        printfn $"❌ Error installing service: {ex.Message}"
                        1
    
    /// Uninstall the Windows service (internal)
    member private self.UninstallServiceInternal() =
        try
            // Stop service if running
            match self.GetServiceStatus() with
            | Some status when status = ServiceControllerStatus.Running ->
                printfn "⏹️ Stopping TARS service..."
                use serviceController = new ServiceController(serviceName)
                serviceController.Stop()
                serviceController.WaitForStatus(ServiceControllerStatus.Stopped, TimeSpan.FromSeconds(30.0))
            | _ -> ()
            
            // Remove service
            let deleteArgs = $"delete \"{serviceName}\""
            let deleteProcess = Process.Start("sc.exe", deleteArgs)
            deleteProcess.WaitForExit()
            
            deleteProcess.ExitCode = 0
        with
        | ex ->
            printfn $"⚠️ Error during uninstall: {ex.Message}"
            false
    
    /// Uninstall the Windows service
    member self.UninstallService() =
        printfn "🗑️ TARS Windows Service Uninstallation"
        printfn "════════════════════════════════════"
        printfn ""
        
        // Check admin privileges
        if not (self.IsRunningAsAdmin()) then
            printfn "❌ Administrator privileges required!"
            printfn "   Please run TARS CLI as Administrator to uninstall the service."
            1
        else
            if not (self.IsServiceInstalled()) then
                printfn "ℹ️ TARS service is not installed."
                0
            else
                if self.UninstallServiceInternal() then
                    printfn "✅ TARS Windows Service uninstalled successfully!"
                    0
                else
                    printfn "❌ Failed to uninstall TARS service."
                    1

    /// Start the Windows service
    member self.StartService() =
        if not (self.IsServiceInstalled()) then
            printfn "❌ TARS service is not installed. Install it first with 'tars service install'"
            1
        else
            try
                use serviceController = new ServiceController(serviceName)
                serviceController.Refresh()

                match serviceController.Status with
                | ServiceControllerStatus.Running ->
                    printfn "ℹ️ TARS service is already running."
                    0
                | _ ->
                    printfn "🚀 Starting TARS service..."
                    serviceController.Start()
                    serviceController.WaitForStatus(ServiceControllerStatus.Running, TimeSpan.FromSeconds(30.0))

                    serviceController.Refresh()
                    if serviceController.Status = ServiceControllerStatus.Running then
                        printfn "✅ TARS service started successfully!"
                        printfn $"🎯 Status: {serviceController.Status}"
                        0
                    else
                        printfn $"⚠️ Service status: {serviceController.Status}"
                        1
            with
            | ex ->
                printfn $"❌ Failed to start service: {ex.Message}"
                printfn "💡 Check Windows Event Log for details"
                1

    /// Stop the Windows service
    member self.StopService() =
        if not (self.IsServiceInstalled()) then
            printfn "❌ TARS service is not installed."
            1
        else
            try
                use serviceController = new ServiceController(serviceName)
                serviceController.Refresh()

                match serviceController.Status with
                | ServiceControllerStatus.Stopped ->
                    printfn "ℹ️ TARS service is already stopped."
                    0
                | _ ->
                    printfn "⏹️ Stopping TARS service..."
                    serviceController.Stop()
                    serviceController.WaitForStatus(ServiceControllerStatus.Stopped, TimeSpan.FromSeconds(30.0))

                    serviceController.Refresh()
                    if serviceController.Status = ServiceControllerStatus.Stopped then
                        printfn "✅ TARS service stopped successfully!"
                        0
                    else
                        printfn $"⚠️ Service status: {serviceController.Status}"
                        1
            with
            | ex ->
                printfn $"❌ Failed to stop service: {ex.Message}"
                1

    /// Restart the Windows service
    member self.RestartService() =
        printfn "🔄 Restarting TARS service..."
        let stopResult = self.StopService()
        if stopResult = 0 then
            System.Threading.Thread.Sleep(2000) // Wait a moment
            self.StartService()
        else
            stopResult

    /// Show service status
    member self.ShowStatus() =
        printfn "📊 TARS Windows Service Status"
        printfn "═════════════════════════════"
        printfn ""

        if not (self.IsServiceInstalled()) then
            printfn "❌ TARS service is not installed."
            printfn "   Install with: tars service install"
            1
        else
            try
                use serviceController = new ServiceController(serviceName)
                serviceController.Refresh()

                printfn $"🤖 Service Name: {serviceController.ServiceName}"
                printfn $"📝 Display Name: {serviceController.DisplayName}"
                printfn $"🎯 Status: {serviceController.Status}"
                printfn $"🔧 Start Type: {serviceController.StartType}"
                printfn $"⚙️ Can Stop: {serviceController.CanStop}"
                printfn $"⏸️ Can Pause: {serviceController.CanPauseAndContinue}"

                // Show additional info based on status
                match serviceController.Status with
                | ServiceControllerStatus.Running ->
                    printfn ""
                    printfn "✅ Service is running normally"
                    printfn "🎯 Available commands: stop, restart, status"
                | ServiceControllerStatus.Stopped ->
                    printfn ""
                    printfn "⏹️ Service is stopped"
                    printfn "🎯 Available commands: start, restart, status"
                | _ ->
                    printfn ""
                    printfn $"⚠️ Service is in {serviceController.Status} state"

                0
            with
            | ex ->
                printfn $"❌ Error getting service status: {ex.Message}"
                1

    /// Execute service command
    member self.Execute(args: string[]) =
        match args with
        | [||] | [| "help" |] ->
            self.ShowHelp()
            0
        | [| "install" |] ->
            self.InstallService(false)
        | [| "install"; "--force" |] | [| "install"; "-f" |] ->
            self.InstallService(true)
        | [| "uninstall" |] ->
            self.UninstallService()
        | [| "start" |] ->
            self.StartService()
        | [| "stop" |] ->
            self.StopService()
        | [| "restart" |] ->
            self.RestartService()
        | [| "status" |] ->
            self.ShowStatus()
        | _ ->
            printfn "❌ Unknown service command. Use 'tars service help' for available commands."
            1

    /// Show help for service commands
    member self.ShowHelp() =
        printfn "🤖 TARS Windows Service Management"
        printfn "═════════════════════════════════"
        printfn ""
        printfn "USAGE:"
        printfn "  tars service <command> [options]"
        printfn ""
        printfn "COMMANDS:"
        printfn "  install [--force]  Install TARS as a Windows service"
        printfn "  uninstall          Uninstall TARS Windows service"
        printfn "  start              Start the TARS service"
        printfn "  stop               Stop the TARS service"
        printfn "  restart            Restart the TARS service"
        printfn "  status             Show service status"
        printfn "  help               Show this help"
        printfn ""
        printfn "OPTIONS:"
        printfn "  --force, -f        Force reinstall if service already exists"
        printfn ""
        printfn "EXAMPLES:"
        printfn "  tars service install           # Install TARS service"
        printfn "  tars service install --force   # Reinstall TARS service"
        printfn "  tars service start             # Start the service"
        printfn "  tars service status            # Check service status"
        printfn "  tars service uninstall         # Remove the service"
        printfn ""
        printfn "NOTE: Service installation requires Administrator privileges."

    // ICommand interface implementation
    interface ICommand with
        member _.Name = "service"

        member _.Description = "Manage TARS Windows Service installation and operation"

        member self.Usage = "tars service <command> [options]"

        member self.Examples = [
            "tars service install           # Install TARS as Windows service"
            "tars service install --force   # Reinstall TARS service"
            "tars service start             # Start the service"
            "tars service stop              # Stop the service"
            "tars service restart           # Restart the service"
            "tars service status            # Show service status"
            "tars service uninstall         # Remove the service"
        ]

        member self.ValidateOptions(options: CommandOptions) =
            // Basic validation - service commands don't need complex validation
            true

        member self.ExecuteAsync(options: CommandOptions) = task {
            let args = options.Arguments |> List.toArray
            let exitCode = self.Execute(args)

            let result =
                if exitCode = 0 then
                    CommandResult.success "Service command completed successfully"
                else
                    CommandResult.failure "Service command failed"

            return result
        }
