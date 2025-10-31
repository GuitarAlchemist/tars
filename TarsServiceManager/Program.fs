open System
open System.IO
open System.Diagnostics
open System.ServiceProcess
open System.Security.Principal

/// Check if running as administrator
let isRunningAsAdmin() =
    let identity = WindowsIdentity.GetCurrent()
    let principal = WindowsPrincipal(identity)
    principal.IsInRole(WindowsBuiltInRole.Administrator)

/// Get the Windows service executable path
let getServiceExecutablePath() =
    let currentDir = Directory.GetCurrentDirectory()
    let servicePath = Path.Combine(currentDir, "TarsEngine.FSharp.WindowsService", "bin", "Release", "net9.0", "TarsEngine.FSharp.WindowsService.exe")
    
    if File.Exists(servicePath) then
        Some servicePath
    else
        // Try debug path
        let debugPath = Path.Combine(currentDir, "TarsEngine.FSharp.WindowsService", "bin", "Debug", "net9.0", "TarsEngine.FSharp.WindowsService.exe")
        if File.Exists(debugPath) then Some debugPath else None

/// Check if service is installed
let isServiceInstalled() =
    try
        use serviceController = new ServiceController("TarsService")
        serviceController.Refresh()
        true
    with
    | :? InvalidOperationException -> false
    | _ -> false

/// Install the Windows service
let installService(force: bool) =
    printfn "🤖 TARS Windows Service Installation"
    printfn "═══════════════════════════════════"
    printfn ""
    
    if not (isRunningAsAdmin()) then
        printfn "❌ Administrator privileges required!"
        printfn "   Please run as Administrator to install the service."
        1
    else
        match getServiceExecutablePath() with
        | None ->
            printfn "❌ Service executable not found!"
            printfn "   Please build the Windows service first:"
            printfn "   dotnet build TarsEngine.FSharp.WindowsService --configuration Release"
            1
        | Some executablePath ->
            printfn $"📁 Service executable: {executablePath}"
            
            if isServiceInstalled() && not force then
                printfn "⚠️ TARS service is already installed!"
                printfn "   Use 'tars service install --force' to reinstall"
                1
            else
                try
                    if isServiceInstalled() && force then
                        printfn "🔄 Reinstalling existing service..."
                        let deleteArgs = "delete \"TarsService\""
                        let deleteProcess = Process.Start("sc.exe", deleteArgs)
                        deleteProcess.WaitForExit()
                        System.Threading.// REAL: Implement actual logic here
                    
                    printfn "🔧 Installing TARS Windows Service..."
                    let createArgs = $"create \"TarsService\" binPath= \"\"{executablePath}\"\" DisplayName= \"TARS Autonomous Development Platform\" start= auto"
                    let createProcess = Process.Start("sc.exe", createArgs)
                    createProcess.WaitForExit()
                    
                    if createProcess.ExitCode = 0 then
                        printfn "✅ TARS Windows Service installed successfully!"
                        printfn ""
                        printfn "🎯 Service Management Commands:"
                        printfn "   Start:     tars service start"
                        printfn "   Stop:      tars service stop"
                        printfn "   Status:    tars service status"
                        printfn "   Uninstall: tars service uninstall"
                        0
                    else
                        printfn $"❌ Failed to install service. Exit code: {createProcess.ExitCode}"
                        1
                with
                | ex ->
                    printfn $"❌ Error installing service: {ex.Message}"
                    1

/// Uninstall the Windows service
let uninstallService() =
    printfn "🗑️ TARS Windows Service Uninstallation"
    printfn "════════════════════════════════════"
    printfn ""
    
    if not (isRunningAsAdmin()) then
        printfn "❌ Administrator privileges required!"
        1
    else
        if not (isServiceInstalled()) then
            printfn "ℹ️ TARS service is not installed."
            0
        else
            try
                // Stop service if running
                use serviceController = new ServiceController("TarsService")
                if serviceController.Status = ServiceControllerStatus.Running then
                    printfn "⏹️ Stopping TARS service..."
                    serviceController.Stop()
                    serviceController.WaitForStatus(ServiceControllerStatus.Stopped, TimeSpan.FromSeconds(30.0))
                
                let deleteArgs = "delete \"TarsService\""
                let deleteProcess = Process.Start("sc.exe", deleteArgs)
                deleteProcess.WaitForExit()
                
                if deleteProcess.ExitCode = 0 then
                    printfn "✅ TARS Windows Service uninstalled successfully!"
                    0
                else
                    printfn "❌ Failed to uninstall TARS service."
                    1
            with
            | ex ->
                printfn $"❌ Error uninstalling service: {ex.Message}"
                1

/// Start the Windows service
let startService() =
    if not (isServiceInstalled()) then
        printfn "❌ TARS service is not installed. Install it first with 'tars service install'"
        1
    else
        try
            use serviceController = new ServiceController("TarsService")
            serviceController.Refresh()
            
            if serviceController.Status = ServiceControllerStatus.Running then
                printfn "ℹ️ TARS service is already running."
                0
            else
                printfn "🚀 Starting TARS service..."
                serviceController.Start()
                serviceController.WaitForStatus(ServiceControllerStatus.Running, TimeSpan.FromSeconds(30.0))
                printfn "✅ TARS service started successfully!"
                0
        with
        | ex ->
            printfn $"❌ Failed to start service: {ex.Message}"
            1

/// Stop the Windows service
let stopService() =
    if not (isServiceInstalled()) then
        printfn "❌ TARS service is not installed."
        1
    else
        try
            use serviceController = new ServiceController("TarsService")
            serviceController.Refresh()
            
            if serviceController.Status = ServiceControllerStatus.Stopped then
                printfn "ℹ️ TARS service is already stopped."
                0
            else
                printfn "⏹️ Stopping TARS service..."
                serviceController.Stop()
                serviceController.WaitForStatus(ServiceControllerStatus.Stopped, TimeSpan.FromSeconds(30.0))
                printfn "✅ TARS service stopped successfully!"
                0
        with
        | ex ->
            printfn $"❌ Failed to stop service: {ex.Message}"
            1

/// Show service status
let showStatus() =
    printfn "📊 TARS Windows Service Status"
    printfn "═════════════════════════════"
    printfn ""
    
    if not (isServiceInstalled()) then
        printfn "❌ TARS service is not installed."
        printfn "   Install with: tars service install"
        1
    else
        try
            use serviceController = new ServiceController("TarsService")
            serviceController.Refresh()
            
            printfn $"🤖 Service Name: {serviceController.ServiceName}"
            printfn $"📝 Display Name: {serviceController.DisplayName}"
            printfn $"🎯 Status: {serviceController.Status}"
            printfn $"🔧 Start Type: {serviceController.StartType}"
            printfn ""
            
            match serviceController.Status with
            | ServiceControllerStatus.Running ->
                printfn "✅ Service is running normally"
            | ServiceControllerStatus.Stopped ->
                printfn "⏹️ Service is stopped"
            | _ ->
                printfn $"⚠️ Service is in {serviceController.Status} state"
            
            0
        with
        | ex ->
            printfn $"❌ Error getting service status: {ex.Message}"
            1

/// Show help
let showHelp() =
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
    printfn "  status             Show service status"
    printfn "  help               Show this help"
    printfn ""
    printfn "EXAMPLES:"
    printfn "  tars service install           # Install TARS service"
    printfn "  tars service install --force   # Reinstall TARS service"
    printfn "  tars service start             # Start the service"
    printfn "  tars service status            # Check service status"
    printfn ""
    printfn "NOTE: Service installation requires Administrator privileges."

/// Main entry point
[<EntryPoint>]
let main args =
    match args with
    | [||] | [| "help" |] ->
        showHelp()
        0
    | [| "service"; "install" |] ->
        installService(false)
    | [| "service"; "install"; "--force" |] ->
        installService(true)
    | [| "service"; "uninstall" |] ->
        uninstallService()
    | [| "service"; "start" |] ->
        startService()
    | [| "service"; "stop" |] ->
        stopService()
    | [| "service"; "status" |] ->
        showStatus()
    | [| "service"; "help" |] ->
        showHelp()
        0
    | _ ->
        printfn "❌ Unknown command. Use 'tars help' for available commands."
        1
