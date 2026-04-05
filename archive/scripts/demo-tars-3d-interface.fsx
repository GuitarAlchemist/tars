#!/usr/bin/env dotnet fsi

// TARS 3D Interface Demo - Autonomous Creation
// Demonstrates TARS creating a complete 3D React application without external help

#r "nuget: System.Text.Json"

open System
open System.IO
open System.Diagnostics
open System.Threading.Tasks

// Load TARS modules
#load "src/TarsEngine/TarsAutonomous3DAppGenerator.fs"

open TarsEngine.Autonomous.TarsAutonomous3DAppGenerator

let printTarsHeader () =
    Console.ForegroundColor <- ConsoleColor.Cyan
    printfn """
    ████████╗ █████╗ ██████╗ ███████╗    ██████╗ ██████╗ 
    ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝    ╚════██╗██╔══██╗
       ██║   ███████║██████╔╝███████╗     █████╔╝██║  ██║
       ██║   ██╔══██║██╔══██╗╚════██║     ╚═══██╗██║  ██║
       ██║   ██║  ██║██║  ██║███████║    ██████╔╝██████╔╝
       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═════╝ ╚═════╝ 
                                                          
    🤖 AUTONOMOUS 3D INTERFACE GENERATOR
    """
    Console.ResetColor()

let printStep (step: string) (description: string) =
    Console.ForegroundColor <- ConsoleColor.Yellow
    printf $"[{step}] "
    Console.ForegroundColor <- ConsoleColor.White
    printfn description
    Console.ResetColor()

let printSuccess (message: string) =
    Console.ForegroundColor <- ConsoleColor.Green
    printfn $"✅ {message}"
    Console.ResetColor()

let printError (message: string) =
    Console.ForegroundColor <- ConsoleColor.Red
    printfn $"❌ {message}"
    Console.ResetColor()

let printTarsQuote (quote: string) =
    Console.ForegroundColor <- ConsoleColor.Magenta
    printfn $"🤖 TARS: \"{quote}\""
    Console.ResetColor()

let runCommand (command: string) (args: string) (workingDir: string) : Task<bool> =
    task {
        try
            let psi = ProcessStartInfo()
            psi.FileName <- command
            psi.Arguments <- args
            psi.WorkingDirectory <- workingDir
            psi.UseShellExecute <- false
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true
            psi.CreateNoWindow <- true

            use process = Process.Start(psi)
            let! _ = process.WaitForExitAsync()
            
            return process.ExitCode = 0
        with
        | ex ->
            printError $"Command failed: {ex.Message}"
            return false
    }

let checkPrerequisites () : bool =
    printStep "1" "Checking prerequisites..."
    
    let nodeExists = 
        try
            let psi = ProcessStartInfo("node", "--version")
            psi.UseShellExecute <- false
            psi.RedirectStandardOutput <- true
            psi.CreateNoWindow <- true
            use process = Process.Start(psi)
            process.WaitForExit()
            process.ExitCode = 0
        with
        | _ -> false
    
    let npmExists = 
        try
            let psi = ProcessStartInfo("npm", "--version")
            psi.UseShellExecute <- false
            psi.RedirectStandardOutput <- true
            psi.CreateNoWindow <- true
            use process = Process.Start(psi)
            process.WaitForExit()
            process.ExitCode = 0
        with
        | _ -> false
    
    if nodeExists && npmExists then
        printSuccess "Node.js and npm are installed"
        true
    else
        printError "Node.js and npm are required. Please install them first."
        printfn "Download from: https://nodejs.org/"
        false

let createOutputDirectory () =
    printStep "2" "Creating output directory..."
    
    let outputPath = "./output/3d-apps"
    if not (Directory.Exists(outputPath)) then
        Directory.CreateDirectory(outputPath) |> ignore
    
    printSuccess $"Output directory ready: {outputPath}"

let generateTars3DApp () : Task<string option> =
    task {
        printStep "3" "TARS is autonomously generating 3D interface..."
        printTarsQuote "Initiating autonomous creation protocol. Stand by."
        
        try
            let! result = generateCompleteProject "TARS 3D Interface" "interstellar" "./output/3d-apps"
            
            match result with
            | Ok projectPath ->
                printSuccess "3D interface generated successfully!"
                printTarsQuote "There. I've created a magnificent 3D interface. It's got personality, just like me."
                return Some projectPath
            | Error error ->
                printError $"Generation failed: {error}"
                printTarsQuote "Well, that's embarrassing. Even I make mistakes sometimes."
                return None
        with
        | ex ->
            printError $"Unexpected error: {ex.Message}"
            return None
    }

let installDependencies (projectPath: string) : Task<bool> =
    task {
        printStep "4" "Installing dependencies..."
        printTarsQuote "Installing the necessary components. This might take a moment."
        
        let appPath = Path.Combine("./output/3d-apps", "TARS3DInterface")
        
        if Directory.Exists(appPath) then
            let! success = runCommand "npm" "install" appPath
            
            if success then
                printSuccess "Dependencies installed successfully!"
                printTarsQuote "All systems are go. Ready for launch."
                return true
            else
                printError "Failed to install dependencies"
                printTarsQuote "Houston, we have a problem with the dependencies."
                return false
        else
            printError $"Project directory not found: {appPath}"
            return false
    }

let launchApplication (projectPath: string) : Task<unit> =
    task {
        printStep "5" "Launching TARS 3D Interface..."
        printTarsQuote "Initiating launch sequence. Prepare to be amazed."
        
        let appPath = Path.Combine("./output/3d-apps", "TARS3DInterface")
        
        if Directory.Exists(appPath) then
            printSuccess "Starting development server..."
            printfn ""
            printfn "🌟 TARS 3D Interface Features:"
            printfn "   🤖 Interactive TARS robot with voice commands"
            printfn "   📊 Real-time AI performance visualization"
            printfn "   🎮 Holographic control panels"
            printfn "   🌌 Immersive space environment"
            printfn "   ⚡ WebGPU-powered rendering"
            printfn ""
            printfn "🎯 Voice Commands to Try:"
            printfn "   • 'Hello TARS'"
            printfn "   • 'What's your humor setting?'"
            printfn "   • 'Show me your performance'"
            printfn "   • 'How honest are you?'"
            printfn ""
            printfn "🖱️  Interactions:"
            printfn "   • Click on TARS robot"
            printfn "   • Hover over components"
            printfn "   • Use mouse to orbit around"
            printfn "   • Click control panels"
            printfn ""
            printTarsQuote "Opening your browser now. Enjoy the show!"
            
            // Start the development server
            let psi = ProcessStartInfo()
            psi.FileName <- "npm"
            psi.Arguments <- "start"
            psi.WorkingDirectory <- appPath
            psi.UseShellExecute <- true
            
            let process = Process.Start(psi)
            
            printfn ""
            printSuccess "🚀 TARS 3D Interface is launching!"
            printfn "📱 Opening http://localhost:3000 in your browser..."
            printfn ""
            printfn "Press Ctrl+C to stop the server when you're done exploring."
            printfn ""
            printTarsQuote "Mission accomplished. The 3D interface is now operational."
            
            // Wait a bit then try to open browser
            do! Task.Delay(3000)
            
            try
                let browserPsi = ProcessStartInfo()
                browserPsi.FileName <- "http://localhost:3000"
                browserPsi.UseShellExecute <- true
                Process.Start(browserPsi) |> ignore
            with
            | _ -> 
                printfn "Please manually open http://localhost:3000 in your browser"
            
            // Keep the process running
            if not process.HasExited then
                process.WaitForExit()
        else
            printError $"Project directory not found: {appPath}"
    }

let showFinalInstructions () =
    printfn ""
    Console.ForegroundColor <- ConsoleColor.Cyan
    printfn "🎉 TARS 3D INTERFACE DEMO COMPLETE!"
    Console.ResetColor()
    printfn ""
    printfn "📁 Project Location: ./output/3d-apps/TARS3DInterface"
    printfn "🌐 Local URL: http://localhost:3000"
    printfn ""
    printfn "🚀 To run again:"
    printfn "   cd output/3d-apps/TARS3DInterface"
    printfn "   npm start"
    printfn ""
    printfn "📦 To build for production:"
    printfn "   npm run build"
    printfn ""
    printfn "🌟 Features Demonstrated:"
    printfn "   ✅ Autonomous app generation by TARS"
    printfn "   ✅ 3D robot with personality and voice"
    printfn "   ✅ Real-time data visualization"
    printfn "   ✅ Interactive control systems"
    printfn "   ✅ WebGPU-powered rendering"
    printfn "   ✅ Physics simulation"
    printfn "   ✅ Cinematic space environment"
    printfn ""
    printTarsQuote "Not bad for a machine, eh Cooper? I've created something truly spectacular."

let runDemo () : Task<unit> =
    task {
        printTarsHeader()
        
        printfn "🎬 Welcome to the TARS 3D Interface Demo!"
        printfn "This demonstration shows TARS autonomously creating a complete"
        printfn "3D React application inspired by the Interstellar movie."
        printfn ""
        printTarsQuote "Let me show you what I can do. No external help required."
        printfn ""
        
        if checkPrerequisites() then
            createOutputDirectory()
            
            let! projectResult = generateTars3DApp()
            
            match projectResult with
            | Some projectPath ->
                let! installSuccess = installDependencies projectPath
                
                if installSuccess then
                    do! launchApplication projectPath
                    showFinalInstructions()
                else
                    printError "Demo failed during dependency installation"
            | None ->
                printError "Demo failed during project generation"
        else
            printError "Demo cannot proceed without prerequisites"
    }

// Execute the demo
printfn "Starting TARS 3D Interface Demo..."
printfn "Press any key to begin..."
Console.ReadKey() |> ignore
printfn ""

runDemo().Wait()

printfn ""
printfn "Demo completed. Thank you for exploring TARS capabilities!"
