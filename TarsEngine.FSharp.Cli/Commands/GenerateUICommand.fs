namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Core
open TarsEngine.FSharp.Cli.UI.TarsElmishGenerator

// ======================================
// TARS AI-Driven UI Generation Command
// ======================================

type GenerateUICommand() =
    
    /// Generate Elmish UI from .trsx or .flux files
    member this.GenerateUIFromFile(inputPath: string, outputPath: string) =
        task {
            try
                if not (File.Exists(inputPath)) then
                    return CommandResult.failure($"Input file not found: {inputPath}")
                else
                    let! content = File.ReadAllTextAsync(inputPath)
                    
                    // Extract UI block from .trsx file
                    let uiBlock = 
                        if inputPath.EndsWith(".trsx") then
                            extractUiBlockFromTrsx content
                        else
                            Some content // Assume entire file is UI DSL
                    
                    match uiBlock with
                    | Some dslContent ->
                        let generatedCode = processUiDsl dslContent
                        
                        // Ensure output directory exists
                        let outputDir = Path.GetDirectoryName(outputPath)
                        if not (String.IsNullOrEmpty(outputDir)) && not (Directory.Exists(outputDir)) then
                            Directory.CreateDirectory(outputDir) |> ignore
                        
                        do! File.WriteAllTextAsync(outputPath, generatedCode)
                        
                        printfn "🎨 AI-Generated Elmish UI Components:"
                        printfn $"   📁 Input:  {inputPath}"
                        printfn $"   📄 Output: {outputPath}"
                        printfn $"   📏 Size:   {generatedCode.Length} characters"
                        printfn "✅ UI generation completed successfully!"
                        
                        return CommandResult.success($"Generated Elmish UI: {outputPath}")
                    
                    | None ->
                        return CommandResult.failure("No UI block found in input file. Expected 'ui { ... }' block.")

            with
            | ex ->
                return CommandResult.failure($"UI generation failed: {ex.Message}")
        }
    
    /// Generate UI from DSL string directly
    member this.GenerateUIFromDsl(dslContent: string, outputPath: string) =
        task {
            try
                let generatedCode = processUiDsl dslContent
                
                // Ensure output directory exists
                let outputDir = Path.GetDirectoryName(outputPath)
                if not (String.IsNullOrEmpty(outputDir)) && not (Directory.Exists(outputDir)) then
                    Directory.CreateDirectory(outputDir) |> ignore
                
                do! File.WriteAllTextAsync(outputPath, generatedCode)
                
                printfn "🎨 AI-Generated Elmish UI from DSL:"
                printfn $"   📄 Output: {outputPath}"
                printfn $"   📏 Size:   {generatedCode.Length} characters"
                printfn "✅ UI generation completed successfully!"
                
                return CommandResult.success($"Generated Elmish UI: {outputPath}")
            
            with
            | ex ->
                return CommandResult.failure($"UI generation failed: {ex.Message}")
        }
    
    /// Generate multiple UI components from a directory
    member this.GenerateUIFromDirectory(inputDir: string, outputDir: string) =
        task {
            try
                if not (Directory.Exists(inputDir)) then
                    return CommandResult.failure($"Input directory not found: {inputDir}")
                else
                    let trsxFiles = Directory.GetFiles(inputDir, "*.trsx", SearchOption.AllDirectories)
                    let fluxFiles = Directory.GetFiles(inputDir, "*.flux", SearchOption.AllDirectories)
                    let allFiles = Array.concat [trsxFiles; fluxFiles]
                    
                    if allFiles.Length = 0 then
                        return CommandResult.failure("No .trsx or .flux files found in input directory")
                    else
                        let mutable successCount = 0
                        let mutable errorCount = 0
                        
                        for inputFile in allFiles do
                            let fileName = Path.GetFileNameWithoutExtension(inputFile)
                            let outputFile = Path.Combine(outputDir, $"{fileName}.fs")
                            
                            let! result = this.GenerateUIFromFile(inputFile, outputFile)
                            if result.Success then
                                successCount <- successCount + 1
                            else
                                errorCount <- errorCount + 1
                                printfn $"❌ Failed to generate UI for {inputFile}: {result.Message}"
                        
                        printfn ""
                        printfn $"📊 UI Generation Summary:"
                        printfn $"   ✅ Successful: {successCount}"
                        printfn $"   ❌ Failed:     {errorCount}"
                        printfn $"   📁 Output:     {outputDir}"
                        
                        return CommandResult.success($"Generated {successCount} UI components")
            
            with
            | ex ->
                return CommandResult.failure($"Batch UI generation failed: {ex.Message}")
        }
    
    /// Create a sample UI DSL file for demonstration
    member this.CreateSampleUIFile(outputPath: string) =
        task {
            try
                let sampleDsl = """ui {
  view_id: "TarsAgentDashboard"
  title: "TARS Agent Activity Dashboard"
  feedback_enabled: true
  real_time_updates: true
  
  header "TARS Agent Monitoring System"
  
  metrics_panel bind(cognitiveMetrics)
  
  thought_flow bind(thoughtPatterns)
  
  table bind(agentRows)
  
  button "Refresh Data" on refreshClicked
  
  line_chart bind(agentPerformance)
  
  threejs bind(agent3DVisualization)
  
  chat_panel bind(agentCommunication)
  
  projects_panel bind(activeProjects)
  
  diagnostics_panel bind(systemDiagnostics)
}"""
                
                // Ensure output directory exists
                let outputDir = Path.GetDirectoryName(outputPath)
                if not (String.IsNullOrEmpty(outputDir)) && not (Directory.Exists(outputDir)) then
                    Directory.CreateDirectory(outputDir) |> ignore
                
                do! File.WriteAllTextAsync(outputPath, sampleDsl)
                
                printfn "📝 Sample UI DSL file created:"
                printfn $"   📄 File: {outputPath}"
                printfn "   🎯 Features:"
                printfn "      • Agent metrics panel"
                printfn "      • Thought flow visualization"
                printfn "      • Real-time data table"
                printfn "      • 3D agent visualization"
                printfn "      • AI chat panel"
                printfn "      • Projects panel"
                printfn "      • Diagnostics panel"
                printfn "      • Feedback system"
                printfn "      • Real-time updates"
                printfn ""
                printfn "💡 Generate Elmish code with:"
                printfn $"   tarscli generate-ui --input {outputPath} --output Views/TarsAgentDashboard.fs"
                
                return CommandResult.success($"Sample UI DSL created: {outputPath}")
            
            with
            | ex ->
                return CommandResult.failure($"Sample creation failed: {ex.Message}")
        }
    
    /// Validate UI DSL syntax
    member this.ValidateUIDsl(inputPath: string) =
        task {
            try
                if not (File.Exists(inputPath)) then
                    return CommandResult.failure($"Input file not found: {inputPath}")
                else
                    let! content = File.ReadAllTextAsync(inputPath)
                    
                    let uiBlock = 
                        if inputPath.EndsWith(".trsx") then
                            extractUiBlockFromTrsx content
                        else
                            Some content
                    
                    match uiBlock with
                    | Some dslContent ->
                        try
                            let ast = parseDsl dslContent
                            
                            printfn "✅ UI DSL Validation Results:"
                            printfn $"   📄 File:       {inputPath}"
                            printfn $"   🎯 View ID:    {ast.ViewId}"
                            printfn $"   📝 Title:      {ast.Title}"
                            printfn $"   🔧 Components: {ast.Components.Length}"
                            printfn $"   💬 Feedback:   {ast.FeedbackEnabled}"
                            printfn $"   ⚡ Real-time:  {ast.RealTimeUpdates}"
                            printfn ""
                            printfn "🧩 Components:"
                            for i, comp in List.indexed ast.Components do
                                let compType = 
                                    match comp with
                                    | Header _ -> "Header"
                                    | Table _ -> "Table"
                                    | Button _ -> "Button"
                                    | LineChart _ -> "Line Chart"
                                    | BarChart _ -> "Bar Chart"
                                    | PieChart _ -> "Pie Chart"
                                    | ThreeJS _ -> "3D Scene"
                                    | VexFlow _ -> "Music Notation"
                                    | D3Visualization _ -> "D3 Visualization"
                                    | MetricsPanel _ -> "Metrics Panel"
                                    | ChatPanel _ -> "Chat Panel"
                                    | ThoughtFlow _ -> "Thought Flow"
                                    | ProjectsPanel _ -> "Projects Panel"
                                    | AgentTeams _ -> "Agent Teams"
                                    | DiagnosticsPanel _ -> "Diagnostics Panel"
                                    | HtmlRaw _ -> "Raw HTML"
                                printfn $"   {i + 1}. {compType}"
                            
                            return CommandResult.success("UI DSL validation passed")
                        
                        with
                        | ex ->
                            return CommandResult.failure($"UI DSL validation failed: {ex.Message}")

                    | None ->
                        return CommandResult.failure("No UI block found in input file")

            with
            | ex ->
                return CommandResult.failure($"Validation failed: {ex.Message}")
        }
    
    interface ICommand with
        member _.Name = "generate-ui"
        
        member _.Description = "Generate Elmish UI components from declarative DSL specifications"
        
        member _.Usage = "tars generate-ui [--input <file>] [--output <file>] [--sample] [--validate] [--batch]"
        
        member _.Examples = [
            "tars generate-ui --input agent_dashboard.trsx --output Views/AgentDashboard.fs"
            "tars generate-ui --sample --output sample_ui.trsx"
            "tars generate-ui --validate --input my_ui.trsx"
            "tars generate-ui --batch --input ui_specs/ --output Views/"
        ]
        
        member _.ValidateOptions(options) = true
        
        member _.ExecuteAsync(options) =
            task {
                let inputPath = options.Options.TryFind("input")
                let outputPath = options.Options.TryFind("output")
                let createSample = options.Options.ContainsKey("sample")
                let validateOnly = options.Options.ContainsKey("validate")
                let batchMode = options.Options.ContainsKey("batch")
                
                match createSample, validateOnly, batchMode with
                | true, _, _ ->
                    let samplePath = outputPath |> Option.defaultValue "sample_ui.trsx"
                    let command = GenerateUICommand()
                    return! command.CreateSampleUIFile(samplePath)

                | _, true, _ ->
                    match inputPath with
                    | Some input ->
                        let command = GenerateUICommand()
                        return! command.ValidateUIDsl(input)
                    | None -> return CommandResult.failure("--input required for validation")

                | _, _, true ->
                    match inputPath, outputPath with
                    | Some input, Some output ->
                        let command = GenerateUICommand()
                        return! command.GenerateUIFromDirectory(input, output)
                    | _ -> return CommandResult.failure("--input and --output required for batch mode")

                | _, _, _ ->
                    match inputPath, outputPath with
                    | Some input, Some output ->
                        let command = GenerateUICommand()
                        return! command.GenerateUIFromFile(input, output)
                    | _ -> return CommandResult.failure("--input and --output required")
            }
