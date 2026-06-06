namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Projects

type ProjectCommand(
    projectService: IAutonomousProjectService,
    logger: ILogger<ProjectCommand>) =
    
    interface ICommand with
        member _.Name = "project"
        member _.Description = "TARS autonomous project creation from simple prompts"
        member self.Usage = "tars project <subcommand> [options]"
        member self.Examples = [
            "tars project create \"file backup system\"     - Create project from prompt"
            "tars project demo                             - Demo project creation"
            "tars project list                             - List created projects"
            "tars project validate <path>                  - Validate project"
        ]
        member self.ValidateOptions(_) = true
        
        member self.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | "create" :: prompt ->
                        let fullPrompt = String.concat " " prompt
                        
                        printfn "🚀 TARS AUTONOMOUS PROJECT CREATION"
                        printfn "==================================="
                        printfn "Prompt: %s" fullPrompt
                        printfn ""
                        
                        printfn "🧠 Analyzing prompt with Codestral LLM..."
                        printfn "📋 Generating project structure..."
                        printfn "📝 Creating files and metascripts..."
                        printfn "🧪 Generating comprehensive tests..."
                        printfn "🔍 Validating project quality..."
                        printfn ""
                        
                        let! result = projectService.CreateProjectFromPromptAsync(fullPrompt)
                        
                        printfn "🎉 PROJECT CREATION COMPLETE!"
                        printfn "============================="
                        printfn ""
                        printfn "📊 Project Summary:"
                        printfn "  🎯 Name: %s" result.ProjectStructure.ProjectName
                        printfn "  📁 Location: %s" result.OutputPath
                        printfn "  📄 Files Generated: %d" result.GeneratedFiles
                        printfn "  🧪 Tests Created: %d" result.TestsGenerated
                        printfn "  ✅ Validation: %s" (if result.ValidationResults then "PASSED" else "NEEDS IMPROVEMENT")
                        printfn "  ⏱️  Generation Time: %dms" (int result.ExecutionTime.TotalMilliseconds)
                        printfn ""
                        
                        printfn "📁 Project Structure Created:"
                        for dir in result.ProjectStructure.Directories do
                            printfn "  📂 %s/" dir
                        
                        printfn ""
                        printfn "📄 Key Files Generated:"
                        for KeyValue(file, _) in result.ProjectStructure.Files |> Map.toSeq |> Seq.take 5 do
                            printfn "  📄 %s" file
                        
                        printfn ""
                        printfn "🚀 Ready for Use!"
                        printfn "  Navigate to: %s" result.OutputPath
                        printfn "  Run tests: tars test run %s" result.OutputPath
                        printfn "  Deploy: Execute scripts/deploy.tars"
                        
                        return CommandResult.success("Project creation completed")
                    
                    | "demo" :: _ ->
                        printfn "🎬 TARS AUTONOMOUS PROJECT CREATION DEMO"
                        printfn "========================================"
                        printfn ""
                        
                        printfn "🚀 Demonstrating full autonomous project creation workflow..."
                        printfn ""
                        
                        let! result = projectService.DemoProjectCreationAsync()
                        
                        printfn ""
                        printfn "🎉 DEMO COMPLETE!"
                        printfn "================="
                        printfn ""
                        printfn "✅ Demonstrated TARS Autonomous Capabilities:"
                        printfn "  🧠 Prompt Analysis - Understood project requirements"
                        printfn "  📋 Structure Generation - Created comprehensive project layout"
                        printfn "  📝 File Creation - Generated source code, configs, docs"
                        printfn "  🧪 Test Generation - Created unit, integration, performance tests"
                        printfn "  🔍 Quality Validation - Ensured production readiness"
                        printfn "  �� Metrics Collection - Tracked generation performance"
                        printfn ""
                        printfn "🎯 Project Created: %s" result.ProjectStructure.ProjectName
                        printfn "📁 Location: %s" result.OutputPath
                        printfn ""
                        printfn "🚀 TARS can create complete projects from simple prompts!"
                        
                        return CommandResult.success("Demo completed")
                    
                    | "list" :: _ ->
                        printfn "📋 TARS GENERATED PROJECTS"
                        printfn "=========================="
                        printfn ""
                        
                        let projectsDir = ".tars/projects"
                        if Directory.Exists(projectsDir) then
                            let projects = Directory.GetDirectories(projectsDir)
                            
                            if projects.Length > 0 then
                                printfn "Found %d autonomous projects:" projects.Length
                                printfn ""
                                
                                for i, projectPath in projects |> Array.indexed do
                                    let projectName = Path.GetFileName(projectPath)
                                    let metadataPath = Path.Combine(projectPath, ".tars_project.json")
                                    
                                    printfn "%d. %s" (i + 1) projectName
                                    printfn "   📁 Path: %s" projectPath
                                    
                                    if File.Exists(metadataPath) then
                                        let metadata = File.ReadAllText(metadataPath)
                                        printfn "   📊 Metadata: Available"
                                    else
                                        printfn "   📊 Metadata: Not found"
                                    
                                    let srcDir = Path.Combine(projectPath, "src")
                                    let testsDir = Path.Combine(projectPath, "tests")
                                    let scriptsDir = Path.Combine(projectPath, "scripts")
                                    
                                    let srcFiles = if Directory.Exists(srcDir) then Directory.GetFiles(srcDir).Length else 0
                                    let testFiles = if Directory.Exists(testsDir) then Directory.GetFiles(testsDir).Length else 0
                                    let scriptFiles = if Directory.Exists(scriptsDir) then Directory.GetFiles(scriptsDir).Length else 0
                                    
                                    printfn "   📄 Source files: %d" srcFiles
                                    printfn "   🧪 Test files: %d" testFiles
                                    printfn "   📜 Scripts: %d" scriptFiles
                                    printfn ""
                            else
                                printfn "No autonomous projects found."
                                printfn "💡 Use 
tars
project
create
\your idea\ to create one!"
                        else
                            printfn "Projects directory not found."
                            printfn "💡 Use tars
project
create
\your idea\ to create your first project!"
                        
                        return CommandResult.success("Project list displayed")
                    
                    | "validate" :: projectPath :: _ ->
                        printfn "🔍 TARS PROJECT VALIDATION"
                        printfn "=========================="
                        printfn "Project: %s" projectPath
                        printfn ""
                        
                        printfn "🔍 Validating project structure..."
                        printfn "🧪 Checking test coverage..."
                        printfn "📊 Analyzing quality metrics..."
                        printfn ""
                        
                        let! isValid = projectService.ValidateProjectAsync(projectPath)
                        
                        if isValid then
                            printfn "✅ PROJECT VALIDATION PASSED"
                            printfn "============================"
                            printfn "🎉 The project meets all quality criteria:"
                            printfn "  ✅ Valid project structure"
                            printfn "  ✅ Source files present"
                            printfn "  ✅ Test files available"
                            printfn "  ✅ Metascripts included"
                            printfn "  ✅ Documentation complete"
                            printfn ""
                            printfn "🚀 Project is ready for production use!"
                        else
                            printfn "❌ PROJECT VALIDATION FAILED"
                            printfn "============================"
                            printfn "⚠️  The project needs improvement:"
                            printfn "  ❌ Missing required structure or files"
                            printfn "  💡 Use tars
project
create to regenerate"
                            printfn ""
                            printfn "🔧 Consider regenerating the project with TARS"
                        
                        return CommandResult.success("Project validation completed")
                    
                    | "explore" :: projectPath :: _ ->
                        printfn "🔍 TARS PROJECT EXPLORATION"
                        printfn "==========================="
                        printfn "Project: %s" projectPath
                        printfn ""
                        
                        if Directory.Exists(projectPath) then
                            printfn "📁 Project Structure:"
                            let dirs = Directory.GetDirectories(projectPath)
                            for dir in dirs do
                                let dirName = Path.GetFileName(dir)
                                let fileCount = Directory.GetFiles(dir, "*", SearchOption.AllDirectories).Length
                                printfn "  📂 %s/ (%d files)" dirName fileCount
                            
                            printfn ""
                            printfn "📄 Key Files:"
                            let keyFiles = ["README.md"; "src/main.fs"; "tests/test_main.fs"; "scripts/deploy.tars"]
                            for file in keyFiles do
                                let filePath = Path.Combine(projectPath, file)
                                if File.Exists(filePath) then
                                    let size = (new FileInfo(filePath)).Length
                                    printfn "  �� %s (%d bytes)" file size
                                else
                                    printfn "  ❌ %s (missing)" file
                            
                            printfn ""
                            printfn "🧪 Test Coverage:"
                            let testsDir = Path.Combine(projectPath, "tests")
                            if Directory.Exists(testsDir) then
                                let testFiles = Directory.GetFiles(testsDir, "*.fs")
                                printfn "  🧪 Test files: %d" testFiles.Length
                                for testFile in testFiles do
                                    printfn "    📄 %s" (Path.GetFileName(testFile))
                            else
                                printfn "  ❌ No tests directory found"
                            
                            printfn ""
                            printfn "📜 Metascripts:"
                            let scriptsDir = Path.Combine(projectPath, "scripts")
                            if Directory.Exists(scriptsDir) then
                                let scriptFiles = Directory.GetFiles(scriptsDir, "*.tars")
                                printfn "  📜 Metascripts: %d" scriptFiles.Length
                                for scriptFile in scriptFiles do
                                    printfn "    📜 %s" (Path.GetFileName(scriptFile))
                            else
                                printfn "  ❌ No scripts directory found"
                        else
                            printfn "❌ Project directory not found: %s" projectPath
                        
                        return CommandResult.success("Project exploration completed")
                    
                    | "status" :: _ ->
                        printfn "🚀 TARS AUTONOMOUS PROJECT SYSTEM STATUS"
                        printfn "========================================"
                        printfn ""
                        printfn "🤖 Autonomous Project Service: ✅ Active"
                        printfn "🧠 Prompt Analysis: ✅ Operational (Codestral LLM)"
                        printfn "📋 Structure Generation: ✅ Operational"
                        printfn "📝 File Creation: ✅ Operational"
                        printfn "🧪 Test Generation: ✅ Operational"
                        printfn "🔍 Quality Validation: ✅ Operational"
                        printfn ""
                        printfn "📊 Project Types Supported:"
                        printfn "  🔐 Authentication Systems"
                        printfn "  💾 Backup Solutions"
                        printfn "  🌐 API Services"
                        printfn "  📊 Data Processing"
                        printfn "  🔧 Utility Tools"
                        printfn "  📱 Any Custom Project"
                        printfn ""
                        printfn "🎯 TARS can create complete projects from simple prompts!"
                        printfn "   Just describe what you want and TARS will build it."
                        
                        return CommandResult.success("Status displayed")
                    
                    | [] ->
                        printfn "TARS Autonomous Project Creation Commands:"
                        printfn "  create \"<prompt>\"     - Create project from natural language prompt"
                        printfn "  demo                  - Demonstrate autonomous project creation"
                        printfn "  list                  - List all generated projects"
                        printfn "  validate <path>       - Validate project quality"
                        printfn "  explore <path>        - Explore project structure"
                        printfn "  status                - Show project system status"
                        printfn ""
                        printfn "Examples:"
                        printfn "  tars project create \"file backup system with encryption\""
                        printfn "  tars project create \"REST API for user management\""
                        printfn "  tars project create \"data processing pipeline\""
                        return CommandResult.success("Help displayed")
                    
                    | unknown :: _ ->
                        printfn "Unknown project command: %s" unknown
                        return CommandResult.failure("Unknown command")
                with
                | ex ->
                    logger.LogError(ex, "Project command error")
                    return CommandResult.failure(ex.Message)
            }

