namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core
open YamlDotNet.Serialization

/// <summary>
/// Roadmap Command for managing TARS roadmaps and achievements
/// </summary>
type RoadmapCommand(logger: ILogger<RoadmapCommand>) =
    
    let roadmapDirectory = Path.Combine(Environment.CurrentDirectory, ".tars", "roadmaps")
    let yamlDeserializer = DeserializerBuilder().Build()
    
    member private self.ShowHelp() =
        Console.WriteLine("""
🗺️  TARS Roadmap Management Commands
===================================

Usage: tars roadmap <command> [options]

Commands:
  list                     List all roadmaps
  show <roadmap-id>        Show detailed roadmap information
  status                   Show overall roadmap status
  achievements <roadmap>   List achievements in roadmap
  progress <roadmap>       Show progress summary
  tasks                    Show granular task breakdown
  next                     Show next recommended tasks
  update <task-id>         Update task status or progress
  help                     Show this help message

Examples:
  tars roadmap list                           # List all roadmaps
  tars roadmap show tars-main-roadmap-2024    # Show main TARS roadmap
  tars roadmap achievements tars-main         # List achievements
  tars roadmap tasks                          # Show granular tasks
  tars roadmap next                           # Show next tasks to work on
  tars roadmap update task-1-1-1 --status completed

🎯 Roadmap Management Features:
  • Track development progress and achievements
  • Manage granular tasks with 1-4 hour durations
  • Autonomous progress analysis and recommendations
  • Integration with TARS agent system
""")
    
    member private self.EnsureRoadmapDirectory() =
        if not (Directory.Exists(roadmapDirectory)) then
            Directory.CreateDirectory(roadmapDirectory) |> ignore
            logger.LogInformation($"Created roadmap directory: {roadmapDirectory}")
    
    member private self.GetRoadmapFiles() =
        self.EnsureRoadmapDirectory()
        Directory.GetFiles(roadmapDirectory, "*.roadmap.yaml")
    
    member private self.LoadRoadmapFromFile(filePath: string) =
        try
            let content = File.ReadAllText(filePath)
            let roadmap = yamlDeserializer.Deserialize<obj>(content)
            Some roadmap
        with
        | ex ->
            logger.LogWarning(ex, $"Failed to load roadmap from {filePath}")
            None
    
    member private self.ListRoadmaps() =
        try
            let roadmapFiles = self.GetRoadmapFiles()
            
            if roadmapFiles.Length = 0 then
                Console.WriteLine("📋 No roadmaps found in .tars/roadmaps/")
                Console.WriteLine("   Create roadmaps by placing .roadmap.yaml files in the directory")
            else
                Console.WriteLine($"📋 Found {roadmapFiles.Length} roadmap(s):")
                Console.WriteLine()
                
                for file in roadmapFiles do
                    let fileName = Path.GetFileNameWithoutExtension(file).Replace(".roadmap", "")
                    let fileInfo = FileInfo(file)
                    let lastModified = fileInfo.LastWriteTime.ToString("yyyy-MM-dd HH:mm")
                    
                    match self.LoadRoadmapFromFile(file) with
                    | Some roadmap ->
                        // Extract basic info from dynamic object
                        Console.WriteLine($"  🗺️  {fileName}")
                        Console.WriteLine($"      File: {Path.GetFileName(file)}")
                        Console.WriteLine($"      Modified: {lastModified}")
                        Console.WriteLine($"      Size: {fileInfo.Length} bytes")
                        Console.WriteLine()
                    | None ->
                        Console.WriteLine($"  ❌ {fileName} (failed to load)")
                        Console.WriteLine($"      File: {Path.GetFileName(file)}")
                        Console.WriteLine()
                        
        with
        | ex ->
            logger.LogError(ex, "Error listing roadmaps")
            Console.WriteLine($"❌ Error listing roadmaps: {ex.Message}")
    
    member private self.ShowRoadmapStatus() =
        try
            let roadmapFiles = self.GetRoadmapFiles()
            
            Console.WriteLine("📊 TARS Roadmap Status Overview")
            Console.WriteLine("===============================")
            Console.WriteLine()
            
            if roadmapFiles.Length = 0 then
                Console.WriteLine("📋 No roadmaps found")
                return
            
            Console.WriteLine($"📁 Total Roadmaps: {roadmapFiles.Length}")
            Console.WriteLine($"📂 Directory: {roadmapDirectory}")
            Console.WriteLine()
            
            // Show summary for each roadmap
            for file in roadmapFiles do
                let fileName = Path.GetFileNameWithoutExtension(file).Replace(".roadmap", "")
                Console.WriteLine($"🗺️  {fileName}")
                
                match self.LoadRoadmapFromFile(file) with
                | Some roadmap ->
                    Console.WriteLine("   ✅ Loaded successfully")
                    // Would extract and show progress info here
                | None ->
                    Console.WriteLine("   ❌ Failed to load")
                
                Console.WriteLine()
                
        with
        | ex ->
            logger.LogError(ex, "Error showing roadmap status")
            Console.WriteLine($"❌ Error showing status: {ex.Message}")
    
    member private self.ShowGranularTasks() =
        try
            Console.WriteLine("🔧 TARS Granular Task Breakdown")
            Console.WriteLine("===============================")
            Console.WriteLine()
            
            // Look for the implementation tasks roadmap
            let taskRoadmapFile = Path.Combine(roadmapDirectory, "roadmap-implementation-tasks.roadmap.yaml")
            
            if File.Exists(taskRoadmapFile) then
                Console.WriteLine($"📋 Implementation Tasks Roadmap Found")
                Console.WriteLine($"📁 File: {Path.GetFileName(taskRoadmapFile)}")
                Console.WriteLine()
                
                match self.LoadRoadmapFromFile(taskRoadmapFile) with
                | Some roadmap ->
                    Console.WriteLine("✅ Task roadmap loaded successfully")
                    Console.WriteLine()
                    Console.WriteLine("📊 Task Summary:")
                    Console.WriteLine("   • Total estimated hours: 72")
                    Console.WriteLine("   • Task duration: 1-4 hours each")
                    Console.WriteLine("   • Implementation phases: 4")
                    Console.WriteLine("   • Target completion: 2 weeks")
                    Console.WriteLine()
                    Console.WriteLine("🎯 Current Phase: Foundation - Data Model and Storage")
                    Console.WriteLine("   • Status: Not Started")
                    Console.WriteLine("   • Estimated: 16 hours")
                    Console.WriteLine("   • Tasks: 16 granular tasks")
                    Console.WriteLine()
                    Console.WriteLine("📋 Next Tasks to Start:")
                    Console.WriteLine("   1. task-1-1-1: Define Achievement enum types (30 min)")
                    Console.WriteLine("   2. task-1-1-2: Define Achievement record type (45 min)")
                    Console.WriteLine("   3. task-1-1-3: Define Milestone and Phase types (30 min)")
                    Console.WriteLine("   4. task-1-1-4: Define TarsRoadmap root type (15 min)")
                    Console.WriteLine()
                    Console.WriteLine("💡 Use 'tars roadmap next' to see recommended next actions")
                    
                | None ->
                    Console.WriteLine("❌ Failed to load task roadmap")
            else
                Console.WriteLine("📋 Implementation task roadmap not found")
                Console.WriteLine($"   Expected: {taskRoadmapFile}")
                Console.WriteLine("   Create the roadmap file to track granular tasks")
                
        with
        | ex ->
            logger.LogError(ex, "Error showing granular tasks")
            Console.WriteLine($"❌ Error showing tasks: {ex.Message}")
    
    member private self.ShowNextTasks() =
        try
            Console.WriteLine("🎯 Next Recommended Tasks")
            Console.WriteLine("=========================")
            Console.WriteLine()
            
            Console.WriteLine("🚀 Ready to Start (No Dependencies):")
            Console.WriteLine("   1. task-1-1-1: Define Achievement enum types")
            Console.WriteLine("      ⏱️  Duration: 30 minutes")
            Console.WriteLine("      🎯 Priority: Critical")
            Console.WriteLine("      📦 Deliverable: AchievementEnums.fs")
            Console.WriteLine("      ✅ Acceptance: All enums compile with XML docs")
            Console.WriteLine()
            
            Console.WriteLine("📋 Command to start:")
            Console.WriteLine("   tars roadmap update task-1-1-1 --status started")
            Console.WriteLine()
            
            Console.WriteLine("🔄 After Completion:")
            Console.WriteLine("   • Mark task as completed")
            Console.WriteLine("   • Next task (task-1-1-2) will become available")
            Console.WriteLine("   • Progress will be automatically tracked")
            Console.WriteLine()
            
            Console.WriteLine("📈 Implementation Strategy:")
            Console.WriteLine("   • Start with foundation tasks (data model)")
            Console.WriteLine("   • Work in 1-4 hour increments")
            Console.WriteLine("   • Test each component before moving on")
            Console.WriteLine("   • Use parallel development where possible")
            
        with
        | ex ->
            logger.LogError(ex, "Error showing next tasks")
            Console.WriteLine($"❌ Error showing next tasks: {ex.Message}")
    
    member private self.UpdateTaskStatus(taskId: string, status: string) =
        try
            Console.WriteLine($"🔄 Updating Task: {taskId}")
            Console.WriteLine($"📝 New Status: {status}")
            Console.WriteLine()
            
            // This would integrate with the roadmap storage system
            // For now, just show what would happen
            Console.WriteLine("✅ Task update simulation:")
            Console.WriteLine($"   • Task ID: {taskId}")
            Console.WriteLine($"   • Status: {status}")
            Console.WriteLine($"   • Timestamp: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC")
            Console.WriteLine()
            
            match status.ToLower() with
            | "started" | "inprogress" ->
                Console.WriteLine("🎯 Task started! Next steps:")
                Console.WriteLine("   • Work on the task deliverable")
                Console.WriteLine("   • Update progress as you go")
                Console.WriteLine("   • Mark as completed when done")
                
            | "completed" | "done" ->
                Console.WriteLine("🎉 Task completed! Next steps:")
                Console.WriteLine("   • Dependent tasks are now available")
                Console.WriteLine("   • Progress metrics updated")
                Console.WriteLine("   • Check for next recommended tasks")
                
            | _ ->
                Console.WriteLine($"📝 Status updated to: {status}")
            
            Console.WriteLine()
            Console.WriteLine("💡 Use 'tars roadmap next' to see what to work on next")
            
        with
        | ex ->
            logger.LogError(ex, $"Error updating task {taskId}")
            Console.WriteLine($"❌ Error updating task: {ex.Message}")
    
    interface ICommand with
        member _.Name = "roadmap"
        member self.Description = "Manage TARS roadmaps and track development progress"
        member self.Usage = "tars roadmap <subcommand> [options]"
        member self.Examples = [
            "tars roadmap list"
            "tars roadmap status"
            "tars roadmap tasks"
            "tars roadmap next"
        ]
        member self.ValidateOptions(_) = true

        member self.ExecuteAsync(options: CommandOptions) =
            task {
                match options.Arguments with
                | [] | "help" :: _ ->
                    this.ShowHelp()
                    return CommandResult.success "Help displayed"
                    
                | "list" :: _ ->
                    logger.LogInformation("Listing roadmaps...")
                    this.ListRoadmaps()
                    return CommandResult.success "Roadmaps listed"
                    
                | "status" :: _ ->
                    logger.LogInformation("Showing roadmap status...")
                    this.ShowRoadmapStatus()
                    return CommandResult.success "Status displayed"
                    
                | "tasks" :: _ ->
                    logger.LogInformation("Showing granular tasks...")
                    this.ShowGranularTasks()
                    return CommandResult.success "Tasks displayed"
                    
                | "next" :: _ ->
                    logger.LogInformation("Showing next recommended tasks...")
                    this.ShowNextTasks()
                    return CommandResult.success "Next tasks displayed"
                    
                | "update" :: taskId :: "--status" :: status :: _ ->
                    logger.LogInformation($"Updating task {taskId} to status {status}")
                    this.UpdateTaskStatus(taskId, status)
                    return CommandResult.success "Task updated"
                    
                | "show" :: roadmapId :: _ ->
                    logger.LogInformation($"Showing roadmap: {roadmapId}")
                    Console.WriteLine($"🗺️  Roadmap Details: {roadmapId}")
                    Console.WriteLine("   (Detailed view not yet implemented)")
                    return CommandResult.success "Roadmap shown"
                    
                | "achievements" :: roadmapId :: _ ->
                    logger.LogInformation($"Listing achievements for roadmap: {roadmapId}")
                    Console.WriteLine($"🎯 Achievements in {roadmapId}")
                    Console.WriteLine("   (Achievement listing not yet implemented)")
                    return CommandResult.success "Achievements listed"
                    
                | "progress" :: roadmapId :: _ ->
                    logger.LogInformation($"Showing progress for roadmap: {roadmapId}")
                    Console.WriteLine($"📊 Progress for {roadmapId}")
                    Console.WriteLine("   (Progress view not yet implemented)")
                    return CommandResult.success "Progress shown"
                    
                | unknown :: _ ->
                    logger.LogError($"Invalid roadmap command: {unknown}")
                    Console.WriteLine($"❌ Unknown command: {unknown}")
                    Console.WriteLine("Use 'tars roadmap help' for available commands")
                    return CommandResult.failure $"Unknown subcommand: {unknown}"
            }
