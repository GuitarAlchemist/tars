namespace TarsEngine.FSharp.Agents

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// TARS Application Code Generator Agent
/// Generates actual executable F# application code for projects
/// </summary>
module ApplicationCodeGenerator =
    
    /// Application type to generate
    type ApplicationType =
        | WebAPI
        | ConsoleApp
        | WebApp
        | Microservice
        | Library
    
    /// Code generation template
    type CodeTemplate = {
        FileName: string
        Content: string
        Directory: string option
    }
    
    /// Generated application structure
    type GeneratedApplication = {
        ApplicationType: ApplicationType
        EntryPoint: string
        SourceFiles: CodeTemplate list
        ConfigFiles: CodeTemplate list
        ProjectUpdates: (string * string) list // (property, value) pairs
        Dependencies: string list
    }
    
    /// <summary>
    /// Application Code Generator Agent
    /// </summary>
    type ApplicationCodeGenerator(logger: ILogger<ApplicationCodeGenerator>) =
        
        /// <summary>
        /// Generate complete executable application code for a project
        /// </summary>
        member this.GenerateApplicationCode(projectPath: string, applicationType: ApplicationType, projectName: string) : Task<GeneratedApplication> =
            task {
                logger.LogInformation("Generating {ApplicationType} code for project {ProjectName}", applicationType, projectName)
                
                match applicationType with
                | WebAPI -> return! this.GenerateWebAPIApplication(projectPath, projectName)
                | ConsoleApp -> return! this.GenerateConsoleApplication(projectPath, projectName)
                | WebApp -> return! this.GenerateWebApplication(projectPath, projectName)
                | Microservice -> return! this.GenerateMicroserviceApplication(projectPath, projectName)
                | Library -> return! this.GenerateLibraryApplication(projectPath, projectName)
            }
        
        /// <summary>
        /// Generate Web API application
        /// </summary>
        member private this.GenerateWebAPIApplication(projectPath: string, projectName: string) : Task<GeneratedApplication> =
            task {
                let programFs = {
                    FileName = "Program.fs"
                    Directory = Some "src"
                    Content = $"""namespace {projectName}

open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Hosting
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Logging

module Program =
    
    [<EntryPoint>]
    let main args =
        let builder = WebApplication.CreateBuilder(args)
        
        // Add services
        builder.Services.AddControllers() |> ignore
        builder.Services.AddEndpointsApiExplorer() |> ignore
        builder.Services.AddSwaggerGen() |> ignore
        
        // Configure logging
        builder.Logging.AddConsole() |> ignore
        
        let app = builder.Build()
        
        // Configure pipeline
        if app.Environment.IsDevelopment() then
            app.UseSwagger() |> ignore
            app.UseSwaggerUI() |> ignore
        
        app.UseHttpsRedirection() |> ignore
        app.UseRouting() |> ignore
        app.MapControllers() |> ignore
        
        // Add health check endpoint
        app.MapGet("/health", fun () -> "{{\"status\": \"healthy\", \"timestamp\": \"{DateTime.UtcNow:yyyy-MM-ddTHH:mm:ssZ}\"}}")
        |> ignore
        
        // Add info endpoint
        app.MapGet("/", fun () -> "{{\"service\": \"{projectName}\", \"version\": \"1.0.0\", \"status\": \"running\"}}")
        |> ignore
        
        printfn "🚀 {projectName} API starting on http://localhost:5000"
        app.Run("http://0.0.0.0:5000")
        
        0
"""
                }
                
                let controllersFs = {
                    FileName = "Controllers.fs"
                    Directory = Some "src"
                    Content = $"""namespace {projectName}.Controllers

open Microsoft.AspNetCore.Mvc
open Microsoft.Extensions.Logging
open System

[<ApiController>]
[<Route("api/[controller]")>]
type ValuesController(logger: ILogger<ValuesController>) =
    inherit ControllerBase()
    
    [<HttpGet>]
    member this.Get() =
        logger.LogInformation("GET /api/values called")
        [| "value1"; "value2"; "value3" |]
    
    [<HttpGet("{{id}}")>]
    member this.Get(id: int) =
        logger.LogInformation("GET /api/values/{{Id}} called", id)
        $"Value {{id}}"
    
    [<HttpPost>]
    member this.Post([<FromBody>] value: string) =
        logger.LogInformation("POST /api/values called with: {{Value}}", value)
        this.Ok($"Created: {{value}}")

[<ApiController>]
[<Route("api/[controller]")>]
type HealthController(logger: ILogger<HealthController>) =
    inherit ControllerBase()
    
    [<HttpGet>]
    member this.Get() =
        logger.LogInformation("Health check requested")
        {{|
            Status = "Healthy"
            Timestamp = DateTime.UtcNow
            Service = "{projectName}"
            Version = "1.0.0"
        |}}
"""
                }
                
                let appSettingsJson = {
                    FileName = "appsettings.json"
                    Directory = None
                    Content = """{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  },
  "AllowedHosts": "*",
  "Kestrel": {
    "Endpoints": {
      "Http": {
        "Url": "http://0.0.0.0:5000"
      }
    }
  }
}"""
                }
                
                let projectUpdates = [
                    ("OutputType", "Exe")
                    ("TargetFramework", "net8.0")
                ]
                
                let dependencies = [
                    "Microsoft.AspNetCore.App"
                    "Swashbuckle.AspNetCore"
                ]
                
                return {
                    ApplicationType = WebAPI
                    EntryPoint = "src/Program.fs"
                    SourceFiles = [programFs; controllersFs]
                    ConfigFiles = [appSettingsJson]
                    ProjectUpdates = projectUpdates
                    Dependencies = dependencies
                }
            }
        
        /// <summary>
        /// Generate Console application
        /// </summary>
        member private this.GenerateConsoleApplication(projectPath: string, projectName: string) : Task<GeneratedApplication> =
            task {
                let programFs = {
                    FileName = "Program.fs"
                    Directory = Some "src"
                    Content = $"""namespace {projectName}

open System

module Program =
    
    let printWelcome() =
        printfn "🤖 Welcome to {projectName}!"
        printfn "Generated by TARS Application Code Generator"
        printfn "Timestamp: %s" (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"))
    
    let processArguments args =
        match args with
        | [||] -> 
            printfn "No arguments provided. Running in interactive mode."
            0
        | [| "--help" |] | [| "-h" |] ->
            printfn "Usage: {projectName} [options]"
            printfn "Options:"
            printfn "  --help, -h    Show this help message"
            printfn "  --version     Show version information"
            0
        | [| "--version" |] ->
            printfn "{projectName} v1.0.0"
            printfn "Built with F# and TARS"
            0
        | _ ->
            printfn "Processing arguments: %A" args
            printfn "Application logic would go here..."
            0
    
    [<EntryPoint>]
    let main args =
        try
            printWelcome()
            processArguments args
        with
        | ex ->
            printfn "Error: %s" ex.Message
            1
"""
                }
                
                return {
                    ApplicationType = ConsoleApp
                    EntryPoint = "src/Program.fs"
                    SourceFiles = [programFs]
                    ConfigFiles = []
                    ProjectUpdates = [("OutputType", "Exe"); ("TargetFramework", "net8.0")]
                    Dependencies = []
                }
            }
        
        /// <summary>
        /// Generate Web application
        /// </summary>
        member private this.GenerateWebApplication(projectPath: string, projectName: string) : Task<GeneratedApplication> =
            task {
                // Similar to WebAPI but with MVC views
                let programFs = {
                    FileName = "Program.fs"
                    Directory = Some "src"
                    Content = $"""namespace {projectName}

open Microsoft.AspNetCore.Builder
open Microsoft.AspNetCore.Hosting
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting

module Program =
    
    [<EntryPoint>]
    let main args =
        let builder = WebApplication.CreateBuilder(args)
        
        // Add services
        builder.Services.AddControllersWithViews() |> ignore
        
        let app = builder.Build()
        
        // Configure pipeline
        if not (app.Environment.IsDevelopment()) then
            app.UseExceptionHandler("/Home/Error") |> ignore
            app.UseHsts() |> ignore
        
        app.UseHttpsRedirection() |> ignore
        app.UseStaticFiles() |> ignore
        app.UseRouting() |> ignore
        
        app.MapControllerRoute(
            name = "default",
            pattern = "{{controller=Home}}/{{action=Index}}/{{id?}}"
        ) |> ignore
        
        printfn "🌐 {projectName} Web App starting on http://localhost:5000"
        app.Run("http://0.0.0.0:5000")
        
        0
"""
                }
                
                return {
                    ApplicationType = WebApp
                    EntryPoint = "src/Program.fs"
                    SourceFiles = [programFs]
                    ConfigFiles = []
                    ProjectUpdates = [("OutputType", "Exe"); ("TargetFramework", "net8.0")]
                    Dependencies = ["Microsoft.AspNetCore.App"]
                }
            }
        
        /// <summary>
        /// Generate Microservice application
        /// </summary>
        member private this.GenerateMicroserviceApplication(projectPath: string, projectName: string) : Task<GeneratedApplication> =
            task {
                // Enhanced WebAPI with microservice patterns
                return! this.GenerateWebAPIApplication(projectPath, projectName)
            }
        
        /// <summary>
        /// Generate Library application
        /// </summary>
        member private this.GenerateLibraryApplication(projectPath: string, projectName: string) : Task<GeneratedApplication> =
            task {
                let libraryFs = {
                    FileName = "Library.fs"
                    Directory = Some "src"
                    Content = $"""namespace {projectName}

/// <summary>
/// Main library module for {projectName}
/// Generated by TARS Application Code Generator
/// </summary>
module Library =
    
    /// <summary>
    /// Sample function that demonstrates the library functionality
    /// </summary>
    let hello name =
        $"Hello {{name}} from {projectName}!"
    
    /// <summary>
    /// Sample computation function
    /// </summary>
    let add x y = x + y
    
    /// <summary>
    /// Sample async function
    /// </summary>
    let asyncHello name = async {{
        do! Async.Sleep(100)
        return hello name
    }}
"""
                }
                
                return {
                    ApplicationType = Library
                    EntryPoint = "src/Library.fs"
                    SourceFiles = [libraryFs]
                    ConfigFiles = []
                    ProjectUpdates = [("OutputType", "Library"); ("TargetFramework", "net8.0")]
                    Dependencies = []
                }
            }
        
        /// <summary>
        /// Apply generated code to project
        /// </summary>
        member this.ApplyGeneratedCode(projectPath: string, generatedApp: GeneratedApplication) : Task<bool> =
            task {
                try
                    logger.LogInformation("Applying generated code to project at {ProjectPath}", projectPath)
                    
                    // Create source files
                    for sourceFile in generatedApp.SourceFiles do
                        let fullDir = 
                            match sourceFile.Directory with
                            | Some dir -> Path.Combine(projectPath, dir)
                            | None -> projectPath
                        
                        Directory.CreateDirectory(fullDir) |> ignore
                        let filePath = Path.Combine(fullDir, sourceFile.FileName)
                        
                        File.WriteAllText(filePath, sourceFile.Content)
                        logger.LogInformation("Created source file: {FilePath}", filePath)
                    
                    // Create config files
                    for configFile in generatedApp.ConfigFiles do
                        let fullDir = 
                            match configFile.Directory with
                            | Some dir -> Path.Combine(projectPath, dir)
                            | None -> projectPath
                        
                        Directory.CreateDirectory(fullDir) |> ignore
                        let filePath = Path.Combine(fullDir, configFile.FileName)
                        
                        File.WriteAllText(filePath, configFile.Content)
                        logger.LogInformation("Created config file: {FilePath}", filePath)
                    
                    // Update project file
                    let! projectUpdated = this.UpdateProjectFile(projectPath, generatedApp.ProjectUpdates, generatedApp.Dependencies)
                    
                    if projectUpdated then
                        logger.LogInformation("✅ Successfully applied generated code to project")
                        return true
                    else
                        logger.LogError("❌ Failed to update project file")
                        return false
                        
                with
                | ex ->
                    logger.LogError(ex, "❌ Error applying generated code")
                    return false
            }
        
        /// <summary>
        /// Update project file with necessary properties and dependencies
        /// </summary>
        member private this.UpdateProjectFile(projectPath: string, updates: (string * string) list, dependencies: string list) : Task<bool> =
            task {
                try
                    let projFiles = Directory.GetFiles(projectPath, "*.fsproj", SearchOption.AllDirectories)
                    
                    if projFiles.Length = 0 then
                        logger.LogWarning("No .fsproj file found in {ProjectPath}", projectPath)
                        return false
                    
                    let projFile = projFiles.[0]
                    let content = File.ReadAllText(projFile)
                    
                    // Simple XML manipulation (in production, use proper XML parsing)
                    let mutable updatedContent = content
                    
                    // Add OutputType if not present
                    if not (content.Contains("<OutputType>")) then
                        let outputType = updates |> List.tryFind (fun (k, _) -> k = "OutputType") |> Option.map snd |> Option.defaultValue "Exe"
                        updatedContent <- updatedContent.Replace("</PropertyGroup>", $"    <OutputType>{outputType}</OutputType>\n  </PropertyGroup>")
                    
                    // Add package references for dependencies
                    for dep in dependencies do
                        if not (content.Contains(dep)) then
                            let packageRef = $"""    <PackageReference Include="{dep}" />"""
                            if content.Contains("</ItemGroup>") then
                                updatedContent <- updatedContent.Replace("</ItemGroup>", $"{packageRef}\n  </ItemGroup>")
                            else
                                updatedContent <- updatedContent.Replace("</Project>", $"  <ItemGroup>\n{packageRef}\n  </ItemGroup>\n</Project>")
                    
                    File.WriteAllText(projFile, updatedContent)
                    logger.LogInformation("Updated project file: {ProjFile}", projFile)
                    
                    return true
                with
                | ex ->
                    logger.LogError(ex, "Error updating project file")
                    return false
            }
