namespace TarsEngine.FSharp.Core

open System
open System.IO
open System.Reflection
open Microsoft.Extensions.Logging

/// Simple component information
type SimpleComponentInfo = {
    Name: string
    Type: string
    FilePath: string
    LineCount: int
    HasTests: bool
}

/// Simple, working component discovery without syntax issues
type SimpleComponentDiscovery(logger: ILogger<SimpleComponentDiscovery>) =
    
    /// Discover F# files in directory
    member this.DiscoverFSharpFiles(directory: string) =
        try
            let fsFiles = Directory.GetFiles(directory, "*.fs", SearchOption.AllDirectories)
            let components = ResizeArray<SimpleComponentInfo>()
            
            for file in fsFiles do
                try
                    let content = File.ReadAllText(file)
                    let lines = content.Split([|'\n'|], StringSplitOptions.None)
                    let lineCount = lines.Length
                    
                    // Check for test indicators
                    let hasTests = 
                        content.Contains("[<Test>]") || 
                        content.Contains("[<Fact>]") || 
                        content.Contains("Test") ||
                        file.Contains("Test")
                    
                    let comp = {
                        Name = Path.GetFileNameWithoutExtension(file)
                        Type = "F# File"
                        FilePath = file
                        LineCount = lineCount
                        HasTests = hasTests
                    }
                    components.Add(comp)
                        
                with
                | ex ->
                    logger.LogWarning(ex, sprintf "Failed to analyze F# file: %s" file)
            
            logger.LogInformation(sprintf "Discovered %d F# components" components.Count)
            components.ToArray()
        with
        | ex ->
            logger.LogError(ex, sprintf "Failed to discover F# components in directory: %s" directory)
            [||]
    
    /// Discover project files
    member this.DiscoverProjectFiles(directory: string) =
        try
            let projectFiles = Directory.GetFiles(directory, "*.fsproj", SearchOption.AllDirectories)
            let components = ResizeArray<SimpleComponentInfo>()
            
            for projectFile in projectFiles do
                try
                    let content = File.ReadAllText(projectFile)
                    let projectName = Path.GetFileNameWithoutExtension(projectFile)
                    
                    let comp = {
                        Name = projectName
                        Type = "F# Project"
                        FilePath = projectFile
                        LineCount = content.Split('\n').Length
                        HasTests = projectName.Contains("Test") || content.Contains("xunit") || content.Contains("NUnit")
                    }
                    components.Add(comp)
                with
                | ex ->
                    logger.LogWarning(ex, sprintf "Failed to analyze project file: %s" projectFile)
            
            logger.LogInformation(sprintf "Discovered %d project components" components.Count)
            components.ToArray()
        with
        | ex ->
            logger.LogError(ex, sprintf "Failed to discover project components in directory: %s" directory)
            [||]
    
    /// Discover loaded assemblies (simplified)
    member this.DiscoverLoadedAssemblies() =
        try
            let assemblies = AppDomain.CurrentDomain.GetAssemblies()
            let components = ResizeArray<SimpleComponentInfo>()
            
            for assembly in assemblies do
                try
                    let assemblyName = assembly.GetName().Name
                    
                    let comp = {
                        Name = assemblyName
                        Type = "Assembly"
                        FilePath = assembly.Location
                        LineCount = 0
                        HasTests = assemblyName.Contains("Test")
                    }
                    components.Add(comp)
                with
                | ex ->
                    logger.LogDebug(ex, sprintf "Could not analyze assembly: %s" (assembly.GetName().Name))
            
            logger.LogInformation(sprintf "Discovered %d assembly components" components.Count)
            components.ToArray()
        with
        | ex ->
            logger.LogError(ex, "Failed to discover loaded assemblies")
            [||]
    
    /// Get comprehensive component discovery
    member this.DiscoverAllComponents(rootDirectory: string) =
        try
            let fsharpComponents = this.DiscoverFSharpFiles(rootDirectory)
            let projectComponents = this.DiscoverProjectFiles(rootDirectory)
            let assemblyComponents = this.DiscoverLoadedAssemblies()
            
            let allComponents = Array.concat [fsharpComponents; projectComponents; assemblyComponents]
            
            // Remove duplicates based on name
            let uniqueComponents = 
                allComponents
                |> Array.groupBy (fun c -> c.Name)
                |> Array.map (fun (_, group) -> group.[0])
            
            logger.LogInformation(sprintf "Total unique components discovered: %d" uniqueComponents.Length)
            
            {|
                TotalComponents = uniqueComponents.Length
                FSharpComponents = fsharpComponents.Length
                ProjectComponents = projectComponents.Length
                AssemblyComponents = assemblyComponents.Length
                ComponentsWithTests = uniqueComponents |> Array.filter (fun c -> c.HasTests) |> Array.length
                Components = uniqueComponents
            |}
        with
        | ex ->
            logger.LogError(ex, "Failed to discover all components")
            {|
                TotalComponents = 0
                FSharpComponents = 0
                ProjectComponents = 0
                AssemblyComponents = 0
                ComponentsWithTests = 0
                Components = [||]
            |}
    
    /// Health check for component discovery
    member this.HealthCheck(rootDirectory: string) =
        try
            let discovery = this.DiscoverAllComponents(rootDirectory)
            
            let isHealthy = 
                discovery.TotalComponents > 0 &&
                discovery.FSharpComponents > 0
            
            logger.LogInformation(sprintf "Component discovery health check: %s" (if isHealthy then "PASS" else "FAIL"))
            
            {|
                IsHealthy = isHealthy
                ComponentCount = discovery.TotalComponents
                Details = sprintf "F#: %d, Projects: %d, Assemblies: %d" discovery.FSharpComponents discovery.ProjectComponents discovery.AssemblyComponents
                Issues = [
                    if discovery.TotalComponents = 0 then "No components discovered"
                    if discovery.FSharpComponents = 0 then "No F# components found"
                ]
            |}
        with
        | ex ->
            logger.LogError(ex, "Component discovery health check failed")
            {|
                IsHealthy = false
                ComponentCount = 0
                Details = "Health check failed"
                Issues = ["Exception during health check: " + ex.Message]
            |}
