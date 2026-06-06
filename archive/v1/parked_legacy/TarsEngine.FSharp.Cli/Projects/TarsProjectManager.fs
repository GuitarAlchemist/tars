namespace TarsEngine.FSharp.Cli.Projects

open System
open System.IO
open System.Text.Json
open System.Collections.Generic

// ============================================================================
// TARS PROJECT MANAGEMENT SYSTEM - REAL .tars DIRECTORY INTEGRATION
// ============================================================================

type ProjectType =
    | WebApplication
    | ConsoleApplication
    | Library
    | Metascript
    | Documentation
    | Research
    | Demo
    | Unknown

type ProjectStatus =
    | Active
    | Completed
    | InProgress
    | Archived
    | Experimental

type TarsProject = {
    Name: string
    Path: string
    Type: ProjectType
    Status: ProjectStatus
    Description: string
    CreatedDate: DateTime
    LastModified: DateTime
    Technologies: string list
    HasReadme: bool
    HasPackageJson: bool
    HasFsproj: bool
    HasCsproj: bool
    HasDockerfile: bool
    HasTarsConfig: bool
    FileCount: int
    SizeBytes: int64
    MetascriptFiles: string list
    DemoFiles: string list
}

type ProjectCategory = {
    Name: string
    Projects: TarsProject list
    Count: int
}

type TarsProjectManager() =
    let tarsDirectory = ".tars"
    let projectsDirectory = Path.Combine(tarsDirectory, "projects")
    
    member this.GetTarsRootPath() =
        let currentDir = Directory.GetCurrentDirectory()
        let rec findTarsRoot (dir: string) =
            if Directory.Exists(Path.Combine(dir, tarsDirectory)) then
                Some dir
            else
                let parent = Directory.GetParent(dir)
                if parent = null then None
                else findTarsRoot parent.FullName
        
        findTarsRoot currentDir
    
    member this.GetProjectsDirectory() =
        match this.GetTarsRootPath() with
        | Some root -> Path.Combine(root, projectsDirectory)
        | None -> projectsDirectory
    
    member private this.DetectProjectType(projectPath: string) =
        let files = Directory.GetFiles(projectPath, "*", SearchOption.AllDirectories)
        let fileNames = files |> Array.map Path.GetFileName |> Set.ofArray
        
        if fileNames.Contains("package.json") && fileNames.Contains("index.html") then WebApplication
        elif fileNames.Contains("package.json") then WebApplication
        elif fileNames.Contains("Program.fs") || fileNames.Contains("Program.cs") then ConsoleApplication
        elif files |> Array.exists (fun f -> f.EndsWith(".fsproj") || f.EndsWith(".csproj")) then Library
        elif files |> Array.exists (fun f -> f.EndsWith(".trsx") || f.EndsWith(".flux")) then Metascript
        elif fileNames.Contains("README.md") && not (fileNames.Contains("package.json")) then Documentation
        elif projectPath.Contains("research") || projectPath.Contains("demo") then Research
        else Unknown
    
    member private this.DetectProjectStatus(projectPath: string) =
        let lastWrite = Directory.GetLastWriteTime(projectPath)
        let daysSinceModified = (DateTime.Now - lastWrite).TotalDays
        
        if daysSinceModified < 7.0 then Active
        elif daysSinceModified < 30.0 then InProgress
        elif projectPath.Contains("archive") || projectPath.Contains("backup") then Archived
        elif projectPath.Contains("experimental") || projectPath.Contains("demo") then Experimental
        else Completed
    
    member private this.DetectTechnologies(projectPath: string) =
        let files = Directory.GetFiles(projectPath, "*", SearchOption.AllDirectories)
        let extensions = files |> Array.map Path.GetExtension |> Set.ofArray
        let technologies = ResizeArray<string>()
        
        if extensions.Contains(".fs") || extensions.Contains(".fsx") then technologies.Add("F#")
        if extensions.Contains(".cs") then technologies.Add("C#")
        if extensions.Contains(".js") || extensions.Contains(".ts") then technologies.Add("JavaScript/TypeScript")
        if extensions.Contains(".py") then technologies.Add("Python")
        if extensions.Contains(".java") then technologies.Add("Java")
        if extensions.Contains(".html") then technologies.Add("HTML")
        if extensions.Contains(".css") then technologies.Add("CSS")
        if extensions.Contains(".json") then technologies.Add("JSON")
        if extensions.Contains(".yml") || extensions.Contains(".yaml") then technologies.Add("YAML")
        if extensions.Contains(".md") then technologies.Add("Markdown")
        if extensions.Contains(".trsx") then technologies.Add("TARS Metascript")
        if extensions.Contains(".flux") then technologies.Add("FLUX Language")
        if extensions.Contains(".tars") then technologies.Add("TARS Config")
        if File.Exists(Path.Combine(projectPath, "Dockerfile")) then technologies.Add("Docker")
        if File.Exists(Path.Combine(projectPath, "package.json")) then technologies.Add("Node.js")
        
        technologies |> List.ofSeq
    
    member private this.GetProjectSize(projectPath: string) =
        try
            let files = Directory.GetFiles(projectPath, "*", SearchOption.AllDirectories)
            files |> Array.sumBy (fun f -> FileInfo(f).Length)
        with
        | _ -> 0L
    
    member private this.GetMetascriptFiles(projectPath: string) =
        try
            Directory.GetFiles(projectPath, "*.trsx", SearchOption.AllDirectories)
            |> Array.append (Directory.GetFiles(projectPath, "*.flux", SearchOption.AllDirectories))
            |> Array.map (fun f -> Path.GetRelativePath(projectPath, f))
            |> List.ofArray
        with
        | _ -> []
    
    member private this.GetDemoFiles(projectPath: string) =
        try
            Directory.GetFiles(projectPath, "*demo*", SearchOption.AllDirectories)
            |> Array.append (Directory.GetFiles(projectPath, "*run*", SearchOption.AllDirectories))
            |> Array.filter (fun f -> 
                let ext = Path.GetExtension(f).ToLower()
                ext = ".cmd" || ext = ".ps1" || ext = ".sh" || ext = ".html")
            |> Array.map (fun f -> Path.GetRelativePath(projectPath, f))
            |> List.ofArray
        with
        | _ -> []
    
    member private this.AnalyzeProject(projectPath: string) =
        let projectName = Path.GetFileName(projectPath)
        let files = Directory.GetFiles(projectPath, "*", SearchOption.AllDirectories)
        
        {
            Name = projectName
            Path = projectPath
            Type = this.DetectProjectType(projectPath)
            Status = this.DetectProjectStatus(projectPath)
            Description = this.GetProjectDescription(projectPath)
            CreatedDate = Directory.GetCreationTime(projectPath)
            LastModified = Directory.GetLastWriteTime(projectPath)
            Technologies = this.DetectTechnologies(projectPath)
            HasReadme = File.Exists(Path.Combine(projectPath, "README.md"))
            HasPackageJson = File.Exists(Path.Combine(projectPath, "package.json"))
            HasFsproj = files |> Array.exists (fun f -> f.EndsWith(".fsproj"))
            HasCsproj = files |> Array.exists (fun f -> f.EndsWith(".csproj"))
            HasDockerfile = File.Exists(Path.Combine(projectPath, "Dockerfile"))
            HasTarsConfig = File.Exists(Path.Combine(projectPath, "tars.yaml")) || File.Exists(Path.Combine(projectPath, ".tars"))
            FileCount = files.Length
            SizeBytes = this.GetProjectSize(projectPath)
            MetascriptFiles = this.GetMetascriptFiles(projectPath)
            DemoFiles = this.GetDemoFiles(projectPath)
        }
    
    member private this.GetProjectDescription(projectPath: string) =
        let readmePath = Path.Combine(projectPath, "README.md")
        if File.Exists(readmePath) then
            try
                let content = File.ReadAllText(readmePath)
                let lines = content.Split('\n')
                // Get first non-empty line that's not a title
                lines 
                |> Array.skip 1
                |> Array.tryFind (fun line -> not (String.IsNullOrWhiteSpace(line)) && not (line.StartsWith("#")))
                |> Option.defaultValue "No description available"
                |> fun desc -> if desc.Length > 100 then desc.Substring(0, 100) + "..." else desc
            with
            | _ -> "No description available"
        else
            "No description available"
    
    member this.GetAllProjects() =
        let projectsDir = this.GetProjectsDirectory()
        if Directory.Exists(projectsDir) then
            Directory.GetDirectories(projectsDir)
            |> Array.filter (fun dir -> not (Path.GetFileName(dir).EndsWith(".zip")))
            |> Array.map this.AnalyzeProject
            |> List.ofArray
        else
            []
    
    member this.GetProjectsByCategory() =
        let projects = this.GetAllProjects()
        
        [
            { Name = "Web Applications"; Projects = projects |> List.filter (fun p -> p.Type = WebApplication); Count = 0 }
            { Name = "Console Applications"; Projects = projects |> List.filter (fun p -> p.Type = ConsoleApplication); Count = 0 }
            { Name = "Libraries"; Projects = projects |> List.filter (fun p -> p.Type = Library); Count = 0 }
            { Name = "Metascripts"; Projects = projects |> List.filter (fun p -> p.Type = Metascript); Count = 0 }
            { Name = "Documentation"; Projects = projects |> List.filter (fun p -> p.Type = Documentation); Count = 0 }
            { Name = "Research & Demos"; Projects = projects |> List.filter (fun p -> p.Type = Research || p.Type = Demo); Count = 0 }
            { Name = "Experimental"; Projects = projects |> List.filter (fun p -> p.Status = Experimental); Count = 0 }
        ]
        |> List.map (fun cat -> { cat with Count = cat.Projects.Length })
        |> List.filter (fun cat -> cat.Count > 0)
    
    member this.GetProjectByName(name: string) =
        this.GetAllProjects() |> List.tryFind (fun p -> p.Name = name)
    
    member this.GetProjectStats() =
        let projects = this.GetAllProjects()
        let totalSize = projects |> List.sumBy (fun p -> p.SizeBytes)
        let totalFiles = projects |> List.sumBy (fun p -> p.FileCount)
        let technologies = projects |> List.collect (fun p -> p.Technologies) |> List.distinct
        
        {|
            TotalProjects = projects.Length
            TotalSizeBytes = totalSize
            TotalFiles = totalFiles
            Technologies = technologies
            ActiveProjects = projects |> List.filter (fun p -> p.Status = Active) |> List.length
            CompletedProjects = projects |> List.filter (fun p -> p.Status = Completed) |> List.length
            ExperimentalProjects = projects |> List.filter (fun p -> p.Status = Experimental) |> List.length
        |}
    
    member this.SearchProjects(query: string) =
        let projects = this.GetAllProjects()
        let lowerQuery = query.ToLower()
        
        projects
        |> List.filter (fun p ->
            p.Name.ToLower().Contains(lowerQuery) ||
            p.Description.ToLower().Contains(lowerQuery) ||
            p.Technologies |> List.exists (fun t -> t.ToLower().Contains(lowerQuery)))
    
    member this.GetRecentProjects(count: int) =
        this.GetAllProjects()
        |> List.sortByDescending (fun p -> p.LastModified)
        |> List.take (min count (this.GetAllProjects().Length))
