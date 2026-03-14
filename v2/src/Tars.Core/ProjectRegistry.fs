/// Project Registry - Manages project lifecycle and persistence
module Tars.Core.ProjectRegistry

open System
open System.IO
open System.Text.Json
open Tars.Core.Project

// ============================================================================
// Registry
// ============================================================================

type ProjectRegistry() =
    let mutable projects: Map<string, Project> = Map.empty
    let mutable projectStates: Map<string, ProjectState> = Map.empty

    let jsonOptions =
        let opts =
            JsonSerializerOptions(WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

        opts.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
        opts

    // ========================================================================
    // Core Operations
    // ========================================================================

    /// Register a new project
    member _.Register(project: Project) : Result<unit, string> =
        if String.IsNullOrWhiteSpace(project.Id) then
            Result.Error "Project ID cannot be empty"
        elif String.IsNullOrWhiteSpace(project.Name) then
            Result.Error "Project Name cannot be empty"
        elif Map.containsKey project.Id projects then
            Result.Error $"Project '{project.Id}' already exists"
        else
            projects <- projects |> Map.add project.Id project
            projectStates <- projectStates |> Map.add project.Id (initProjectState project)
            Result.Ok()

    /// Get a project by ID
    member _.Get(id: string) : Project option = Map.tryFind id projects

    /// Get project state by ID
    member _.GetState(id: string) : ProjectState option = Map.tryFind id projectStates

    /// Update project state
    member _.UpdateState(id: string, state: ProjectState) : Result<unit, string> =
        if not (Map.containsKey id projects) then
            Result.Error $"Project '{id}' not found"
        else
            projectStates <- projectStates |> Map.add id state
            Result.Ok()

    /// List all projects
    member _.List() : Project list = projects |> Map.toList |> List.map snd

    /// Remove a project
    member _.Remove(id: string) : bool =
        if Map.containsKey id projects then
            projects <- projects |> Map.remove id
            projectStates <- projectStates |> Map.remove id
            true
        else
            false

    /// Check if project exists
    member _.Exists(id: string) : bool = Map.containsKey id projects

    /// Get count of projects
    member _.Count: int = Map.count projects

    // ========================================================================
    // Serialization
    // ========================================================================

    /// Save all projects to a directory
    member this.SaveToDirectory(dir: string) : Result<int, string> =
        try
            if not (Directory.Exists dir) then
                Directory.CreateDirectory dir |> ignore

            let mutable count = 0

            for project in this.List() do
                let path = Path.Combine(dir, $"{project.Id}.json")
                let json = JsonSerializer.Serialize(project, jsonOptions)
                File.WriteAllText(path, json)
                count <- count + 1

            Result.Ok count
        with ex ->
            Result.Error $"Failed to save projects: {ex.Message}"

    /// Load projects from a directory
    member this.LoadFromDirectory(dir: string) : Result<int, string> =
        try
            if not (Directory.Exists dir) then
                Result.Error $"Directory not found: {dir}"
            else
                let mutable count = 0

                for file in Directory.GetFiles(dir, "*.json") do
                    try
                        let json = File.ReadAllText file
                        let project = JsonSerializer.Deserialize<Project>(json, jsonOptions)

                        match this.Register project with
                        | Result.Ok() -> count <- count + 1
                        | Result.Error _ -> ()
                    with _ ->
                        ()

                Result.Ok count
        with ex ->
            Result.Error $"Failed to load projects: {ex.Message}"

// ============================================================================
// Storage Location
// ============================================================================

/// Get the default projects home directory
let getProjectsHome () =
    let tarsHome =
        match Environment.GetEnvironmentVariable("TARS_HOME") with
        | null
        | "" -> Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars")
        | path -> path

    Path.Combine(tarsHome, "projects")

/// Ensure projects directory exists
let ensureProjectsDir () =
    let dir = getProjectsHome ()

    if not (Directory.Exists dir) then
        Directory.CreateDirectory dir |> ignore

    dir

// ============================================================================
// Global Registry Instance with Auto-Persistence
// ============================================================================

/// Default global project registry with persistence
let defaultRegistry =
    let registry = ProjectRegistry()
    // Auto-load on startup
    let projectsDir = getProjectsHome ()

    if Directory.Exists projectsDir then
        registry.LoadFromDirectory(projectsDir) |> ignore

    registry

/// Save all projects to disk
let saveProjects () =
    let dir = ensureProjectsDir ()
    defaultRegistry.SaveToDirectory(dir)

/// Save a single project to disk
let saveProject (project: Project) =
    let dir = ensureProjectsDir ()
    let path = Path.Combine(dir, $"{project.Id}.json")

    let opts =
        JsonSerializerOptions(WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

    opts.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
    let json = JsonSerializer.Serialize(project, opts)
    File.WriteAllText(path, json)

/// Delete project file from disk
let deleteProjectFile (id: string) =
    let dir = getProjectsHome ()
    let path = Path.Combine(dir, $"{id}.json")

    if File.Exists path then
        File.Delete path

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create and register a new project (with auto-save)
let createAndRegister id name rootPath template mode =
    let project = createProject id name rootPath template mode

    match defaultRegistry.Register project with
    | Result.Ok() ->
        saveProject project // Auto-save to disk
        Result.Ok project
    | Result.Error e -> Result.Error e

/// Get a project by ID
let getProject id = defaultRegistry.Get id

/// Get project state
let getProjectState id = defaultRegistry.GetState id

/// List all projects
let listProjects () = defaultRegistry.List()

/// Update project to next stage
let advanceStage (projectId: string) : Result<PipelineStage option, string> =
    match defaultRegistry.GetState projectId with
    | None -> Result.Error $"Project '{projectId}' not found"
    | Some state ->
        let stages = templateStages state.Project.Template

        match state.CurrentStage with
        | None when not stages.IsEmpty ->
            // Start first stage
            let firstStage = stages.Head

            let newState =
                { state with
                    CurrentStage = Some firstStage
                    StartedAt = Some DateTime.UtcNow }

            match defaultRegistry.UpdateState(projectId, newState) with
            | Result.Ok() -> Result.Ok(Some firstStage)
            | Result.Error e -> Result.Error e
        | Some current ->
            match nextStage stages current with
            | Some next ->
                let newState = { state with CurrentStage = Some next }

                match defaultRegistry.UpdateState(projectId, newState) with
                | Result.Ok() -> Result.Ok(Some next)
                | Result.Error e -> Result.Error e
            | None ->
                // Pipeline complete
                let newState =
                    { state with
                        CurrentStage = None
                        CompletedAt = Some DateTime.UtcNow }

                match defaultRegistry.UpdateState(projectId, newState) with
                | Result.Ok() -> Result.Ok None
                | Result.Error e -> Result.Error e
        | None -> Result.Error "No stages in pipeline"
