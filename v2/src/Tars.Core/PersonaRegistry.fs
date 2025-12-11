/// AI Persona Registry
/// Manages registration and retrieval of AI personas
module Tars.Core.PersonaRegistry

open System
open System.IO
open System.Text.Json
open Tars.Core.Persona

// ============================================================================
// Registry State
// ============================================================================

type PersonaRegistry() =
    let mutable personas: Map<string, Persona> = Map.empty

    let jsonOptions =
        let opts =
            JsonSerializerOptions(WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

        opts.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
        opts

    // ========================================================================
    // Core Operations
    // ========================================================================

    /// Register a persona (overwrites if exists)
    member _.Register(persona: Persona) : Result<unit, string> =
        if String.IsNullOrWhiteSpace(persona.Id) then
            Result.Error "Persona ID cannot be empty"
        elif String.IsNullOrWhiteSpace(persona.Role) then
            Result.Error "Persona Role cannot be empty"
        else
            personas <- personas |> Map.add persona.Id persona
            Result.Ok()

    /// Get a persona by ID
    member _.Get(id: string) : Persona option = Map.tryFind id personas

    /// List all registered personas
    member _.List() : Persona list = personas |> Map.toList |> List.map snd

    /// Remove a persona by ID
    member _.Remove(id: string) : bool =
        if Map.containsKey id personas then
            personas <- personas |> Map.remove id
            true
        else
            false

    /// Check if a persona exists
    member _.Exists(id: string) : bool = Map.containsKey id personas

    /// Get count of registered personas
    member _.Count: int = Map.count personas

    // ========================================================================
    // Serialization
    // ========================================================================

    /// Load personas from a JSON file
    member this.LoadFromFile(path: string) : Result<int, string> =
        try
            if not (File.Exists path) then
                Result.Error $"File not found: {path}"
            else
                let json = File.ReadAllText path
                let loaded = JsonSerializer.Deserialize<Persona list>(json, jsonOptions)
                loaded |> List.iter (fun p -> this.Register p |> ignore)
                Result.Ok loaded.Length
        with ex ->
            Result.Error $"Failed to load personas: {ex.Message}"

    /// Save all personas to a JSON file
    member this.SaveToFile(path: string) : Result<unit, string> =
        try
            let dir = Path.GetDirectoryName path

            if not (String.IsNullOrEmpty dir) && not (Directory.Exists dir) then
                Directory.CreateDirectory dir |> ignore

            let json = JsonSerializer.Serialize(this.List(), jsonOptions)
            File.WriteAllText(path, json)
            Result.Ok()
        with ex ->
            Result.Error $"Failed to save personas: {ex.Message}"

    /// Load personas from a directory (each .json file)
    member this.LoadFromDirectory(dir: string) : Result<int, string> =
        try
            if not (Directory.Exists dir) then
                Result.Error $"Directory not found: {dir}"
            else
                let mutable count = 0

                for file in Directory.GetFiles(dir, "*.json") do
                    match this.LoadFromFile file with
                    | Result.Ok n -> count <- count + n
                    | Result.Error _ -> () // Skip invalid files

                Result.Ok count
        with ex ->
            Result.Error $"Failed to load from directory: {ex.Message}"

    // ========================================================================
    // Initialization
    // ========================================================================

    /// Initialize with built-in personas
    member this.LoadBuiltIns() =
        BuiltIn.all |> List.iter (fun p -> this.Register p |> ignore)

// ============================================================================
// Global Registry Instance
// ============================================================================

/// Default global persona registry
let defaultRegistry =
    let reg = PersonaRegistry()
    reg.LoadBuiltIns()
    reg

// ============================================================================
// Convenience Functions
// ============================================================================

/// Get a persona from the default registry
let getPersona (id: string) = defaultRegistry.Get id

/// List all personas from the default registry
let listPersonas () = defaultRegistry.List()

/// Create and use a persona with RTFD prompt
let withPersona (personaId: string) (task: string) (format: OutputFormat option) (details: string option) =
    match defaultRegistry.Get personaId with
    | Some persona ->
        let rtfd =
            { Persona = persona
              Task = task
              Format = format
              Details = details }

        Result.Ok(buildRtfdPrompt rtfd)
    | None -> Result.Error $"Persona not found: {personaId}"
