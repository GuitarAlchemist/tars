namespace Tars.Core

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization

// ===================================
// Phase 14: Constitution Loading
// ===================================

module ConstitutionLoader =

    let private options =
        let opts = JsonSerializerOptions(WriteIndented = true)

        opts.Converters.Add(JsonFSharpConverter())

        opts

    /// <summary>
    /// Loads an agent constitution from a JSON file.
    /// </summary>
    let load (path: string) : Result<AgentConstitution, string> =
        try
            if not (File.Exists path) then
                Result.Error $"Constitution file not found: {path}"
            else
                let json = File.ReadAllText path
                let constitution = JsonSerializer.Deserialize<AgentConstitution>(json, options)
                Result.Ok constitution
        with ex ->
            Result.Error $"Failed to load constitution: {ex.Message}"

    /// <summary>
    /// Saves an agent constitution to a JSON file.
    /// </summary>
    let save (path: string) (constitution: AgentConstitution) : Result<unit, string> =
        try
            let dir = Path.GetDirectoryName(path)

            if not (String.IsNullOrEmpty dir) && not (Directory.Exists dir) then
                Directory.CreateDirectory dir |> ignore

            let json = JsonSerializer.Serialize(constitution, options)
            File.WriteAllText(path, json)
            Result.Ok()
        with ex ->
            Result.Error $"Failed to save constitution: {ex.Message}"

    /// <summary>
    /// creates a default safe constitution for standard agents
    /// </summary>
    let createDefault (agentId: string) =
        let id = AgentId(Guid.NewGuid()) // Corrected AgentId constructor to take only Guid
        let role = NeuralRole.GeneralReasoning

        let baseCon = AgentConstitution.Create(id, role)

        // Add basic safety defaults
        { baseCon with
            Prohibitions =
                [ Prohibition.CannotModifyCore
                  Prohibition.CannotDeleteData
                  Prohibition.CannotAccessPath "/etc/passwd"
                  Prohibition.CannotAccessPath "C:/Windows" ]
            Permissions =
                [ Permission.ReadCode "*"
                  Permission.CallTool "read_file"
                  Permission.CallTool "search_web"
                  Permission.CallTool "validate_puzzle_answer" ]
            Invariants = [ ConstitutionInvariant.TestPassing ] }
