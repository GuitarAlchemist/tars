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

    /// Templates (`constitutions/templates/*.json`) describe a constitution with no agent:
    /// `role` plus `contract { prohibitions, permissions, resourceBounds }`, each entry tagged
    /// by `type`. The agent is assigned at instantiation, so a fresh AgentId is generated.
    let private fromTemplate (root: JsonElement) : AgentConstitution =
        let text (e: JsonElement) (name: string) = e.GetProperty(name).GetString()

        // Every list is required: an empty Permissions list means permissive mode in
        // ContractEnforcement, so a missing or misspelled key must not load as "allow all".
        let items (name: string) (map: JsonElement -> 'T) =
            match root.GetProperty("contract").TryGetProperty name with
            | true, arr -> [ for e in arr.EnumerateArray() -> map e ]
            | _ -> failwith $"Template contract is missing '{name}' (use [] for none)"

        let prohibition e =
            match text e "type" with
            | "CannotModifyCore" -> Prohibition.CannotModifyCore
            | "CannotDeleteData" -> Prohibition.CannotDeleteData
            | "CannotAccessNetwork" -> Prohibition.CannotAccessNetwork
            | "CannotSpawnUnlimited" -> Prohibition.CannotSpawnUnlimited
            | "CannotExceedBudget" -> Prohibition.CannotExceedBudget
            | "CannotUseTool" -> Prohibition.CannotUseTool(text e "toolName")
            | "CannotAccessPath" -> Prohibition.CannotAccessPath(text e "path")
            | other -> failwith $"Unknown prohibition type '{other}'"

        let permission e =
            match text e "type" with
            | "ReadKnowledgeGraph" -> Permission.ReadKnowledgeGraph
            | "ModifyKnowledgeGraph" -> Permission.ModifyKnowledgeGraph
            | "ReadCode" -> Permission.ReadCode(text e "pattern")
            | "ModifyCode" -> Permission.ModifyCode(text e "pattern")
            | "SpawnAgent" -> Permission.SpawnAgent(text e "agentType")
            | "CallTool" -> Permission.CallTool(text e "toolName")
            | "AccessSecret" -> Permission.AccessSecret(text e "secretName")
            | "ExecuteShellCommand" -> Permission.ExecuteShellCommand(text e "pattern")
            | "All" -> Permission.All
            | other -> failwith $"Unknown permission type '{other}'"

        let limit (e: JsonElement) =
            let value = e.GetProperty "value"

            match text e "type" with
            | "MaxIterations" -> ResourceLimit.MaxIterations(value.GetInt32())
            | "MaxTokens" -> ResourceLimit.MaxTokens(value.GetInt32())
            | "MaxTimeMinutes" -> ResourceLimit.MaxTimeMinutes(value.GetInt32())
            | "MaxMemoryMB" -> ResourceLimit.MaxMemoryMB(value.GetInt64())
            | "MaxCpuPercent" -> ResourceLimit.MaxCpuPercent(value.GetInt32())
            | "MaxDiskWritesMB" -> ResourceLimit.MaxDiskWritesMB(value.GetInt32())
            | "MaxCost" -> ResourceLimit.MaxCost(value.GetDecimal())
            | other -> failwith $"Unknown resource bound type '{other}'"

        let role =
            match text root "role" with
            | "GeneralReasoning" -> NeuralRole.GeneralReasoning
            | other -> failwith $"Unsupported template role '{other}'"

        { AgentConstitution.Create(AgentId(Guid.NewGuid()), role) with
            Prohibitions = items "prohibitions" prohibition
            Permissions = items "permissions" permission
            HardResourceBounds = items "resourceBounds" limit }

    /// <summary>
    /// Loads an agent constitution from a JSON file.
    /// </summary>
    let load (path: string) : Result<AgentConstitution, string> =
        try
            if not (File.Exists path) then
                Result.Error $"Constitution file not found: {path}"
            else
                let json = File.ReadAllText path
                use doc = JsonDocument.Parse json

                match doc.RootElement.TryGetProperty "contract" with
                | true, _ -> Result.Ok(fromTemplate doc.RootElement)
                | _ -> Result.Ok(JsonSerializer.Deserialize<AgentConstitution>(json, options))
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
