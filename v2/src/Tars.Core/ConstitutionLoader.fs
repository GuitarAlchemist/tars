namespace Tars.Core

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization

module ConstitutionLoader =

    [<CLIMutable>]
    type ResourceLimitDto = { Type: string; Value: JsonElement } // Use JsonElement to handle int, float, decimal dynamically

    [<CLIMutable>]
    type PermissionDto =
        { Type: string
          Pattern: string option
          ToolName: string option
          AgentType: string option
          SecretName: string option }

    [<CLIMutable>]
    type ProhibitionDto =
        { Type: string
          ToolName: string option
          Path: string option
          InvariantName: string option } // For CannotViolateInvariant

    [<CLIMutable>]
    type ContractDto =
        { Prohibitions: ProhibitionDto list
          Permissions: PermissionDto list
          ResourceBounds: ResourceLimitDto list }

    [<CLIMutable>]
    type ConstitutionTemplateDto =
        { Name: string
          Description: string
          Role: string
          Contract: ContractDto }

    let private parseResourceLimit (dto: ResourceLimitDto) : ResourceLimit option =
        match dto.Type with
        | "MaxIterations" -> Some(ResourceLimit.MaxIterations(dto.Value.GetInt32()))
        | "MaxTokens" -> Some(ResourceLimit.MaxTokens(dto.Value.GetInt32()))
        | "MaxTimeMinutes" -> Some(ResourceLimit.MaxTimeMinutes(dto.Value.GetInt32()))
        | "MaxMemoryMB" -> Some(ResourceLimit.MaxMemoryMB(dto.Value.GetInt64()))
        | "MaxCpuPercent" -> Some(ResourceLimit.MaxCpuPercent(dto.Value.GetInt32()))
        | "MaxDiskWritesMB" -> Some(ResourceLimit.MaxDiskWritesMB(dto.Value.GetInt32()))
        | "MaxCost" -> Some(ResourceLimit.MaxCost(dto.Value.GetDecimal()))
        | _ -> None

    let private parsePermission (dto: PermissionDto) : Permission option =
        match dto.Type with
        | "ReadKnowledgeGraph" -> Some Permission.ReadKnowledgeGraph
        | "ModifyKnowledgeGraph" -> Some Permission.ModifyKnowledgeGraph
        | "ReadCode" -> Some(Permission.ReadCode(Option.defaultValue "*" dto.Pattern))
        | "ModifyCode" -> Some(Permission.ModifyCode(Option.defaultValue "*" dto.Pattern))
        | "SpawnAgent" -> Some(Permission.SpawnAgent(Option.defaultValue "Generic" dto.AgentType))
        | "CallTool" -> Some(Permission.CallTool(Option.defaultValue "*" dto.ToolName))
        | "AccessSecret" -> Some(Permission.AccessSecret(Option.defaultValue "" dto.SecretName))
        | "ExecuteShellCommand" -> Some(Permission.ExecuteShellCommand(Option.defaultValue "" dto.Pattern))
        | "All" -> Some Permission.All
        | _ -> None

    let private parseProhibition (dto: ProhibitionDto) : Prohibition option =
        match dto.Type with
        | "CannotModifyCore" -> Some Prohibition.CannotModifyCore
        | "CannotDeleteData" -> Some Prohibition.CannotDeleteData
        | "CannotAccessNetwork" -> Some Prohibition.CannotAccessNetwork
        | "CannotSpawnUnlimited" -> Some Prohibition.CannotSpawnUnlimited
        | "CannotExceedBudget" -> Some Prohibition.CannotExceedBudget
        | "CannotUseTool" -> Some(Prohibition.CannotUseTool(Option.defaultValue "" dto.ToolName))
        | "CannotAccessPath" -> Some(Prohibition.CannotAccessPath(Option.defaultValue "" dto.Path))
        // CannotViolateInvariant needs more complex parsing, skipping for now
        | _ -> None

    let private parseRole (roleParams: string) : NeuralRole =
        match roleParams with
        | "GeneralReasoning" -> NeuralRole.GeneralReasoning
        | _ -> NeuralRole.GeneralReasoning // Default for now

    let loadFromJson (json: string) : Result<AgentConstitution, string> =
        try
            let options = JsonSerializerOptions()
            options.PropertyNameCaseInsensitive <- true
            let dto = JsonSerializer.Deserialize<ConstitutionTemplateDto>(json, options)

            let resources = dto.Contract.ResourceBounds |> List.choose parseResourceLimit
            let permissions = dto.Contract.Permissions |> List.choose parsePermission
            let prohibitions = dto.Contract.Prohibitions |> List.choose parseProhibition
            let role = parseRole dto.Role

            // Create dummy AgentId for template
            let templateId = AgentId(Guid.NewGuid())

            let constitution =
                { AgentConstitution.Create(templateId, role) with
                    Permissions = permissions
                    Prohibitions = prohibitions
                    HardResourceBounds = resources // Map contract bounds to hard bounds for simplicity in template
                    SymbolicContract =
                        { SymbolicContract.Empty with
                            ResourceBounds = resources } }

            FSharp.Core.Ok constitution
        with ex ->
            FSharp.Core.Error(sprintf "Failed to parse constitution: %s" ex.Message)

    let loadFromFile (path: string) : Result<AgentConstitution, string> =
        try
            if File.Exists(path) then
                let json = File.ReadAllText(path)
                loadFromJson json
            else
                FSharp.Core.Error(sprintf "File not found: %s" path)
        with ex ->
            FSharp.Core.Error ex.Message
