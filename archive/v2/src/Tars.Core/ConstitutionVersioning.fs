namespace Tars.Core

open System
// =============================================================================
// PHASE 14.4: CONSTITUTION VERSIONING & HISTORY
// =============================================================================
//
// Provides version control for agent constitutions, enabling:
// - Audit trail of all constitutional changes
// - Rollback capability
// - Amendment tracking with approval status
// Reference: docs/3_Roadmap/2_Phases/phase_14_agent_constitutions.md

module ConstitutionVersioning =

    /// A single version of an agent's constitution
    type ConstitutionVersion =
        { Version: int
          Constitution: AgentConstitution
          CreatedAt: DateTimeOffset
          CreatedBy: string
          Reason: string
          AmendmentType: AmendmentType }

    and AmendmentType =
        | Initial
        | PermissionGrant
        | PermissionRevoke
        | ProhibitionAdd
        | ProhibitionRemove
        | ResourceLimitChange
        | RoleChange
        | EmergencyOverride
        | HumanApproved of approver: string

    /// Constitution change event for event sourcing
    type ConstitutionEvent =
        | Created of agentId: AgentId * constitution: AgentConstitution * createdBy: string
        | Amended of
            agentId: AgentId *
            newConstitution: AgentConstitution *
            reason: string *
            amendmentType: AmendmentType *
            amendedBy: string
        | Suspended of agentId: AgentId * reason: string * suspendedBy: string
        | Restored of agentId: AgentId * version: int * restoredBy: string

    /// In-memory constitution history store
    type ConstitutionHistory() =
        let histories =
            System.Collections.Concurrent.ConcurrentDictionary<AgentId, ConstitutionVersion list>()

        /// Get current version number for an agent
        member _.GetCurrentVersion(agentId: AgentId) =
            match histories.TryGetValue(agentId) with
            | true, versions -> versions |> List.tryHead |> Option.map (fun v -> v.Version)
            | false, _ -> None

        /// Get all versions for an agent
        member _.GetHistory(agentId: AgentId) =
            match histories.TryGetValue(agentId) with
            | true, versions -> versions |> List.rev // Oldest first
            | false, _ -> []

        /// Get a specific version
        member _.GetVersion(agentId: AgentId, version: int) =
            match histories.TryGetValue(agentId) with
            | true, versions -> versions |> List.tryFind (fun v -> v.Version = version)
            | false, _ -> None

        /// Get the current (latest) constitution
        member _.GetCurrent(agentId: AgentId) =
            match histories.TryGetValue(agentId) with
            | true, versions -> versions |> List.tryHead |> Option.map (fun v -> v.Constitution)
            | false, _ -> None

        /// Record initial constitution
        member _.RecordInitial(agentId: AgentId, constitution: AgentConstitution, createdBy: string) =
            let version =
                { Version = 1
                  Constitution = constitution
                  CreatedAt = DateTimeOffset.UtcNow
                  CreatedBy = createdBy
                  Reason = "Initial constitution"
                  AmendmentType = Initial }

            histories.[agentId] <- [ version ]
            Metrics.recordSimple "constitution.version" "created" (Some agentId) (Some 1.0) None
            version

        /// Record an amendment
        member this.RecordAmendment
            (
                agentId: AgentId,
                newConstitution: AgentConstitution,
                reason: string,
                amendmentType: AmendmentType,
                amendedBy: string
            ) =

            let currentVersion = this.GetCurrentVersion(agentId) |> Option.defaultValue 0

            let newVersion =
                { Version = currentVersion + 1
                  Constitution = newConstitution
                  CreatedAt = DateTimeOffset.UtcNow
                  CreatedBy = amendedBy
                  Reason = reason
                  AmendmentType = amendmentType }

            histories.AddOrUpdate(agentId, [ newVersion ], fun _ existing -> newVersion :: existing)
            |> ignore

            Metrics.recordSimple "constitution.version" "amended" (Some agentId) (Some(float newVersion.Version)) None
            newVersion

        /// Rollback to a specific version
        member this.Rollback(agentId: AgentId, targetVersion: int, rolledBackBy: string) =
            match this.GetVersion(agentId, targetVersion) with
            | None -> FSharp.Core.Error $"Version {targetVersion} not found for agent {agentId}"
            | Some oldVersion ->
                let restoredVersion =
                    this.RecordAmendment(
                        agentId,
                        oldVersion.Constitution,
                        $"Rollback to version {targetVersion}",
                        EmergencyOverride,
                        rolledBackBy
                    )

                Metrics.recordSimple "constitution.version" "rollback" (Some agentId) (Some(float targetVersion)) None
                FSharp.Core.Ok restoredVersion

        /// Compare two versions and return differences
        member this.Diff(agentId: AgentId, fromVersion: int, toVersion: int) =
            match this.GetVersion(agentId, fromVersion), this.GetVersion(agentId, toVersion) with
            | Some from, Some to' ->
                let fromC = from.Constitution
                let toC = to'.Constitution

                let addedProhibitions =
                    toC.Prohibitions
                    |> List.filter (fun p -> not (List.contains p fromC.Prohibitions))

                let removedProhibitions =
                    fromC.Prohibitions
                    |> List.filter (fun p -> not (List.contains p toC.Prohibitions))

                let addedPermissions =
                    toC.Permissions
                    |> List.filter (fun p -> not (List.contains p fromC.Permissions))

                let removedPermissions =
                    fromC.Permissions
                    |> List.filter (fun p -> not (List.contains p toC.Permissions))

                let addedLimits =
                    toC.HardResourceBounds
                    |> List.filter (fun l -> not (List.contains l fromC.HardResourceBounds))

                let removedLimits =
                    fromC.HardResourceBounds
                    |> List.filter (fun l -> not (List.contains l toC.HardResourceBounds))

                let roleChanged = fromC.NeuralRole <> toC.NeuralRole

                FSharp.Core.Ok
                    {| FromVersion = fromVersion
                       ToVersion = toVersion
                       AddedProhibitions = addedProhibitions
                       RemovedProhibitions = removedProhibitions
                       AddedPermissions = addedPermissions
                       RemovedPermissions = removedPermissions
                       AddedLimits = addedLimits
                       RemovedLimits = removedLimits
                       RoleChanged = roleChanged
                       OldRole = if roleChanged then Some fromC.NeuralRole else None
                       NewRole = if roleChanged then Some toC.NeuralRole else None |}
            | None, _ -> FSharp.Core.Error $"Version {fromVersion} not found"
            | _, None -> FSharp.Core.Error $"Version {toVersion} not found"

    // =========================================================================
    // Amendment Helpers
    // =========================================================================

    /// Safely grant a new permission with versioning
    let grantPermission
        (history: ConstitutionHistory)
        (agentId: AgentId)
        (currentConstitution: AgentConstitution)
        (permission: Permission)
        (grantedBy: string)
        =

        if List.contains permission currentConstitution.Permissions then
            FSharp.Core.Error "Permission already granted"
        else
            let newConstitution =
                { currentConstitution with
                    Permissions = permission :: currentConstitution.Permissions }

            let version =
                history.RecordAmendment(
                    agentId,
                    newConstitution,
                    $"Granted permission: {permission}",
                    PermissionGrant,
                    grantedBy
                )

            FSharp.Core.Ok(newConstitution, version)

    /// Safely revoke a permission with versioning
    let revokePermission
        (history: ConstitutionHistory)
        (agentId: AgentId)
        (currentConstitution: AgentConstitution)
        (permission: Permission)
        (revokedBy: string)
        =

        if not (List.contains permission currentConstitution.Permissions) then
            FSharp.Core.Error "Permission not currently granted"
        else
            let newConstitution =
                { currentConstitution with
                    Permissions = currentConstitution.Permissions |> List.filter ((<>) permission) }

            let version =
                history.RecordAmendment(
                    agentId,
                    newConstitution,
                    $"Revoked permission: {permission}",
                    PermissionRevoke,
                    revokedBy
                )

            FSharp.Core.Ok(newConstitution, version)

    /// Add a new prohibition with versioning
    let addProhibition
        (history: ConstitutionHistory)
        (agentId: AgentId)
        (currentConstitution: AgentConstitution)
        (prohibition: Prohibition)
        (addedBy: string)
        =

        if List.contains prohibition currentConstitution.Prohibitions then
            FSharp.Core.Error "Prohibition already exists"
        else
            let newConstitution =
                { currentConstitution with
                    Prohibitions = prohibition :: currentConstitution.Prohibitions }

            let version =
                history.RecordAmendment(
                    agentId,
                    newConstitution,
                    $"Added prohibition: {prohibition}",
                    ProhibitionAdd,
                    addedBy
                )

            FSharp.Core.Ok(newConstitution, version)

    /// Update resource limits with versioning
    let updateResourceLimits
        (history: ConstitutionHistory)
        (agentId: AgentId)
        (currentConstitution: AgentConstitution)
        (newLimits: ResourceLimit list)
        (updatedBy: string)
        =

        let newConstitution =
            { currentConstitution with
                HardResourceBounds = newLimits }

        let version =
            history.RecordAmendment(agentId, newConstitution, "Updated resource limits", ResourceLimitChange, updatedBy)

        FSharp.Core.Ok(newConstitution, version)
