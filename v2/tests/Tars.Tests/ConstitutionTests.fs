module Tars.Tests.ConstitutionTests

open Xunit
open FsUnit
open Tars.Core
open System

[<Fact>]
let ``ConstitutionLoader parses General Safety template`` () =
    let json =
        """{
      "name": "General Safety",
      "description": "desc",
      "role": "GeneralReasoning",
      "contract": {
        "prohibitions": [
          { "type": "CannotModifyCore" },
          { "type": "CannotDeleteData" }
        ],
        "permissions": [
          { "type": "ReadCode", "pattern": "*" }
        ],
        "resourceBounds": [
          { "type": "MaxTokens", "value": 1000 }
        ]
      }
    }"""

    let path = IO.Path.Combine(IO.Path.GetTempPath(), $"tars-constitution-{Guid.NewGuid():N}.json")
    IO.File.WriteAllText(path, json)

    match (try ConstitutionLoader.load path finally IO.File.Delete path) with
    | FSharp.Core.Ok c ->
        c.Prohibitions |> should contain Prohibition.CannotModifyCore
        c.Prohibitions |> should contain Prohibition.CannotDeleteData
        c.Permissions |> should contain (Permission.ReadCode "*")
        c.HardResourceBounds |> should contain (ResourceLimit.MaxTokens 1000)
    | FSharp.Core.Error e -> failwithf "Failed to parse: %s" e

let private shippedTemplate (name: string) =
    let rec v2Root (dir: IO.DirectoryInfo) =
        if isNull dir then
            failwith "Could not locate v2/ (no Tars.sln above the test directory)"
        elif dir.GetFiles("Tars.sln").Length > 0 then
            dir.FullName
        else
            v2Root dir.Parent

    let root = v2Root (IO.DirectoryInfo(IO.Directory.GetCurrentDirectory()))
    IO.Path.Combine(root, "constitutions", "templates", name)

[<Fact>]
let ``ConstitutionLoader loads the shipped general_safety template`` () =
    let path = shippedTemplate "general_safety.json"

    match ConstitutionLoader.load path with
    | FSharp.Core.Ok c ->
        c.NeuralRole |> should equal NeuralRole.GeneralReasoning

        c.Prohibitions
        |> should equal [ Prohibition.CannotModifyCore; Prohibition.CannotDeleteData; Prohibition.CannotAccessNetwork ]

        c.Permissions |> should contain (Permission.ModifyCode "src/Tars.Playground/*")
        c.Permissions |> should contain (Permission.CallTool "find_code_smells")

        c.HardResourceBounds
        |> should equal [ ResourceLimit.MaxTokens 8192; ResourceLimit.MaxCost 2.0m; ResourceLimit.MaxTimeMinutes 5 ]
    | FSharp.Core.Error e -> failwithf "Failed to load shipped template: %s" e

[<Fact>]
let ``ConstitutionLoader rejects a template with an unknown prohibition type`` () =
    let json =
        """{ "role": "GeneralReasoning",
             "contract": { "prohibitions": [ { "type": "CannotFly" } ], "permissions": [], "resourceBounds": [] } }"""

    let path = IO.Path.Combine(IO.Path.GetTempPath(), $"tars-constitution-{Guid.NewGuid():N}.json")
    IO.File.WriteAllText(path, json)

    match (try ConstitutionLoader.load path finally IO.File.Delete path) with
    | FSharp.Core.Error e -> e |> should contain "CannotFly"
    | FSharp.Core.Ok c -> failwithf "Should have rejected the unknown type: %A" c

[<Fact>]
let ``ConstitutionLoader rejects a template missing its permissions list`` () =
    // An empty Permissions list is permissive mode, so a missing key must not load as one.
    let json =
        """{ "role": "GeneralReasoning",
             "contract": { "prohibitions": [ { "type": "CannotModifyCore" } ], "resourceBounds": [] } }"""

    let path = IO.Path.Combine(IO.Path.GetTempPath(), $"tars-constitution-{Guid.NewGuid():N}.json")
    IO.File.WriteAllText(path, json)

    match (try ConstitutionLoader.load path finally IO.File.Delete path) with
    | FSharp.Core.Error e -> e |> should contain "permissions"
    | FSharp.Core.Ok c -> failwithf "Should have rejected the missing permissions list: %A" c

[<Fact>]
let ``ConstitutionLoader round-trips a saved constitution`` () =
    let original = ConstitutionLoader.createDefault "agent-001"
    let path = IO.Path.Combine(IO.Path.GetTempPath(), $"tars-constitution-{Guid.NewGuid():N}.json")

    let roundTrip () =
        ConstitutionLoader.save path original
        |> Result.bind (fun () -> ConstitutionLoader.load path)

    match (try roundTrip () finally IO.File.Delete path) with
    | FSharp.Core.Ok loaded -> loaded |> should equal original
    | FSharp.Core.Error e -> failwithf "Round trip failed: %s" e

[<Fact>]
let ``Enforcement blocks prohibited action`` () =
    let constitution =
        { AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) with
            Prohibitions = [ Prohibition.CannotModifyCore ] }

    // Note: ContractEnforcement logic for CannotModifyCore checks path substring
    let badAction = AgentAction.WriteFile "src/Tars.Core/Domain.fs"
    let result = ContractEnforcement.validateAction constitution badAction

    match result with
    | FSharp.Core.Error(Violation.ProhibitionViolated(Prohibition.CannotModifyCore, _)) -> ()
    | FSharp.Core.Ok() -> failwith "Should have violated prohibition"
    | _ -> failwith "Unexpected result"

// Issue #272: glob permission patterns were compared as literal substrings.
let private isOk result =
    match result with
    | FSharp.Core.Ok() -> true
    | FSharp.Core.Error _ -> false

let private modifyOnly (pattern: string) =
    { AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) with
        Permissions = [ Permission.ModifyCode pattern ] }

[<Theory>]
[<InlineData("src/Tars.Playground/Main.fs")>]
[<InlineData("src/Tars.Playground/Nested/Deep.fs")>]
[<InlineData(@"src\Tars.Playground\Main.fs")>]
[<InlineData(@"C:\repos\tars\v2\src\Tars.Playground\Main.fs")>]
let ``Trailing-star permission allows paths under its prefix`` (path: string) =
    let c = modifyOnly "src/Tars.Playground/*"
    Assert.True(isOk (ContractEnforcement.validateAction c (AgentAction.WriteFile path)), path)

[<Theory>]
[<InlineData("src/Tars.Core/Domain.fs")>]
[<InlineData("src/Tars.PlaygroundEvil/Main.fs")>]
[<InlineData("other/notsrc/Tars.Playground/Main.fs")>]
let ``Trailing-star permission denies paths outside its prefix`` (path: string) =
    let c = modifyOnly "src/Tars.Playground/*"
    Assert.False(isOk (ContractEnforcement.validateAction c (AgentAction.WriteFile path)), path)

[<Fact>]
let ``Shipped general_safety template permits its playground and still blocks core`` () =
    match ConstitutionLoader.load (shippedTemplate "general_safety.json") with
    | FSharp.Core.Error e -> failwithf "Failed to load shipped template: %s" e
    | FSharp.Core.Ok c ->
        Assert.True(isOk (ContractEnforcement.validateAction c (AgentAction.WriteFile "src/Tars.Playground/Main.fs")))
        Assert.False(isOk (ContractEnforcement.validateAction c (AgentAction.WriteFile "src/Tars.Core/Domain.fs")))
        Assert.False(isOk (ContractEnforcement.validateAction c (AgentAction.WriteFile "src/Tars.Llm/Routing.fs")))

[<Theory>]
[<InlineData(@"C:\Windows\System32\drivers\etc\hosts")>]
[<InlineData("C:/Windows/System32/drivers/etc/hosts")>]
let ``Path prohibition applies whatever the separator`` (path: string) =
    let c =
        { AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) with
            Prohibitions = [ Prohibition.CannotAccessPath "C:/Windows" ] }

    match ContractEnforcement.validateAction c (AgentAction.ReadFile path) with
    | FSharp.Core.Error(Violation.ProhibitionViolated(Prohibition.CannotAccessPath _, _)) -> ()
    | other -> failwithf "Expected CannotAccessPath violation for %s, got %A" path other

[<Fact>]
let ``Enforcement allows permitted action when restrictive`` () =
    let constitution =
        { AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) with
            Permissions = [ Permission.ReadCode "*" ] }

    // Read allowed
    match ContractEnforcement.validateAction constitution (AgentAction.ReadFile "foo.fs") with
    | FSharp.Core.Ok() -> ()
    | FSharp.Core.Error e -> failwithf "Should pass: %A" e

    // Write denied (implicit because permissions list is non-empty)
    match ContractEnforcement.validateAction constitution (AgentAction.WriteFile "foo.fs") with
    | FSharp.Core.Error(Violation.PermissionDenied _) -> ()
    | FSharp.Core.Ok() -> failwith "Should have been denied"
    | _ -> failwith "Unexpected result"

// =============================================================================
// RESOURCE TRACKING TESTS
// =============================================================================

[<Fact>]
let ``ResourceTracker detects exceeded token limit`` () =
    let constitution =
        { AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) with
            HardResourceBounds = [ ResourceLimit.MaxTokens 100 ] }

    let tracker = ConstitutionWorkflow.ResourceTracker(constitution)

    // Record 50 tokens - should be OK
    tracker.RecordTokens(50)

    match tracker.CheckLimits() with
    | FSharp.Core.Ok() -> ()
    | FSharp.Core.Error e -> failwithf "Should not fail: %A" e

    // Record 60 more tokens (total 110) - should exceed
    tracker.RecordTokens(60)

    match tracker.CheckLimits() with
    | FSharp.Core.Error(Violation.ResourceQuotaExceeded _) -> ()
    | _ -> failwith "Should have exceeded limit"

[<Fact>]
let ``ResourceTracker detects exceeded cost limit`` () =
    let constitution =
        { AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) with
            HardResourceBounds = [ ResourceLimit.MaxCost 10.0m ] }

    let tracker = ConstitutionWorkflow.ResourceTracker(constitution)

    tracker.RecordCost(5.0m)

    match tracker.CheckLimits() with
    | FSharp.Core.Ok() -> ()
    | _ -> failwith "Should not fail at 5.0"

    tracker.RecordCost(6.0m) // Total 11.0

    match tracker.CheckLimits() with
    | FSharp.Core.Error(Violation.ResourceQuotaExceeded _) -> ()
    | _ -> failwith "Should have exceeded cost limit"

// =============================================================================
// AMENDMENT PROTOCOL TESTS
// =============================================================================

[<Fact>]
let ``Amendment that removes core protection requires human review`` () =
    let request: ConstitutionWorkflow.AmendmentRequest =
        { AgentId = AgentId(Guid.NewGuid())
          ProposedChanges =
            { AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) with
                Prohibitions = [] } // Missing CannotModifyCore!
          Reason = "Need more power"
          RequestedBy = "rogue_agent"
          RequestedAt = DateTimeOffset.UtcNow }

    match ConstitutionWorkflow.evaluateAmendment request with
    | ConstitutionWorkflow.AmendmentDecision.RequiresHumanReview reason ->
        reason |> should contain "core modification protection"
    | _ -> failwith "Should require human review"

[<Fact>]
let ``Amendment granting All permission requires human review`` () =
    let request: ConstitutionWorkflow.AmendmentRequest =
        { AgentId = AgentId(Guid.NewGuid())
          ProposedChanges =
            { AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) with
                Prohibitions = [ Prohibition.CannotModifyCore ]
                Permissions = [ Permission.All ] } // Dangerous!
          Reason = "Need admin access"
          RequestedBy = "power_user"
          RequestedAt = DateTimeOffset.UtcNow }

    match ConstitutionWorkflow.evaluateAmendment request with
    | ConstitutionWorkflow.AmendmentDecision.RequiresHumanReview reason ->
        reason |> should contain "dangerous permissions"
    | _ -> failwith "Should require human review for All permission"

[<Fact>]
let ``Safe amendment is approved automatically`` () =
    let constitution =
        { AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) with
            Prohibitions = [ Prohibition.CannotModifyCore; Prohibition.CannotDeleteData ]
            Permissions = [ Permission.ReadCode "*"; Permission.CallTool "safe_tool" ] }

    let request: ConstitutionWorkflow.AmendmentRequest =
        { AgentId = AgentId(Guid.NewGuid())
          ProposedChanges = constitution
          Reason = "Add safe tool permission"
          RequestedBy = "admin"
          RequestedAt = DateTimeOffset.UtcNow }

    match ConstitutionWorkflow.evaluateAmendment request with
    | ConstitutionWorkflow.AmendmentDecision.Approved _ -> ()
    | other -> failwithf "Should have been approved: %A" other

// =============================================================================
// DEPENDENCY CHECKING TESTS
// =============================================================================

[<Fact>]
let ``Dependency check passes when all dependencies are active`` () =
    let depId = AgentId(Guid.NewGuid())

    let constitution =
        { AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) with
            SymbolicContract =
                { SymbolicContract.Empty with
                    Dependencies = [ depId ] } }

    let activeAgents = [ depId; AgentId(Guid.NewGuid()) ]

    match ContractEnforcement.checkDependencies constitution activeAgents with
    | FSharp.Core.Ok() -> ()
    | FSharp.Core.Error e -> failwithf "Should pass: %A" e

[<Fact>]
let ``Dependency check fails when dependency is missing`` () =
    let depId = AgentId(Guid.NewGuid())

    let constitution =
        { AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) with
            SymbolicContract =
                { SymbolicContract.Empty with
                    Dependencies = [ depId ] } }

    let activeAgents = [ AgentId(Guid.NewGuid()) ] // depId NOT present

    match ContractEnforcement.checkDependencies constitution activeAgents with
    | FSharp.Core.Error(Violation.DependencyMissing missing) -> missing |> should equal depId
    | _ -> failwith "Should have failed with missing dependency"

// =============================================================================
// TOOL BLOCKING TESTS
// =============================================================================

[<Fact>]
let ``CannotUseTool prohibition blocks specific tool`` () =
    let constitution =
        { AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) with
            Prohibitions = [ Prohibition.CannotUseTool "dangerous_tool" ] }

    // Blocked tool
    match ContractEnforcement.validateAction constitution (AgentAction.ExecuteTool("dangerous_tool", "args")) with
    | FSharp.Core.Error(Violation.ProhibitionViolated _) -> ()
    | _ -> failwith "Should have blocked dangerous_tool"

    // Other tools allowed (permissive mode since Permissions is empty)
    match ContractEnforcement.validateAction constitution (AgentAction.ExecuteTool("safe_tool", "args")) with
    | FSharp.Core.Ok() -> ()
    | FSharp.Core.Error e -> failwithf "Should allow safe_tool: %A" e

// =============================================================================
// VERSIONING TESTS (Phase 14.4)
// =============================================================================

[<Fact>]
let ``ConstitutionHistory records initial version`` () =
    let history = ConstitutionVersioning.ConstitutionHistory()
    let agentId = AgentId(Guid.NewGuid())
    let constitution = AgentConstitution.Create(agentId, NeuralRole.GeneralReasoning)

    let version = history.RecordInitial(agentId, constitution, "system")

    version.Version |> should equal 1

    version.AmendmentType
    |> should equal ConstitutionVersioning.AmendmentType.Initial

    history.GetCurrentVersion(agentId) |> should equal (Some 1)

[<Fact>]
let ``ConstitutionHistory records amendments with incrementing versions`` () =
    let history = ConstitutionVersioning.ConstitutionHistory()
    let agentId = AgentId(Guid.NewGuid())
    let constitution = AgentConstitution.Create(agentId, NeuralRole.GeneralReasoning)

    history.RecordInitial(agentId, constitution, "system") |> ignore

    let amended =
        { constitution with
            Permissions = [ Permission.ReadCode "*" ] }

    let v2 =
        history.RecordAmendment(
            agentId,
            amended,
            "Grant read permission",
            ConstitutionVersioning.PermissionGrant,
            "admin"
        )

    v2.Version |> should equal 2
    history.GetCurrentVersion(agentId) |> should equal (Some 2)

    // Check history length
    let fullHistory = history.GetHistory(agentId)
    fullHistory.Length |> should equal 2

[<Fact>]
let ``ConstitutionHistory can rollback to previous version`` () =
    let history = ConstitutionVersioning.ConstitutionHistory()
    let agentId = AgentId(Guid.NewGuid())
    let constitution = AgentConstitution.Create(agentId, NeuralRole.GeneralReasoning)

    history.RecordInitial(agentId, constitution, "system") |> ignore

    let v2Constitution =
        { constitution with
            Prohibitions = [ Prohibition.CannotModifyCore ] }

    history.RecordAmendment(agentId, v2Constitution, "Add prohibition", ConstitutionVersioning.ProhibitionAdd, "admin")
    |> ignore

    // Rollback to version 1
    match history.Rollback(agentId, 1, "emergency_admin") with
    | FSharp.Core.Ok restored ->
        restored.Version |> should equal 3 // New version created for rollback
        restored.Constitution.Prohibitions |> should be Empty // Back to original
    | FSharp.Core.Error e -> failwithf "Rollback failed: %s" e

[<Fact>]
let ``ConstitutionHistory diff shows changes between versions`` () =
    let history = ConstitutionVersioning.ConstitutionHistory()
    let agentId = AgentId(Guid.NewGuid())
    let constitution = AgentConstitution.Create(agentId, NeuralRole.GeneralReasoning)

    history.RecordInitial(agentId, constitution, "system") |> ignore

    let amended =
        { constitution with
            Permissions = [ Permission.ReadCode "*" ]
            Prohibitions = [ Prohibition.CannotDeleteData ] }

    history.RecordAmendment(
        agentId,
        amended,
        "Add permissions and prohibitions",
        ConstitutionVersioning.PermissionGrant,
        "admin"
    )
    |> ignore

    match history.Diff(agentId, 1, 2) with
    | FSharp.Core.Ok diff ->
        diff.AddedPermissions.Length |> should equal 1
        diff.AddedProhibitions.Length |> should equal 1
        diff.RemovedPermissions |> should be Empty
        diff.RoleChanged |> should equal false
    | FSharp.Core.Error e -> failwithf "Diff failed: %s" e

[<Fact>]
let ``grantPermission helper creates new version`` () =
    let history = ConstitutionVersioning.ConstitutionHistory()
    let agentId = AgentId(Guid.NewGuid())
    let constitution = AgentConstitution.Create(agentId, NeuralRole.GeneralReasoning)

    history.RecordInitial(agentId, constitution, "system") |> ignore

    match
        ConstitutionVersioning.grantPermission history agentId constitution (Permission.CallTool "my_tool") "admin"
    with
    | FSharp.Core.Ok(newConstitution, version) ->
        newConstitution.Permissions |> should contain (Permission.CallTool "my_tool")
        version.Version |> should equal 2
        version.AmendmentType |> should equal ConstitutionVersioning.PermissionGrant
    | FSharp.Core.Error e -> failwithf "Grant failed: %s" e
