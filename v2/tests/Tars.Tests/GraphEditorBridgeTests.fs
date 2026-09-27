namespace Tars.Tests

open System
open System.Text.Json
open Xunit
open Tars.Core
open Tars.Cortex
open Tars.Cortex.WoTTypes

/// The editor-facing trio: what can go in a graph, whether a graph is sound, and
/// whether it may run.
///
/// These are the contract the GA graph editor is built against, so where the shape
/// on the wire is the point, the assertion is against the JSON rather than the F#
/// record behind it.
module GraphEditorBridgeTests =

    // =========================================================================
    // Fixtures
    // =========================================================================

    let private tool name description : Tool =
        { Name = name
          Description = description
          Version = "1.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Execute = fun _ -> async { return Result.Ok "done" } }

    /// A registry holding exactly the named tools.
    let private registryOf (names: (string * string) list) =
        let tools = names |> List.map (fun (n, d) -> tool n d)

        { new IToolRegistry with
            member _.Register(_) = ()

            member _.Get(name) =
                tools |> List.tryFind (fun t -> t.Name = name)

            member _.GetAll() = tools }

    /// Apply these descriptions to a clean sidecar.
    ///
    /// The registry is process-wide, so every test states the whole world it expects
    /// rather than inheriting whatever ran before it.
    let private withDescriptors (descriptors: ToolMetadata.ToolDescriptor list) =
        ToolMetadata.clear ()
        ToolMetadata.describeAll descriptors

    let private readOnlyTool: ToolMetadata.ToolDescriptor =
        { Name = "read_file"
          InputSchema = ToolMetadata.objectSchema [ "path", "string", "The file." ] [ "path" ]
          Required = [ "path" ]
          Approval = ToolMetadata.Approval.observes "reads a file from disk" }

    let private writingTool: ToolMetadata.ToolDescriptor =
        { Name = "write_code"
          InputSchema =
            ToolMetadata.objectSchema
                [ "path", "string", "The file."; "content", "string", "The contents." ]
                [ "path"; "content" ]
          Required = [ "path"; "content" ]
          Approval = ToolMetadata.Approval.mutates "writes a file to disk" }

    let private spec nodes edges : GraphEditorBridge.PlanSpec =
        { Id = None
          Goal = "a goal"
          EntryNode = None
          Nodes = nodes
          Edges = edges
          Policy = None }

    let private edge from target : GraphEditorBridge.EdgeSpec =
        { From = from
          To = target
          Label = None
          Confidence = None }

    let private jsonString (value: string) =
        JsonDocument.Parse(JsonSerializer.Serialize value).RootElement.Clone()

    let private toolNode id toolName args : GraphEditorBridge.NodeSpec =
        { Id = id
          Kind = "tool"
          Prompt = None
          Hint = None
          Tool = Some toolName
          Arguments = args |> List.map (fun (k, v) -> k, jsonString v) |> Map.ofList |> Some
          Label = None
          Tags = None }

    let private reasonNode id prompt : GraphEditorBridge.NodeSpec =
        { Id = id
          Kind = "reason"
          Prompt = Some prompt
          Hint = None
          Tool = None
          Arguments = None
          Label = None
          Tags = None }

    // =========================================================================
    // Catalog
    // =========================================================================

    [<Fact>]
    let ``the catalog offers described tools and says how many are missing`` () =
        withDescriptors [ readOnlyTool ]

        let registry =
            registryOf [ "read_file", "Reads a file."; "undescribed_tool", "Nobody described this." ]

        let catalog = GraphEditorBridge.catalog registry

        // Two tools registered, one described. The other is not broken — it is just
        // not offered here, which is what makes a gradual backfill safe.
        Assert.Equal(2, catalog.TotalTools)
        Assert.Equal(1, catalog.DescribedTools)

        let names = catalog.Nodes |> List.map (fun n -> n.Name)
        Assert.Contains("read_file", names)
        Assert.DoesNotContain("undescribed_tool", names)

    [<Fact>]
    let ``the catalog separates nodes that act from nodes that only think`` () =
        withDescriptors [ readOnlyTool ]

        let catalog =
            GraphEditorBridge.catalog (registryOf [ "read_file", "Reads a file." ])

        let kindOf name =
            catalog.Nodes |> List.find (fun n -> n.Name = name) |> (fun n -> n.NodeKind)

        // A reasoning node cannot call tools — the executor builds its prompt with
        // none attached — so an editor that showed it as just another node would let
        // someone wire up something that narrates instead of acting (tars#326).
        Assert.Equal("reason", kindOf "reason")
        Assert.Equal("tool", kindOf "read_file")

    [<Fact>]
    let ``the catalog states the effect of every node it offers`` () =
        withDescriptors [ readOnlyTool; writingTool ]

        let catalog =
            GraphEditorBridge.catalog (registryOf [ "read_file", "r"; "write_code", "w" ])

        for entry in catalog.Nodes do
            Assert.False(String.IsNullOrWhiteSpace entry.Approval.Effect, $"{entry.Name} has no stated effect")
            Assert.Contains(entry.Approval.Tier, [ "read_only"; "mutates"; "escapes" ])

        let writeEntry = catalog.Nodes |> List.find (fun n -> n.Name = "write_code")
        Assert.Equal("mutates", writeEntry.Approval.Tier)
        Assert.False(writeEntry.Approval.AutoApproved)

    // =========================================================================
    // Validation
    // =========================================================================

    [<Fact>]
    let ``a sound graph validates and comes back with an order`` () =
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "Reads a file." ]

        let graph =
            spec
                [ toolNode "read" "read_file" [ "path", "README.md" ]
                  reasonNode "summarise" "Summarise what you were given." ]
                [ edge "read" "summarise" ]

        let verdict = GraphEditorBridge.validate registry graph

        Assert.True(verdict.Valid, verdict.Errors |> List.map (fun e -> e.Message) |> String.concat "; ")
        Assert.Equal<string>([ "read"; "summarise" ], verdict.ExecutionOrder)

    [<Fact>]
    let ``an undescribed tool is refused by the editor and only by the editor`` () =
        withDescriptors [] // nothing described
        let registry = registryOf [ "run_shell", "Runs a command." ]

        let verdict =
            GraphEditorBridge.validate registry (spec [ toolNode "go" "run_shell" [] ] [])

        Assert.False(verdict.Valid)
        Assert.Contains(verdict.Errors, fun e -> e.Code = "undescribed_tool")

        // The point of failing closed *here*: the tool itself is untouched and still
        // resolves from the registry, so nothing else in TARS stopped working.
        Assert.True((registry.Get "run_shell").IsSome)

    [<Fact>]
    let ``a cycle is reported rather than run forever`` () =
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "Reads a file." ]

        let graph =
            spec [ reasonNode "a" "first"; reasonNode "b" "second" ] [ edge "a" "b"; edge "b" "a" ]

        let verdict = GraphEditorBridge.validate registry graph

        Assert.False(verdict.Valid)
        Assert.Contains(verdict.Errors, fun e -> e.Code = "cycle")
        Assert.Empty(verdict.ExecutionOrder)

    [<Fact>]
    let ``a missing required argument is named`` () =
        withDescriptors [ writingTool ]
        let registry = registryOf [ "write_code", "Writes a file." ]

        // path supplied, content not.
        let verdict =
            GraphEditorBridge.validate registry (spec [ toolNode "w" "write_code" [ "path", "out.fs" ] ] [])

        Assert.False(verdict.Valid)

        let missing = verdict.Errors |> List.filter (fun e -> e.Code = "missing_argument")

        Assert.Single(missing) |> ignore
        Assert.Contains("content", missing.Head.Message)

    [<Fact>]
    let ``a node kind we cannot round-trip is reported, never quietly dropped`` () =
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "r" ]

        let control =
            { reasonNode "branch" "" with
                Kind = "control"
                Prompt = None }

        let verdict = GraphEditorBridge.validate registry (spec [ control ] [])

        // `ControlPayload.Observe` carries a live `(string -> string) option`, so such
        // a plan cannot survive JSON at all. Saying so beats handing back a plan that
        // has silently lost a node.
        Assert.False(verdict.Valid)
        Assert.Single(verdict.Unsupported) |> ignore
        Assert.Equal("control", verdict.Unsupported.Head.Kind)

    [<Fact>]
    let ``an edge to nowhere is an error`` () =
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "r" ]

        let verdict =
            GraphEditorBridge.validate registry (spec [ reasonNode "a" "think" ] [ edge "a" "ghost" ])

        Assert.False(verdict.Valid)
        Assert.Contains(verdict.Errors, fun e -> e.Code = "unknown_edge_endpoint")

    [<Fact>]
    let ``what needs approval is listed before anything runs`` () =
        withDescriptors [ readOnlyTool; writingTool ]
        let registry = registryOf [ "read_file", "r"; "write_code", "w" ]

        let graph =
            spec
                [ toolNode "read" "read_file" [ "path", "in.txt" ]
                  toolNode "write" "write_code" [ "path", "out.txt"; "content", "x" ] ]
                []

        let verdict = GraphEditorBridge.validate registry graph

        Assert.True(verdict.Valid, verdict.Errors |> List.map (fun e -> e.Message) |> String.concat "; ")

        // Reading is automatic; writing is not, and the notice says what it does so a
        // person can decide without reading the tool's source.
        Assert.Single(verdict.NeedsApproval) |> ignore
        Assert.Equal("write", verdict.NeedsApproval.Head.NodeId)
        Assert.Contains("writes a file", verdict.NeedsApproval.Head.Effect)

    // =========================================================================
    // Running
    // =========================================================================

    /// Records the plans it is asked to run, and reports success without running any.
    let private recordingExecutor () =
        let seen = ResizeArray<WoTPlan>()

        let executor (plan: WoTPlan) =
            async {
                seen.Add plan

                return
                    { Output = "ran"
                      Success = true
                      Trace =
                        { RunId = Guid.NewGuid()
                          Plan = plan
                          Steps =
                            plan.Nodes
                            |> List.map (fun n ->
                                { NodeId = n.Id
                                  NodeType = string n.Kind
                                  StartedAt = DateTime.UtcNow
                                  Status = Completed("ok", 1L)
                                  Input = None
                                  Output = Some "ok"
                                  Confidence = None
                                  TokensUsed = None })
                          StartedAt = DateTime.UtcNow
                          CompletedAt = Some DateTime.UtcNow
                          FinalStatus = "completed" }
                      TriplesDelta = []
                      ToolsUsed = []
                      Metrics =
                        { TotalSteps = plan.Nodes.Length
                          SuccessfulSteps = plan.Nodes.Length
                          FailedSteps = 0
                          TotalTokens = 0
                          TotalDurationMs = 0L
                          BranchingFactor = 0.0
                          ConstraintScore = None }
                      Warnings = []
                      Errors = []
                      CognitiveStateAfter = None }
            }

        executor, seen

    [<Fact>]
    let ``a graph that writes will not run until that node is approved by name`` () =
        withDescriptors [ writingTool ]
        let registry = registryOf [ "write_code", "Writes a file." ]
        let executor, ran = recordingExecutor ()

        let graph =
            spec [ toolNode "w" "write_code" [ "path", "out.fs"; "content", "x" ] ] []

        let refused =
            GraphEditorBridge.run executor registry graph [] |> Async.RunSynchronously

        Assert.False(refused.Success)
        Assert.Empty(ran) // nothing ran at all, not even partly
        Assert.Contains("writes a file", String.concat " " refused.Errors)

        let allowed =
            GraphEditorBridge.run executor registry graph [ "w" ] |> Async.RunSynchronously

        Assert.True(allowed.Success, String.concat "; " allowed.Errors)
        Assert.Single(ran) |> ignore

    [<Fact>]
    let ``approving one node does not approve another`` () =
        withDescriptors [ writingTool ]
        let registry = registryOf [ "write_code", "Writes a file." ]
        let executor, ran = recordingExecutor ()

        let graph =
            spec
                [ toolNode "first" "write_code" [ "path", "a"; "content", "x" ]
                  toolNode "second" "write_code" [ "path", "b"; "content", "y" ] ]
                []

        let result =
            GraphEditorBridge.run executor registry graph [ "first" ]
            |> Async.RunSynchronously

        Assert.False(result.Success)
        Assert.Empty(ran)
        Assert.Contains("second", String.concat " " result.Errors)

    [<Fact>]
    let ``an invalid graph never reaches the executor`` () =
        withDescriptors []
        let registry = registryOf [ "run_shell", "Runs a command." ]
        let executor, ran = recordingExecutor ()

        let result =
            GraphEditorBridge.run executor registry (spec [ toolNode "go" "run_shell" [] ] []) []
            |> Async.RunSynchronously

        Assert.False(result.Success)
        Assert.Empty(ran)

    [<Fact>]
    let ``the plan handed to the executor keeps the graph's shape`` () =
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "r" ]
        let executor, ran = recordingExecutor ()

        let graph =
            { spec
                  [ toolNode "read" "read_file" [ "path", "README.md" ]
                    reasonNode "summarise" "Summarise it." ]
                  [ { edge "read" "summarise" with
                        Label = Some "next" } ] with
                EntryNode = Some "read"
                Policy = Some [ "no-network" ] }

        GraphEditorBridge.run executor registry graph []
        |> Async.RunSynchronously
        |> ignore

        let plan = Assert.Single ran

        // Edges, entry node and policy all survive. A flat step list with `depends_on`
        // could not have carried the last two.
        Assert.Equal("read", plan.EntryNode)
        Assert.Equal<string>([ "no-network" ], plan.Policy)
        Assert.Single plan.Edges |> ignore
        Assert.Equal(Some "next", plan.Edges.Head.Label)
        Assert.Equal<string>([ "read"; "summarise" ], plan.Nodes |> List.map (fun n -> n.Id))

    // =========================================================================
    // The wire
    // =========================================================================

    [<Fact>]
    let ``the JSON entry points answer in the shape the editor expects`` () =
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "Reads a file." ]

        let catalogJson = GraphEditorBridge.catalogJson registry
        Assert.Contains("\"described_tools\"", catalogJson)
        Assert.Contains("\"node_kind\"", catalogJson)
        Assert.Contains("\"auto_approved\"", catalogJson)

        let validateJson =
            GraphEditorBridge.validateJson
                registry
                """{"goal":"g","nodes":[{"id":"a","kind":"reason","prompt":"think"}],"edges":[]}"""

        Assert.Contains("\"valid\":true", validateJson)
        Assert.Contains("\"execution_order\":[\"a\"]", validateJson)

    [<Fact>]
    let ``malformed JSON is an error, not an exception`` () =
        withDescriptors []
        let registry = registryOf []

        let answer = GraphEditorBridge.validateJson registry "{ not json at all"

        Assert.Contains("\"valid\":false", answer)
        Assert.Contains("malformed", answer)
