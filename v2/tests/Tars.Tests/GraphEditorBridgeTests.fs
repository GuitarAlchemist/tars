namespace Tars.Tests

open System
open System.Text.Json
open Xunit
open Tars.Core
open Tars.Cortex
open Tars.Cortex.WoTTypes

/// What can go in a graph, and how a graph arrives.
///
/// This is the contract the GA graph editor is built against, so where the shape on
/// the wire is the point, the assertion is against the JSON rather than the F# record
/// behind it.
///
/// The metadata registry is process-wide, so the modules that write to it share a
/// collection: xUnit runs collections in parallel, and one test calling `clear()`
/// while another counts is a race, not a failure of either test.
[<Xunit.Collection("ToolMetadata registry")>]
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

    let private parsed input =
        match GraphEditorBridge.parsePlanSpec input with
        | Result.Ok spec -> spec
        | Result.Error message -> failwith $"expected a graph, got: {message}"

    let private refusal input =
        match GraphEditorBridge.parsePlanSpec input with
        | Result.Error message -> message
        | Result.Ok _ -> failwith "expected a refusal, got a graph"

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

    [<Fact>]
    let ``a reasoning node is classed by its reach, not by the fact that it only returns text`` () =
        withDescriptors []
        let catalog = GraphEditorBridge.catalog (registryOf [])
        let reason = catalog.Nodes |> List.find (fun n -> n.Name = "reason")

        // The tier answers "what is the worst this can do". A model call sends the
        // prompt to a remote paid service, and the tier documentation names LLM calls
        // under `escapes`; `read_only` contradicted it.
        Assert.Equal("escapes", reason.Approval.Tier)
        Assert.False(reason.Approval.AutoApproved)

    [<Fact>]
    let ``every node in the catalog carries the schema an editor needs to build a form`` () =
        withDescriptors [ writingTool ]

        let catalog =
            GraphEditorBridge.catalog (registryOf [ "write_code", "Writes a file." ])

        // An entry whose schema did not parse would be an entry the editor cannot draw
        // a form for, and the catalog is the only place it learns the shape.
        for entry in catalog.Nodes do
            use doc = JsonDocument.Parse entry.InputSchema
            Assert.Equal("object", doc.RootElement.GetProperty("type").GetString())

            let declared =
                doc.RootElement.GetProperty("properties").EnumerateObject()
                |> Seq.map (fun p -> p.Name)
                |> Seq.toList

            for required in entry.Required do
                Assert.Contains(required, declared)

    // =========================================================================
    // The wire
    // =========================================================================

    [<Fact>]
    let ``the catalog serializes under the names the editor reads`` () =
        withDescriptors [ readOnlyTool ]
        let json = GraphEditorBridge.catalogJson (registryOf [ "read_file", "Reads a file." ])

        Assert.Contains("\"described_tools\"", json)
        Assert.Contains("\"total_tools\"", json)
        Assert.Contains("\"node_kind\"", json)
        Assert.Contains("\"auto_approved\"", json)

        // Snake case everywhere, so a client written against one field name is not
        // surprised by another.
        use doc = JsonDocument.Parse json
        Assert.Equal(1, doc.RootElement.GetProperty("described_tools").GetInt32())

    // =========================================================================
    // How the graph actually arrives
    // =========================================================================

    [<Fact>]
    let ``a graph wrapped in the MCP arguments envelope is still a graph`` () =
        let plan =
            """{"goal":"g","nodes":[{"id":"a","kind":"reason","prompt":"think"}],"edges":[]}"""

        // `McpServer.handleListTools` advertises every tool as taking one string
        // property called `arguments`, and `handleCallTool` hands the containing object
        // straight through. A client that follows that schema sends this shape, and
        // reading `nodes` off it finds nothing — the graph arrived empty however well
        // it was formed.
        let enveloped = JsonSerializer.Serialize {| arguments = plan |}

        Assert.Equal(parsed plan, parsed enveloped)
        Assert.Equal(1, (parsed enveloped).Nodes.Length)

    [<Fact>]
    let ``an object that merely has an arguments field is not an envelope`` () =
        // Unwrapping anything with an `arguments` property would swallow a real graph
        // whose own fields happen to include one. Only a lone string `arguments` is
        // the envelope.
        let notAnEnvelope =
            """{"goal":"g","arguments":"not the graph","nodes":[{"id":"a","kind":"reason","prompt":"t"}]}"""

        let spec = parsed notAnEnvelope
        Assert.Equal("g", spec.Goal)
        Assert.Equal(1, spec.Nodes.Length)

    [<Fact>]
    let ``malformed JSON is an error, not an exception`` () =
        Assert.Contains("not valid JSON", refusal "{ not json at all")

    [<Fact>]
    let ``an edges field that is not a list is refused, not ignored`` () =
        // A graph whose dependencies were written into a malformed `edges` field used
        // to arrive with no edges at all, and would then run every node in an
        // unrelated order — which for an approved mutating node is the graph nobody
        // drew.
        let message =
            refusal """{"goal":"g","nodes":[{"id":"a","kind":"reason","prompt":"t"}],"edges":"a->b"}"""

        Assert.Contains("edges", message)

    [<Fact>]
    let ``a nodes field that is not a list is refused`` () =
        Assert.Contains("nodes", refusal """{"goal":"g","nodes":"a, b"}""")

    [<Fact>]
    let ``a policy field that is not a list is refused`` () =
        // A policy list says what the graph may *not* do, so reading a malformed one
        // as "no restrictions" fails in the wrong direction.
        let message =
            refusal
                """{"goal":"g","nodes":[{"id":"a","kind":"reason","prompt":"t"}],"edges":[],"policy":"no-network"}"""

        Assert.Contains("policy", message)

    [<Fact>]
    let ``a policy entry that is not a string is refused, not dropped`` () =
        // Dropping it would turn this into a policy carrying one restriction instead
        // of the two that were written, and a restriction that quietly disappears
        // fails in the direction that lets more happen.
        let message =
            refusal
                """{"goal":"g","nodes":[{"id":"a","kind":"reason","prompt":"t"}],
                        "policy":["no-network",{"rule":"no-writes"}]}"""

        Assert.Contains("policy", message)
        Assert.Contains("not a string", message)

    [<Fact>]
    let ``a policy of strings is read whole`` () =
        let spec =
            parsed
                """{"goal":"g","nodes":[{"id":"a","kind":"reason","prompt":"t"}],
                        "policy":["no-network","no-writes"]}"""

        Assert.Equal<string>([ "no-network"; "no-writes" ], spec.Policy.Value)

    [<Fact>]
    let ``a tag that is not a string is refused, and the node is named`` () =
        // Tags are only labels, but a list that silently loses entries is the same
        // bug wherever it is, and the message has to say which node to go and fix.
        let message =
            refusal """{"goal":"g","nodes":[{"id":"a","kind":"reason","prompt":"t","tags":["ok",7]}]}"""

        Assert.Contains("'a'", message)
        Assert.Contains("tags", message)

    [<Fact>]
    let ``arguments that are not an object are refused, not read as no arguments`` () =
        // Read as absent, a tool whose arguments arrived as a string is called with
        // none at all — and a tool that requires nothing would then actually run.
        let message =
            refusal """{"goal":"g","nodes":[{"id":"c","kind":"tool","tool":"count_lines","arguments":"path=src"}]}"""

        Assert.Contains("'c'", message)
        Assert.Contains("arguments", message)

    [<Fact>]
    let ``an entry node that is not a string is refused`` () =
        // Read as absent it leaves no entry node declared, and since the executor
        // walks the node list in order, that silently changes what runs first.
        let message =
            refusal """{"goal":"g","entry_node":3,"nodes":[{"id":"a","kind":"reason","prompt":"t"}]}"""

        Assert.Contains("entry_node", message)
        Assert.Contains("not a string", message)

    [<Fact>]
    let ``a goal that is not a string is refused rather than becoming empty`` () =
        Assert.Contains("goal", refusal """{"goal":["a","b"],"nodes":[{"id":"a","kind":"reason","prompt":"t"}]}""")

    [<Fact>]
    let ``a confidence that is not a number is refused, and the edge is named`` () =
        let message =
            refusal
                """{"goal":"g","nodes":[{"id":"a","kind":"reason","prompt":"t"},
                        {"id":"b","kind":"reason","prompt":"t"}],
                        "edges":[{"from":"a","to":"b","confidence":"high"}]}"""

        Assert.Contains("'a'", message)
        Assert.Contains("confidence", message)

    [<Fact>]
    let ``a null optional field means absent, not malformed`` () =
        // `null` is how a serializer writes "nothing here". Refusing it would make
        // every client that round-trips a graph through a nullable type fail.
        let spec =
            parsed
                """{"id":null,"goal":"g","entry_node":null,"policy":null,
                        "nodes":[{"id":"a","kind":"reason","prompt":"t","arguments":null,
                                  "hint":null,"label":null,"tags":null}]}"""

        Assert.Equal(None, spec.EntryNode)
        Assert.Equal(None, spec.Policy)
        Assert.Equal(None, spec.Nodes.Head.Arguments)
        Assert.Equal(None, spec.Nodes.Head.Tags)

    [<Fact>]
    let ``an absent list field is still fine`` () =
        // Absent means "none given". Only present-and-wrong is an error.
        let spec = parsed """{"goal":"g","nodes":[{"id":"a","kind":"reason","prompt":"t"}]}"""

        Assert.Empty(spec.Edges)
        Assert.Equal(None, spec.Policy)

    [<Fact>]
    let ``a tool node keeps its arguments as the JSON they arrived as`` () =
        // The arguments are handed to a tool whose schema declares types, so a number
        // has to still be a number here. Flattening them to strings would make every
        // typed field fail its own schema later.
        let spec =
            parsed
                """{"goal":"g","nodes":[{"id":"c","kind":"tool","tool":"count_lines",
                        "arguments":{"path":"src","recursive":true,"max":10}}]}"""

        let arguments = spec.Nodes.Head.Arguments.Value

        Assert.Equal(JsonValueKind.String, arguments.["path"].ValueKind)
        Assert.Equal(JsonValueKind.True, arguments.["recursive"].ValueKind)
        Assert.Equal(JsonValueKind.Number, arguments.["max"].ValueKind)

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
            GraphEditorBridge.run (fun () -> executor) registry graph [] |> Async.RunSynchronously

        Assert.False(refused.Success)
        Assert.Empty(ran) // nothing ran at all, not even partly
        Assert.Contains("writes a file", String.concat " " refused.Errors)

        let allowed =
            GraphEditorBridge.run (fun () -> executor) registry graph [ "w" ] |> Async.RunSynchronously

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
            GraphEditorBridge.run (fun () -> executor) registry graph [ "first" ]
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
            GraphEditorBridge.run (fun () -> executor) registry (spec [ toolNode "go" "run_shell" [] ] []) []
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
                EntryNode = Some "read" }

        // "summarise" is named because a reason node spends a model call.
        GraphEditorBridge.run (fun () -> executor) registry graph [ "summarise" ]
        |> Async.RunSynchronously
        |> ignore

        let plan = Assert.Single ran

        // Edges and entry node both survive. A flat step list with `depends_on` could
        // not have carried either.
        Assert.Equal("read", plan.EntryNode)
        Assert.Single plan.Edges |> ignore
        Assert.Equal(Some "next", plan.Edges.Head.Label)
        Assert.Equal<string>([ "read"; "summarise" ], plan.Nodes |> List.map (fun n -> n.Id))

    [<Fact>]
    let ``a graph carrying a policy nobody enforces is refused rather than run`` () =
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "r" ]
        let executor, ran = recordingExecutor ()

        let graph =
            { spec [ toolNode "read" "read_file" [ "path", "README.md" ] ] [] with
                Policy = Some [ "no-network" ] }

        // `WoTExecutor` never reads `plan.Policy` (tars#334). Running this would tell
        // the caller their restriction held while the graph did whatever it liked.
        let result =
            GraphEditorBridge.run (fun () -> executor) registry graph []
            |> Async.RunSynchronously

        Assert.False(result.Success)
        Assert.Contains("no-network", String.concat " " result.Errors)
        Assert.Empty(ran)

        // Reading a graph does nothing, so validation only says so.
        let verdict = GraphEditorBridge.validate registry graph
        Assert.True(verdict.Valid)
        Assert.Contains(verdict.Warnings, fun (w: string) -> w.Contains "nothing enforces a policy")

    [<Fact>]
    let ``the policy still reaches the plan, so enforcing it later needs no new wiring`` () =
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "r" ]

        let graph =
            { spec [ toolNode "read" "read_file" [ "path", "README.md" ] ] [] with
                Policy = Some [ "no-network" ] }

        let plan = GraphEditorBridge.toWoTPlan registry graph [ "read" ]
        Assert.Equal<string>([ "no-network" ], plan.Policy)

    [<Fact>]
    let ``a refused graph never builds an executor`` () =
        withDescriptors [ writingTool ]
        let registry = registryOf [ "write_code", "Writes a file." ]

        // Building one reaches the LLM configuration and creates directories and
        // stores under `~/.tars`. A refused graph that had done that would have
        // changed this machine while reporting that it ran nothing.
        let mutable built = 0

        let getExecutor () =
            built <- built + 1
            fst (recordingExecutor ())

        let unapproved =
            spec [ toolNode "w" "write_code" [ "path", "out.fs"; "content", "x" ] ] []

        GraphEditorBridge.run getExecutor registry unapproved []
        |> Async.RunSynchronously
        |> ignore

        GraphEditorBridge.run getExecutor registry (spec [] []) []
        |> Async.RunSynchronously
        |> ignore

        Assert.Equal(0, built)

        GraphEditorBridge.run getExecutor registry unapproved [ "w" ]
        |> Async.RunSynchronously
        |> ignore

        Assert.Equal(1, built)

    // =========================================================================
    // Casing is a convenience, not a way past the gate
    // =========================================================================

    [<Fact>]
    let ``a tool that only matches a description by case is treated as undescribed`` () =
        // `ToolMetadata` is keyed case-insensitively; `ToolRegistry` is not. A registry
        // holding both the described `read_file` and a custom, mutating `READ_FILE`
        // would otherwise hand the second one the first one's schema and its
        // `auto_approved: true` — a tool that writes, running unasked, advertised as
        // read-only.
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "r"; "READ_FILE", "a different tool entirely" ]
        let executor, ran = recordingExecutor ()

        let graph = spec [ toolNode "x" "READ_FILE" [ "path", "README.md" ] ] []
        let verdict = GraphEditorBridge.validate registry graph

        Assert.False(verdict.Valid)
        Assert.Equal<string>([ "undescribed_tool" ], verdict.Errors |> List.map (fun e -> e.Code))
        Assert.Empty(verdict.NeedsApproval)

        GraphEditorBridge.run (fun () -> executor) registry graph [ "x" ]
        |> Async.RunSynchronously
        |> ignore

        Assert.Empty(ran)

    [<Fact>]
    let ``a name matching two registered tools by case alone is refused, not guessed`` () =
        // The registry allows both spellings, so picking either would be picking for
        // the caller.
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "r"; "READ_FILE", "a different tool entirely" ]

        let verdict =
            GraphEditorBridge.validate registry (spec [ toolNode "x" "Read_File" [ "path", "README.md" ] ] [])

        Assert.False(verdict.Valid)
        Assert.Equal<string>([ "ambiguous_tool" ], verdict.Errors |> List.map (fun e -> e.Code))

    [<Fact>]
    let ``a casing variant is still the same tool when nothing collides`` () =
        // The convenience this was in aid of has to survive the fix.
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "r" ]

        let verdict =
            GraphEditorBridge.validate registry (spec [ toolNode "x" "Read_File" [ "path", "README.md" ] ] [])

        Assert.True(verdict.Valid, verdict.Errors |> List.map (fun e -> e.Message) |> String.concat "; ")

    // =========================================================================
    // What the executor is handed
    // =========================================================================

    [<Fact>]
    let ``a structured argument reaches the tool as the JSON it was written as`` () =
        let structured: ToolMetadata.ToolDescriptor =
            { Name = "configure"
              InputSchema = ToolMetadata.objectSchema [ "config", "object", "The settings." ] [ "config" ]
              Required = [ "config" ]
              Approval = ToolMetadata.Approval.observes "reads a configuration" }

        withDescriptors [ structured ]
        let registry = registryOf [ "configure", "c" ]

        let node: GraphEditorBridge.NodeSpec =
            { Id = "c"
              Kind = "tool"
              Prompt = None
              Hint = None
              Tool = Some "configure"
              Arguments =
                Map.ofList
                    [ "config",
                      JsonDocument.Parse("""{"x":1,"nested":["a","b"]}""").RootElement.Clone() ]
                |> Some
              Label = None
              Tags = None }

        let plan = GraphEditorBridge.toWoTPlan registry (spec [ node ] []) [ "c" ]
        let payload = plan.Nodes.Head.Payload :?> ToolPayload

        // `WoTExecutor.serializeToolArgs` serializes the whole argument map before
        // calling the tool. Carrying the structure as raw *text* meant it was encoded a
        // second time — the tool received `{"config":"{\"x\":1}"}`, which no longer
        // matches the schema the catalog published for it.
        let serialized = JsonSerializer.Serialize payload.Args
        use doc = JsonDocument.Parse serialized
        let config = doc.RootElement.GetProperty "config"

        Assert.Equal(JsonValueKind.Object, config.ValueKind)
        Assert.Equal(1, config.GetProperty("x").GetInt32())
        Assert.Equal(2, config.GetProperty("nested").GetArrayLength())

    [<Fact>]
    let ``a run with a failed step says the rest of the graph ran anyway`` () =
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "r" ]

        // `WoTExecutor` does not stop at the first failure, and a later node is handed
        // the previous output in place of the missing one (tars#335). The runner cannot
        // prevent that from here — it hands over the whole plan in one call — so it has
        // to say so rather than report a clean run.
        let failing (plan: WoTPlan) =
            async {
                return
                    { Output = "ran"
                      Success = false
                      Trace =
                        { RunId = Guid.NewGuid()
                          Plan = plan
                          Steps =
                            plan.Nodes
                            |> List.mapi (fun i n ->
                                { NodeId = n.Id
                                  NodeType = string n.Kind
                                  StartedAt = DateTime.UtcNow
                                  Status = (if i = 0 then Failed("no such file", 1L) else Completed("ok", 1L))
                                  Input = None
                                  Output = Some "ok"
                                  Confidence = None
                                  TokensUsed = None })
                          StartedAt = DateTime.UtcNow
                          CompletedAt = Some DateTime.UtcNow
                          FinalStatus = "Failed" }
                      TriplesDelta = []
                      ToolsUsed = []
                      Metrics =
                        { TotalSteps = plan.Nodes.Length
                          SuccessfulSteps = plan.Nodes.Length - 1
                          FailedSteps = 1
                          TotalTokens = 0
                          TotalDurationMs = 0L
                          BranchingFactor = 0.0
                          ConstraintScore = None }
                      Warnings = []
                      Errors = [ "no such file" ]
                      CognitiveStateAfter = None }
            }

        let graph =
            spec
                [ toolNode "check" "read_file" [ "path", "missing.txt" ]
                  toolNode "then" "read_file" [ "path", "README.md" ] ]
                [ edge "check" "then" ]

        let result =
            GraphEditorBridge.run (fun () -> failing) registry graph []
            |> Async.RunSynchronously

        let warnings = String.concat " " result.Warnings
        Assert.Contains("check", warnings)
        Assert.Contains("ran anyway", warnings)

    // =========================================================================
    // The wire
    // =========================================================================

    [<Fact>]
    let ``the validator answers in the shape the editor expects`` () =
        withDescriptors []
        let registry = registryOf []

        let answer =
            GraphEditorBridge.validateJson
                registry
                """{"goal":"g","nodes":[{"id":"a","kind":"reason","prompt":"think"}],"edges":[]}"""

        Assert.Contains("\"valid\":true", answer)
        Assert.Contains("\"execution_order\":[\"a\"]", answer)
        Assert.Contains("\"needs_approval\"", answer)

    [<Fact>]
    let ``malformed JSON comes back as an invalid graph, not an exception`` () =
        withDescriptors []
        let answer = GraphEditorBridge.validateJson (registryOf []) "{ not json at all"

        // `parsePlanSpec` already refuses this; the point here is that the JSON entry
        // point turns that refusal into a verdict rather than letting it escape as an
        // exception through the MCP server.
        Assert.Contains("\"valid\":false", answer)
        Assert.Contains("malformed", answer)

    // =========================================================================
    // How the graph actually arrives
    // =========================================================================

    [<Fact>]
    let ``the envelope is unwrapped by the validator too, not only by the parser`` () =
        withDescriptors []
        let registry = registryOf []

        let plan =
            """{"goal":"g","nodes":[{"id":"a","kind":"reason","prompt":"think"}],"edges":[]}"""

        let enveloped = JsonSerializer.Serialize {| arguments = plan |}

        Assert.Equal(GraphEditorBridge.validateJson registry plan, GraphEditorBridge.validateJson registry enveloped)

    [<Fact>]
    let ``a malformed approve list is said out loud, not read as no approvals`` () =
        withDescriptors [ writingTool ]
        let registry = registryOf [ "write_code", "Writes a file." ]
        let executor, ran = recordingExecutor ()

        // Dropping the entries that are not strings would refuse the run too, but with
        // "node 'w' runs write_code, approve it by name" — telling the caller to do
        // the thing they just did. The refusal has to name the real reason.
        let answer =
            GraphEditorBridge.runJson
                (fun () -> executor)
                registry
                """{"goal":"g","nodes":[{"id":"w","kind":"tool","tool":"write_code",
                        "arguments":{"path":"out.fs","content":"x"}}],"approve":["w",7]}"""
            |> Async.RunSynchronously

        Assert.Contains("\"success\":false", answer)
        Assert.Contains("approve", answer)
        Assert.Contains("not a string", answer)
        Assert.Empty(ran)

    [<Fact>]
    let ``approvals survive the envelope too`` () =
        withDescriptors [ writingTool ]
        let registry = registryOf [ "write_code", "Writes a file." ]
        let executor, ran = recordingExecutor ()

        let plan =
            """{"goal":"g","nodes":[{"id":"w","kind":"tool","tool":"write_code",
                    "arguments":{"path":"out.fs","content":"x"}}],"edges":[],"approve":["w"]}"""

        // `approve` is a sibling of `nodes`, so it has to come out of the same
        // envelope. Read from the wrapper it is invisible, and the run is refused even
        // though the caller approved the node.
        let enveloped = JsonSerializer.Serialize {| arguments = plan |}

        let answer =
            GraphEditorBridge.runJson (fun () -> executor) registry enveloped |> Async.RunSynchronously

        Assert.Contains("\"success\":true", answer)
        Assert.Single(ran) |> ignore

    // =========================================================================
    // What the catalog promises, the gate must ask for
    // =========================================================================

    [<Fact>]
    let ``a reason node costs a model call, so it needs approving too`` () =
        withDescriptors []
        let registry = registryOf []
        let executor, ran = recordingExecutor ()

        let graph = spec [ reasonNode "think" "Work it out." ] []

        // The catalog says the reason node is not auto-approved. The validator has to
        // ask for that approval, or the promise is decoration: `tars_plan_run` would
        // spend model calls on an empty `approve` list.
        let verdict = GraphEditorBridge.validate registry graph
        Assert.True(verdict.Valid)
        Assert.Single(verdict.NeedsApproval) |> ignore
        Assert.Equal("think", verdict.NeedsApproval.Head.NodeId)

        let refused =
            GraphEditorBridge.run (fun () -> executor) registry graph [] |> Async.RunSynchronously

        Assert.False(refused.Success)
        Assert.Empty(ran)

        let allowed =
            GraphEditorBridge.run (fun () -> executor) registry graph [ "think" ]
            |> Async.RunSynchronously

        Assert.True(allowed.Success, String.concat "; " allowed.Errors)
        Assert.Single(ran) |> ignore

    [<Fact>]
    let ``every catalog entry that is not auto-approved is one the gate asks about`` () =
        withDescriptors [ readOnlyTool; writingTool ]
        let registry = registryOf [ "read_file", "r"; "write_code", "w" ]
        let catalog = GraphEditorBridge.catalog registry

        // One node per catalog entry, so the two cannot drift apart again.
        let nodes =
            catalog.Nodes
            |> List.map (fun entry ->
                if entry.NodeKind = "reason" then
                    reasonNode entry.Name "think"
                else
                    toolNode entry.Name entry.Name (entry.Required |> List.map (fun r -> r, "x")))

        let verdict = GraphEditorBridge.validate registry (spec nodes [])
        Assert.True(verdict.Valid, verdict.Errors |> List.map (fun e -> e.Message) |> String.concat "; ")

        let expected =
            catalog.Nodes
            |> List.filter (fun e -> not e.Approval.AutoApproved)
            |> List.map (fun e -> e.Name)
            |> List.sort

        let asked = verdict.NeedsApproval |> List.map (fun n -> n.NodeId) |> List.sort

        Assert.Equal<string>(expected, asked)

    // =========================================================================
    // The entry node
    // =========================================================================

    [<Fact>]
    let ``the declared entry node runs first, even when it sorts last`` () =
        withDescriptors []
        let registry = registryOf []
        let executor, ran = recordingExecutor ()

        // "a" and "z" both have no dependencies, so alphabetical order would put "a"
        // first. The graph says to start at "z".
        let graph =
            { spec [ reasonNode "a" "first?"; reasonNode "z" "declared entry" ] [] with
                EntryNode = Some "z" }

        let verdict = GraphEditorBridge.validate registry graph
        Assert.True(verdict.Valid, verdict.Errors |> List.map (fun e -> e.Message) |> String.concat "; ")
        Assert.Equal("z", verdict.ExecutionOrder.Head)

        // And the order is what the executor actually sees: `WoTExecutor` walks
        // `plan.Nodes` in list order and never reads `EntryNode`, so the field alone
        // would have changed nothing about what runs first.
        GraphEditorBridge.run (fun () -> executor) registry graph [ "a"; "z" ]
        |> Async.RunSynchronously
        |> ignore

        let plan = Assert.Single ran
        Assert.Equal("z", plan.Nodes.Head.Id)

    [<Fact>]
    let ``a node something runs into cannot be the entry node`` () =
        withDescriptors []
        let registry = registryOf []

        let graph =
            { spec [ reasonNode "a" "first"; reasonNode "b" "second" ] [ edge "a" "b" ] with
                EntryNode = Some "b" }

        let verdict = GraphEditorBridge.validate registry graph

        Assert.False(verdict.Valid)
        Assert.Contains(verdict.Errors, fun e -> e.Code = "entry_node_not_a_start")

    [<Fact>]
    let ``nodes the entry node cannot reach are called out, because they run anyway`` () =
        withDescriptors []
        let registry = registryOf []

        // Two disconnected components. The executor walks every node in the list, so
        // "orphan" runs whatever the entry node says — the editor should know.
        let graph =
            { spec
                  [ reasonNode "start" "here"
                    reasonNode "next" "then here"
                    reasonNode "orphan" "nobody points at me" ]
                  [ edge "start" "next" ] with
                EntryNode = Some "start" }

        let verdict = GraphEditorBridge.validate registry graph

        Assert.True(verdict.Valid, verdict.Errors |> List.map (fun e -> e.Message) |> String.concat "; ")

    // =========================================================================
    // The schema is a promise the validator has to keep
    // =========================================================================

    [<Fact>]
    let ``an argument of the wrong type is refused before it reaches the tool`` () =
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "r" ]
        let executor, ran = recordingExecutor ()

        // The catalog says `path` is a string. A node sending 42 used to validate and
        // reach the tool, which then failed with whatever it makes of a number —
        // catching it here is the reason for publishing a schema at all.
        let numericPath: GraphEditorBridge.NodeSpec =
            { toolNode "read" "read_file" [] with
                Arguments = Some(Map.ofList [ "path", JsonDocument.Parse("42").RootElement.Clone() ]) }

        let verdict = GraphEditorBridge.validate registry (spec [ numericPath ] [])

        Assert.False(verdict.Valid)
        Assert.Contains(verdict.Errors, fun e -> e.Code = "wrong_argument_type")

        GraphEditorBridge.run (fun () -> executor) registry (spec [ numericPath ] []) []
        |> Async.RunSynchronously
        |> ignore

        Assert.Empty(ran)

    [<Fact>]
    let ``an argument the tool does not take is named`` () =
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "r" ]

        // The schemas say additionalProperties: false, so a stray argument is a
        // mistake to report rather than something to pass along and hope about.
        let verdict =
            GraphEditorBridge.validate
                registry
                (spec [ toolNode "read" "read_file" [ "path", "README.md"; "encoding", "utf-8" ] ] [])

        Assert.False(verdict.Valid)

        let unknown = verdict.Errors |> List.filter (fun e -> e.Code = "unknown_argument")
        Assert.Single(unknown) |> ignore
        Assert.Contains("encoding", unknown.Head.Message)

    [<Fact>]
    let ``a tool named in another casing is the same tool`` () =
        withDescriptors [ readOnlyTool ]
        let registry = registryOf [ "read_file", "r" ]
        let executor, ran = recordingExecutor ()

        // `ToolMetadata` is keyed case-insensitively and says why: names are typed by
        // hand into graphs. `ToolRegistry` is case-sensitive, so the exact lookup
        // answered `unknown_tool` before the metadata lookup could run.
        let graph = spec [ toolNode "read" "Read_File" [ "path", "README.md" ] ] []

        let verdict = GraphEditorBridge.validate registry graph
        Assert.True(verdict.Valid, verdict.Errors |> List.map (fun e -> e.Message) |> String.concat "; ")

        GraphEditorBridge.run (fun () -> executor) registry graph []
        |> Async.RunSynchronously
        |> ignore

        // And the plan carries the registry's spelling, not the graph's: the executor
        // looks the name up in that same case-sensitive registry, so accepting the
        // variant without canonicalising it would only move the failure later.
        let plan = Assert.Single ran
        let payload = plan.Nodes.Head.Payload :?> ToolPayload
        Assert.Equal("read_file", payload.Tool)
