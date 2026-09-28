namespace Tars.Tests

open System
open System.Text.Json
open Xunit
open Tars.Core
open Tars.Cortex

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
