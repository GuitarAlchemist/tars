namespace Tars.Cortex

open System
open System.Text.Json
open System.Text.Json.Serialization
open Tars.Core
open Tars.Cortex.WoTTypes

/// The editor-facing surface: a catalog of what can go in a graph, a validator for
/// a graph, and a runner for one.
///
/// Three things this is deliberately *not*:
///
/// 1. It is not `ClaudeCodeBridge`. That bridge serializes a plan for a caller that
///    already has one and walks it step by step, and its manifest is lossy on
///    purpose — edges become a `Next` list, `Policy` and `Metadata` are dropped, tool
///    arguments are `Map<string, string>`. An editor has to hand back what it was
///    given, so this module has its own wire shape that round-trips.
///
/// 2. It is not a second privileged path. Everything runs through `IToolRegistry`
///    and the same approval rules apply to a graph an agent proposed and a graph a
///    person drew.
///
/// 3. It does not gate `ToolRegistry`. A tool with no `ToolMetadata` description is
///    absent from *this* catalog and refused by *this* validator; every existing
///    caller keeps working. That is what makes the backfill safe to do gradually.
///
/// Supported node kinds are `Reason` and `Tool`. `Control` and `Validate` are
/// reported as `unsupported` rather than silently flattened: `ControlPayload.Observe`
/// carries a `(string -> string) option`, a live function, so a graph containing one
/// cannot survive a round trip through JSON at all, and quietly dropping it would
/// hand back a different plan than the one that was opened.
module GraphEditorBridge =

    let private jsonOptions =
        let o =
            JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower)

        o.DefaultIgnoreCondition <- JsonIgnoreCondition.WhenWritingNull
        o.WriteIndented <- false
        o

    // =========================================================================
    // Wire shapes
    // =========================================================================

    /// One entry in the node catalog.
    ///
    /// `node_kind` separates the two things a graph can hold, because they are not
    /// interchangeable and the difference is invisible from a tool list alone: a
    /// `tool` entry acts, a `reason` entry only produces text. `WoTExecutor`'s
    /// reasoning path builds a prompt with no tools attached, so a reasoning node
    /// asked to "deploy" can only write prose claiming it did (tars#326).
    type CatalogEntry =
        {
            Name: string
            Description: string
            /// "tool" or "reason"
            NodeKind: string
            /// JSON Schema for this node's arguments. Reason nodes take a prompt.
            InputSchema: string
            Required: string list
            Approval: ApprovalWire
        }

    and ApprovalWire =
        { Tier: string
          Effect: string
          AutoApproved: bool }

    type CatalogResponse =
        {
            Nodes: CatalogEntry list
            /// How many of the registry's tools carry a description. The rest are not
            /// broken — they are simply not offered here yet.
            DescribedTools: int
            TotalTools: int
        }

    /// A node as the editor sends it back.
    type NodeSpec =
        {
            Id: string
            /// "reason" or "tool"
            Kind: string
            /// Reason nodes only.
            Prompt: string option
            Hint: string option
            /// Tool nodes only.
            Tool: string option
            Arguments: Map<string, JsonElement> option
            Label: string option
            Tags: string list option
        }

    type EdgeSpec =
        { From: string
          To: string
          Label: string option
          Confidence: float option }

    /// A whole graph, in the shape the editor holds it.
    ///
    /// WoTPlan-native — nodes, edges, entry node, policy — rather than a flat list of
    /// steps with `depends_on`. A `WoTPlan` has an entry node and a policy list that a
    /// step list cannot carry, and flattening would lose them on the first save.
    type PlanSpec =
        { Id: string option
          Goal: string
          EntryNode: string option
          Nodes: NodeSpec list
          Edges: EdgeSpec list
          Policy: string list option }

    // =========================================================================
    // Reading JSON
    // =========================================================================

    /// A string property, treating anything that is not one as absent.
    ///
    /// Only for the fields whose absence is *itself* refused further on — `id`,
    /// `kind`, `from`, `to`, `prompt`, `tool`. There, present-but-wrong and absent
    /// both end in a refusal, so the distinction buys nothing.
    let private tryGetString (name: string) (root: JsonElement) =
        match root.TryGetProperty name with
        | true, v when v.ValueKind = JsonValueKind.String -> Some(v.GetString())
        | _ -> None

    /// A string property, or an error when it is present and is not one.
    ///
    /// For every field whose absence is *benign*. Reading a malformed one as absent
    /// is how a graph quietly becomes a different graph: `"entry_node": 3` would
    /// simply leave no entry node declared, and since `WoTExecutor` walks the node
    /// list in order, that changes what runs first without saying anything.
    let private requireStringOrMissing (name: string) (root: JsonElement) =
        match root.TryGetProperty name with
        | true, v when v.ValueKind = JsonValueKind.Null -> Result.Ok None
        | true, v when v.ValueKind = JsonValueKind.String -> Result.Ok(Some(v.GetString()))
        | true, v -> Result.Error $"'{name}' is present but is not a string (it is {v.ValueKind})"
        | _ -> Result.Ok None

    /// A list of strings, or an error when it is present and is not one.
    ///
    /// Absent is fine and means "none given". Present-but-wrong is not, and that holds
    /// one element at a time: dropping the entries that are not strings would turn
    /// `["no-network", {"rule": "x"}]` into a policy carrying one restriction instead
    /// of the two that were written, which fails in the direction that lets more
    /// happen.
    let private requireStringsOrMissing (name: string) (root: JsonElement) =
        match root.TryGetProperty name with
        | true, v when v.ValueKind = JsonValueKind.Null -> Result.Ok None
        | true, v when v.ValueKind = JsonValueKind.Array ->
            let entries = v.EnumerateArray() |> Seq.toList

            match entries |> List.tryFind (fun e -> e.ValueKind <> JsonValueKind.String) with
            | Some wrong -> Result.Error $"'{name}' has an entry that is not a string (it is {wrong.ValueKind})"
            | None -> Result.Ok(Some(entries |> List.map (fun e -> e.GetString())))
        | true, v -> Result.Error $"'{name}' is present but is not a list (it is {v.ValueKind})"
        | _ -> Result.Ok None

    let private parseNode (element: JsonElement) : Result<NodeSpec, string> =
        match tryGetString "id" element, tryGetString "kind" element with
        | None, _ -> Result.Error "a node has no 'id'"
        | _, None -> Result.Error "a node has no 'kind'"
        | Some id, Some kind ->
            // Present-but-not-an-object is an error, not an empty argument map. A tool
            // whose arguments arrived as a string would otherwise be called with none
            // at all — and a tool that requires nothing would then actually run.
            let arguments =
                match element.TryGetProperty "arguments" with
                | true, v when v.ValueKind = JsonValueKind.Object ->
                    v.EnumerateObject()
                    |> Seq.map (fun p -> p.Name, p.Value.Clone())
                    |> Map.ofSeq
                    |> Some
                    |> Result.Ok
                | true, v when v.ValueKind = JsonValueKind.Null -> Result.Ok None
                | true, v -> Result.Error $"'arguments' is present but is not an object (it is {v.ValueKind})"
                | _ -> Result.Ok None

            let fields =
                arguments
                |> Result.bind (fun args ->
                    requireStringOrMissing "hint" element
                    |> Result.bind (fun hint ->
                        requireStringOrMissing "label" element
                        |> Result.bind (fun label ->
                            requireStringsOrMissing "tags" element
                            |> Result.map (fun tags -> args, hint, label, tags))))

            match fields with
            | Result.Error message -> Result.Error $"node '{id}': {message}"
            | Result.Ok(args, hint, label, tags) ->
                Result.Ok
                    { Id = id
                      Kind = kind.ToLowerInvariant()
                      Prompt = tryGetString "prompt" element
                      Hint = hint
                      Tool = tryGetString "tool" element
                      Arguments = args
                      Label = label
                      Tags = tags }

    let private parseEdge (element: JsonElement) : Result<EdgeSpec, string> =
        match tryGetString "from" element, tryGetString "to" element with
        | Some from, Some target ->
            let confidence =
                match element.TryGetProperty "confidence" with
                | true, v when v.ValueKind = JsonValueKind.Number -> Result.Ok(Some(v.GetDouble()))
                | true, v when v.ValueKind = JsonValueKind.Null -> Result.Ok None
                | true, v -> Result.Error $"'confidence' is present but is not a number (it is {v.ValueKind})"
                | _ -> Result.Ok None

            match requireStringOrMissing "label" element, confidence with
            | Result.Error message, _
            | _, Result.Error message -> Result.Error $"edge '{from}' -> '{target}': {message}"
            | Result.Ok label, Result.Ok confidence ->
                Result.Ok
                    { From = from
                      To = target
                      Label = label
                      Confidence = confidence }
        | None, _ -> Result.Error "an edge has no 'from'"
        | _, None -> Result.Error "an edge has no 'to'"

    /// An array property, or an error when it is present and is not an array.
    ///
    /// Absent is fine and means "none given". Present-but-wrong-kind is not: a graph
    /// whose dependencies were written into a malformed `edges` field would otherwise
    /// validate with no edges at all and run every node in an unrelated order, which
    /// is exactly the graph the caller did not draw.
    let private requireArrayOrMissing (name: string) (root: JsonElement) =
        match root.TryGetProperty name with
        | true, v when v.ValueKind = JsonValueKind.Array -> Result.Ok(v.EnumerateArray() |> Seq.toList)
        | true, v when v.ValueKind = JsonValueKind.Null -> Result.Ok []
        | true, v -> Result.Error $"'{name}' is present but is not a list (it is {v.ValueKind})"
        | _ -> Result.Ok []

    /// Strip the envelope the MCP server wraps tool arguments in, if it is there.
    ///
    /// `McpServer.handleListTools` advertises every tool as taking one string property
    /// called `arguments`, and `handleCallTool` hands the containing object straight to
    /// the tool. A client that follows the advertised schema therefore sends
    /// `{"arguments": "{\"goal\": ..., \"nodes\": [...]}"}`, and reading `nodes` off
    /// that finds nothing — the graph would come back as `empty_graph` however well it
    /// was formed. A client that ignores the schema and sends the plan directly works.
    /// Both shapes are accepted here rather than only the second.
    ///
    /// The test is narrow on purpose: exactly one property, named `arguments`, holding
    /// a string. A plan has `goal` and `nodes` at its top level and no `arguments`
    /// there, so there is nothing to confuse it with.
    let private unwrapArguments (json: string) =
        try
            use doc = JsonDocument.Parse(json)
            let root = doc.RootElement

            if root.ValueKind <> JsonValueKind.Object then
                json
            else
                match root.EnumerateObject() |> Seq.toList with
                | [ only ] when only.Name = "arguments" && only.Value.ValueKind = JsonValueKind.String ->
                    only.Value.GetString()
                | _ -> json
        with _ ->
            json

    /// Read a graph out of the JSON the editor sends.
    let parsePlanSpec (input: string) : Result<PlanSpec, string> =
        try
            let json = unwrapArguments input
            use doc = JsonDocument.Parse(json)
            let root = doc.RootElement

            // Every field is read strictly, and the first one that is present and
            // wrong refuses the whole graph. A policy list says what a graph may *not*
            // do, so reading a malformed one as "no restrictions" fails in the
            // direction that lets more happen; `entry_node` read as absent changes
            // what runs first. Silence is the failure mode worth spending code on.
            let policyField = requireStringsOrMissing "policy" root
            let idField = requireStringOrMissing "id" root
            let goalField = requireStringOrMissing "goal" root
            let entryField = requireStringOrMissing "entry_node" root

            let malformedField =
                [ "nodes"; "edges" ]
                |> List.tryPick (fun name ->
                    match requireArrayOrMissing name root with
                    | Result.Error message -> Some message
                    | Result.Ok _ -> None)
                |> Option.orElse (
                    [ policyField |> Result.map ignore
                      idField |> Result.map ignore
                      goalField |> Result.map ignore
                      entryField |> Result.map ignore ]
                    |> List.tryPick (function
                        | Result.Error message -> Some message
                        | Result.Ok _ -> None)
                )

            match malformedField, requireArrayOrMissing "nodes" root, requireArrayOrMissing "edges" root with
            | Some message, _, _ -> Result.Error message
            | _, Result.Error message, _
            | _, _, Result.Error message -> Result.Error message
            | None, Result.Ok rawNodes, Result.Ok rawEdges ->

                let nodes = rawNodes |> List.map parseNode

                let edges = rawEdges |> List.map parseEdge

                let firstError =
                    (nodes
                     |> List.tryPick (function
                         | Result.Error e -> Some e
                         | _ -> None))
                    |> Option.orElse (
                        edges
                        |> List.tryPick (function
                            | Result.Error e -> Some e
                            | _ -> None)
                    )

                match firstError with
                | Some e -> Result.Error e
                | None ->
                    // `malformedField` has already turned any of these being wrong into an
                    // error, so this branch only ever sees good ones.
                    let orNone =
                        function
                        | Result.Ok value -> value
                        | Result.Error _ -> None

                    Result.Ok
                        { Id = orNone idField
                          Goal = orNone goalField |> Option.defaultValue ""
                          EntryNode = orNone entryField
                          Nodes =
                            nodes
                            |> List.choose (function
                                | Result.Ok n -> Some n
                                | _ -> None)
                          Edges =
                            edges
                            |> List.choose (function
                                | Result.Ok e -> Some e
                                | _ -> None)
                          Policy = orNone policyField }
        with ex ->
            Result.Error $"the graph is not valid JSON: {ex.Message}"

    // =========================================================================
    // Catalog
    // =========================================================================

    /// What running a reasoning node does.
    ///
    /// `Escapes`, not `ReadOnly`. The tier is about reach, and a model call sends the
    /// prompt to whatever service is configured — usually a remote, paid one. Calling
    /// it read-only because it returns text rather than writing any would contradict
    /// the tier documentation, which names LLM calls under `Escapes` explicitly.
    ///
    /// A binding of its own rather than four literals inside `reasonEntry`, because
    /// the validator will have to ask for exactly the approval the catalog advertised,
    /// and two copies of that answer would drift.
    let private reasonApproval: ToolMetadata.Approval =
        { Tier = ToolMetadata.Escapes
          Effect =
            "sends the prompt to the configured language model, which is usually a remote paid service, "
            + "and returns its text"
          AutoApproved = false }

    /// The one reasoning node, offered alongside the tools.
    ///
    /// It is in the catalog because a graph that cannot think is not much of a graph,
    /// and it is marked `reason` because it cannot act.
    let private reasonEntry =
        { Name = "reason"
          Description =
            "Ask the model to think about something and produce text. Cannot call tools "
            + "or change anything; its output is prose."
          NodeKind = "reason"
          InputSchema =
            ToolMetadata.objectSchema
                [ "prompt", "string", "What to think about."
                  "hint", "string", "Model to prefer: fast, smart, reasoning, or a model name." ]
                [ "prompt" ]
          Required = [ "prompt" ]
          Approval =
            { Tier = reasonApproval.Tier.Wire
              Effect = reasonApproval.Effect
              AutoApproved = reasonApproval.AutoApproved } }

    /// What can go in a graph: every described tool, plus the reasoning node.
    ///
    /// Undescribed tools are left out rather than guessed at. An editor that offered
    /// a tool without knowing what it does would be inviting a caller to run it blind.
    let catalog (registry: IToolRegistry) : CatalogResponse =
        let tools = registry.GetAll()

        let described =
            tools
            |> List.choose (fun tool ->
                ToolMetadata.tryFind tool.Name
                |> Option.map (fun descriptor ->
                    { Name = tool.Name
                      Description = tool.Description
                      NodeKind = "tool"
                      InputSchema = descriptor.InputSchema
                      Required = descriptor.Required
                      Approval =
                        { Tier = descriptor.Approval.Tier.Wire
                          Effect = descriptor.Approval.Effect
                          AutoApproved = descriptor.Approval.AutoApproved } }))

        { Nodes = reasonEntry :: (described |> List.sortBy (fun e -> e.Name))
          DescribedTools = described.Length
          TotalTools = tools.Length }

    let catalogJson (registry: IToolRegistry) =
        JsonSerializer.Serialize(catalog registry, jsonOptions)
