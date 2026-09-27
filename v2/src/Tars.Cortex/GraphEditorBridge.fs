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

    type ValidationError =
        { Code: string
          Step: string option
          Message: string }

    /// A node the editor may show but must not edit or save.
    type UnsupportedNode =
        { Id: string
          Kind: string
          Reason: string }

    type ValidateResponse =
        {
            Valid: bool
            Errors: ValidationError list
            Warnings: string list
            /// Node ids in an order that respects the edges. Empty when invalid.
            ExecutionOrder: string list
            Unsupported: UnsupportedNode list
            /// Nodes that will not run without a human saying yes, with the reason.
            NeedsApproval: ApprovalNotice list
        }

    and ApprovalNotice =
        { NodeId: string
          Tool: string
          Tier: string
          Effect: string }

    type RunResponse =
        { Success: bool
          Output: string
          Errors: string list
          Warnings: string list
          StepsRun: int
          ToolsUsed: string list }

    // =========================================================================
    // Reading JSON
    // =========================================================================

    let private tryGetString (name: string) (root: JsonElement) =
        match root.TryGetProperty name with
        | true, v when v.ValueKind = JsonValueKind.String -> Some(v.GetString())
        | _ -> None

    let private tryGetArray (name: string) (root: JsonElement) =
        match root.TryGetProperty name with
        | true, v when v.ValueKind = JsonValueKind.Array -> Some(v.EnumerateArray() |> Seq.toList)
        | _ -> None

    let private stringList (name: string) (root: JsonElement) =
        tryGetArray name root
        |> Option.map (
            List.choose (fun (e: JsonElement) ->
                if e.ValueKind = JsonValueKind.String then
                    Some(e.GetString())
                else
                    None)
        )

    let private parseNode (element: JsonElement) : Result<NodeSpec, string> =
        match tryGetString "id" element, tryGetString "kind" element with
        | None, _ -> Result.Error "a node has no 'id'"
        | _, None -> Result.Error "a node has no 'kind'"
        | Some id, Some kind ->
            let arguments =
                match element.TryGetProperty "arguments" with
                | true, v when v.ValueKind = JsonValueKind.Object ->
                    v.EnumerateObject()
                    |> Seq.map (fun p -> p.Name, p.Value.Clone())
                    |> Map.ofSeq
                    |> Some
                | _ -> None

            Result.Ok
                { Id = id
                  Kind = kind.ToLowerInvariant()
                  Prompt = tryGetString "prompt" element
                  Hint = tryGetString "hint" element
                  Tool = tryGetString "tool" element
                  Arguments = arguments
                  Label = tryGetString "label" element
                  Tags = stringList "tags" element }

    let private parseEdge (element: JsonElement) : Result<EdgeSpec, string> =
        match tryGetString "from" element, tryGetString "to" element with
        | Some from, Some target ->
            Result.Ok
                { From = from
                  To = target
                  Label = tryGetString "label" element
                  Confidence =
                    match element.TryGetProperty "confidence" with
                    | true, v when v.ValueKind = JsonValueKind.Number -> Some(v.GetDouble())
                    | _ -> None }
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

            // A policy list is what a graph is *not* allowed to do, so a malformed one
            // is refused alongside the rest rather than read as "no restrictions".
            let listFields =
                [ "nodes"; "edges"; "policy" ]
                |> List.tryPick (fun name ->
                    match requireArrayOrMissing name root with
                    | Result.Error message -> Some message
                    | Result.Ok _ -> None)

            match listFields, requireArrayOrMissing "nodes" root, requireArrayOrMissing "edges" root with
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
                    Result.Ok
                        { Id = tryGetString "id" root
                          Goal = tryGetString "goal" root |> Option.defaultValue ""
                          EntryNode = tryGetString "entry_node" root
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
                          Policy = stringList "policy" root }
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

    // =========================================================================
    // Validation
    // =========================================================================

    /// Node ids in an order that respects the edges, or the cycle that prevents one.
    ///
    /// Kahn's algorithm. The editor needs the order to show what runs when, and the
    /// cycle detection is the reason a graph editor needs a validator at all — a cycle
    /// is easy to draw and impossible to run.
    ///
    /// `first` is the declared entry node, and it goes ahead of its peers the moment
    /// it is ready. That matters because `WoTExecutor` walks `plan.Nodes` in list
    /// order and never reads `EntryNode`: this list *is* the entry node's only effect
    /// on what actually runs, so ties are broken in its favour rather than
    /// alphabetically.
    /// Find a tool by name, ignoring case, and return the registry's own spelling.
    ///
    /// `ToolMetadata` is keyed case-insensitively and says why: names are typed by hand
    /// into graphs, and `Read_File` should not be a different tool from `read_file`.
    /// `ToolRegistry` is a plain `ConcurrentDictionary`, so it is case-sensitive, and an
    /// exact-only lookup answered `unknown_tool` before the metadata lookup could run.
    ///
    /// The registry's spelling is what matters: accepting a variant without
    /// canonicalising it would only move the failure to the executor, which looks the
    /// name up in that same case-sensitive registry.
    let private resolveTool (registry: IToolRegistry) (name: string) =
        match registry.Get name with
        | Some tool -> Some tool
        | None ->
            registry.GetAll()
            |> List.tryFind (fun t -> String.Equals(t.Name, name, StringComparison.OrdinalIgnoreCase))

    /// The type each property is declared as, or `None` when the schema says nothing.
    ///
    /// `None` means "no type information here", so nothing is checked against it. An
    /// unreadable schema must not turn every argument into an error.
    let private declaredTypes (schema: string) =
        try
            use doc = JsonDocument.Parse schema

            match doc.RootElement.TryGetProperty "properties" with
            | true, properties when properties.ValueKind = JsonValueKind.Object ->
                let byName =
                    properties.EnumerateObject()
                    |> Seq.choose (fun p ->
                        match p.Value.TryGetProperty "type" with
                        | true, t when t.ValueKind = JsonValueKind.String -> Some(p.Name, t.GetString())
                        | _ -> None)
                    |> Map.ofSeq

                if byName.IsEmpty then None else Some byName
            | _ -> None
        with _ ->
            None

    /// Does this value match the JSON Schema type the tool declared?
    let private matchesDeclaredType (declared: string) (value: JsonElement) =
        match declared, value.ValueKind with
        | "string", JsonValueKind.String
        | "boolean", JsonValueKind.True
        | "boolean", JsonValueKind.False
        | "number", JsonValueKind.Number
        | "object", JsonValueKind.Object
        | "array", JsonValueKind.Array
        | "null", JsonValueKind.Null -> true
        | "integer", JsonValueKind.Number ->
            match value.TryGetInt64() with
            | true, _ -> true
            | _ -> false
        | _ -> false

    let private topologicalOrder (first: string option) (nodeIds: string list) (edges: EdgeSpec list) =
        let incoming =
            nodeIds
            |> List.map (fun id ->
                id,
                edges
                |> List.filter (fun e -> e.To = id)
                |> List.map (fun e -> e.From)
                |> Set.ofList)
            |> Map.ofList

        let rec walk (remaining: Map<string, Set<string>>) (ordered: string list) =
            if Map.isEmpty remaining then
                Result.Ok(List.rev ordered)
            else
                let ready =
                    remaining
                    |> Map.toList
                    |> List.filter (fun (_, deps) -> deps |> Set.forall (fun d -> not (remaining.ContainsKey d)))
                    |> List.map fst
                    |> List.sortBy (fun id -> (first <> Some id), id)

                match ready with
                | [] ->
                    // Everything left depends on something else left: a cycle.
                    Result.Error(remaining |> Map.toList |> List.map fst |> List.sort)
                | _ ->
                    let next = remaining |> Map.filter (fun id _ -> not (List.contains id ready))
                    walk next (List.rev ready @ ordered)

        walk incoming []

    /// Check a graph, and say what would happen if it ran.
    let validate (registry: IToolRegistry) (spec: PlanSpec) : ValidateResponse =
        let mutable errors: ValidationError list = []
        let mutable warnings: string list = []

        let error code step message =
            errors <-
                { Code = code
                  Step = step
                  Message = message }
                :: errors

        // Ids must be unique, or an edge cannot say which node it means.
        let duplicates =
            spec.Nodes
            |> List.countBy (fun n -> n.Id)
            |> List.filter (fun (_, count) -> count > 1)
            |> List.map fst

        for duplicate in duplicates do
            error "duplicate_node_id" (Some duplicate) $"more than one node has the id '{duplicate}'"

        let unsupported =
            spec.Nodes
            |> List.filter (fun n -> n.Kind <> "reason" && n.Kind <> "tool")
            |> List.map (fun n ->
                { Id = n.Id
                  Kind = n.Kind
                  Reason =
                    $"'{n.Kind}' nodes are not editable yet. They are shown so a plan is never "
                    + "silently changed, but this graph cannot be saved or run while one is present." })

        for node in unsupported do
            error "unsupported_node" (Some node.Id) node.Reason

        let mutable needsApproval: ApprovalNotice list = []

        for node in spec.Nodes do
            match node.Kind with
            | "reason" ->
                if node.Prompt |> Option.forall String.IsNullOrWhiteSpace then
                    error "missing_prompt" (Some node.Id) "a reason node needs a prompt"

                // A reasoning node reaches nothing and still spends money every run,
                // which is exactly what the gate is for.
                needsApproval <-
                    { NodeId = node.Id
                      Tool = "reason"
                      Tier = reasonApproval.Tier.Wire
                      Effect = reasonApproval.Effect }
                    :: needsApproval

            | "tool" ->
                match node.Tool with
                | None
                | Some "" -> error "missing_tool" (Some node.Id) "a tool node needs a tool name"
                | Some givenName ->
                    match resolveTool registry givenName with
                    | None -> error "unknown_tool" (Some node.Id) $"no tool named '{givenName}' is registered"
                    | Some tool ->
                        // The registry's spelling from here on, so a casing variant in
                        // the graph is reported and approved under the real name.
                        let toolName = tool.Name

                        match ToolMetadata.tryFind toolName with
                        | None ->
                            // Fail closed: an undescribed tool is refused here, while
                            // remaining perfectly usable everywhere else in TARS.
                            error
                                "undescribed_tool"
                                (Some node.Id)
                                $"'{toolName}' has no description, so what it expects and what it does are unknown"
                        | Some descriptor ->
                            let arguments = node.Arguments |> Option.defaultValue Map.empty

                            for required in descriptor.Required do
                                if not (arguments.ContainsKey required) then
                                    error "missing_argument" (Some node.Id) $"'{toolName}' requires '{required}'"

                            // Names alone are not enough: a schema that says `path` is a
                            // string and a node that sends `42` used to validate and
                            // reach the tool, which then failed with whatever error it
                            // makes of a number. Catching it here is the whole point of
                            // having the schema.
                            match declaredTypes descriptor.InputSchema with
                            | None -> () // the schema declares no types; nothing to check against
                            | Some declared ->
                                for KeyValue(name, value) in arguments do
                                    match declared.TryFind name with
                                    | None ->
                                        // The schemas say `additionalProperties: false`.
                                        error
                                            "unknown_argument"
                                            (Some node.Id)
                                            $"'{toolName}' takes no argument called '{name}'"
                                    | Some expected when not (matchesDeclaredType expected value) ->
                                        error
                                            "wrong_argument_type"
                                            (Some node.Id)
                                            $"'{toolName}' expects '{name}' to be {expected}, not {value.ValueKind}"
                                    | Some _ -> ()

                            if not descriptor.Approval.AutoApproved then
                                needsApproval <-
                                    { NodeId = node.Id
                                      Tool = toolName
                                      Tier = descriptor.Approval.Tier.Wire
                                      Effect = descriptor.Approval.Effect }
                                    :: needsApproval

            | _ -> () // already reported as unsupported

        // Edges must join nodes that exist.
        let ids = spec.Nodes |> List.map (fun n -> n.Id) |> Set.ofList

        for edge in spec.Edges do
            if not (ids.Contains edge.From) then
                error "unknown_edge_endpoint" (Some edge.From) $"an edge starts at '{edge.From}', which is not a node"

            if not (ids.Contains edge.To) then
                error "unknown_edge_endpoint" (Some edge.To) $"an edge ends at '{edge.To}', which is not a node"

        match spec.EntryNode with
        | Some entry when not (ids.Contains entry) ->
            error "unknown_entry_node" (Some entry) $"the entry node '{entry}' is not one of the nodes"
        | Some entry when spec.Edges |> List.exists (fun e -> e.To = entry) ->
            // Nothing can run before the node a graph starts at.
            error
                "entry_node_not_a_start"
                (Some entry)
                $"'{entry}' is the entry node but something runs into it, so it cannot be where the graph starts"
        | _ -> ()

        if spec.Nodes.IsEmpty then
            error "empty_graph" None "a graph needs at least one node"

        let order =
            match topologicalOrder spec.EntryNode (spec.Nodes |> List.map (fun n -> n.Id)) spec.Edges with
            | Result.Ok ordered -> ordered
            | Result.Error cycle ->
                error
                    "cycle"
                    None
                    ("these nodes form a cycle, so no order can run them: "
                     + String.concat ", " cycle)

                []

        if spec.EntryNode.IsNone && not spec.Nodes.IsEmpty then
            warnings <-
                "no entry node given; the first node in execution order will be used"
                :: warnings

        // Everything in the graph runs, reachable from the entry node or not, because
        // `WoTExecutor` walks every node in the list. An editor that drew two separate
        // components would otherwise expect only the entry's own component to run.
        match spec.EntryNode with
        | Some entry when not order.IsEmpty ->
            let rec reach (seen: Set<string>) (frontier: string list) =
                match frontier with
                | [] -> seen
                | current :: rest ->
                    let next =
                        spec.Edges
                        |> List.filter (fun e -> e.From = current)
                        |> List.map (fun e -> e.To)
                        |> List.filter (fun id -> not (seen.Contains id))

                    reach (Set.union seen (Set.ofList next)) (rest @ next)

            let reachable = reach (Set.singleton entry) [ entry ]
            let stranded = order |> List.filter (fun id -> not (reachable.Contains id))

            if not stranded.IsEmpty then
                warnings <-
                    ("these nodes cannot be reached from the entry node and will run anyway: "
                     + String.concat ", " stranded)
                    :: warnings
        | _ -> ()

        { Valid = errors.IsEmpty
          Errors = List.rev errors
          Warnings = List.rev warnings
          ExecutionOrder = (if errors.IsEmpty then order else [])
          Unsupported = unsupported
          NeedsApproval = List.rev needsApproval }

    let validateJson (registry: IToolRegistry) (json: string) =
        match parsePlanSpec json with
        | Result.Error message ->
            let response =
                { Valid = false
                  Errors =
                    [ { Code = "malformed"
                        Step = None
                        Message = message } ]
                  Warnings = []
                  ExecutionOrder = []
                  Unsupported = []
                  NeedsApproval = [] }

            JsonSerializer.Serialize(response, jsonOptions)
        | Result.Ok spec -> JsonSerializer.Serialize(validate registry spec, jsonOptions)
    // =========================================================================
    // Running
    // =========================================================================

    /// A JSON argument as the tool registry wants it.
    ///
    /// `ToolPayload.Args` is `Map<string, obj>`, so the JSON types are narrowed here
    /// rather than at the point of use. Objects and arrays keep their JSON text: a
    /// tool receives its arguments serialized anyway, so re-serializing a structure we
    /// had already parsed would only risk changing it.
    let private toArgValue (element: JsonElement) : obj =
        match element.ValueKind with
        | JsonValueKind.String -> box (element.GetString())
        | JsonValueKind.True -> box true
        | JsonValueKind.False -> box false
        | JsonValueKind.Null -> null
        | JsonValueKind.Number ->
            match element.TryGetInt64() with
            | true, i -> box i
            | _ -> box (element.GetDouble())
        | _ -> box (element.GetRawText())

    let private toHint (hint: string option) =
        match hint |> Option.map (fun h -> h.ToLowerInvariant()) with
        | Some "fast" -> Some Fast
        | Some "smart" -> Some Smart
        | Some "reasoning" -> Some Reasoning
        | Some other when not (String.IsNullOrWhiteSpace other) -> Some(Specific other)
        | _ -> None

    /// Turn a validated graph into the plan the executor runs.
    ///
    /// Only call this on a spec that `validate` accepted: it assumes node kinds are
    /// `reason` or `tool` and that tool nodes name a tool.
    ///
    /// The registry is needed for the name, not for the tools: a graph may spell a tool
    /// in any case, and the plan has to carry the registry's own spelling because the
    /// executor looks it up in that same case-sensitive registry.
    let toWoTPlan (registry: IToolRegistry) (spec: PlanSpec) (executionOrder: string list) : WoTPlan =
        let node (n: NodeSpec) : WoTNode =
            match n.Kind with
            | "tool" ->
                { Id = n.Id
                  Kind = WoTNodeKind.Tool
                  Payload =
                    box
                        { ToolPayload.Tool =
                            n.Tool
                            |> Option.bind (fun given -> resolveTool registry given |> Option.map (fun t -> t.Name))
                            |> Option.orElse n.Tool
                            |> Option.defaultValue ""
                          Args =
                            n.Arguments
                            |> Option.map (Map.map (fun _ v -> toArgValue v))
                            |> Option.defaultValue Map.empty }
                  Metadata =
                    { Label = n.Label
                      Tags = n.Tags |> Option.defaultValue []
                      Extra = Map.empty } }
            | _ ->
                { Id = n.Id
                  Kind = WoTNodeKind.Reason
                  Payload =
                    box
                        { Prompt = n.Prompt |> Option.defaultValue ""
                          Hint = toHint n.Hint }
                  Metadata =
                    { Label = n.Label
                      Tags = n.Tags |> Option.defaultValue []
                      Extra = Map.empty } }

        // The executor walks `Nodes` in list order, so the list is put in the order
        // validation worked out. The edges still carry what each node may see.
        let ordered =
            match executionOrder with
            | [] -> spec.Nodes
            | order -> order |> List.choose (fun id -> spec.Nodes |> List.tryFind (fun n -> n.Id = id))

        let nodes = ordered |> List.map node

        { Id =
            match
                spec.Id
                |> Option.bind (fun i ->
                    match Guid.TryParse i with
                    | true, g -> Some g
                    | _ -> None)
            with
            | Some existing -> existing
            | None -> Guid.NewGuid()
          Nodes = nodes
          Edges =
            spec.Edges
            |> List.map (fun e ->
                { From = e.From
                  To = e.To
                  Label = e.Label
                  Confidence = e.Confidence })
          EntryNode =
            match spec.EntryNode with
            | Some entry -> entry
            | None -> nodes |> List.tryHead |> Option.map (fun n -> n.Id) |> Option.defaultValue ""
          Metadata =
            { Kind = Custom "GraphEditor"
              SourceGoal = spec.Goal
              CompiledAt = DateTime.UtcNow
              EstimatedTokens = None
              EstimatedSteps = Some nodes.Length }
          Policy = spec.Policy |> Option.defaultValue [] }

    /// Validate a graph and run it.
    ///
    /// `approved` names the nodes the caller has explicitly allowed. A node whose tool
    /// is not auto-approved and is not named there stops the run before anything
    /// happens — the whole graph, not just that node, because a partial run of a graph
    /// someone refused is worse than no run at all.
    let run
        (executor: WoTPlan -> Async<WoTResult>)
        (registry: IToolRegistry)
        (spec: PlanSpec)
        (approved: string list)
        : Async<RunResponse> =
        async {
            let verdict = validate registry spec

            if not verdict.Valid then
                return
                    { Success = false
                      Output = ""
                      Errors = verdict.Errors |> List.map (fun e -> e.Message)
                      Warnings = verdict.Warnings
                      StepsRun = 0
                      ToolsUsed = [] }
            else
                let approvedSet = Set.ofList approved

                let refused =
                    verdict.NeedsApproval
                    |> List.filter (fun notice -> not (approvedSet.Contains notice.NodeId))

                if not refused.IsEmpty then
                    return
                        { Success = false
                          Output = ""
                          Errors =
                            refused
                            |> List.map (fun r ->
                                $"node '%s{r.NodeId}' runs '%s{r.Tool}', which %s{r.Effect}. Approve it by name to run this graph.")
                          Warnings = verdict.Warnings
                          StepsRun = 0
                          ToolsUsed = [] }
                else
                    let plan = toWoTPlan registry spec verdict.ExecutionOrder
                    let! result = executor plan

                    return
                        { Success = result.Success
                          Output = result.Output
                          Errors = result.Errors
                          Warnings = verdict.Warnings @ result.Warnings
                          StepsRun = result.Trace.Steps.Length
                          ToolsUsed = result.ToolsUsed }
        }

    let runJson (executor: WoTPlan -> Async<WoTResult>) (registry: IToolRegistry) (json: string) =
        async {
            match parsePlanSpec json with
            | Result.Error message ->
                return
                    JsonSerializer.Serialize(
                        { Success = false
                          Output = ""
                          Errors = [ message ]
                          Warnings = []
                          StepsRun = 0
                          ToolsUsed = [] },
                        jsonOptions
                    )
            | Result.Ok spec ->
                let approved =
                    try
                        // Same envelope as the plan: `approve` is a sibling of `nodes`,
                        // so it has to be read from the unwrapped object or every
                        // approval a caller gave would be invisible and the run refused.
                        use doc = JsonDocument.Parse(unwrapArguments json)
                        stringList "approve" doc.RootElement |> Option.defaultValue []
                    with _ ->
                        []

                let! response = run executor registry spec approved
                return JsonSerializer.Serialize(response, jsonOptions)
        }
