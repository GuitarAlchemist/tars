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

    /// A string property, treating anything that is not one as absent.
    ///
    /// Only for `prompt` and `tool`, and only because each matters to one node kind: a
    /// reason node with a non-string `prompt` gets `missing_prompt`, a tool node with a
    /// non-string `tool` gets `missing_tool`. On the *other* kind the field is unused,
    /// so a malformed one is ignored rather than refused — the same tolerance a node
    /// already has for any field it does not use.
    ///
    /// Every other field goes through `requireStringOrMissing`, including `id`, `kind`,
    /// `from` and `to`. Those four are refused either way, so reading them leniently
    /// cost nothing for the *decision* — but it made the only thing the caller actually
    /// receives, the message, wrong: `{"id": 7}` was reported as a node with no `id`,
    /// sending whoever wrote it to check that the field is serialized at all. On an
    /// editor whose node ids are numbers, that is the likeliest mistake of the set.
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
        match requireStringOrMissing "id" element, requireStringOrMissing "kind" element with
        | Result.Error message, _
        | _, Result.Error message -> Result.Error message
        | Result.Ok None, _ -> Result.Error "a node has no 'id'"
        | _, Result.Ok None -> Result.Error "a node has no 'kind'"
        | Result.Ok(Some id), Result.Ok(Some kind) ->
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
        match requireStringOrMissing "from" element, requireStringOrMissing "to" element with
        | Result.Error message, _
        | _, Result.Error message -> Result.Error message
        | Result.Ok None, _ -> Result.Error "an edge has no 'from'"
        | _, Result.Ok None -> Result.Error "an edge has no 'to'"
        | Result.Ok(Some from), Result.Ok(Some target) ->
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
            // Read once each, then matched together. The or-patterns below report the
            // first field that is wrong in this order, which is the order they are read
            // in; an earlier version checked the same fields twice to build the message
            // and then needed a local `orNone` to unwrap values it had already proved
            // good, with a comment explaining that its error branch could not be reached.
            // A branch that cannot happen is a branch nobody can test.
            let nodesField = requireArrayOrMissing "nodes" root
            let edgesField = requireArrayOrMissing "edges" root
            let policyField = requireStringsOrMissing "policy" root
            let idField = requireStringOrMissing "id" root
            let goalField = requireStringOrMissing "goal" root
            let entryField = requireStringOrMissing "entry_node" root

            match nodesField, edgesField, policyField, idField, goalField, entryField with
            | Result.Error message, _, _, _, _, _
            | _, Result.Error message, _, _, _, _
            | _, _, Result.Error message, _, _, _
            | _, _, _, Result.Error message, _, _
            | _, _, _, _, Result.Error message, _
            | _, _, _, _, _, Result.Error message -> Result.Error message
            | Result.Ok rawNodes, Result.Ok rawEdges, Result.Ok policy, Result.Ok id, Result.Ok goal, Result.Ok entry ->

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
                        { Id = id
                          Goal = goal |> Option.defaultValue ""
                          EntryNode = entry
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
                          Policy = policy }
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
            + "and returns its text" }

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

    /// What a name in a graph resolves to.
    // Tool resolution and the approval decision live in `Tars.Core.ToolGate`, shared
    // with `ClaudeCodeBridge`. They used to live here, which is why that bridge had no
    // gate at all: not a decision anybody took, just where the code happened to sit.

    /// What a schema declares: every property name, and the subset that names a type.
    ///
    /// `None` means "this schema says nothing about properties", so nothing is checked
    /// against it — an unreadable schema must not turn every argument into an error.
    ///
    /// A schema that declares *no* properties is emphatically not that case. It says,
    /// positively, that the tool takes nothing, so it returns an empty set and
    /// `unknown_argument` refuses everything. Folding the two together switched both
    /// argument checks off for exactly the tools that accept nothing — and
    /// `ToolHelpers.parseStringArg` falls back to any single property it is handed
    /// whatever that property is called, so a no-argument `observes` tool would have run
    /// with a caller-chosen argument, unapproved, on a graph the validator called valid.
    ///
    /// Names and types are kept separate because a property may be declared without one:
    /// `{"x": {}}` declares `x`, so `x` is not unknown, and there is nothing to check its
    /// type against. Deriving the names from the type map would have refused it.
    let private declaredProperties (schema: string) =
        try
            use doc = JsonDocument.Parse schema

            match doc.RootElement.TryGetProperty "properties" with
            | true, properties when properties.ValueKind = JsonValueKind.Object ->
                let all = properties.EnumerateObject() |> Seq.toList

                let types =
                    all
                    |> List.choose (fun p ->
                        match p.Value.TryGetProperty "type" with
                        | true, t when t.ValueKind = JsonValueKind.String -> Some(p.Name, t.GetString())
                        | _ -> None)
                    |> Map.ofList

                Some(all |> List.map (fun p -> p.Name) |> Set.ofList, types)
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

    // =========================================================================
    // Validation, in three named pieces
    //
    // These were one two-hundred-line function with three mutable accumulators, and
    // the approval decision - the one line anyone reviewing this file has to find -
    // sat ninety lines into a loop inside a five-level match. Nothing below changed
    // except where it lives.
    // =========================================================================

    /// What one reason node needs, and what it costs.
    let private checkReasonNode (node: NodeSpec) : ValidationError list * ApprovalNotice list =
        let errors =
            if node.Prompt |> Option.forall String.IsNullOrWhiteSpace then
                [ { Code = "missing_prompt"
                    Step = Some node.Id
                    Message = "a reason node needs a prompt" } ]
            else
                []

        // A reasoning node reaches nothing and still spends money every run, which is
        // exactly what the gate is for.
        let approvals =
            [ { NodeId = node.Id
                Tool = "reason"
                Tier = reasonApproval.Tier.Wire
                Effect = reasonApproval.Effect } ]

        errors, approvals

    /// What one tool node needs, and whether it may run without being asked.
    ///
    /// **This is the gate.** Everything the editor is allowed to do without a person
    /// saying yes is decided by the last five lines of this function.
    let private checkToolNode (registry: IToolRegistry) (node: NodeSpec) : ValidationError list * ApprovalNotice list =
        let mutable errors: ValidationError list = []
        let mutable approvals: ApprovalNotice list = []

        let error code message =
            errors <-
                { Code = code
                  Step = Some node.Id
                  Message = message }
                :: errors

        match node.Tool with
        | None
        | Some "" -> error "missing_tool" "a tool node needs a tool name"
        | Some givenName ->
            match ToolGate.inspect registry givenName with
            | ToolGate.Unresolved ToolGate.NotFound ->
                error "unknown_tool" $"no tool named '{givenName}' is registered"
            | ToolGate.Unresolved(ToolGate.Ambiguous names) ->
                error
                    "ambiguous_tool"
                    ($"'{givenName}' matches more than one registered tool, differing only by case: "
                     + String.concat ", " names
                     + ". Name one of them exactly.")
            | ToolGate.Unresolved(ToolGate.Resolved _) -> () // `inspect` never returns this
            | ToolGate.Undescribed(tool, Some nearMiss) ->
                error
                    "undescribed_tool"
                    ($"'{tool.Name}' has no description of its own; '{nearMiss.Name}' differs from it "
                     + "only by case and describes a different tool")
            | ToolGate.Undescribed(tool, None) ->
                // Fail closed, which is this surface's posture and not the gate's: the
                // catalog only offers described tools, so an undescribed one in a graph
                // means the graph was not built from the catalog. `ClaudeCodeBridge`
                // decides the opposite, for reasons written down there.
                error
                    "undescribed_tool"
                    $"'{tool.Name}' has no description, so what it expects and what it does are unknown"
            | ToolGate.Allowed(tool, descriptor)
            | ToolGate.NeedsApproval(tool, descriptor) ->
                // The registry's spelling from here on, so a casing variant in the
                // graph is reported and approved under the real name.
                let toolName = tool.Name
                let arguments = node.Arguments |> Option.defaultValue Map.empty

                for required in descriptor.Required do
                    if not (arguments.ContainsKey required) then
                        error "missing_argument" $"'{toolName}' requires '{required}'"

                // Names alone are not enough: a schema that says `path` is a string and
                // a node that sends `42` used to validate and reach the tool, which
                // then failed with whatever error it makes of a number. Catching it
                // here is the whole point of having the schema.
                match declaredProperties descriptor.InputSchema with
                | None -> () // the schema says nothing about properties; nothing to check
                | Some(declaredNames, declaredTypes) ->
                    for KeyValue(name, value) in arguments do
                        if not (declaredNames.Contains name) then
                            // The schemas say `additionalProperties: false`.
                            error "unknown_argument" $"'{toolName}' takes no argument called '{name}'"
                        else
                            match declaredTypes.TryFind name with
                            | Some expected when not (matchesDeclaredType expected value) ->
                                error
                                    "wrong_argument_type"
                                    $"'{toolName}' expects '{name}' to be {expected}, not {value.ValueKind}"
                            | _ -> ()

                if not descriptor.Approval.AutoApproved then
                    approvals <-
                        [ { NodeId = node.Id
                            Tool = toolName
                            Tier = descriptor.Approval.Tier.Wire
                            Effect = descriptor.Approval.Effect } ]

        List.rev errors, approvals

    let private checkNode (registry: IToolRegistry) (node: NodeSpec) =
        match node.Kind with
        | "reason" -> checkReasonNode node
        | "tool" -> checkToolNode registry node
        | _ -> [], [] // already reported as unsupported

    /// The graph as a whole: how its nodes are joined, where it starts, and whether an
    /// order exists at all. Returns that order, empty when a cycle means there is none.
    let private checkShape (spec: PlanSpec) : ValidationError list * string list =
        let mutable errors: ValidationError list = []

        let error code step message =
            errors <-
                { Code = code
                  Step = step
                  Message = message }
                :: errors

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

        List.rev errors, order

    /// Everything true of the graph that does not stop it running.
    let private warningsFor (spec: PlanSpec) (order: string list) : string list =
        let mutable warnings: string list = []

        // `goal` and `id` are the two fields a graph can get wrong without hearing
        // about it. Neither stops the graph running, so neither is an error; saying
        // nothing at all is what makes them worth a line here.
        if String.IsNullOrWhiteSpace spec.Goal then
            // Both MCP tool descriptions present `goal` as part of the input, and it
            // ends up in `WoTPlan.Metadata.SourceGoal`, which is how a run is found
            // again afterwards. An empty one costs nothing now and everything later.
            warnings <- "no 'goal' was given, so this run will be unlabelled in the trace" :: warnings

        match spec.Id with
        | Some id when not (fst (Guid.TryParse id)) ->
            // `WoTPlan.Id` is a GUID, so `toWoTPlan` mints a fresh one for anything
            // else. It is the only field in the format that does not come back out the
            // way it went in, and an editor that keys its own state on the id it sent
            // would silently lose track of the run.
            warnings <-
                $"'id' is not a GUID ('{id}'), so the run will be given a fresh one and this value will not come back"
                :: warnings
        | _ -> ()

        if spec.EntryNode.IsNone && not spec.Nodes.IsEmpty then
            warnings <-
                "no entry node given; the first node in execution order will be used"
                :: warnings

        // Reading a graph does nothing, so an unenforceable policy is only a warning
        // here. `run` refuses it outright, because running under a restriction nobody
        // applies is claiming a guarantee that does not exist (tars#334).
        match spec.Policy with
        | Some policy when not (policy |> List.filter (String.IsNullOrWhiteSpace >> not)).IsEmpty ->
            warnings <-
                ("nothing enforces a policy: the executor never reads it, so "
                 + String.concat ", " policy
                 + " will not be applied. `tars_plan_run` refuses a graph carrying one (tars#334)")
                :: warnings
        | _ -> ()

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

        List.rev warnings

    /// Check a graph, and say what would happen if it ran.
    let validate (registry: IToolRegistry) (spec: PlanSpec) : ValidateResponse =
        // Ids must be unique, or an edge cannot say which node it means.
        let duplicateIds =
            spec.Nodes
            |> List.countBy (fun n -> n.Id)
            |> List.filter (fun (_, count) -> count > 1)
            |> List.map fst

        let duplicateErrors =
            duplicateIds
            |> List.map (fun duplicate ->
                { Code = "duplicate_node_id"
                  Step = Some duplicate
                  Message = $"more than one node has the id '{duplicate}'" })

        let unsupported =
            spec.Nodes
            |> List.filter (fun n -> n.Kind <> "reason" && n.Kind <> "tool")
            |> List.map (fun n ->
                { Id = n.Id
                  Kind = n.Kind
                  Reason =
                    $"'{n.Kind}' nodes are not editable yet. They are shown so a plan is never "
                    + "silently changed, but this graph cannot be saved or run while one is present." })

        let unsupportedErrors =
            unsupported
            |> List.map (fun node ->
                { Code = "unsupported_node"
                  Step = Some node.Id
                  Message = node.Reason })

        let perNode = spec.Nodes |> List.map (checkNode registry)
        let shapeErrors, order = checkShape spec

        // Concatenated in the order the single function used to append them, because
        // the first error is the one a caller reads.
        let errors =
            duplicateErrors
            @ unsupportedErrors
            @ (perNode |> List.collect fst)
            @ shapeErrors

        { Valid = errors.IsEmpty
          Errors = errors
          Warnings = warningsFor spec order
          ExecutionOrder = (if errors.IsEmpty then order else [])
          Unsupported = unsupported
          NeedsApproval = perNode |> List.collect snd }

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
    /// rather than at the point of use.
    ///
    /// An object or an array stays a `JsonElement`. Handing on its raw *text* instead
    /// looked harmless — a tool receives its arguments serialized anyway — but
    /// `WoTExecutor.serializeToolArgs` serializes the whole map, so that text was
    /// encoded a second time: `{"config":{"x":1}}` reached the tool as
    /// `{"config":"{\"x\":1}"}`, which no longer matches the schema the catalog
    /// published for it. `JsonElement` serializes back as the JSON it came from.
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
        | _ -> box (element.Clone())

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
    /// `Extra = Map.empty` on every node, below, is load-bearing rather than tidy.
    ///
    /// It is the only thing between a caller and the executor's two hidden controls:
    /// `WoTExecutor.groupIntoSegments` reads `parallel_group` out of `Extra` and runs
    /// those nodes under `Async.Parallel`, and `evaluateCondition` reads `condition` and
    /// skips the node outright. Either would let a graph run in an order the validator
    /// never computed, or not run at all while reporting success. The caller's own `tags`
    /// go to `Metadata.Tags`, which the executor never reads.
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
                            |> Option.bind (fun given ->
                                match ToolGate.resolve registry given with
                                | ToolGate.Resolved tool -> Some tool.Name
                                | ToolGate.NotFound
                                | ToolGate.Ambiguous _ -> None)
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

    let private refuse errors warnings : RunResponse =
        { Success = false
          Output = ""
          Errors = errors
          Warnings = warnings
          StepsRun = 0
          ToolsUsed = [] }

    /// Validate a graph and run it.
    ///
    /// `approved` names the nodes the caller has explicitly allowed. A node whose tool
    /// is not auto-approved and is not named there stops the run before anything
    /// happens — the whole graph, not just that node, because a partial run of a graph
    /// someone refused is worse than no run at all.
    ///
    /// `getExecutor` is a thunk, not an executor, and it is forced only once every
    /// refusal is past. Building one reaches the LLM configuration and creates
    /// directories and stores under `~/.tars`; a refused graph that had already done
    /// that would have changed this machine while reporting that it ran nothing.
    let run
        (getExecutor: unit -> (WoTPlan -> Async<WoTResult>))
        (registry: IToolRegistry)
        (spec: PlanSpec)
        (approved: string list)
        : Async<RunResponse> =
        async {
            let verdict = validate registry spec

            if not verdict.Valid then
                return refuse (verdict.Errors |> List.map (fun e -> e.Message)) verdict.Warnings
            else
                let approvedSet = Set.ofList approved

                let refused =
                    verdict.NeedsApproval
                    |> List.filter (fun notice -> not (approvedSet.Contains notice.NodeId))

                // A policy list says what the graph may *not* do, and nothing reads it:
                // `WoTExecutor` never looks at `plan.Policy`. Accepting `no-network`
                // and then running a graph that reaches the network is worse than
                // refusing, because the caller believes a restriction is in force.
                // Validation only warns — reading a graph is not doing anything — but
                // running one under a restriction nobody enforces is refused outright.
                let policy = spec.Policy |> Option.defaultValue [] |> List.filter (String.IsNullOrWhiteSpace >> not)

                if not policy.IsEmpty then
                    return
                        refuse
                            [ "this graph carries a policy ("
                              + String.concat ", " policy
                              + "), and nothing here enforces one: the executor never reads it. "
                              + "Running the graph would claim a restriction that is not in force. "
                              + "Remove the policy to run it (tars#334)." ]
                            verdict.Warnings
                elif not refused.IsEmpty then
                    return
                        refuse
                            (refused
                             |> List.map (fun r ->
                                 $"node '%s{r.NodeId}' runs '%s{r.Tool}', which %s{r.Effect}. Approve it by name to run this graph."))
                            verdict.Warnings
                else
                    let plan = toWoTPlan registry spec verdict.ExecutionOrder
                    let! result = getExecutor () plan

                    // `WoTExecutor` skips every node downstream of a failure and keeps
                    // running the branches that do not depend on it (tars#335). A caller
                    // who approved a node as a step *after* a guard still needs to hear
                    // that it did not run, and why - an approved node that silently did
                    // nothing reads like one that did.
                    let failed =
                        result.Trace.Steps
                        |> List.filter (fun step ->
                            match step.Status with
                            | Failed _ -> true
                            | _ -> false)
                        |> List.map (fun step -> step.NodeId)

                    let skipped =
                        result.Trace.Steps
                        |> List.choose (fun step ->
                            match step.Status with
                            | Skipped reason when reason.EndsWith "which it depends on, did not complete" -> Some step.NodeId
                            | _ -> None)

                    let failureWarning =
                        if failed.IsEmpty then
                            []
                        else
                            [ "these nodes failed: "
                              + String.concat ", " failed
                              + (if skipped.IsEmpty then
                                     ". Nothing depended on them."
                                 else
                                     ". These were skipped because they depend on one of them: "
                                     + String.concat ", " skipped) ]

                    return
                        { Success = result.Success
                          Output = result.Output
                          Errors = result.Errors
                          Warnings = verdict.Warnings @ result.Warnings @ failureWarning
                          StepsRun = result.Trace.Steps.Length
                          ToolsUsed = result.ToolsUsed }
        }

    let runJson (getExecutor: unit -> (WoTPlan -> Async<WoTResult>)) (registry: IToolRegistry) (json: string) =
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
                        requireStringsOrMissing "approve" doc.RootElement
                    with ex ->
                        Result.Error $"'approve' could not be read: {ex.Message}"

                match approved with
                // A malformed `approve` is said out loud rather than read as "nothing
                // approved". Both refuse the run, but only one tells the caller that
                // the approval they wrote never arrived.
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
                | Result.Ok approved ->
                    let! response = run getExecutor registry spec (approved |> Option.defaultValue [])
                    return JsonSerializer.Serialize(response, jsonOptions)
        }
