namespace Tars.Core

open System

/// The one place that answers "which tool is this, and may it run without anyone
/// being asked".
///
/// Before this module, that question was answered inside `GraphEditorBridge` and not
/// asked at all by `ClaudeCodeBridge`, which hands a tool straight to
/// `ToolExecution.runDefault`. Two bridges is not yet a pattern; three would be, and
/// the second one having no gate was not a decision anybody took — it is simply where
/// the gate happened to be written. So the decision moved here and the bridges call
/// it.
///
/// **What is shared is the decision. What is not shared is the posture.** Each surface
/// still chooses what to do with an `Undescribed` verdict, and they choose differently
/// on purpose:
///
/// - the graph editor refuses it, because its catalog only ever offers described tools,
///   so an undescribed one in a graph means the graph was not built from the catalog;
/// - the Claude Code bridge allows it, because it drives the whole registry — some two
///   hundred tools, six of them described — and refusing everything undescribed would
///   refuse every plan it runs today.
///
/// That difference is a real one and is stated at both call sites rather than hidden
/// behind a single `bool`. What must not differ, and no longer can, is *resolution*:
/// which tool a name means, and whether a descriptor belongs to it.
module ToolGate =

    /// Which tool a name means.
    type Resolution =
        | Resolved of Tool
        | NotFound
        /// More than one registered tool differs from the name only by case.
        | Ambiguous of string list

    /// Find a tool by name, ignoring case, and return the registry's own spelling.
    ///
    /// `ToolMetadata` is keyed case-insensitively and says why: names are typed by hand
    /// into graphs, and `Read_File` should not be a different tool from `read_file`.
    /// `ToolRegistry` is a plain `ConcurrentDictionary`, so it is case-sensitive, and an
    /// exact-only lookup answers "unknown tool" before the metadata lookup can run.
    ///
    /// The registry's spelling is what matters: accepting a variant without
    /// canonicalising it would only move the failure to the executor, which looks the
    /// name up in that same case-sensitive registry.
    ///
    /// Two registered tools whose names differ only by case make the request ambiguous,
    /// and picking either one would be picking for the caller. The registry allows it —
    /// it is a case-sensitive dictionary — so this has to say so rather than guess.
    let resolve (registry: IToolRegistry) (name: string) =
        match registry.Get name with
        | Some tool -> Resolved tool
        | None ->
            match
                registry.GetAll()
                |> List.filter (fun t -> String.Equals(t.Name, name, StringComparison.OrdinalIgnoreCase))
            with
            | [] -> NotFound
            | [ only ] -> Resolved only
            | many -> Ambiguous(many |> List.map (fun t -> t.Name) |> List.sort)

    /// What the gate has to say about one named tool.
    type Verdict =
        /// Described, and its tier says it may run unasked.
        | Allowed of Tool * ToolMetadata.ToolDescriptor
        /// Described, and somebody has to say yes first.
        | NeedsApproval of Tool * ToolMetadata.ToolDescriptor
        /// The tool exists but nothing describes it, so what it expects and what it
        /// does are both unknown. Each caller decides what that is worth.
        ///
        /// The second field carries a descriptor that matched the name only by case and
        /// therefore describes a *different* tool. It is the difference between "nobody
        /// has described this" and "something that looks like it is described, and it
        /// is not this", and a caller told the first when the second is true will go
        /// looking in the wrong place.
        | Undescribed of Tool * ToolMetadata.ToolDescriptor option
        /// No such tool, or more than one.
        | Unresolved of Resolution

    /// Resolve a name and decide whether running it needs asking.
    ///
    /// The descriptor has to belong to *this* tool, not to one whose name matches
    /// case-insensitively. `ToolMetadata` is keyed loosely, so a registry holding both
    /// the described `read_file` and a custom, mutating `READ_FILE` would otherwise
    /// hand the second one the first one's schema and its `auto_approved: true` — a
    /// tool that writes, running unasked, advertised as read-only. That check is the
    /// single most important line in this module and is why it is only written once.
    let inspect (registry: IToolRegistry) (name: string) : Verdict =
        match resolve registry name with
        | NotFound -> Unresolved NotFound
        | Ambiguous names -> Unresolved(Ambiguous names)
        | Resolved tool ->
            match ToolMetadata.tryFind tool.Name with
            | Some nearMiss when nearMiss.Name <> tool.Name -> Undescribed(tool, Some nearMiss)
            | None -> Undescribed(tool, None)
            | Some descriptor ->
                if descriptor.Approval.AutoApproved then
                    Allowed(tool, descriptor)
                else
                    NeedsApproval(tool, descriptor)

    /// One plain sentence for a caller who was refused, naming the tool and what
    /// running it would do. Shared so that two surfaces cannot describe the same
    /// refusal differently.
    let refusalMessage (tool: Tool) (descriptor: ToolMetadata.ToolDescriptor) =
        $"'{tool.Name}' {descriptor.Approval.Effect} ({descriptor.Approval.Tier.Wire}), "
        + "so it needs approving before it runs"
