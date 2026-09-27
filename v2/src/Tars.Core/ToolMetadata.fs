namespace Tars.Core

open System
open System.Collections.Concurrent

/// What a tool is allowed to do, and what callers must send it.
///
/// `Tool` (Domain.fs) carries a name, a description and `Execute: string -> ...`.
/// That is enough to call a tool you already understand, and not enough to offer
/// one in an editor: there is no way to know what the string should contain, which
/// fields are required, or whether running it writes to disk, spends money or
/// leaves the machine.
///
/// This is a sidecar keyed by tool name rather than a change to `Tool`, so a tool
/// can be described without touching where it is defined, and an undescribed tool
/// is simply absent rather than wrong.
///
/// It gates the editor-facing surface only. `ToolRegistry` and every existing
/// caller keep working exactly as before whether a tool is described or not —
/// otherwise adding this file would switch off every tool that has not been
/// described yet, which is most of them.
module ToolMetadata =

    /// How far a tool's effects reach.
    ///
    /// The question each tier answers is "what is the worst this can do", not "how
    /// useful is it": a tool is placed by its reach, not by how likely the damage is.
    type ApprovalTier =
        /// Observes without changing anything: reads a file, parses, computes.
        | ReadOnly
        /// Changes state this machine owns: writes files, commits, installs, runs a
        /// shell command, alters configuration.
        | Mutates
        /// Reaches outside the machine or spends money: network requests, LLM calls,
        /// anything that publishes.
        | Escapes

        /// The wire spelling. Kept explicit rather than derived from the case name so
        /// that renaming a case cannot silently change a published contract.
        member this.Wire =
            match this with
            | ReadOnly -> "read_only"
            | Mutates -> "mutates"
            | Escapes -> "escapes"

    /// What running this tool actually does.
    type Approval =
        { Tier: ApprovalTier
          /// One plain sentence naming the effect, for a human deciding whether to
          /// allow it: "writes files under the working directory", not "file I/O".
          Effect: string
          /// May a graph run this without asking first?
          ///
          /// Separate from the tier because the two answer different questions. A
          /// ReadOnly tool that calls an LLM reaches nothing and still costs money on
          /// every run, so it is not automatically approved.
          AutoApproved: bool }

        /// Observes, costs nothing, reaches nothing: safe to run unasked.
        static member observes(effect: string) =
            { Tier = ReadOnly
              Effect = effect
              AutoApproved = true }

        /// Observes but costs money or time on each call — describable, not automatic.
        static member observesAtACost(effect: string) =
            { Tier = ReadOnly
              Effect = effect
              AutoApproved = false }

        /// Changes something this machine owns.
        static member mutates(effect: string) =
            { Tier = Mutates
              Effect = effect
              AutoApproved = false }

        /// Leaves the machine or spends money.
        static member escapes(effect: string) =
            { Tier = Escapes
              Effect = effect
              AutoApproved = false }

    /// Everything the editor needs about one tool, beyond its name and description.
    type ToolDescriptor =
        { Name: string
          /// JSON Schema (draft 2020-12) for the object a caller sends.
          ///
          /// `Tool.Execute` still receives that object serialized as a string — the
          /// schema describes what to send, it does not change how tools are invoked.
          /// A tool whose input is genuinely a bare string does NOT belong here: the
          /// executor serializes a node arguments to JSON before calling the tool, so
          /// describing one as an object would hand it the JSON instead of the string.
          InputSchema: string
          /// Property names that must be present. Kept alongside the schema rather
          /// than only inside it so a caller can check without a schema parser.
          Required: string list
          Approval: Approval }

    /// Tool name -> descriptor. Ordinal-case-insensitive, because tool names are
    /// typed by hand in graphs and `Git_Commit` should not be a different tool.
    let private descriptors =
        ConcurrentDictionary<string, ToolDescriptor>(StringComparer.OrdinalIgnoreCase)

    /// Describe a tool, replacing any previous description of the same name.
    let describe (descriptor: ToolDescriptor) =
        descriptors.[descriptor.Name] <- descriptor

    /// Describe several at once.
    let describeAll (all: ToolDescriptor seq) = all |> Seq.iter describe

    /// The description of one tool, if it has one.
    let tryFind (name: string) : ToolDescriptor option =
        if String.IsNullOrWhiteSpace name then
            None
        else
            match descriptors.TryGetValue(name.Trim()) with
            | true, descriptor -> Some descriptor
            | _ -> None

    /// Has this tool been described?
    let isDescribed (name: string) = (tryFind name).IsSome

    /// Every description, by name.
    let all () =
        descriptors.Values |> Seq.sortBy (fun d -> d.Name) |> Seq.toList

    /// How many tools are described. Paired with the registry's total, this is the
    /// "N of M tools described" an editor shows while the backfill is incomplete.
    let describedCount () = descriptors.Count

    /// Forget every description. For tests; nothing in the running system calls it.
    let internal clear () = descriptors.Clear()

    // =========================================================================
    // Schema helpers
    // =========================================================================

    /// A JSON Schema for an object with these properties.
    ///
    /// Hand-writing the same schema skeleton several hundred times invites the kind
    /// of typo that a schema is supposed to catch, so the shape is built here and the
    /// descriptors below say only what differs.
    let objectSchema (properties: (string * string * string) list) (required: string list) =
        let property (name: string, jsonType: string, description: string) =
            let escape (s: string) =
                s.Replace("\\", "\\\\").Replace("\"", "\\\"")

            $"\"{escape name}\":{{\"type\":\"{escape jsonType}\",\"description\":\"{escape description}\"}}"

        let body = properties |> List.map property |> String.concat ","

        let requiredList =
            required |> List.map (fun r -> $"\"{r}\"") |> String.concat ","

        $"""{{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{{{body}}},"required":[{requiredList}],"additionalProperties":false}}"""
