namespace Tars.Core

open System
open System.Collections.Concurrent
open System.Text.Json

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
    ///
    /// **These tiers are about integrity, not confidentiality.** They say what a tool
    /// can *change* and how far that reaches. They say nothing about what it can
    /// *see*, and the gate reads only these tiers, so:
    ///
    /// - a `ReadOnly` tool runs without anyone being asked, and can read every file
    ///   the TARS process can read - `read_file` on an id_rsa, a .env, a password
    ///   manager's export, is `ReadOnly` and auto-approved, correctly by this scale;
    /// - what stops that today is not the tier but the catalog: `read_file` is one of
    ///   six described tools, and whoever draws the graph is the person whose files
    ///   they are.
    ///
    /// That holds while the editor and the files share an owner. It stops holding the
    /// moment a graph arrives from somewhere its author does not own - a shared
    /// editor, a plan pasted in, a model writing the graph. At that point reading is
    /// exfiltration and `ReadOnly` is the wrong answer, and the fix is a second axis
    /// on `Approval` rather than a new tier, because reach and sensitivity are
    /// genuinely independent.
    ///
    /// This is written down rather than fixed because nothing in TARS has that shape
    /// yet, and a dimension added before there is a case to test it against would be
    /// guessed rather than designed.
    type ApprovalTier =
        /// Observes without changing anything: reads a file, parses, computes. Says
        /// nothing about *what* it may observe - see the note on the type.
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
        {
            Tier: ApprovalTier
            /// One plain sentence naming the effect, for a human deciding whether to
            /// allow it: "writes files under the working directory", not "file I/O".
            Effect: string
        }

        /// May a graph run this without asking first?
        ///
        /// Derived from the tier, not stored beside it. An earlier version carried this
        /// as a field, arguing that "how far do the effects reach" and "should a person
        /// be asked" are different questions. They are — but no tool answers them
        /// differently, and the gate reads only this one. Two fields for one fact means
        /// the gate can be told `true` by a descriptor whose tier says `Escapes`, and the
        /// field it reads is the weaker of the two.
        ///
        /// If a tool ever needs to be observed-only and still confirmed, give it a tier
        /// that says so. Do not put back a field the gate can disagree with.
        member this.AutoApproved = (this.Tier = ReadOnly)

        /// Observes, costs nothing, changes nothing: safe to run unasked.
        ///
        /// "Reaches nothing" would be the wrong word: a reader reaches whatever the
        /// process can read. See the note on `ApprovalTier`.
        static member observes(effect: string) = { Tier = ReadOnly; Effect = effect }

        /// Changes something this machine owns.
        static member mutates(effect: string) = { Tier = Mutates; Effect = effect }

        /// Leaves the machine or spends money.
        static member escapes(effect: string) = { Tier = Escapes; Effect = effect }

    /// Everything the editor needs about one tool, beyond its name and description.
    type ToolDescriptor =
        {
            Name: string
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
            Approval: Approval
        }

    /// Tool name -> descriptor. Ordinal-case-insensitive, because tool names are
    /// typed by hand in graphs and `Git_Commit` should not be a different tool.
    let private descriptors =
        ConcurrentDictionary<string, ToolDescriptor>(StringComparer.OrdinalIgnoreCase)

    /// Describe a tool, replacing any previous description of the same name.
    ///
    /// The name is trimmed going in because `tryFind` trims coming out. Without this, a
    /// descriptor stored under `" read_file"` is unfindable, and the editor reports the
    /// tool as *undescribed* rather than as misnamed — the one failure mode this sidecar
    /// exists to stop happening quietly.
    let describe (descriptor: ToolDescriptor) =
        let name = if isNull descriptor.Name then "" else descriptor.Name.Trim()
        descriptors.[name] <- { descriptor with Name = name }

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
    /// descriptors say only what differs.
    ///
    /// Written through `Utf8JsonWriter` rather than by concatenating strings. The
    /// concatenated version escaped backslashes and quotes in the descriptions and
    /// nothing else: a newline or a tab in a description produced invalid JSON, and
    /// required property names were interpolated with no escaping at all. Since these
    /// schemas are what an editor parses, a description someone writes later must not
    /// be able to break the catalog.
    let objectSchema (properties: (string * string * string) list) (required: string list) =
        use buffer = new IO.MemoryStream()

        (use writer = new Utf8JsonWriter(buffer)

         writer.WriteStartObject()
         writer.WriteString("$schema", "https://json-schema.org/draft/2020-12/schema")
         writer.WriteString("type", "object")

         writer.WriteStartObject("properties")

         for name, jsonType, description in properties do
             writer.WriteStartObject(name)
             writer.WriteString("type", jsonType)
             writer.WriteString("description", description)
             writer.WriteEndObject()

         writer.WriteEndObject()

         writer.WriteStartArray("required")

         for name in required do
             writer.WriteStringValue(name)

         writer.WriteEndArray()

         writer.WriteBoolean("additionalProperties", false)
         writer.WriteEndObject())

        Text.Encoding.UTF8.GetString(buffer.ToArray())
