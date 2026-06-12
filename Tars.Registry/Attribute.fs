namespace Tars.Registry

open System

/// Marks a public static method as a discoverable Tars skill.
///
/// The registry scans loaded assemblies at startup and exposes every
/// annotated method through `Tars.Registry.Registry.all ()` /
/// `Tars.Registry.Registry.byName`.
///
/// Mirrors the `#[ix_skill]` proc-macro from the ix Rust workspace.
/// See `docs/MIRROR-TO-ECOSYSTEM.md` in the ix repo for the full pattern.
[<AttributeUsage(AttributeTargets.Method, AllowMultiple = false)>]
[<AllowNullLiteral>]
type TarsSkillAttribute(name: string, domain: string) =
    inherit Attribute()

    /// Dotted MCP-style name (e.g. "grammar.promote").
    member _.Name = name

    /// High-level domain bucket (e.g. "grammar", "ingestion", "ml").
    member _.Domain = domain

    /// Optional human-readable description; defaults to the empty string.
    member val Description = "" with get, set
