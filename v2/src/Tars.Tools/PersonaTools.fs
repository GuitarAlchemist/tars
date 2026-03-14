/// Persona Tools - Functions for managing and using AI personas
module Tars.Tools.PersonaTools

open Tars.Core.Persona
open Tars.Core.PersonaRegistry

// ============================================================================
// Pure Functions (synchronous)
// ============================================================================

/// List all registered personas
let listPersonas () : string =
    let personas = defaultRegistry.List()

    let truncate maxLen (s: string) =
        if s.Length > maxLen then
            s.Substring(0, maxLen - 3) + "..."
        else
            s

    let formatted =
        personas
        |> List.map (fun p ->
            let desc = p.Description |> Option.defaultValue p.Role |> truncate 80
            $"- **{p.Name}** (`{p.Id}`): {desc}")
        |> String.concat "\n"

    $"## Registered Personas ({personas.Length})\n\n{formatted}"

/// Get details of a specific persona
let getPersonaDetails (id: string) : string =
    match defaultRegistry.Get id with
    | Some p ->
        let constraints =
            if p.Constraints.IsEmpty then
                "None"
            else
                p.Constraints |> List.map (sprintf "- %s") |> String.concat "\n"

        let tags = p.Tags |> String.concat ", "

        let examplesText =
            if p.Examples.IsEmpty then
                "No examples defined"
            else
                $"{p.Examples.Length} examples"

        $"""## {p.Name}

**ID:** `{p.Id}`
**Role:** {p.Role}
**Default Format:** {p.DefaultFormat}
**Temperature:** {p.Temperature |> Option.map string |> Option.defaultValue "default"}
**Tags:** {tags}

### Constraints
{constraints}

### Examples
{examplesText}"""
    | None -> $"Error: Persona '{id}' not found. Use `list_personas` to see available personas."

/// Generate an RTFD prompt using a persona
let generateWithPersona
    (personaId: string)
    (taskDesc: string)
    (format: string option)
    (details: string option)
    : Result<string, string> =
    let outputFormat =
        match format with
        | Some "markdown"
        | Some "md" -> Some Markdown
        | Some "json" -> Some JSON
        | Some "table" -> Some Table
        | Some "bullets"
        | Some "bullet" -> Some BulletPoints
        | Some "prose"
        | Some "text" -> Some Prose
        | Some other -> Some(Custom other)
        | None -> None

    withPersona personaId taskDesc outputFormat details

/// Create and register a new persona
let createNewPersona
    (id: string)
    (name: string)
    (role: string)
    (format: string option)
    (constraints: string list)
    : Result<string, string> =
    let outputFormat =
        match format with
        | Some "markdown"
        | Some "md" -> Markdown
        | Some "json" -> JSON
        | Some "table" -> Table
        | Some "bullets" -> BulletPoints
        | Some "prose" -> Prose
        | _ -> Markdown

    let persona =
        { Id = id
          Name = name
          Role = role
          Description = Some $"Custom persona: {name}"
          DefaultFormat = outputFormat
          Constraints = constraints
          Temperature = None
          Examples = []
          Tags = [ "custom" ] }

    match defaultRegistry.Register persona with
    | Result.Ok() -> Result.Ok $"Persona '{name}' registered with ID '{id}'"
    | Result.Error e -> Result.Error e

// Note: MCP tool registration will be added when integrating with McpServer
