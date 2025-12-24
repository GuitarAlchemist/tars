namespace Tars.Tools

open System
open System.IO
open System.Threading.Tasks
open System.Text.RegularExpressions

/// Agent Skills specification support (per agentskills.io)
/// A skill is a directory containing a SKILL.md file with YAML frontmatter + Markdown instructions
module AgentSkills =

    /// Represents a parsed Agent Skill
    type Skill =
        { Name: string
          Version: string
          Description: string
          Author: string option
          Tags: string list
          Dependencies: string list
          Instructions: string
          Scripts: Map<string, string>
          Resources: string list }

    /// Parse SKILL.md frontmatter (YAML between ---)
    let private parseFrontmatter (content: string) =
        let fmPattern = Regex(@"^---\s*\n(.*?)\n---\s*\n", RegexOptions.Singleline)
        let m = fmPattern.Match(content)

        if m.Success then
            let yaml = m.Groups.[1].Value
            let body = content.Substring(m.Length)

            // Simple YAML key: value parsing
            let lines = yaml.Split('\n') |> Array.filter (fun l -> l.Contains(":"))

            let pairs =
                lines
                |> Array.choose (fun l ->
                    let idx = l.IndexOf(':')

                    if idx > 0 then
                        Some(l.Substring(0, idx).Trim(), l.Substring(idx + 1).Trim())
                    else
                        None)
                |> Map.ofArray

            Some(pairs, body)
        else
            None

    /// Load a skill from a directory
    let loadSkill (directory: string) : Result<Skill, string> =
        let skillPath = Path.Combine(directory, "SKILL.md")

        if not (File.Exists skillPath) then
            Error $"SKILL.md not found in {directory}"
        else
            try
                let content = File.ReadAllText(skillPath)

                match parseFrontmatter content with
                | Some(meta, body) ->
                    let name =
                        meta |> Map.tryFind "name" |> Option.defaultValue (Path.GetFileName directory)

                    let version = meta |> Map.tryFind "version" |> Option.defaultValue "1.0.0"
                    let desc = meta |> Map.tryFind "description" |> Option.defaultValue ""
                    let author = meta |> Map.tryFind "author"

                    let tags =
                        meta
                        |> Map.tryFind "tags"
                        |> Option.map (fun s -> s.Split(',') |> Array.map (fun t -> t.Trim()) |> Array.toList)
                        |> Option.defaultValue []

                    // Load any scripts in the directory
                    let scripts =
                        Directory.GetFiles(directory, "*.ps1")
                        |> Array.append (Directory.GetFiles(directory, "*.sh"))
                        |> Array.append (Directory.GetFiles(directory, "*.py"))
                        |> Array.map (fun f -> Path.GetFileName f, File.ReadAllText f)
                        |> Map.ofArray

                    Ok
                        { Name = name
                          Version = version
                          Description = desc
                          Author = author
                          Tags = tags
                          Dependencies = []
                          Instructions = body
                          Scripts = scripts
                          Resources = [] }
                | None ->
                    // No frontmatter, treat whole file as instructions
                    Ok
                        { Name = Path.GetFileName directory
                          Version = "1.0.0"
                          Description = ""
                          Author = None
                          Tags = []
                          Dependencies = []
                          Instructions = content
                          Scripts = Map.empty
                          Resources = [] }
            with ex ->
                Error $"Failed to load skill: {ex.Message}"

    /// Discover all skills in a directory
    let discoverSkills (rootPath: string) : Skill list =
        if not (Directory.Exists rootPath) then
            []
        else
            Directory.GetDirectories(rootPath)
            |> Array.choose (fun dir ->
                match loadSkill dir with
                | Ok skill -> Some skill
                | Error _ -> None)
            |> Array.toList

    /// Create a tool from a skill
    let skillToTool (skill: Skill) : Tars.Core.Tool =
        { Name = $"skill_{skill.Name}"
          Description = $"[Skill] {skill.Description}"
          Version = skill.Version
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Execute =
            fun _ ->
                async {
                    // Return skill instructions for the agent to follow
                    return
                        Result.Ok
                            $"""
## Skill: {skill.Name}

{skill.Instructions}

### Available Scripts:
{skill.Scripts |> Map.toList |> List.map fst |> String.concat ", "}
"""
                }
          ThingDescription = None }

module AgentSkillsTools =

    open AgentSkills

    let private skillsDir =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "skills")

    [<TarsToolAttribute("list_skills", "Lists all available Agent Skills. No input required.")>]
    let listSkills (_: string) : Task<string> =
        task {
            let skills = discoverSkills skillsDir

            if skills.IsEmpty then
                return $"No skills found in {skillsDir}. Create a skill by adding a directory with SKILL.md file."
            else
                let lines =
                    skills |> List.map (fun s -> $"- {s.Name} (v{s.Version}): {s.Description}")

                return "Available Agent Skills:\n" + (String.Join("\n", lines))
        }

    [<TarsToolAttribute("load_skill", "Loads an Agent Skill by name. Input: skill name")>]
    let loadSkillTool (args: string) : Task<string> =
        task {
            let name = ToolHelpers.parseStringArg args "name"
            let skillPath = Path.Combine(skillsDir, name)

            match loadSkill skillPath with
            | Ok skill ->
                return
                    $"""
## Skill Loaded: {skill.Name}

**Version**: {skill.Version}
**Tags**: {String.Join(", ", skill.Tags)}

### Instructions:
{skill.Instructions}
"""
            | Error e -> return $"Error: {e}"
        }

    [<TarsToolAttribute("create_skill",
                        "Creates a new Agent Skill. Input JSON: { \"name\": \"...\", \"description\": \"...\", \"instructions\": \"...\" }")>]
    let createSkill (args: string) : Task<string> =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement
                let name = root.GetProperty("name").GetString()
                let desc = root.GetProperty("description").GetString()
                let instr = root.GetProperty("instructions").GetString()

                let skillPath = Path.Combine(skillsDir, name)
                Directory.CreateDirectory(skillPath) |> ignore

                let skillMd =
                    $"""---
name: {name}
version: 1.0.0
description: {desc}
author: TARS
tags: custom
---

{instr}
"""

                File.WriteAllText(Path.Combine(skillPath, "SKILL.md"), skillMd)
                return $"✅ Skill '{name}' created at {skillPath}"
            with ex ->
                return $"Error creating skill: {ex.Message}"
        }

    [<TarsToolAttribute("search_skills_registry",
                        "Searches the Agent Skills registry for available skills. Input: search query")>]
    let searchSkillsRegistry (args: string) : Task<string> =
        task {
            let query = (ToolHelpers.parseStringArg args "query").ToLower()

            // Curated list of available skills from the ecosystem
            // In production, this would query agentskills.io or a similar registry
            let registeredSkills =
                [ ("code-review", "Canva", "Review code for quality, bugs, and style")
                  ("stripe-payments", "Stripe", "Handle Stripe payment integrations")
                  ("notion-docs", "Notion", "Create and manage Notion documents")
                  ("figma-design", "Figma", "Generate and modify Figma designs")
                  ("github-pr", "GitHub", "Manage GitHub pull requests and reviews")
                  ("slack-notify", "Slack", "Send notifications to Slack channels")
                  ("zapier-automation", "Zapier", "Create automation workflows")
                  ("jira-tickets", "Atlassian", "Create and manage Jira tickets") ]

            let matches =
                registeredSkills
                |> List.filter (fun (name, _, desc) -> name.Contains(query) || desc.ToLower().Contains(query))
                |> List.map (fun (name, author, desc) -> $"- {name} by {author}: {desc}")

            if matches.IsEmpty then
                return "No skills found matching query. Try: 'code', 'payment', 'github', or 'slack'."
            else
                return
                    "Found Skills in Registry:\n"
                    + (String.Join("\n", matches))
                    + "\n\nUse install_skill to add a skill to TARS."
        }
