namespace Tars.Core

open System
open System.IO
open System.Text.RegularExpressions

/// Declarative agent definition loaded from Markdown + YAML frontmatter.
/// Compatible with GA's SKILL.md pattern and Spring AI's agent definition format.
[<RequireQualifiedAccess>]
type AgentSkill =
    | Reasoning
    | Planning
    | Critique
    | Verification
    | Communication
    | Analysis
    | Coding
    | Custom of string

/// Parsed agent definition from a .md file.
type AgentDefinition =
    { /// Unique agent identifier (e.g., "planner", "qa-engineer")
      Id: string
      /// Display name
      Name: string
      /// Agent role for routing
      Role: string
      /// One-line description
      Description: string
      /// Model hint for routing (e.g., "reasoning", "fast", "coding")
      ModelHint: string option
      /// Sampling temperature override
      Temperature: float option
      /// Declared capabilities
      Capabilities: AgentSkill list
      /// System prompt (from markdown body)
      SystemPrompt: string
      /// Optional version string
      Version: string option
      /// Source file path
      SourcePath: string option }

module AgentDefinitionParser =

    /// Parse a YAML-like frontmatter value, handling quoted and unquoted strings.
    let private parseValue (raw: string) : string =
        let v = raw.Trim()
        if v.StartsWith("\"") && v.EndsWith("\"") then v.[1..v.Length-2]
        elif v.StartsWith("'") && v.EndsWith("'") then v.[1..v.Length-2]
        else v

    /// Parse capabilities from a YAML list (inline [...] or multi-line - items).
    let private parseCapability (s: string) : AgentSkill =
        match s.Trim().ToLowerInvariant() with
        | "reasoning" -> AgentSkill.Reasoning
        | "planning" -> AgentSkill.Planning
        | "critique" -> AgentSkill.Critique
        | "verification" -> AgentSkill.Verification
        | "communication" -> AgentSkill.Communication
        | "analysis" -> AgentSkill.Analysis
        | "coding" -> AgentSkill.Coding
        | other -> AgentSkill.Custom other

    /// Parse inline YAML list: [item1, item2, item3]
    let private parseInlineList (s: string) : string list =
        if s.StartsWith("[") && s.EndsWith("]") then
            s.[1..s.Length-2].Split(',')
            |> Array.map (fun x -> x.Trim().Trim('"').Trim('\''))
            |> Array.filter (fun x -> x.Length > 0)
            |> Array.toList
        else
            [s]

    /// Parse a markdown file with YAML frontmatter into an AgentDefinition.
    let parse (content: string) (sourcePath: string option) : Result<AgentDefinition, string> =
        let lines = content.Replace("\r\n", "\n").Split('\n')

        // Find frontmatter delimiters
        if lines.Length < 3 || lines.[0].Trim() <> "---" then
            Error "Missing YAML frontmatter (must start with ---)"
        else

        let mutable endIdx = -1
        for i in 1 .. lines.Length - 1 do
            if endIdx = -1 && lines.[i].Trim() = "---" then
                endIdx <- i

        if endIdx < 0 then
            Error "Missing closing --- for frontmatter"
        else

        // Parse frontmatter key-value pairs
        let frontmatter = System.Collections.Generic.Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        let mutable capLines = ResizeArray<string>()
        let mutable inCapList = false

        for i in 1 .. endIdx - 1 do
            let line = lines.[i]
            let trimmed = line.Trim()
            if inCapList then
                if trimmed.StartsWith("- ") then
                    capLines.Add(trimmed.[2..].Trim())
                elif trimmed.Length > 0 && not (trimmed.StartsWith("#")) then
                    inCapList <- false
                    let colonIdx = trimmed.IndexOf(':')
                    if colonIdx > 0 then
                        let key = trimmed.[..colonIdx-1].Trim()
                        let value = trimmed.[colonIdx+1..].Trim()
                        frontmatter.[key] <- value
            else
                let colonIdx = trimmed.IndexOf(':')
                if colonIdx > 0 then
                    let key = trimmed.[..colonIdx-1].Trim()
                    let value = trimmed.[colonIdx+1..].Trim()
                    if key.Equals("capabilities", StringComparison.OrdinalIgnoreCase) && value = "" then
                        inCapList <- true
                    else
                        frontmatter.[key] <- value

        // Extract body (everything after frontmatter)
        let body =
            lines.[endIdx + 1 ..]
            |> Array.toList
            |> List.skipWhile (fun l -> l.Trim() = "")
            |> String.concat "\n"
            |> fun s -> s.TrimEnd()

        // Build definition
        let tryGet key = match frontmatter.TryGetValue(key) with true, v -> Some (parseValue v) | _ -> None
        let require key =
            match tryGet key with
            | Some v -> Ok v
            | None -> Error $"Missing required frontmatter field: {key}"

        match require "id", require "name", require "role" with
        | Ok id, Ok name, Ok role ->
            let caps =
                if capLines.Count > 0 then
                    capLines |> Seq.map parseCapability |> Seq.toList
                else
                    match tryGet "capabilities" with
                    | Some v -> parseInlineList v |> List.map parseCapability
                    | None -> []

            let temp =
                tryGet "temperature"
                |> Option.bind (fun v -> match Double.TryParse(v) with true, f -> Some f | _ -> None)

            Ok
                { Id = id
                  Name = name
                  Role = role
                  Description = tryGet "description" |> Option.defaultValue ""
                  ModelHint = tryGet "model_hint"
                  Temperature = temp
                  Capabilities = caps
                  SystemPrompt = body
                  Version = tryGet "version"
                  SourcePath = sourcePath }
        | Error e, _, _ | _, Error e, _ | _, _, Error e ->
            Error e

    /// Load an agent definition from a .md file.
    let loadFile (path: string) : Result<AgentDefinition, string> =
        try
            let content = File.ReadAllText(path)
            parse content (Some path)
        with ex ->
            Error $"Failed to read {path}: {ex.Message}"

module AgentDefinitionDiscovery =

    /// Search directories for agent .md files.
    let discover (searchPaths: string list) : AgentDefinition list =
        searchPaths
        |> List.collect (fun dir ->
            if Directory.Exists(dir) then
                Directory.GetFiles(dir, "*.md")
                |> Array.choose (fun path ->
                    match AgentDefinitionParser.loadFile path with
                    | Ok def -> Some def
                    | Error _ -> None)
                |> Array.toList
            else [])

    /// Default search paths: ./agents/, ~/.tars/agents/
    let defaultSearchPaths (workingDir: string) : string list =
        let home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)
        [ Path.Combine(workingDir, "agents")
          Path.Combine(home, ".tars", "agents") ]
