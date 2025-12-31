namespace Tars.Core.SelfExtension

open System
open System.IO
open System.Threading.Tasks
open System.Text.Json
open System.Text.Json.Serialization

/// A generated extension (metascript, tool, or grammar)
type Extension =
    { Id: Guid
      Name: string
      Type: ExtensionType
      Description: string
      CreatedAt: DateTimeOffset
      CreatedBy: string
      Version: string
      FilePath: string option
      Status: ExtensionStatus }

and ExtensionType =
    | Metascript
    | DynamicTool
    | Grammar
    | Workflow

and ExtensionStatus =
    | Draft
    | Validated
    | Active
    | Deprecated
    | Failed of reason: string

/// JSON-serializable version of Extension
type SerializableExtension =
    { Id: string
      Name: string
      Type: string
      Description: string
      CreatedAt: string
      CreatedBy: string
      Version: string
      FilePath: string
      Status: string }

/// Manifest tracking all self-generated extensions (serializable)
type SerializableManifest =
    { LastUpdated: string
      Extensions: SerializableExtension list
      TotalCreated: int
      TotalActive: int }

/// Manifest tracking all self-generated extensions
type ExtensionManifest =
    { LastUpdated: DateTimeOffset
      Extensions: Extension list
      TotalCreated: int
      TotalActive: int }

/// Definition for a dynamically generated tool
type DynamicToolDefinition =
    { Name: string
      Description: string
      Version: string
      InputSchema: string option
      OutputSchema: string option
      Implementation: ToolImplementation }

and ToolImplementation =
    | FSharpScript of code: string
    | MetascriptRef of name: string
    | ExternalCommand of command: string

/// Definition for a custom DSL block type
type BlockDefinition =
    { Name: string // e.g., "BELIEF", "INVARIANT", "WORKFLOW"
      Description: string
      Version: string
      Parameters: BlockParameter list
      ContentType: BlockContentType
      EbnfRule: string // EBNF grammar rule for this block
      CompileToIR: string } // F# code that compiles block to IR

and BlockParameter =
    { Name: string
      Type: string // "string", "float", "bool", "list"
      Required: bool
      Default: string option }

and BlockContentType =
    | Freeform // Any text content
    | Structured of schema: string // JSON schema
    | Expression // F# expression
    | NestedBlocks // Can contain other blocks

/// JSON-serializable version of DynamicToolDefinition
type SerializableTool =
    { Name: string
      Description: string
      Version: string
      InputSchema: string
      OutputSchema: string
      ImplementationType: string // "FSharpScript", "MetascriptRef", "ExternalCommand"
      ImplementationValue: string } // The actual code/name/command

module ToolSerialization =
    open System.Text.Json

    /// Convert DynamicToolDefinition to serializable format
    let toSerializable (tool: DynamicToolDefinition) : SerializableTool =
        let implType, implValue =
            match tool.Implementation with
            | FSharpScript code -> "FSharpScript", code
            | MetascriptRef name -> "MetascriptRef", name
            | ExternalCommand cmd -> "ExternalCommand", cmd

        { Name = tool.Name
          Description = tool.Description
          Version = tool.Version
          InputSchema = tool.InputSchema |> Option.defaultValue ""
          OutputSchema = tool.OutputSchema |> Option.defaultValue ""
          ImplementationType = implType
          ImplementationValue = implValue }

    /// Convert serializable format to DynamicToolDefinition
    let fromSerializable (st: SerializableTool) : DynamicToolDefinition =
        let impl =
            match st.ImplementationType with
            | "FSharpScript" -> FSharpScript st.ImplementationValue
            | "MetascriptRef" -> MetascriptRef st.ImplementationValue
            | "ExternalCommand" -> ExternalCommand st.ImplementationValue
            | _ -> FSharpScript st.ImplementationValue

        { Name = st.Name
          Description = st.Description
          Version = st.Version
          InputSchema =
            if String.IsNullOrEmpty(st.InputSchema) then
                None
            else
                Some st.InputSchema
          OutputSchema =
            if String.IsNullOrEmpty(st.OutputSchema) then
                None
            else
                Some st.OutputSchema
          Implementation = impl }

    /// Serialize tool definition to JSON
    let serializeToJson (tool: DynamicToolDefinition) : string =
        let options = JsonSerializerOptions(WriteIndented = true)
        let st = toSerializable tool
        JsonSerializer.Serialize(st, options)

    /// Deserialize tool definition from JSON
    let deserializeFromJson (json: string) : Result<DynamicToolDefinition, string> =
        try
            let st = JsonSerializer.Deserialize<SerializableTool>(json)
            Ok(fromSerializable st)
        with ex ->
            Error ex.Message

    /// Load all tools from the generated tools directory
    let loadAllTools (toolsDir: string) : DynamicToolDefinition list =
        if Directory.Exists(toolsDir) then
            Directory.GetFiles(toolsDir, "*.json")
            |> Array.choose (fun path ->
                try
                    let json = File.ReadAllText(path)

                    match deserializeFromJson json with
                    | Ok tool -> Some tool
                    | Error _ -> None
                with _ ->
                    None)
            |> Array.toList
        else
            []

/// Result of generating an extension
type GenerationResult<'T> = Result<'T * Extension, string>

/// Service for TARS self-extension capabilities
module SelfExtensionService =

    /// Paths for self-generated content
    type ExtensionPaths =
        { MetascriptsDir: string
          GrammarsDir: string
          ToolsDir: string
          ManifestPath: string }

    let defaultPaths (baseDir: string) =
        { MetascriptsDir = Path.Combine(baseDir, "metascripts", "generated")
          GrammarsDir = Path.Combine(baseDir, "grammars", "generated")
          ToolsDir = Path.Combine(baseDir, "tools", "generated")
          ManifestPath = Path.Combine(baseDir, "extensions", "manifest.json") }

    /// Ensure directories exist
    let ensureDirectories (paths: ExtensionPaths) =
        Directory.CreateDirectory(paths.MetascriptsDir) |> ignore
        Directory.CreateDirectory(paths.GrammarsDir) |> ignore
        Directory.CreateDirectory(paths.ToolsDir) |> ignore
        Directory.CreateDirectory(Path.GetDirectoryName(paths.ManifestPath)) |> ignore

    /// Convert Extension to serializable format
    let toSerializable (ext: Extension) : SerializableExtension =
        { Id = ext.Id.ToString()
          Name = ext.Name
          Type =
            match ext.Type with
            | ExtensionType.Metascript -> "Metascript"
            | ExtensionType.DynamicTool -> "DynamicTool"
            | ExtensionType.Grammar -> "Grammar"
            | ExtensionType.Workflow -> "Workflow"
          Description = ext.Description
          CreatedAt = ext.CreatedAt.ToString("o")
          CreatedBy = ext.CreatedBy
          Version = ext.Version
          FilePath = ext.FilePath |> Option.defaultValue ""
          Status =
            match ext.Status with
            | ExtensionStatus.Active -> "Active"
            | ExtensionStatus.Draft -> "Draft"
            | ExtensionStatus.Validated -> "Validated"
            | ExtensionStatus.Deprecated -> "Deprecated"
            | ExtensionStatus.Failed reason -> $"Failed: {reason}" }

    /// Convert serializable format to Extension
    let fromSerializable (se: SerializableExtension) : Extension =
        { Id = Guid.Parse(se.Id)
          Name = se.Name
          Type =
            match se.Type with
            | "Metascript" -> ExtensionType.Metascript
            | "DynamicTool" -> ExtensionType.DynamicTool
            | "Grammar" -> ExtensionType.Grammar
            | "Workflow" -> ExtensionType.Workflow
            | _ -> ExtensionType.Metascript
          Description = se.Description
          CreatedAt = DateTimeOffset.Parse(se.CreatedAt)
          CreatedBy = se.CreatedBy
          Version = se.Version
          FilePath =
            if String.IsNullOrEmpty(se.FilePath) then
                None
            else
                Some se.FilePath
          Status =
            match se.Status with
            | "Active" -> ExtensionStatus.Active
            | "Draft" -> ExtensionStatus.Draft
            | "Validated" -> ExtensionStatus.Validated
            | "Deprecated" -> ExtensionStatus.Deprecated
            | s when s.StartsWith("Failed:") -> ExtensionStatus.Failed(s.Substring(8))
            | _ -> ExtensionStatus.Active }

    /// Load the extension manifest
    let loadManifest (paths: ExtensionPaths) : ExtensionManifest =
        if File.Exists(paths.ManifestPath) then
            let json = File.ReadAllText(paths.ManifestPath)

            try
                let sm = JsonSerializer.Deserialize<SerializableManifest>(json)

                { LastUpdated = DateTimeOffset.Parse(sm.LastUpdated)
                  Extensions = sm.Extensions |> List.map fromSerializable
                  TotalCreated = sm.TotalCreated
                  TotalActive = sm.TotalActive }
            with _ ->
                { LastUpdated = DateTimeOffset.UtcNow
                  Extensions = []
                  TotalCreated = 0
                  TotalActive = 0 }
        else
            { LastUpdated = DateTimeOffset.UtcNow
              Extensions = []
              TotalCreated = 0
              TotalActive = 0 }

    /// Save the extension manifest
    let saveManifest (paths: ExtensionPaths) (manifest: ExtensionManifest) =
        let sm: SerializableManifest =
            { LastUpdated = manifest.LastUpdated.ToString("o")
              Extensions = manifest.Extensions |> List.map toSerializable
              TotalCreated = manifest.TotalCreated
              TotalActive = manifest.TotalActive }

        let options = JsonSerializerOptions(WriteIndented = true)
        let json = JsonSerializer.Serialize(sm, options)
        File.WriteAllText(paths.ManifestPath, json)

    /// Generate a new metascript from a description
    let generateMetascript
        (paths: ExtensionPaths)
        (name: string)
        (description: string)
        (blocks: string list)
        : Extension =

        ensureDirectories paths

        let safeName = name.Replace(" ", "_").ToLowerInvariant()
        let filePath = Path.Combine(paths.MetascriptsDir, $"{safeName}.tars")

        // Generate metascript content
        let content =
            [ $"meta {{"
              $"    name = \"{name}\""
              $"    description = \"{description}\""
              $"    author = \"TARS-self-extension\""
              $"    version = \"1.0\""
              $"    created = \"{DateTimeOffset.UtcNow:o}\""
              $"}}"
              ""
              yield! blocks ]
            |> String.concat Environment.NewLine

        File.WriteAllText(filePath, content)

        let extension: Extension =
            { Id = Guid.NewGuid()
              Name = name
              Type = ExtensionType.Metascript
              Description = description
              CreatedAt = DateTimeOffset.UtcNow
              CreatedBy = "TARS"
              Version = "1.0"
              FilePath = Some filePath
              Status = ExtensionStatus.Active }

        // Update manifest
        let manifest = loadManifest paths

        let updatedManifest =
            { manifest with
                LastUpdated = DateTimeOffset.UtcNow
                Extensions = extension :: manifest.Extensions
                TotalCreated = manifest.TotalCreated + 1
                TotalActive = manifest.TotalActive + 1 }

        saveManifest paths updatedManifest

        extension

    /// Generate a new grammar from EBNF definition
    let generateGrammar (paths: ExtensionPaths) (name: string) (description: string) (ebnfContent: string) : Extension =

        ensureDirectories paths

        let safeName = name.Replace(" ", "_").ToLowerInvariant()
        let filePath = Path.Combine(paths.GrammarsDir, $"{safeName}.ebnf")

        // Add header comment
        let content =
            [ $"// Grammar: {name}"
              $"// Description: {description}"
              $"// Generated by: TARS Self-Extension"
              $"// Created: {DateTimeOffset.UtcNow:o}"
              ""
              ebnfContent ]
            |> String.concat Environment.NewLine

        File.WriteAllText(filePath, content)

        let extension: Extension =
            { Id = Guid.NewGuid()
              Name = name
              Type = ExtensionType.Grammar
              Description = description
              CreatedAt = DateTimeOffset.UtcNow
              CreatedBy = "TARS"
              Version = "1.0"
              FilePath = Some filePath
              Status = ExtensionStatus.Active }

        // Update manifest
        let manifest = loadManifest paths

        let updatedManifest =
            { manifest with
                LastUpdated = DateTimeOffset.UtcNow
                Extensions = extension :: manifest.Extensions
                TotalCreated = manifest.TotalCreated + 1
                TotalActive = manifest.TotalActive + 1 }

        saveManifest paths updatedManifest

        extension

    /// Generate a dynamic tool definition
    let generateTool (paths: ExtensionPaths) (toolDef: DynamicToolDefinition) : Extension =

        ensureDirectories paths

        let safeName = toolDef.Name.Replace(" ", "_").ToLowerInvariant()
        let filePath = Path.Combine(paths.ToolsDir, $"{safeName}.json")

        // Use proper serialization to handle discriminated union
        let json = ToolSerialization.serializeToJson toolDef
        File.WriteAllText(filePath, json)

        let extension: Extension =
            { Id = Guid.NewGuid()
              Name = toolDef.Name
              Type = ExtensionType.DynamicTool
              Description = toolDef.Description
              CreatedAt = DateTimeOffset.UtcNow
              CreatedBy = "TARS"
              Version = toolDef.Version
              FilePath = Some filePath
              Status = ExtensionStatus.Active }

        // Update manifest
        let manifest = loadManifest paths

        let updatedManifest =
            { manifest with
                LastUpdated = DateTimeOffset.UtcNow
                Extensions = extension :: manifest.Extensions
                TotalCreated = manifest.TotalCreated + 1
                TotalActive = manifest.TotalActive + 1 }

        saveManifest paths updatedManifest

        extension

    /// Generate a new DSL block type
    let generateBlock (paths: ExtensionPaths) (blockDef: BlockDefinition) : Extension =

        ensureDirectories paths

        let safeName = blockDef.Name.Replace(" ", "_").ToLowerInvariant()
        let grammarPath = Path.Combine(paths.GrammarsDir, "block_" + safeName + ".ebnf")

        let metascriptPath =
            Path.Combine(paths.MetascriptsDir, "block_" + safeName + "_handler.tars")

        // Build parameter string
        let paramsStr =
            blockDef.Parameters
            |> List.map (fun p -> p.Name + "=\"...\"")
            |> String.concat ", "

        let createdAt = DateTimeOffset.UtcNow.ToString("o")

        // Build EBNF content using StringBuilder
        let sb = System.Text.StringBuilder()
        sb.AppendLine("// Block: " + blockDef.Name) |> ignore
        sb.AppendLine("// Description: " + blockDef.Description) |> ignore
        sb.AppendLine("// Generated by: TARS Self-Extension") |> ignore
        sb.AppendLine("// Created: " + createdAt) |> ignore
        sb.AppendLine() |> ignore

        sb.AppendLine("// Usage: " + blockDef.Name + "(" + paramsStr + ") { content }")
        |> ignore

        sb.AppendLine() |> ignore
        sb.AppendLine(blockDef.EbnfRule) |> ignore

        File.WriteAllText(grammarPath, sb.ToString())

        // Build handler content
        let hsb = System.Text.StringBuilder()
        hsb.AppendLine("meta {") |> ignore
        hsb.AppendLine("    name = \"" + blockDef.Name + "_handler\"") |> ignore

        hsb.AppendLine("    description = \"Handler for " + blockDef.Name + " block type\"")
        |> ignore

        hsb.AppendLine("    author = \"TARS-self-extension\"") |> ignore
        hsb.AppendLine("    version = \"" + blockDef.Version + "\"") |> ignore
        hsb.AppendLine("}") |> ignore
        hsb.AppendLine() |> ignore
        hsb.AppendLine("// This metascript processes the block content") |> ignore
        hsb.AppendLine("// and compiles it to the HybridBrain IR") |> ignore
        hsb.AppendLine() |> ignore
        hsb.AppendLine("FSHARP(output=\"ir\") {") |> ignore
        hsb.AppendLine("    // Block compilation logic") |> ignore
        hsb.AppendLine("    // Block type: " + blockDef.Name) |> ignore
        hsb.AppendLine("    let content = \"${block_content}\"") |> ignore
        hsb.AppendLine("    // Return IR representation") |> ignore
        hsb.AppendLine("    content") |> ignore
        hsb.AppendLine("}") |> ignore

        File.WriteAllText(metascriptPath, hsb.ToString())

        let extension: Extension =
            { Id = Guid.NewGuid()
              Name = blockDef.Name
              Type = ExtensionType.Grammar
              Description = "DSL Block: " + blockDef.Description
              CreatedAt = DateTimeOffset.UtcNow
              CreatedBy = "TARS"
              Version = blockDef.Version
              FilePath = Some grammarPath
              Status = ExtensionStatus.Active }

        let manifest = loadManifest paths

        let updatedManifest =
            { manifest with
                LastUpdated = DateTimeOffset.UtcNow
                Extensions = extension :: manifest.Extensions
                TotalCreated = manifest.TotalCreated + 1
                TotalActive = manifest.TotalActive + 1 }

        saveManifest paths updatedManifest

        extension

    /// List all extensions
    let listExtensions (paths: ExtensionPaths) : ExtensionManifest = loadManifest paths

    /// Get extension by name
    let getExtension (paths: ExtensionPaths) (name: string) : Extension option =
        let manifest = loadManifest paths
        manifest.Extensions |> List.tryFind (fun e -> e.Name = name)

    /// Deprecate an extension
    let deprecateExtension (paths: ExtensionPaths) (name: string) : bool =
        let manifest = loadManifest paths

        let updated =
            manifest.Extensions
            |> List.map (fun e ->
                if e.Name = name then
                    { e with
                        Status = ExtensionStatus.Deprecated }
                else
                    e)

        let newManifest =
            { manifest with
                Extensions = updated
                TotalActive =
                    updated
                    |> List.filter (fun e -> e.Status = ExtensionStatus.Active)
                    |> List.length }

        saveManifest paths newManifest
        true

    /// Generate a summary report of all extensions
    let generateReport (paths: ExtensionPaths) : string =
        let manifest = loadManifest paths
        let sb = System.Text.StringBuilder()

        sb.AppendLine("═══════════════════════════════════════════════════════════════")
        |> ignore

        sb.AppendLine("            TARS SELF-EXTENSION MANIFEST                        ")
        |> ignore

        sb.AppendLine("═══════════════════════════════════════════════════════════════")
        |> ignore

        sb.AppendLine() |> ignore
        let dateStr = manifest.LastUpdated.ToString("yyyy-MM-dd HH:mm:ss")
        sb.AppendLine($"Last Updated: {dateStr}") |> ignore
        sb.AppendLine($"Total Created: {manifest.TotalCreated}") |> ignore
        sb.AppendLine($"Total Active: {manifest.TotalActive}") |> ignore
        sb.AppendLine() |> ignore

        let byType = manifest.Extensions |> List.groupBy (fun e -> e.Type)

        for (extType, extensions) in byType do
            sb.AppendLine($"┌─ {extType} ({extensions.Length})") |> ignore

            for ext in extensions do
                let statusIcon =
                    match ext.Status with
                    | Active -> "✅"
                    | Draft -> "📝"
                    | Validated -> "✓"
                    | Deprecated -> "⚠️"
                    | Failed _ -> "❌"

                sb.AppendLine($"│  {statusIcon} {ext.Name} v{ext.Version}") |> ignore
                sb.AppendLine($"│     {ext.Description}") |> ignore

            sb.AppendLine("└─") |> ignore
            sb.AppendLine() |> ignore

        sb.AppendLine("═══════════════════════════════════════════════════════════════")
        |> ignore

        sb.ToString()
