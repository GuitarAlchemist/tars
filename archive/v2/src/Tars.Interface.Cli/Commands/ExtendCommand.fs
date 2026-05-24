namespace Tars.Interface.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Spectre.Console
open Tars.Core.SelfExtension

/// CLI command for TARS self-extension capabilities
module ExtendCommand =

    type ExtendAction =
        | CreateMetascript of name: string * description: string
        | CreateGrammar of name: string * description: string
        | CreateTool of name: string * description: string
        | CreateBlock of name: string * description: string
        | ListExtensions
        | ShowReport
        | Help

    /// Parse command arguments
    let parseArgs (args: string list) : ExtendAction =
        match args with
        | "metascript" :: name :: rest when rest.Length > 0 ->
            CreateMetascript(name, String.Join(" ", rest))
        | "grammar" :: name :: rest when rest.Length > 0 ->
            CreateGrammar(name, String.Join(" ", rest))
        | "tool" :: name :: rest when rest.Length > 0 ->
            CreateTool(name, String.Join(" ", rest))
        | "block" :: name :: rest when rest.Length > 0 ->
            CreateBlock(name, String.Join(" ", rest))
        | ["list"] | ["ls"] -> ListExtensions
        | ["report"] | ["status"] -> ShowReport
        | _ -> Help

    /// Execute the extend command
    let execute (args: string list) : Task<int> =
        task {
            let baseDir = Directory.GetCurrentDirectory()
            let paths = SelfExtensionService.defaultPaths baseDir
            
            match parseArgs args with
            | Help ->
                printfn """
TARS Extend Command - Self-Extension Capabilities

USAGE:
    tars extend <action> [arguments]

ACTIONS:
    metascript <name> <description>  Create a new metascript workflow
    grammar <name> <description>     Create a new grammar rule
    tool <name> <description>        Create a new dynamic tool
    block <name> <description>       Create a new DSL block type
    list | ls                        List all extensions
    report | status                  Show extension report

EXAMPLES:
    tars extend metascript "code_analyzer" "Analyzes code for patterns"
    tars extend grammar "json_response" "Grammar for JSON outputs"
    tars extend block "BELIEF" "Block for declaring beliefs with confidence"
    tars extend list
    tars extend report

DESCRIPTION:
    The extend command allows TARS to create new capabilities without
    modifying core code. Extensions are stored in:
    
    /metascripts/generated/   - Metascript workflows
    /grammars/generated/      - Grammar rules + DSL blocks
    /tools/generated/         - Dynamic tool definitions
    /extensions/manifest.json - Extension registry

GRAMMAR SELF-EXTENSION:
    TARS can extend its own DSL by creating new block types:
    
    tars extend block "INVARIANT" "Declares a constraint invariant"
    
    This creates both grammar rules and handler metascripts.
"""
                return 0
                
            | ListExtensions ->
                let manifest = SelfExtensionService.listExtensions paths
                
                AnsiConsole.Write(new Rule("[bold blue]TARS Extensions[/]"))
                
                if manifest.Extensions.Length = 0 then
                    AnsiConsole.MarkupLine("[dim]No extensions created yet.[/]")
                    AnsiConsole.MarkupLine("[dim]Use 'tars extend metascript <name> <description>' to create one.[/]")
                else
                    let table = new Table()
                    table.AddColumn("Name") |> ignore
                    table.AddColumn("Type") |> ignore
                    table.AddColumn("Status") |> ignore
                    table.AddColumn("Created") |> ignore
                    
                    for ext in manifest.Extensions do
                        let status = 
                            match ext.Status with
                            | ExtensionStatus.Active -> "[green]Active[/]"
                            | ExtensionStatus.Draft -> "[yellow]Draft[/]"
                            | ExtensionStatus.Validated -> "[blue]Validated[/]"
                            | ExtensionStatus.Deprecated -> "[dim]Deprecated[/]"
                            | ExtensionStatus.Failed reason -> $"[red]Failed: {reason}[/]"
                        
                        let typeName =
                            match ext.Type with
                            | ExtensionType.Metascript -> "Metascript"
                            | ExtensionType.DynamicTool -> "Tool"
                            | ExtensionType.Grammar -> "Grammar"
                            | ExtensionType.Workflow -> "Workflow"
                        
                        table.AddRow([| 
                            Markup(ext.Name) :> Rendering.IRenderable
                            Markup(typeName) :> Rendering.IRenderable
                            Markup(status) :> Rendering.IRenderable
                            Markup(ext.CreatedAt.ToString("yyyy-MM-dd")) :> Rendering.IRenderable
                        |]) |> ignore
                    
                    AnsiConsole.Write(table)
                    AnsiConsole.MarkupLine($"[dim]Total: {manifest.TotalCreated} created, {manifest.TotalActive} active[/]")
                
                return 0
                
            | ShowReport ->
                let report = SelfExtensionService.generateReport paths
                AnsiConsole.WriteLine(report)
                return 0
                
            | CreateMetascript(name, description) ->
                AnsiConsole.MarkupLine($"[bold]Creating metascript:[/] {name}")
                AnsiConsole.MarkupLine($"[dim]Description: {description}[/]")
                
                // Generate a template metascript
                let blocks = [
                    ""
                    "// Template metascript - customize the blocks below"
                    ""
                    "FSHARP(output=\"result\") {"
                    "    // Add F# logic here"
                    "    \"Hello from TARS self-extension!\""
                    "}"
                    ""
                    "QUERY(output=\"analysis\") {"
                    "    Based on the result: ${result}"
                    "    "
                    "    Provide your analysis."
                    "}"
                ]
                
                let ext = SelfExtensionService.generateMetascript paths name description blocks
                
                AnsiConsole.MarkupLine($"[green]✓[/] Created: {ext.FilePath.Value}")
                AnsiConsole.MarkupLine($"[dim]Edit the file to customize your metascript.[/]")
                return 0
                
            | CreateGrammar(name, description) ->
                AnsiConsole.MarkupLine($"[bold]Creating grammar:[/] {name}")
                AnsiConsole.MarkupLine($"[dim]Description: {description}[/]")
                
                // Generate a template grammar
                let ebnf = $"""
// Template grammar for: {name}
// Customize the rules below

root        ::= item+
item        ::= "- " content "\\n"
content     ::= [^\\n]+
"""
                
                let ext = SelfExtensionService.generateGrammar paths name description ebnf
                
                AnsiConsole.MarkupLine($"[green]✓[/] Created: {ext.FilePath.Value}")
                AnsiConsole.MarkupLine($"[dim]Edit the file to define your grammar rules.[/]")
                return 0
                
            | CreateTool(name, description) ->
                AnsiConsole.MarkupLine($"[bold]Creating tool:[/] {name}")
                AnsiConsole.MarkupLine($"[dim]Description: {description}[/]")
                
                let toolDef : DynamicToolDefinition = {
                    Name = name
                    Description = description
                    Version = "1.0"
                    InputSchema = Some """{"type": "string", "description": "Input parameter"}"""
                    OutputSchema = Some """{"type": "string", "description": "Result"}"""
                    Implementation = ToolImplementation.FSharpScript "// Add F# implementation here"
                }
                
                let ext = SelfExtensionService.generateTool paths toolDef
                
                AnsiConsole.MarkupLine($"[green]✓[/] Created: {ext.FilePath.Value}")
                AnsiConsole.MarkupLine($"[dim]Edit the file to implement your tool.[/]")
                return 0
            
            | CreateBlock(name, description) ->
                AnsiConsole.MarkupLine($"[bold]Creating DSL block type:[/] {name}")
                AnsiConsole.MarkupLine($"[dim]Description: {description}[/]")
                
                // Generate template block definition
                let blockDef : BlockDefinition = {
                    Name = name.ToUpperInvariant()
                    Description = description
                    Version = "1.0"
                    Parameters = [
                        { Name = "id"; Type = "string"; Required = false; Default = None }
                        { Name = "confidence"; Type = "float"; Required = false; Default = Some "1.0" }
                    ]
                    ContentType = BlockContentType.Freeform
                    EbnfRule = $"""
// EBNF rule for {name} block
{name.ToLowerInvariant()}_block ::= "{name.ToUpperInvariant()}" "(" parameters? ")" "{{" content "}}"
parameters ::= parameter ("," parameter)*
parameter  ::= identifier "=" value
content    ::= [^{{}}]*
"""
                    CompileToIR = "// F# compilation logic"
                }
                
                let ext = SelfExtensionService.generateBlock paths blockDef
                
                AnsiConsole.MarkupLine($"[green]✓[/] Created block grammar: {ext.FilePath.Value}")
                AnsiConsole.MarkupLine($"[dim]TARS can now use {name.ToUpperInvariant()}{{ }} blocks in metascripts![/]")
                return 0
        }
