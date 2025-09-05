namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Commands

/// FLUX language processing module for the chatbot
module ChatbotFLUX =

    /// Parse a FLUX expression
    let parseFluxExpression (expression: string) : Result<FluxValue, string> =
        try
            // Simple expression parsing
            let trimmed = expression.Trim()
            
            // Try to parse as number
            match FluxValueUtils.tryParseNumber trimmed with
            | Some value -> Ok value
            | None ->
                // Try to parse as boolean
                match FluxValueUtils.tryParseBoolean trimmed with
                | Some value -> Ok value
                | None ->
                    // Try to parse as mathematical expression
                    match ChatbotMath.evaluateExpression trimmed with
                    | Ok result -> 
                        match FluxValueUtils.tryParseNumber result.Output with
                        | Some numValue -> Ok numValue
                        | None -> Ok (StringValue result.Output)
                    | Error _ ->
                        // Default to string value
                        Ok (StringValue trimmed)
        with
        | ex -> Error ex.Message

    /// Execute a FLUX assignment
    let executeAssignment (context: FluxContext) (variable: string) (expression: string) : Result<FluxContext * FluxValue, string> =
        match parseFluxExpression expression with
        | Ok value ->
            let newContext = FluxContextUtils.assignVariable context variable value
            Ok (newContext, value)
        | Error err -> Error err

    /// Execute a FLUX expression
    let executeExpression (context: FluxContext) (expression: string) : Result<FluxValue, string> =
        // Check if it's a variable reference
        if expression.StartsWith("$") then
            let varName = expression.Substring(1)
            match FluxContextUtils.getVariable context varName with
            | Some value -> Ok value
            | None -> Error $"Variable '{varName}' not found"
        else
            parseFluxExpression expression

    /// Execute a FLUX command
    let executeFluxCommand (context: FluxContext) (command: FluxCommand) : Result<FluxContext * FluxValue option, string> =
        match command with
        | Assignment (variable, expression) ->
            match executeAssignment context variable expression with
            | Ok (newContext, value) -> Ok (newContext, Some value)
            | Error err -> Error err
        | Expression expression ->
            match executeExpression context expression with
            | Ok value -> Ok (context, Some value)
            | Error err -> Error err
        | FunctionCall (name, args) ->
            // Simple function call handling
            match name.ToLower() with
            | "add" when args.Length = 2 ->
                match executeExpression context args.[0], executeExpression context args.[1] with
                | Ok (NumberValue a), Ok (NumberValue b) -> Ok (context, Some (NumberValue (a + b)))
                | _ -> Error "Add function requires two numbers"
            | "concat" when args.Length = 2 ->
                match executeExpression context args.[0], executeExpression context args.[1] with
                | Ok a, Ok b -> 
                    let result = FluxValueUtils.fluxValueToString a + FluxValueUtils.fluxValueToString b
                    Ok (context, Some (StringValue result))
                | _ -> Error "Concat function requires two values"
            | _ -> Error $"Unknown function: {name}"
        | Query query ->
            // Simple query handling
            if query.StartsWith("list") then
                let variables = 
                    context.Variables 
                    |> Map.toList 
                    |> List.map (fun (k, v) -> $"{k} = {FluxValueUtils.fluxValueToString v}")
                    |> String.concat "\n"
                Ok (context, Some (StringValue variables))
            else
                Error $"Unknown query: {query}"

    /// Process a FLUX input
    let processFluxInput (input: string) (session: ChatbotSession) : Task<ChatbotResult> =
        task {
            try
                match CommandParsingUtils.parseFluxCommand input with
                | Some command ->
                    match executeFluxCommand session.FluxContext command with
                    | Ok (newContext, Some value) ->
                        let valueStr = FluxValueUtils.fluxValueToString value
                        
                        let resultContent = $"""[bold green]FLUX Result:[/]
[yellow]Input:[/] {Markup.Escape(input)}
[yellow]Output:[/] {Markup.Escape(valueStr)}
[dim]Processed by FLUX Engine[/]"""
                        
                        let resultPanel = Panel(resultContent)
                        resultPanel.Header <- PanelHeader("[bold blue]⚡ FLUX Language Engine[/]")
                        resultPanel.Border <- BoxBorder.Rounded
                        AnsiConsole.Write(resultPanel)
                        AnsiConsole.MarkupLine("[green]✅ FLUX command executed[/]")

                        let updatedSession = ChatbotSessionUtils.updateFluxContext session newContext
                        return ResultUtils.success "FLUX command executed successfully" (Some updatedSession)

                    | Ok (newContext, None) ->
                        AnsiConsole.MarkupLine("[green]✅ FLUX command executed (no output)[/]")
                        let updatedSession = ChatbotSessionUtils.updateFluxContext session newContext
                        return ResultUtils.success "FLUX command executed" (Some updatedSession)

                    | Error err ->
                        AnsiConsole.MarkupLine($"[red]❌ FLUX Error: {err}[/]")
                        return ResultUtils.failure $"FLUX Error: {err}"

                | None ->
                    return ResultUtils.failure "Invalid FLUX command syntax"

            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ FLUX Exception: {ex.Message}[/]")
                return ResultUtils.failure $"FLUX Exception: {ex.Message}"
        }

    /// Display FLUX context variables
    let displayFluxContext (context: FluxContext) : unit =
        if context.Variables.IsEmpty then
            AnsiConsole.MarkupLine("[dim]No FLUX variables defined[/]")
        else
            let variablesTable = Table()
            variablesTable.AddColumn("Variable")
            variablesTable.AddColumn("Value")
            variablesTable.AddColumn("Type")

            for KeyValue(name, value) in context.Variables do
                let valueStr = FluxValueUtils.fluxValueToString value
                let typeStr = 
                    match value with
                    | StringValue _ -> "String"
                    | NumberValue _ -> "Number"
                    | BooleanValue _ -> "Boolean"
                    | ListValue _ -> "List"
                    | ObjectValue _ -> "Object"
                    | FunctionValue _ -> "Function"

                variablesTable.AddRow(name, valueStr, typeStr)

            AnsiConsole.Write(variablesTable)

    /// Get FLUX context status
    let getFluxStatus (context: FluxContext) : string =
        let varCount = context.Variables.Count
        let funcCount = context.Functions.Count
        $"✅ FLUX Context - Variables: {varCount}, Functions: {funcCount}"

    /// Clear FLUX context
    let clearFluxContext () : FluxContext =
        FluxContextUtils.emptyContext

    /// Export FLUX context to string
    let exportFluxContext (context: FluxContext) : string =
        context.Variables
        |> Map.toList
        |> List.map (fun (name, value) -> $"{name} = {FluxValueUtils.fluxValueToString value}")
        |> String.concat "\n"

    /// Import FLUX context from string
    let importFluxContext (contextStr: string) : Result<FluxContext, string> =
        try
            let lines = contextStr.Split('\n', StringSplitOptions.RemoveEmptyEntries)
            let mutable context = FluxContextUtils.emptyContext

            for line in lines do
                if line.Contains("=") then
                    let parts = line.Split('=', 2)
                    if parts.Length = 2 then
                        let varName = parts.[0].Trim()
                        let valueStr = parts.[1].Trim()
                        let value = FluxValueUtils.parseFluxValue valueStr
                        context <- FluxContextUtils.assignVariable context varName value

            Ok context
        with
        | ex -> Error ex.Message

    /// Validate FLUX syntax
    let validateFluxSyntax (input: string) : Result<unit, string> =
        try
            match CommandParsingUtils.parseFluxCommand input with
            | Some _ -> Ok ()
            | None -> Error "Invalid FLUX syntax"
        with
        | ex -> Error ex.Message

    /// Get FLUX help text
    let getFluxHelp () : string =
        """[bold blue]FLUX Language Help[/]

[yellow]Variable Assignment:[/]
  variable = value
  x = 42
  name = "TARS"

[yellow]Expressions:[/]
  flux 2 + 3
  flux $x * 2

[yellow]Functions:[/]
  add(5, 3)
  concat("Hello", " World")

[yellow]Queries:[/]
  list variables
  
[yellow]Variable References:[/]
  $variable_name"""
