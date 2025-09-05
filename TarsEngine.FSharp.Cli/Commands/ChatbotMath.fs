namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Commands

/// Mathematical computation module for the chatbot
module ChatbotMath =

    /// Evaluate a mathematical expression using AngouriMath
    let evaluateExpression (expression: string) : Result<MathResult, string> =
        try
            // Try to use AngouriMath if available
            #if ANGOURIMATH_AVAILABLE
            let entity = AngouriMath.MathS.FromString(expression)
            let simplified = entity.Simplify()
            let result = simplified.ToString()
            Ok (ResultUtils.mathSuccess expression result)
            #else
            // Fallback to basic arithmetic evaluation
            evaluateBasicArithmetic expression
            #endif
        with
        | ex -> Error ex.Message

    /// Basic arithmetic evaluation (fallback)
    let evaluateBasicArithmetic (expression: string) : Result<MathResult, string> =
        try
            let cleanExpr = expression.Replace(" ", "")
            
            // Simple arithmetic operations
            if cleanExpr.Contains("+") then
                let parts = cleanExpr.Split('+')
                if parts.Length = 2 then
                    match Double.TryParse(parts.[0]), Double.TryParse(parts.[1]) with
                    | (true, a), (true, b) -> 
                        let result = a + b
                        Ok (ResultUtils.mathSuccess expression (result.ToString()))
                    | _ -> Error "Invalid numbers in addition"
                else
                    Error "Invalid addition expression"
            
            elif cleanExpr.Contains("-") && not (cleanExpr.StartsWith("-")) then
                let parts = cleanExpr.Split('-')
                if parts.Length = 2 then
                    match Double.TryParse(parts.[0]), Double.TryParse(parts.[1]) with
                    | (true, a), (true, b) -> 
                        let result = a - b
                        Ok (ResultUtils.mathSuccess expression (result.ToString()))
                    | _ -> Error "Invalid numbers in subtraction"
                else
                    Error "Invalid subtraction expression"
            
            elif cleanExpr.Contains("*") then
                let parts = cleanExpr.Split('*')
                if parts.Length = 2 then
                    match Double.TryParse(parts.[0]), Double.TryParse(parts.[1]) with
                    | (true, a), (true, b) -> 
                        let result = a * b
                        Ok (ResultUtils.mathSuccess expression (result.ToString()))
                    | _ -> Error "Invalid numbers in multiplication"
                else
                    Error "Invalid multiplication expression"
            
            elif cleanExpr.Contains("/") then
                let parts = cleanExpr.Split('/')
                if parts.Length = 2 then
                    match Double.TryParse(parts.[0]), Double.TryParse(parts.[1]) with
                    | (true, a), (true, b) when b <> 0.0 -> 
                        let result = a / b
                        Ok (ResultUtils.mathSuccess expression (result.ToString()))
                    | (true, _), (true, 0.0) -> Error "Division by zero"
                    | _ -> Error "Invalid numbers in division"
                else
                    Error "Invalid division expression"
            
            else
                // Try to parse as a single number
                match Double.TryParse(cleanExpr) with
                | true, value -> Ok (ResultUtils.mathSuccess expression (value.ToString()))
                | false, _ -> Error "Unsupported mathematical expression"

        with
        | ex -> Error ex.Message

    /// Process a mathematical input
    let processMathInput (input: string) (session: ChatbotSession) : Task<ChatbotResult> =
        task {
            try
                match CommandParsingUtils.parseMathExpression input with
                | Some (SimpleExpression expr) ->
                    match evaluateExpression expr with
                    | Ok result ->
                        let resultContent = $"""[bold green]Mathematical Result:[/]
[yellow]Input:[/] {Markup.Escape(result.Input)}
[yellow]Output:[/] {Markup.Escape(result.Output)}
[dim]Computed using Mathematical Engine[/]"""
                        
                        let resultPanel = Panel(resultContent)
                        resultPanel.Header <- PanelHeader("[bold blue]🧮 Mathematical Engine[/]")
                        resultPanel.Border <- BoxBorder.Rounded
                        AnsiConsole.Write(resultPanel)
                        AnsiConsole.MarkupLine("[green]✅ Mathematical computation completed[/]")

                        return ResultUtils.success "Mathematical computation completed" (Some session)

                    | Error err ->
                        AnsiConsole.MarkupLine($"[red]❌ Math Error: {err}[/]")
                        return ResultUtils.failure $"Math Error: {err}"

                | Some (VariableExpression (variable, expr)) ->
                    match evaluateExpression expr with
                    | Ok result ->
                        // Try to parse the result as a FLUX value
                        let fluxValue = FluxValueUtils.parseFluxValue result.Output
                        let newContext = FluxContextUtils.assignVariable session.FluxContext variable fluxValue
                        
                        AnsiConsole.MarkupLine($"[green]✅ Variable assigned: {variable} = {FluxValueUtils.fluxValueToString fluxValue}[/]")
                        
                        let updatedSession = ChatbotSessionUtils.updateFluxContext session newContext
                        return ResultUtils.success "Variable assigned successfully" (Some updatedSession)

                    | Error err ->
                        AnsiConsole.MarkupLine($"[red]❌ Math Error: {err}[/]")
                        return ResultUtils.failure $"Math Error: {err}"

                | Some (FunctionExpression (name, args)) ->
                    // Handle mathematical functions
                    match name.ToLower() with
                    | "sqrt" when args.Length = 1 ->
                        match evaluateExpression args.[0] with
                        | Ok result ->
                            match Double.TryParse(result.Output) with
                            | true, value when value >= 0.0 ->
                                let sqrtResult = Math.Sqrt(value)
                                let mathResult = ResultUtils.mathSuccess $"sqrt({result.Output})" (sqrtResult.ToString())
                                
                                let resultContent = $"""[bold green]Mathematical Result:[/]
[yellow]Input:[/] {Markup.Escape(mathResult.Input)}
[yellow]Output:[/] {Markup.Escape(mathResult.Output)}
[dim]Computed using Mathematical Engine[/]"""
                                
                                let resultPanel = Panel(resultContent)
                                resultPanel.Header <- PanelHeader("[bold blue]🧮 Mathematical Engine[/]")
                                resultPanel.Border <- BoxBorder.Rounded
                                AnsiConsole.Write(resultPanel)
                                AnsiConsole.MarkupLine("[green]✅ Mathematical computation completed[/]")

                                return ResultUtils.success "Mathematical computation completed" (Some session)
                            | true, _ -> 
                                return ResultUtils.failure "Cannot take square root of negative number"
                            | false, _ -> 
                                return ResultUtils.failure "Invalid number for square root"
                        | Error err ->
                            return ResultUtils.failure $"Math Error: {err}"
                    | _ ->
                        return ResultUtils.failure $"Unknown mathematical function: {name}"

                | None ->
                    return ResultUtils.failure "Invalid mathematical expression syntax"

            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ Math Exception: {ex.Message}[/]")
                return ResultUtils.failure $"Math Exception: {ex.Message}"
        }

    /// Get mathematical capabilities
    let getMathCapabilities () : string list =
        [
            "Basic arithmetic (+, -, *, /)"
            "Square root (sqrt)"
            "Variable assignment"
            "Expression evaluation"
        ]

    /// Get math help text
    let getMathHelp () : string =
        """[bold blue]Mathematical Engine Help[/]

[yellow]Basic Operations:[/]
  math 2 + 3
  calculate 10 * 5
  solve 100 / 4

[yellow]Functions:[/]
  math sqrt(16)
  calculate sqrt(25)

[yellow]Variable Assignment:[/]
  x = math 2 + 3
  result = calculate 10 * 5

[yellow]Supported Operations:[/]
  + Addition
  - Subtraction
  * Multiplication
  / Division
  sqrt() Square root"""

    /// Validate mathematical expression
    let validateMathExpression (expression: string) : Result<unit, string> =
        try
            match evaluateExpression expression with
            | Ok _ -> Ok ()
            | Error err -> Error err
        with
        | ex -> Error ex.Message

    /// Get mathematical constants
    let getMathConstants () : Map<string, float> =
        Map [
            ("pi", Math.PI)
            ("e", Math.E)
            ("tau", 2.0 * Math.PI)
        ]

    /// Evaluate expression with constants
    let evaluateWithConstants (expression: string) : Result<MathResult, string> =
        try
            let constants = getMathConstants()
            let mutable expr = expression.ToLower()
            
            // Replace constants
            for KeyValue(name, value) in constants do
                expr <- expr.Replace(name, value.ToString())
            
            evaluateExpression expr
        with
        | ex -> Error ex.Message
