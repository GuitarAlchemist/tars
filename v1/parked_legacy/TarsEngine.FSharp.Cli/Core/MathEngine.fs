namespace TarsEngine.FSharp.Cli.Core

open System
open System.Text.RegularExpressions
open AngouriMath

/// Powerful free and open source mathematical computation engine for FLUX
/// Uses AngouriMath (MIT License) - full-featured computer algebra system
module MathEngine =

    /// Parse mathematical expressions using AngouriMath
    let parseExpression (expr: string) : Result<obj, string> =
        try
            let parsed = MathS.FromString expr
            Ok (parsed :> obj)
        with
        | ex -> Error $"Parse error: {ex.Message}"

    /// Helper function to extract content between parentheses, handling nested parentheses
    let extractBetweenParentheses (input: string) (startIndex: int) : string * int =
        let mutable parenCount = 0
        let mutable i = startIndex
        let mutable found = false

        // Find opening parenthesis
        while i < input.Length && not found do
            if input.[i] = '(' then
                found <- true
                parenCount <- 1
                i <- i + 1
            else
                i <- i + 1

        let startContent = i

        // Find matching closing parenthesis
        while i < input.Length && parenCount > 0 do
            if input.[i] = '(' then
                parenCount <- parenCount + 1
            elif input.[i] = ')' then
                parenCount <- parenCount - 1
            i <- i + 1

        let content = if startContent < i - 1 then input.Substring(startContent, i - 1 - startContent) else ""
        (content, i)

    /// Parse mathematical operations with standard notation (improved for quality)
    let parseOperation (expr: string) : Result<string * obj * string option, string> =
        let expr = expr.Trim()

        // Handle integrate(expression, variable) or ∫(expression, variable)
        if expr.StartsWith("integrate(") || expr.StartsWith("∫(") then
            let startIdx = if expr.StartsWith("integrate(") then 9 else 1
            let (content, _) = extractBetweenParentheses expr startIdx
            let parts = content.Split(',')
            if parts.Length = 2 then
                let expression = parts.[0].Trim()
                let variable = parts.[1].Trim()
                match parseExpression expression with
                | Ok mathExpr -> Ok ("integrate", mathExpr, Some variable)
                | Error err -> Error $"Could not parse expression '{expression}': {err}"
            else
                Error "Integration syntax: integrate(expression, variable)"

        // Handle diff(expression, variable) or d/dx(expression)
        elif expr.StartsWith("diff(") then
            let (content, _) = extractBetweenParentheses expr 4
            let parts = content.Split(',')
            if parts.Length >= 1 then
                let expression = parts.[0].Trim()
                let variable = if parts.Length > 1 then parts.[1].Trim() else "x"
                match parseExpression expression with
                | Ok mathExpr -> Ok ("differentiate", mathExpr, Some variable)
                | Error err -> Error $"Could not parse expression '{expression}': {err}"
            else
                Error "Differentiation syntax: diff(expression, variable)"

        // Handle solve(equation, variable)
        elif expr.StartsWith("solve(") then
            let (content, _) = extractBetweenParentheses expr 5
            let parts = content.Split(',')
            if parts.Length = 2 then
                let equation = parts.[0].Trim()
                let variable = parts.[1].Trim()
                match parseExpression equation with
                | Ok mathExpr -> Ok ("solve", mathExpr, Some variable)
                | Error err -> Error $"Could not parse equation '{equation}': {err}"
            else
                Error "Equation solving syntax: solve(equation, variable)"

        // Handle limit(expression, variable, value)
        elif expr.StartsWith("limit(") then
            let (content, _) = extractBetweenParentheses expr 5
            let parts = content.Split(',')
            if parts.Length = 3 then
                let expression = parts.[0].Trim()
                let variable = parts.[1].Trim()
                let value = parts.[2].Trim()
                match parseExpression expression with
                | Ok mathExpr -> Ok ("limit", mathExpr, Some $"{variable},{value}")
                | Error err -> Error $"Could not parse expression '{expression}': {err}"
            else
                Error "Limit syntax: limit(expression, variable, value)"

        // Handle simplify(expression)
        elif expr.StartsWith("simplify(") then
            let (content, _) = extractBetweenParentheses expr 8
            match parseExpression content with
            | Ok mathExpr -> Ok ("simplify", mathExpr, None)
            | Error err -> Error $"Could not parse expression '{content}': {err}"

        // Handle simple evaluation
        else
            match parseExpression expr with
            | Ok mathExpr -> Ok ("evaluate", mathExpr, None)
            | Error err -> Error $"Could not parse expression: {err}"



    /// Perform symbolic integration using AngouriMath
    let integrate (expr: obj) (variable: string) : Result<string, string> =
        try
            let entity = expr :?> AngouriMath.Entity
            let varSymbol = MathS.Var variable
            let result = entity.Integrate(varSymbol)
            Ok (result.ToString())
        with
        | ex -> Error $"Integration failed: {ex.Message}"

    /// Perform symbolic differentiation using AngouriMath
    let differentiate (expr: obj) (variable: string) : Result<string, string> =
        try
            let entity = expr :?> AngouriMath.Entity
            let varSymbol = MathS.Var variable
            let result = entity.Differentiate(varSymbol)
            Ok (result.ToString())
        with
        | ex -> Error $"Differentiation failed: {ex.Message}"

    /// Solve equations using AngouriMath
    let solve (expr: obj) (variable: string) : Result<string, string> =
        try
            let entity = expr :?> AngouriMath.Entity
            let varSymbol = MathS.Var variable
            let solutions = entity.SolveEquation(varSymbol)
            let solutionStr = solutions.ToString()
            if String.IsNullOrEmpty(solutionStr) || solutionStr = "{}" then
                Ok "No solutions found"
            else
                Ok solutionStr
        with
        | ex -> Error $"Equation solving failed: {ex.Message}"

    /// Calculate limits using AngouriMath
    let calculateLimit (expr: obj) (variable: string) (value: string) : Result<string, string> =
        try
            let entity = expr :?> AngouriMath.Entity
            let varSymbol = MathS.Var variable
            let limitValue = MathS.FromString value
            let result = entity.Limit(varSymbol, limitValue)
            Ok (result.ToString())
        with
        | ex -> Error $"Limit calculation failed: {ex.Message}"

    /// Simplify expressions using AngouriMath
    let simplify (expr: obj) : string =
        try
            let entity = expr :?> AngouriMath.Entity
            let simplified = entity.Simplify()
            simplified.ToString()
        with
        | _ -> expr.ToString() // Fallback to original if simplification fails

    /// Main computation function using AngouriMath (enhanced for quality)
    let computeExpression (expr: string) : Result<string, string> =
        match parseOperation expr with
        | Ok ("integrate", mathExpr, Some variable) ->
            match integrate mathExpr variable with
            | Ok result -> Ok $"∫ {expr} d{variable} = {result} + C"
            | Error err -> Error err
        | Ok ("differentiate", mathExpr, Some variable) ->
            match differentiate mathExpr variable with
            | Ok result -> Ok $"d/d{variable} [{expr}] = {result}"
            | Error err -> Error err
        | Ok ("solve", mathExpr, Some variable) ->
            match solve mathExpr variable with
            | Ok result -> Ok $"Solutions for {variable}: {result}"
            | Error err -> Error err
        | Ok ("limit", mathExpr, Some varAndValue) ->
            let parts = varAndValue.Split(',')
            if parts.Length = 2 then
                let variable = parts.[0].Trim()
                let value = parts.[1].Trim()
                match calculateLimit mathExpr variable value with
                | Ok result -> Ok $"lim({variable} → {value}) = {result}"
                | Error err -> Error err
            else
                Error "Invalid limit parameters"
        | Ok ("simplify", mathExpr, _) ->
            let simplified = simplify mathExpr
            Ok $"Simplified: {simplified}"
        | Ok ("evaluate", mathExpr, _) ->
            let simplified = simplify mathExpr
            Ok $"Simplified: {simplified}"
        | Error msg -> Error msg
        | _ -> Error "Unsupported operation. Supported: diff(expr, var), integrate(expr, var), solve(equation, var), limit(expr, var, value), simplify(expr)"
