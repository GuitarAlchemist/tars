namespace TarsEngine.DSL

open System
open System.Text.RegularExpressions
open Ast

/// Module containing the expression evaluator for the TARS DSL
module Evaluator =
    /// Evaluate a string with variable interpolation
    let rec evaluateString (str: string) (env: Map<string, PropertyValue>) =
        // Match ${variable} or ${variable.property} patterns
        let regex = Regex(@"\${([^}]+)}")

        // Replace all matches with their evaluated values
        let result = regex.Replace(str, fun m ->
            let expr = m.Groups.[1].Value

            // Check if this is a simple variable or an expression
            if expr.Contains(" + ") then
                // This is an expression like "${a + b}"
                let parts = expr.Split([|" + "|], StringSplitOptions.None)
                if parts.Length = 2 then
                    let var1 = parts.[0].Trim()
                    let var2 = parts.[1].Trim()

                    // Get the values of the variables
                    let value1 =
                        match env.TryFind(var1) with
                        | Some(NumberValue n) -> n
                        | Some(StringValue s) ->
                            match Double.TryParse(s) with
                            | true, n -> n
                            | _ -> 0.0
                        | _ -> 0.0

                    let value2 =
                        match env.TryFind(var2) with
                        | Some(NumberValue n) -> n
                        | Some(StringValue s) ->
                            match Double.TryParse(s) with
                            | true, n -> n
                            | _ -> 0.0
                        | _ ->
                            // Check if var2 is a number literal
                            match Double.TryParse(var2) with
                            | true, n -> n
                            | _ -> 0.0

                    // Calculate the result
                    let sum = value1 + value2
                    sum.ToString()
                else
                    // Just a regular path
                    let path = expr.Split('.')

                    // Get the root variable
                    match env.TryFind(path.[0]) with
                    | Some value ->
                        // If there are property accesses, navigate through them
                        let mutable current = value
                        let mutable success = true

                        for i in 1 .. path.Length - 1 do
                            match current with
                            | ObjectValue props ->
                                match props.TryFind(path.[i]) with
                                | Some v -> current <- v
                                | None ->
                                    success <- false
                                    current <- StringValue("undefined")
                            | ListValue items ->
                                // Try to parse the property as an index
                                match Int32.TryParse(path.[i]) with
                                | true, idx when idx >= 0 && idx < items.Length ->
                                    current <- items.[idx]
                                | _ ->
                                    success <- false
                                    current <- StringValue("undefined")
                            | _ ->
                                success <- false
                                current <- StringValue("undefined")

                        // Convert the final value to a string
                        if success then
                            match current with
                            | StringValue s -> s
                            | NumberValue n -> n.ToString()
                            | BoolValue b -> b.ToString().ToLower()
                            | ListValue _ -> "[array]"
                            | ObjectValue _ -> "[object]"
                        else
                            "undefined"
                    | None -> "undefined"
            else
                // Just a regular path
                let path = expr.Split('.')

                // Get the root variable
                match env.TryFind(path.[0]) with
                | Some value ->
                    // If there are property accesses, navigate through them
                    let mutable current = value
                    let mutable success = true

                    for i in 1 .. path.Length - 1 do
                        match current with
                        | ObjectValue props ->
                            match props.TryFind(path.[i]) with
                            | Some v -> current <- v
                            | None ->
                                success <- false
                                current <- StringValue("undefined")
                        | ListValue items ->
                            // Try to parse the property as an index
                            match Int32.TryParse(path.[i]) with
                            | true, idx when idx >= 0 && idx < items.Length ->
                                current <- items.[idx]
                            | _ ->
                                success <- false
                                current <- StringValue("undefined")
                        | _ ->
                            success <- false
                            current <- StringValue("undefined")

                    // Convert the final value to a string
                    if success then
                        match current with
                        | StringValue s -> s
                        | NumberValue n -> n.ToString()
                        | BoolValue b -> b.ToString().ToLower()
                        | ListValue _ -> "[array]"
                        | ObjectValue _ -> "[object]"
                    else
                        "undefined"
                | None -> "undefined")

        result

    /// Evaluate a boolean expression
    let rec evaluateBooleanExpression (expr: string) (env: Map<string, PropertyValue>) =
        // Handle simple variable references like "${var}"
        if expr.StartsWith("${") && expr.EndsWith("}") then
            let varName = expr.Substring(2, expr.Length - 3)
            match env.TryFind(varName) with
            | Some(BoolValue b) -> b
            | Some(StringValue s) -> not (String.IsNullOrEmpty(s))
            | Some(NumberValue n) -> n <> 0.0
            | Some(ListValue l) -> not (List.isEmpty l)
            | Some(ObjectValue o) -> not (Map.isEmpty o)
            | None -> false
        // Handle equality comparisons like "${var} == 'value'"
        elif expr.Contains("==") then
            let parts = expr.Split([|"=="|], StringSplitOptions.None)
            if parts.Length = 2 then
                let left = evaluateString (parts.[0].Trim()) env
                let right = evaluateString (parts.[1].Trim()) env
                left = right
            else
                false
        // Handle inequality comparisons like "${var} != 'value'"
        elif expr.Contains("!=") then
            let parts = expr.Split([|"!="|], StringSplitOptions.None)
            if parts.Length = 2 then
                let left = evaluateString (parts.[0].Trim()) env
                let right = evaluateString (parts.[1].Trim()) env
                left <> right
            else
                false
        // Handle greater than comparisons like "${var} > 5"
        elif expr.Contains(">") && not (expr.Contains(">=")) then
            let parts = expr.Split([|">"|], StringSplitOptions.None)
            if parts.Length = 2 then
                let left = evaluateString (parts.[0].Trim()) env
                let right = evaluateString (parts.[1].Trim()) env
                match Double.TryParse(left), Double.TryParse(right) with
                | (true, leftNum), (true, rightNum) -> leftNum > rightNum
                | _ -> String.Compare(left, right) > 0
            else
                false
        // Handle less than comparisons like "${var} < 5"
        elif expr.Contains("<") && not (expr.Contains("<=")) then
            let parts = expr.Split([|"<"|], StringSplitOptions.None)
            if parts.Length = 2 then
                let left = evaluateString (parts.[0].Trim()) env
                let right = evaluateString (parts.[1].Trim()) env
                match Double.TryParse(left), Double.TryParse(right) with
                | (true, leftNum), (true, rightNum) -> leftNum < rightNum
                | _ -> String.Compare(left, right) < 0
            else
                false
        // Handle greater than or equal comparisons like "${var} >= 5"
        elif expr.Contains(">=") then
            let parts = expr.Split([|">="|], StringSplitOptions.None)
            if parts.Length = 2 then
                let left = evaluateString (parts.[0].Trim()) env
                let right = evaluateString (parts.[1].Trim()) env
                match Double.TryParse(left), Double.TryParse(right) with
                | (true, leftNum), (true, rightNum) -> leftNum >= rightNum
                | _ -> String.Compare(left, right) >= 0
            else
                false
        // Handle less than or equal comparisons like "${var} <= 5"
        elif expr.Contains("<=") then
            let parts = expr.Split([|"<="|], StringSplitOptions.None)
            if parts.Length = 2 then
                let left = evaluateString (parts.[0].Trim()) env
                let right = evaluateString (parts.[1].Trim()) env
                match Double.TryParse(left), Double.TryParse(right) with
                | (true, leftNum), (true, rightNum) -> leftNum <= rightNum
                | _ -> String.Compare(left, right) <= 0
            else
                false
        // Handle includes checks like "${array}.includes('value')"
        elif expr.Contains(".includes(") && expr.EndsWith(")") then
            let pattern = @"\${([^}]+)}.includes\(([^)]+)\)"
            let m = Regex.Match(expr, pattern)
            if m.Success then
                let arrayName = m.Groups.[1].Value
                let searchValue = evaluateString m.Groups.[2].Value env

                match env.TryFind(arrayName) with
                | Some(ListValue items) ->
                    items |> List.exists (function
                        | StringValue s -> s = searchValue
                        | _ -> false)
                | Some(StringValue s) -> s.Contains(searchValue)
                | _ -> false
            else
                false
        // Handle logical AND like "${var1} && ${var2}"
        elif expr.Contains("&&") then
            let parts = expr.Split([|"&&"|], StringSplitOptions.None)
            parts |> Array.forall (fun p -> evaluateBooleanExpression (p.Trim()) env)
        // Handle logical OR like "${var1} || ${var2}"
        elif expr.Contains("||") then
            let parts = expr.Split([|"||"|], StringSplitOptions.None)
            parts |> Array.exists (fun p -> evaluateBooleanExpression (p.Trim()) env)
        // Handle logical NOT like "!${var}"
        elif expr.StartsWith("!") then
            not (evaluateBooleanExpression (expr.Substring(1).Trim()) env)
        // Handle literal true/false
        elif expr.ToLower() = "true" then
            true
        elif expr.ToLower() = "false" then
            false
        // Default case
        else
            let evaluated = evaluateString expr env
            not (String.IsNullOrEmpty(evaluated) || evaluated = "false" || evaluated = "0" || evaluated = "undefined")

    /// Evaluate a property value with variable interpolation
    let rec evaluatePropertyValue (value: PropertyValue) (env: Map<string, PropertyValue>) =
        match value with
        | StringValue s -> StringValue(evaluateString s env)
        | NumberValue n -> NumberValue(n)  // Numbers don't need interpolation
        | BoolValue b -> BoolValue(b)      // Booleans don't need interpolation
        | ListValue items ->
            ListValue(items |> List.map (fun item -> evaluatePropertyValue item env))
        | ObjectValue props ->
            let evaluatedProps =
                props
                |> Map.toSeq
                |> Seq.map (fun (k, v) -> (k, evaluatePropertyValue v env))
                |> Map.ofSeq
            ObjectValue(evaluatedProps)

    /// Parse a string into a PropertyValue
    let parseValue (str: string) =
        // Try to parse as number
        match Double.TryParse(str) with
        | true, num -> NumberValue(num)
        | _ ->
            // Try to parse as boolean
            match str.ToLower() with
            | "true" -> BoolValue(true)
            | "false" -> BoolValue(false)
            | _ ->
                // Try to parse as JSON array or object
                if (str.StartsWith("[") && str.EndsWith("]")) || (str.StartsWith("{") && str.EndsWith("}")) then
                    try
                        let json = Newtonsoft.Json.JsonConvert.DeserializeObject(str)
                        match json with
                        | :? System.Collections.IList as list ->
                            let items =
                                [for i in 0 .. list.Count - 1 do
                                    match list.[i] with
                                    | :? string as s -> StringValue(s)
                                    | :? int as i -> NumberValue(float i)
                                    | :? double as d -> NumberValue(d)
                                    | :? bool as b -> BoolValue(b)
                                    | _ -> StringValue(list.[i].ToString())]
                            ListValue(items)
                        | :? System.Collections.IDictionary as dict ->
                            let props =
                                dict.Keys
                                |> Seq.cast<obj>
                                |> Seq.map (fun k ->
                                    let key = k.ToString()
                                    let value =
                                        match dict.[k] with
                                        | :? string as s -> StringValue(s)
                                        | :? int as i -> NumberValue(float i)
                                        | :? double as d -> NumberValue(d)
                                        | :? bool as b -> BoolValue(b)
                                        | _ -> StringValue(dict.[k].ToString())
                                    (key, value))
                                |> Map.ofSeq
                            ObjectValue(props)
                        | _ -> StringValue(str)
                    with _ ->
                        StringValue(str)
                else
                    StringValue(str)
