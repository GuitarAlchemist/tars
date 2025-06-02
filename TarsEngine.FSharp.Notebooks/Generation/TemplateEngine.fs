namespace TarsEngine.FSharp.Notebooks.Generation

open System
open System.Collections.Generic
open System.Text.RegularExpressions
open TarsEngine.FSharp.Notebooks.Types

/// <summary>
/// Template engine for generating notebook content from templates
/// </summary>

/// Template variable
type TemplateVariable = {
    Name: string
    Value: obj
    Type: VariableType
}

/// Variable types
and VariableType = 
    | String
    | Number
    | Boolean
    | List
    | Object

/// Template context
type TemplateContext = {
    Variables: Map<string, TemplateVariable>
    Functions: Map<string, obj list -> obj>
    Metadata: Map<string, obj>
}

/// Template processing result
type TemplateResult = {
    Success: bool
    Content: string
    Errors: string list
    Warnings: string list
}

/// Template engine
type TemplateEngine() =
    
    /// Process template with context
    member _.ProcessTemplate(template: string, context: TemplateContext) : TemplateResult =
        try
            let mutable content = template
            let errors = ResizeArray<string>()
            let warnings = ResizeArray<string>()
            
            // Process variables
            content <- this.ProcessVariables(content, context.Variables, errors, warnings)
            
            // Process functions
            content <- this.ProcessFunctions(content, context.Functions, errors, warnings)
            
            // Process conditionals
            content <- this.ProcessConditionals(content, context.Variables, errors, warnings)
            
            // Process loops
            content <- this.ProcessLoops(content, context.Variables, errors, warnings)
            
            {
                Success = errors.Count = 0
                Content = content
                Errors = errors |> List.ofSeq
                Warnings = warnings |> List.ofSeq
            }
            
        with
        | ex ->
            {
                Success = false
                Content = template
                Errors = [ex.Message]
                Warnings = []
            }
    
    /// Process variable substitutions
    member private _.ProcessVariables(
        content: string,
        variables: Map<string, TemplateVariable>,
        errors: ResizeArray<string>,
        warnings: ResizeArray<string>) : string =
        
        let pattern = @"\{\{(\w+)\}\}"
        let regex = Regex(pattern)
        
        regex.Replace(content, fun m ->
            let varName = m.Groups.[1].Value
            match variables.TryFind(varName) with
            | Some variable -> this.FormatVariable(variable)
            | None ->
                warnings.Add($"Variable '{varName}' not found in context")
                m.Value
        )
    
    /// Process function calls
    member private _.ProcessFunctions(
        content: string,
        functions: Map<string, obj list -> obj>,
        errors: ResizeArray<string>,
        warnings: ResizeArray<string>) : string =
        
        let pattern = @"\{\{(\w+)\((.*?)\)\}\}"
        let regex = Regex(pattern)
        
        regex.Replace(content, fun m ->
            let funcName = m.Groups.[1].Value
            let argsStr = m.Groups.[2].Value
            
            match functions.TryFind(funcName) with
            | Some func ->
                try
                    let args = this.ParseArguments(argsStr)
                    let result = func args
                    result.ToString()
                with
                | ex ->
                    errors.Add($"Error calling function '{funcName}': {ex.Message}")
                    m.Value
            | None ->
                warnings.Add($"Function '{funcName}' not found in context")
                m.Value
        )
    
    /// Process conditional blocks
    member private _.ProcessConditionals(
        content: string,
        variables: Map<string, TemplateVariable>,
        errors: ResizeArray<string>,
        warnings: ResizeArray<string>) : string =
        
        let pattern = @"\{\%\s*if\s+(\w+)\s*\%\}(.*?)\{\%\s*endif\s*\%\}"
        let regex = Regex(pattern, RegexOptions.Singleline)
        
        regex.Replace(content, fun m ->
            let varName = m.Groups.[1].Value
            let ifContent = m.Groups.[2].Value
            
            match variables.TryFind(varName) with
            | Some variable ->
                if this.IsTruthy(variable) then ifContent else ""
            | None ->
                warnings.Add($"Variable '{varName}' not found in conditional")
                ""
        )
    
    /// Process loop blocks
    member private _.ProcessLoops(
        content: string,
        variables: Map<string, TemplateVariable>,
        errors: ResizeArray<string>,
        warnings: ResizeArray<string>) : string =
        
        let pattern = @"\{\%\s*for\s+(\w+)\s+in\s+(\w+)\s*\%\}(.*?)\{\%\s*endfor\s*\%\}"
        let regex = Regex(pattern, RegexOptions.Singleline)
        
        regex.Replace(content, fun m ->
            let itemVar = m.Groups.[1].Value
            let listVar = m.Groups.[2].Value
            let loopContent = m.Groups.[3].Value
            
            match variables.TryFind(listVar) with
            | Some variable when variable.Type = List ->
                match variable.Value with
                | :? (obj list) as items ->
                    items
                    |> List.mapi (fun i item ->
                        let itemContext = Map.add itemVar {
                            Name = itemVar
                            Value = item
                            Type = this.InferType(item)
                        } variables
                        this.ProcessVariables(loopContent, itemContext, errors, warnings)
                    )
                    |> String.concat ""
                | _ ->
                    errors.Add($"Variable '{listVar}' is not a list")
                    ""
            | Some _ ->
                errors.Add($"Variable '{listVar}' is not a list")
                ""
            | None ->
                warnings.Add($"Variable '{listVar}' not found in loop")
                ""
        )
    
    /// Format variable value
    member private _.FormatVariable(variable: TemplateVariable) : string =
        match variable.Type with
        | String -> variable.Value.ToString()
        | Number -> variable.Value.ToString()
        | Boolean -> if (variable.Value :?> bool) then "true" else "false"
        | List -> 
            match variable.Value with
            | :? (obj list) as items -> String.Join(", ", items)
            | _ -> variable.Value.ToString()
        | Object -> variable.Value.ToString()
    
    /// Check if variable is truthy
    member private _.IsTruthy(variable: TemplateVariable) : bool =
        match variable.Type with
        | Boolean -> variable.Value :?> bool
        | String -> not (String.IsNullOrEmpty(variable.Value.ToString()))
        | Number -> 
            match variable.Value with
            | :? int as i -> i <> 0
            | :? float as f -> f <> 0.0
            | :? decimal as d -> d <> 0m
            | _ -> true
        | List ->
            match variable.Value with
            | :? (obj list) as items -> not items.IsEmpty
            | _ -> true
        | Object -> variable.Value <> null
    
    /// Parse function arguments
    member private _.ParseArguments(argsStr: string) : obj list =
        if String.IsNullOrWhiteSpace(argsStr) then
            []
        else
            argsStr.Split(',')
            |> Array.map (fun arg -> arg.Trim().Trim('"', '\'') :> obj)
            |> List.ofArray
    
    /// Infer variable type from value
    member private _.InferType(value: obj) : VariableType =
        match value with
        | :? string -> String
        | :? bool -> Boolean
        | :? int | :? float | :? decimal -> Number
        | :? (obj list) -> List
        | _ -> Object

/// Template utilities
module TemplateUtils =
    
    /// Create template context
    let createContext variables functions metadata =
        {
            Variables = variables
            Functions = functions
            Metadata = metadata
        }
    
    /// Create template variable
    let createVariable name value varType =
        {
            Name = name
            Value = value
            Type = varType
        }
    
    /// Create string variable
    let stringVar name value = createVariable name (value :> obj) String
    
    /// Create number variable
    let numberVar name value = createVariable name (value :> obj) Number
    
    /// Create boolean variable
    let boolVar name value = createVariable name (value :> obj) Boolean
    
    /// Create list variable
    let listVar name value = createVariable name (value :> obj) List
    
    /// Create object variable
    let objectVar name value = createVariable name (value :> obj) Object
    
    /// Built-in template functions
    let builtInFunctions = Map.ofList [
        ("upper", fun args ->
            match args with
            | [str] -> str.ToString().ToUpper() :> obj
            | _ -> "" :> obj
        )
        
        ("lower", fun args ->
            match args with
            | [str] -> str.ToString().ToLower() :> obj
            | _ -> "" :> obj
        )
        
        ("length", fun args ->
            match args with
            | [str] -> str.ToString().Length :> obj
            | _ -> 0 :> obj
        )
        
        ("date", fun args ->
            match args with
            | [] -> DateTime.Now.ToString("yyyy-MM-dd") :> obj
            | [format] -> DateTime.Now.ToString(format.ToString()) :> obj
            | _ -> DateTime.Now.ToString() :> obj
        )
        
        ("join", fun args ->
            match args with
            | separator :: items -> String.Join(separator.ToString(), items |> List.map (fun x -> x.ToString())) :> obj
            | _ -> "" :> obj
        )
    ]

/// Notebook template generator
type NotebookTemplateGenerator() =
    
    let templateEngine = TemplateEngine()
    
    /// Generate notebook from template
    member _.GenerateFromTemplate(
        templateNotebook: JupyterNotebook,
        context: TemplateContext) : Async<JupyterNotebook> = async {
        
        let processedCells = 
            templateNotebook.Cells
            |> List.map (fun cell ->
                match cell with
                | CodeCell codeData ->
                    let processedSource = 
                        codeData.Source
                        |> List.map (fun line ->
                            let result = templateEngine.ProcessTemplate(line, context)
                            result.Content
                        )
                    CodeCell { codeData with Source = processedSource }
                
                | MarkdownCell markdownData ->
                    let processedSource = 
                        markdownData.Source
                        |> List.map (fun line ->
                            let result = templateEngine.ProcessTemplate(line, context)
                            result.Content
                        )
                    MarkdownCell { markdownData with Source = processedSource }
                
                | RawCell rawData ->
                    let processedSource = 
                        rawData.Source
                        |> List.map (fun line ->
                            let result = templateEngine.ProcessTemplate(line, context)
                            result.Content
                        )
                    RawCell { rawData with Source = processedSource }
            )
        
        // Process metadata
        let processedMetadata = 
            match templateNotebook.Metadata.Title with
            | Some title ->
                let result = templateEngine.ProcessTemplate(title, context)
                { templateNotebook.Metadata with Title = Some result.Content }
            | None -> templateNotebook.Metadata
        
        return {
            templateNotebook with
                Cells = processedCells
                Metadata = processedMetadata
        }
    }
    
    /// Create template context from analysis
    member _.CreateContextFromAnalysis(analysis: MetascriptAnalysis) : TemplateContext =
        let variables = Map.ofList [
            ("title", TemplateUtils.stringVar "title" analysis.Narrative.Title)
            ("objective", TemplateUtils.stringVar "objective" analysis.Narrative.Objective)
            ("agent_count", TemplateUtils.numberVar "agent_count" analysis.Complexity.AgentCount)
            ("action_count", TemplateUtils.numberVar "action_count" analysis.Complexity.ActionCount)
            ("variable_count", TemplateUtils.numberVar "variable_count" analysis.Complexity.VariableCount)
            ("complexity_score", TemplateUtils.numberVar "complexity_score" analysis.Complexity.ComplexityScore)
        ]
        
        TemplateUtils.createContext variables TemplateUtils.builtInFunctions Map.empty
