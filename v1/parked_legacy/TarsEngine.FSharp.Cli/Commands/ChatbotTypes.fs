namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Collections.Generic

/// FLUX value types for the chatbot
type FluxValue =
    | StringValue of string
    | NumberValue of float
    | BooleanValue of bool
    | ListValue of FluxValue list
    | ObjectValue of Map<string, FluxValue>
    | FunctionValue of (FluxValue list -> FluxValue)

/// FLUX execution context
type FluxContext = {
    Variables: Map<string, FluxValue>
    Functions: Map<string, FluxValue list -> FluxValue>
}

/// Mathematical computation result
type MathResult = {
    Input: string
    Output: string
    Success: bool
    Error: string option
}

/// LLM request for chatbot
type ChatbotLLMRequest = {
    Model: string
    Prompt: string
    SystemPrompt: string option
    Temperature: float option
    MaxTokens: int option
    Context: string option
}

/// LLM response for chatbot
type ChatbotLLMResponse = {
    Content: string
    Success: bool
    Error: string option
    Model: string
}

/// Chatbot session state
type ChatbotSession = {
    FluxContext: FluxContext
    ConversationHistory: (string * string) list
    CurrentModel: string option
}

/// Chatbot command result
type ChatbotResult = {
    Success: bool
    Message: string
    Session: ChatbotSession option
}

/// FLUX command types
type FluxCommand =
    | Assignment of variable: string * expression: string
    | Expression of expression: string
    | FunctionCall of name: string * args: string list
    | Query of query: string

/// Mathematical expression types
type MathExpression =
    | SimpleExpression of string
    | VariableExpression of variable: string * expression: string
    | FunctionExpression of name: string * args: string list

/// Utility functions for FLUX values
module FluxValueUtils =
    
    let rec fluxValueToString (value: FluxValue) : string =
        match value with
        | StringValue s -> s
        | NumberValue n -> n.ToString()
        | BooleanValue b -> b.ToString()
        | ListValue items -> 
            items 
            |> List.map fluxValueToString 
            |> String.concat ", "
            |> sprintf "[%s]"
        | ObjectValue map -> 
            map 
            |> Map.toList 
            |> List.map (fun (k, v) -> sprintf "%s: %s" k (fluxValueToString v))
            |> String.concat ", "
            |> sprintf "{%s}"
        | FunctionValue _ -> "<function>"

    let tryParseNumber (str: string) : FluxValue option =
        match Double.TryParse(str) with
        | true, value -> Some (NumberValue value)
        | false, _ -> None

    let tryParseBoolean (str: string) : FluxValue option =
        match Boolean.TryParse(str) with
        | true, value -> Some (BooleanValue value)
        | false, _ -> None

    let parseFluxValue (str: string) : FluxValue =
        match tryParseNumber str with
        | Some value -> value
        | None ->
            match tryParseBoolean str with
            | Some value -> value
            | None -> StringValue str

/// Utility functions for FLUX context
module FluxContextUtils =
    
    let emptyContext : FluxContext = {
        Variables = Map.empty
        Functions = Map.empty
    }

    let assignVariable (context: FluxContext) (name: string) (value: FluxValue) : FluxContext =
        { context with Variables = Map.add name value context.Variables }

    let getVariable (context: FluxContext) (name: string) : FluxValue option =
        Map.tryFind name context.Variables

    let hasVariable (context: FluxContext) (name: string) : bool =
        Map.containsKey name context.Variables

/// Utility functions for chatbot sessions
module ChatbotSessionUtils =
    
    let emptySession : ChatbotSession = {
        FluxContext = FluxContextUtils.emptyContext
        ConversationHistory = []
        CurrentModel = None
    }

    let addToHistory (session: ChatbotSession) (userInput: string) (botResponse: string) : ChatbotSession =
        { session with ConversationHistory = (userInput, botResponse) :: session.ConversationHistory }

    let setModel (session: ChatbotSession) (model: string) : ChatbotSession =
        { session with CurrentModel = Some model }

    let updateFluxContext (session: ChatbotSession) (context: FluxContext) : ChatbotSession =
        { session with FluxContext = context }

/// Command parsing utilities
module CommandParsingUtils =
    
    let isFluxCommand (input: string) : bool =
        input.Contains("=") || input.StartsWith("flux ") || input.Contains("$")

    let isMathCommand (input: string) : bool =
        input.StartsWith("math ") || input.StartsWith("calculate ") || input.StartsWith("solve ")

    let parseFluxCommand (input: string) : FluxCommand option =
        if input.Contains("=") then
            let parts = input.Split('=', 2)
            if parts.Length = 2 then
                let variable = parts.[0].Trim()
                let expression = parts.[1].Trim()
                Some (Assignment (variable, expression))
            else
                None
        elif input.StartsWith("flux ") then
            let expression = input.Substring(5).Trim()
            Some (Expression expression)
        else
            None

    let parseMathExpression (input: string) : MathExpression option =
        if input.StartsWith("math ") then
            let expression = input.Substring(5).Trim()
            Some (SimpleExpression expression)
        elif input.StartsWith("calculate ") then
            let expression = input.Substring(10).Trim()
            Some (SimpleExpression expression)
        elif input.StartsWith("solve ") then
            let expression = input.Substring(6).Trim()
            Some (SimpleExpression expression)
        else
            None

/// Result creation utilities
module ResultUtils =
    
    let success (message: string) (session: ChatbotSession option) : ChatbotResult =
        { Success = true; Message = message; Session = session }

    let failure (message: string) : ChatbotResult =
        { Success = false; Message = message; Session = None }

    let mathSuccess (input: string) (output: string) : MathResult =
        { Input = input; Output = output; Success = true; Error = None }

    let mathFailure (input: string) (error: string) : MathResult =
        { Input = input; Output = ""; Success = false; Error = Some error }

    let llmSuccess (content: string) (model: string) : ChatbotLLMResponse =
        { Content = content; Success = true; Error = None; Model = model }

    let llmFailure (error: string) (model: string) : ChatbotLLMResponse =
        { Content = ""; Success = false; Error = Some error; Model = model }
