namespace Tars.Metascript

open System
open System.Text.Json
open System.Text.Json.Serialization
open Domain

/// <summary>
/// Parser for Metascript workflow definitions.
/// Supports JSON format with optional YAML-like shortcuts.
/// </summary>
module Parser =

    /// <summary>Parse result with detailed error information</summary>
    type ParseResult<'T> =
        | Success of 'T
        | ParseError of line: int * column: int * message: string
        | ValidationError of errors: string list

    /// <summary>JSON serialization options for workflows</summary>
    let private jsonOptions =
        let opts = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
        opts.Converters.Add(JsonFSharpConverter())
        opts.AllowTrailingCommas <- true
        opts.ReadCommentHandling <- JsonCommentHandling.Skip
        opts

    /// <summary>Parse a workflow from JSON string</summary>
    let parseJson (json: string) : ParseResult<Workflow> =
        try
            let workflow = JsonSerializer.Deserialize<Workflow>(json, jsonOptions)

            match Validation.validateWorkflow workflow with
            | Ok w -> Success w
            | Error errors -> ValidationError errors
        with
        | :? JsonException as ex ->
            // Extract line/column from exception (Nullable<int64>)
            let line =
                if ex.LineNumber.HasValue then
                    int ex.LineNumber.Value
                else
                    0

            let col =
                if ex.BytePositionInLine.HasValue then
                    int ex.BytePositionInLine.Value
                else
                    0

            ParseError(line, col, ex.Message)
        | ex -> ParseError(0, 0, ex.Message)

    /// <summary>Serialize a workflow to JSON</summary>
    let toJson (workflow: Workflow) : string =
        let opts = JsonSerializerOptions(WriteIndented = true)
        opts.Converters.Add(JsonFSharpConverter())
        JsonSerializer.Serialize(workflow, opts)

    /// <summary>Create a simple agent step</summary>
    let agentStep (id: string) (agent: string) (instruction: string) : WorkflowStep =
        { Id = id
          Type = "agent"
          Agent = Some agent
          Tool = None
          Instruction = Some instruction
          Params = None
          Context = None
          DependsOn = None
          Outputs = Some [ "result" ]
          Tools = None }

    /// <summary>Create a tool step</summary>
    let toolStep (id: string) (tool: string) (parameters: Map<string, string>) : WorkflowStep =
        { Id = id
          Type = "tool"
          Agent = None
          Tool = Some tool
          Instruction = None
          Params = Some parameters
          Context = None
          DependsOn = None
          Outputs = Some [ "result" ]
          Tools = None }

    /// <summary>Create a retrieval step</summary>
    let retrievalStep (id: string) (query: string) : WorkflowStep =
        { Id = id
          Type = "retrieval"
          Agent = None
          Tool = None
          Instruction = Some query
          Params = None
          Context = None
          DependsOn = None
          Outputs = Some [ "context" ]
          Tools = None }

    /// <summary>Create a decision step</summary>
    let decisionStep (id: string) (condition: string) : WorkflowStep =
        { Id = id
          Type = "decision"
          Agent = None
          Tool = None
          Instruction = Some condition
          Params = None
          Context = None
          DependsOn = None
          Outputs = Some [ "branch" ]
          Tools = None }

    /// <summary>Create a loop step</summary>
    let loopStep (id: string) (items: string) : WorkflowStep =
        { Id = id
          Type = "loop"
          Agent = None
          Tool = None
          Instruction = Some items
          Params = None
          Context = None
          DependsOn = None
          Outputs = Some [ "item"; "index" ]
          Tools = None }

    /// <summary>Workflow builder for fluent API</summary>
    type WorkflowBuilder(name: string) =
        let mutable description = ""
        let mutable version = "1.0"
        let mutable inputs: WorkflowInput list = []
        let mutable steps: WorkflowStep list = []

        member this.Description(desc: string) =
            description <- desc
            this

        member this.Version(ver: string) =
            version <- ver
            this

        member this.Input(n: string, typ: string, desc: string) =
            inputs <-
                inputs
                @ [ { Name = n
                      Type = typ
                      Description = desc } ]

            this

        member this.Step(step: WorkflowStep) =
            steps <- steps @ [ step ]
            this

        member this.Agent(id: string, agent: string, instruction: string) =
            steps <- steps @ [ agentStep id agent instruction ]
            this

        member this.Tool(id: string, tool: string, parameters: Map<string, string>) =
            steps <- steps @ [ toolStep id tool parameters ]
            this

        member this.Retrieval(id: string, query: string) =
            steps <- steps @ [ retrievalStep id query ]
            this

        member _.Build() : Workflow =
            { Name = name
              Description = description
              Version = version
              Inputs = inputs
              Steps = steps }

    /// <summary>Start building a workflow</summary>
    let workflow (name: string) = WorkflowBuilder(name)
