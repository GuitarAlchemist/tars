namespace TarsEngine.DSL

/// Module containing the Abstract Syntax Tree (AST) for the TARS DSL
module Ast =
    /// Property value types in the DSL
    type PropertyValue =
        | StringValue of string
        | NumberValue of float
        | BoolValue of bool
        | ListValue of PropertyValue list
        | ObjectValue of Map<string, PropertyValue>

    /// Block types in the DSL
    type BlockType =
        | Config
        | Prompt
        | Action
        | Task
        | Agent
        | AutoImprove
        | Describe
        | SpawnAgent
        | Message
        | SelfImprove
        | Tars
        | Communication
        | Variable
        | If
        | Else
        | For
        | While
        | Function
        | Call
        | Return
        | Import
        | Export
        | Unknown of string

    /// A block in the DSL
    type TarsBlock = {
        Type: BlockType
        Name: string option
        Content: string
        Properties: Map<string, PropertyValue>
        NestedBlocks: TarsBlock list
    }

    /// A TARS program consisting of blocks
    type TarsProgram = {
        Blocks: TarsBlock list
    }

    /// Expression types in the DSL
    type Expression =
        | Literal of PropertyValue
        | Variable of string
        | FunctionCall of string * Expression list
        | BinaryOp of Expression * string * Expression
        | UnaryOp of string * Expression

    /// Statement types in the DSL
    type Statement =
        | Assignment of string * Expression
        | ExpressionStatement of Expression
        | IfStatement of Expression * Statement list * Statement list option
        | ForStatement of string * Expression * Statement list
        | WhileStatement of Expression * Statement list
        | ReturnStatement of Expression option
