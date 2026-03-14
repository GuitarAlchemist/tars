namespace TarsEngine.DSL

<<<<<<< HEAD:v1/parked_legacy/TarsEngine.DSL/Ast.fs
open Ast

<<<<<<<< HEAD:v1/parked_legacy/TarsEngine.DSL/Ast.fs
=======
/// Module containing the Abstract Syntax Tree (AST) for the TARS DSL
module Ast =
    /// Property value types in the DSL
    type PropertyValue =
        | StringValue of string
        | NumberValue of float
        | BoolValue of bool
        | ListValue of PropertyValue list
        | ObjectValue of Map<string, PropertyValue>

>>>>>>> origin/main:TarsEngine.DSL/Ast.fs
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
<<<<<<< HEAD:v1/parked_legacy/TarsEngine.DSL/Ast.fs
========
/// Module containing utility functions for the TARS DSL
module Library =
    /// Create a new property value from a string
    let createStringValue (value: string) = StringValue value
>>>>>>>> origin/main:TarsEngine.DSL/Library.fs

    /// Create a new property value from a number
    let createNumberValue (value: float) = NumberValue value

    /// Create a new property value from a boolean
    let createBoolValue (value: bool) = BoolValue value

    /// Create a new property value from a list
    let createListValue (values: PropertyValue list) = ListValue values

    /// Create a new property value from a map
    let createObjectValue (values: Map<string, PropertyValue>) = ObjectValue values

    /// Create a new block
    let createBlock blockType name content properties nestedBlocks =
        {
            Type = blockType
            Name = name
            Content = content
            Properties = properties
            NestedBlocks = nestedBlocks
        }

    /// Create a new program
    let createProgram blocks =
        { Blocks = blocks }
=======

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
>>>>>>> origin/main:TarsEngine.DSL/Ast.fs
