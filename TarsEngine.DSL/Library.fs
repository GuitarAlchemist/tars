namespace TarsEngine.DSL

open Ast

/// Module containing utility functions for the TARS DSL
module Library =
    /// Create a new property value from a string
    let createStringValue (value: string) = StringValue value

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
