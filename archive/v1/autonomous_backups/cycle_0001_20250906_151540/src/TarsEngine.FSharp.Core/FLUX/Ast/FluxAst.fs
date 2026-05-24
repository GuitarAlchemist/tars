namespace TarsEngine.FSharp.FLUX.Ast

open System
open System.Collections.Generic

/// FLUX AST - Simplified version for standalone testing
module FluxAst =

    /// FLUX value types
    type FluxValue =
        | StringValue of string
        | IntValue of int
        | FloatValue of float
        | BoolValue of bool
        | ListValue of FluxValue list
        | MapValue of Map<string, FluxValue>
        | NullValue

    /// FLUX type system
    type FluxType =
        | StringType
        | IntType
        | FloatType
        | BoolType
        | ListType of FluxType
        | MapType of FluxType * FluxType
        | FunctionType of FluxType list * FluxType
        | CustomType of string

    /// Language Block - Multi-modal language execution
    type LanguageBlock = {
        Language: string
        Content: string
        LineNumber: int
        Variables: Map<string, FluxValue>
    }

    /// Meta Block - Script metadata and configuration
    type MetaBlock = {
        Properties: MetaProperty list
        LineNumber: int
    }

    and MetaProperty = {
        Name: string
        Value: FluxValue
    }

    /// FLUX Block types
    type FluxBlock =
        | MetaBlock of MetaBlock
        | LanguageBlock of LanguageBlock

    /// FLUX Script
    type FluxScript = {
        Blocks: FluxBlock list
        FileName: string option
        ParsedAt: DateTime
        Version: string
        Metadata: Map<string, FluxValue>
    }

    /// FLUX Execution Result
    type FluxExecutionResult = {
        Success: bool
        Result: FluxValue option
        ExecutionTime: TimeSpan
        BlocksExecuted: int
        ErrorMessage: string option
        Trace: string list
        GeneratedArtifacts: Map<string, string>
        AgentOutputs: Map<string, FluxValue>
        DiagnosticResults: Map<string, FluxValue>
        ReflectionInsights: string list
    }
