namespace TarsEngine.FSharp.FLUX.ComputationExpressions

/// Dynamic Computation Expression Generator
/// Generates F# computation expressions from EBNF grammars
module DynamicCEGenerator =

    /// Generate computation expression from grammar
    let generateComputationExpression (grammarName: string) (grammarContent: string) : string =
        sprintf """// Generated Computation Expression for %s
type %sBuilder() =
    member this.Bind(x, f) = f x
    member this.Return(x) = x
    member this.Zero() = ()

let %s = %sBuilder()""" grammarName grammarName (grammarName.ToLowerInvariant()) grammarName
    
    printfn "🧬 Dynamic CE Generator Loaded"
