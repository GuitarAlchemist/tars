namespace TarsEngine.FSharp.FLUX.Metaprogramming

open System
open TarsEngine.FSharp.FLUX.Ast.FluxAst

/// Simplified metaprogramming utilities for FLUX
/// (Full Myriad integration will be added later)
module FluxMyriadGenerators =

    /// Generate computation expression from grammar
    let generateComputationExpression (grammarName: string) (grammarContent: string) : string =
        let builderName = sprintf "%sBuilder" grammarName
        let instanceName = grammarName.ToLowerInvariant()

        sprintf "// Generated Computation Expression for %s\n// Grammar: %s\n\ntype %s() =\n    member this.Bind(value, f) = f value\n    member this.Return(value) = value\n    member this.Zero() = ()\n\nlet %s = %s()"
            grammarName grammarContent builderName instanceName builderName
    /// Generate agent implementation
    let generateAgentImplementation (agentName: string) (capabilities: string) (role: string) : string =
        sprintf "// Generated Agent Implementation: %s\n// Role: %s\n// Capabilities: %s\n\ntype %sAgent() =\n    member this.Name = \"%s\"\n    member this.Role = \"%s\"\n    member this.Capabilities = %s"
            agentName role capabilities agentName agentName role capabilities

    printfn "🧬 FLUX Metaprogramming Utilities Loaded"
